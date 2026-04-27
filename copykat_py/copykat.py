"""Main CopyKAT function: end-to-end copy number inference from scRNA-seq data.

Faithfully reimplements the R copykat() function workflow:
1. Read and filter data
2. Annotate gene coordinates
3. Freeman-Tukey transformation + DLM smoothing
4. Baseline estimation (normal cell identification)
5. MCMC segmentation with KS breakpoint detection
6. Convert to genomic bins (hg20)
7. Baseline adjustment
8. Final prediction (aneuploid vs diploid)
9. Output files + heatmap
"""

import time
import os
import json
import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy import sparse
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster
import pickle

try:
    import pyarrow as pa
    import pyarrow.csv as pa_csv
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

from copykat_py.annotation import annotate_genes
from copykat_py.smoothing import dlm_smooth, get_last_dlm_smooth_info
from copykat_py.baseline import (
    AUTO_PCA_CELL_COUNT_CUTOFF,
    AUTO_PCA_LARGE_SAMPLE,
    AUTO_PCA_SMALL_SAMPLE,
    FULL_CLUSTER_MAX_CELLS,
    baseline_norm_cl,
    baseline_gmm,
    baseline_synthetic,
    _hierarchical_cluster,
    get_last_cluster_info,
    resolve_adaptive_pca_components,
)
from copykat_py.segmentation import cna_mcmc, get_last_cna_mcmc_info
from copykat_py.convert_bins import convert_to_bins, get_last_convert_bins_info
from copykat_py.data_loader import load_cyclegenes


def _write_cna_csv(df, path, float_fmt="%.6f"):
    """Write CNA DataFrame as TSV using pyarrow (fast) with 6 d.p. float precision.

    Falls back to pandas .to_csv() when pyarrow is not available.
    """
    if HAS_PYARROW:
        anno_cols = [c for c in df.columns if not pd.api.types.is_float_dtype(df[c])]
        float_cols = [c for c in df.columns if c not in anno_cols]
        rounded = df.copy()
        if float_cols:
            rounded[float_cols] = rounded[float_cols].round(6)
        table = pa.Table.from_pandas(rounded, preserve_index=False)
        with open(path, "wb") as f:
            pa_csv.write_csv(table, f, write_options=pa_csv.WriteOptions(delimiter="\t"))
    else:
        df.to_csv(path, sep="\t", index=False, float_format=float_fmt)


def _meta_with_pred(meta_csv, pred_dict, sample_name):
    """Read *meta_csv*, append copykat-py predictions as the last column.

    Returns the path to a new CSV written alongside the original outputs.
    Cells absent from *pred_dict* receive ``"not.defined"``.
    """
    import pandas as pd
    meta = pd.read_csv(meta_csv)
    cell_col = meta.columns[0]
    meta = meta.set_index(cell_col)
    if pred_dict is not None:
        meta["copykat_pred_py"] = meta.index.map(pred_dict).fillna("not.defined")
    else:
        meta["copykat_pred_py"] = "not.defined"
    out_path = f"{sample_name}meta_with_pred.csv"
    meta.reset_index().to_csv(out_path, index=False)
    return out_path


def _run_plot_heatmap(mat_adj, chrom_info, predictions, sample_name, distance, n_cores, WNS1, WNS, output_path):
    from copykat_py.plotting import plot_heatmap

    plot_heatmap(
        mat_adj, chrom_info,
        predictions=predictions,
        sample_name=sample_name,
        distance=distance,
        n_cores=n_cores,
        WNS1=WNS1,
        WNS=WNS,
        output_path=output_path,
    )


def _load_matrix(rawmat):
    """Load raw matrix from various input formats.
    
    Supports: pd.DataFrame, scipy sparse, numpy array, dict (matrix/genes/barcodes), or file path (mtx/csv/tsv).
    """
    if isinstance(rawmat, dict):
        # Dict format from CLI with sparse matrix + gene/barcode names
        mat = rawmat.get("matrix")
        genes = rawmat.get("genes")
        barcodes = rawmat.get("barcodes")
        
        if mat is None:
            raise ValueError("Dict input must contain 'matrix' key")
        
        # Convert to dense if sparse
        if hasattr(mat, "toarray"):
            mat = mat.toarray()
        
        df = pd.DataFrame(mat, index=genes, columns=barcodes)
        return df
    elif isinstance(rawmat, pd.DataFrame):
        return rawmat
    elif isinstance(rawmat, np.ndarray):
        return pd.DataFrame(rawmat)
    elif hasattr(rawmat, "toarray"):
        # scipy sparse
        return pd.DataFrame(rawmat.toarray())
    elif isinstance(rawmat, str):
        if rawmat.endswith(".mtx") or rawmat.endswith(".mtx.gz"):
            mat = mmread(rawmat)
            return pd.DataFrame(mat.toarray() if hasattr(mat, "toarray") else mat)
        elif rawmat.endswith(".csv"):
            return pd.read_csv(rawmat, index_col=0)
        elif rawmat.endswith(".tsv") or rawmat.endswith(".txt"):
            return pd.read_csv(rawmat, sep="\t", index_col=0)
        else:
            return pd.read_csv(rawmat, sep="\t", index_col=0)
    else:
        raise ValueError(f"Unsupported rawmat type: {type(rawmat)}")


def _format_seconds(seconds):
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, rem = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {rem:.1f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {rem:.1f}s"


def _record_step(runtime_info, step, start_time, parallel_info=None, extra=None):
    elapsed = time.perf_counter() - start_time
    entry = {"step": step, "seconds": round(float(elapsed), 4)}
    if parallel_info:
        entry["parallel"] = bool(parallel_info.get("parallel", False))
        entry["requested_cores"] = int(parallel_info.get("requested_cores", 1))
        entry["effective_cores"] = int(parallel_info.get("effective_cores", 1))
        for key in ("tasks", "chunk_size", "mc_samples", "engine", "approximate"):
            if key in parallel_info:
                entry[key] = parallel_info[key]
    if extra:
        entry.update(extra)
    runtime_info["steps"].append(entry)
    return elapsed


def _preN_to_names(preN, cell_names):
    if preN is None:
        return set()

    if isinstance(preN, (str, bytes)):
        return {str(preN)}

    values = list(preN)
    if not values:
        return set()

    names = []
    for value in values:
        if isinstance(value, (int, np.integer)):
            idx = int(value)
            if 0 <= idx < len(cell_names):
                names.append(cell_names[idx])
        else:
            names.append(str(value))
    return set(names)


def _assign_binary_labels(cluster_labels, scores, high_label, low_label):
    labels = np.empty(len(cluster_labels), dtype=object)
    labels[:] = ""
    cluster_vals = np.asarray(sorted(set(cluster_labels)))
    score_arr = np.asarray(scores, dtype=float)
    max_score = np.max(score_arr)
    min_score = np.min(score_arr)

    for cluster_val in cluster_vals[score_arr == max_score]:
        labels[cluster_labels == cluster_val] = high_label
    for cluster_val in cluster_vals[score_arr == min_score]:
        labels[cluster_labels == cluster_val] = low_label
    return labels


def _aggregate_duplicate_genes_sparse(mat, genes):
    genes = np.asarray(genes, dtype=object)
    unique_genes, inverse = np.unique(genes.astype(str), return_inverse=True)
    if len(unique_genes) == len(genes):
        return mat, genes

    row_map = sparse.csr_matrix(
        (np.ones(len(genes), dtype=np.float64), (inverse, np.arange(len(genes)))),
        shape=(len(unique_genes), len(genes)),
    )
    return row_map @ mat, unique_genes.astype(object)


def _aggregate_duplicate_genes_frame(df):
    if not df.index.has_duplicates:
        return df

    # Match R rowsum(..., reorder=TRUE): sum duplicate symbols and sort groups.
    return df.groupby(level=0, sort=True).sum()


def _prepare_input_dataframe(rawmat, min_gene_per_cell, low_dr):
    if isinstance(rawmat, dict):
        mat = rawmat.get("matrix")
        genes = np.asarray(rawmat.get("genes"))
        barcodes = np.asarray(rawmat.get("barcodes"))
        if mat is None:
            raise ValueError("Dict input must contain 'matrix' key")
        if sparse.issparse(mat):
            mat = mat.tocsc(copy=False)
            mat, genes = _aggregate_duplicate_genes_sparse(mat, genes)
            mat = mat.tocsc(copy=False)
            original_cell_names = barcodes.tolist()
            genes_per_cell = np.asarray(mat.getnnz(axis=0)).ravel()
            if (genes_per_cell > min_gene_per_cell).sum() == 0:
                raise ValueError("No cells have more than min_gene_per_cell genes")
            keep_cells = genes_per_cell >= min_gene_per_cell
            filtered_cells = int((~keep_cells).sum())
            mat = mat[:, keep_cells]
            barcodes = barcodes[keep_cells]
            detection_rate = np.asarray(mat.getnnz(axis=1)).ravel() / max(mat.shape[1], 1)
            keep_genes = detection_rate > low_dr
            filtered_gene_rows = int((~keep_genes).sum())
            mat = mat[keep_genes, :]
            genes = genes[keep_genes]
            df = pd.DataFrame(mat.toarray(), index=genes.tolist(), columns=barcodes.tolist())
            return df, original_cell_names, {
                "input_type": "sparse_dict",
                "filtered_cells": filtered_cells,
                "filtered_gene_rows": filtered_gene_rows,
            }

    loaded = _load_matrix(rawmat)
    loaded = _aggregate_duplicate_genes_frame(loaded)
    original_cell_names = list(loaded.columns)
    genes_per_cell = (loaded > 0).sum(axis=0)
    if (genes_per_cell > min_gene_per_cell).sum() == 0:
        raise ValueError("No cells have more than min_gene_per_cell genes")
    low_gene_cells = genes_per_cell < min_gene_per_cell
    filtered_cells = int(low_gene_cells.sum())
    if filtered_cells > 0:
        loaded = loaded.loc[:, ~low_gene_cells]
    detection_rate = (loaded > 0).sum(axis=1) / loaded.shape[1]
    keep_genes = detection_rate > low_dr
    filtered_gene_rows = int((~keep_genes).sum())
    if keep_genes.sum() >= 1:
        loaded = loaded.loc[keep_genes]
    return loaded, original_cell_names, {
        "input_type": "dense",
        "filtered_cells": filtered_cells,
        "filtered_gene_rows": filtered_gene_rows,
    }


def _keep_cells_by_chr_coverage(values, chroms, ngene_chr):
    chrom_codes, unique_chroms = pd.factorize(chroms, sort=False)
    nonzero = values != 0
    counts = np.vstack([
        nonzero[chrom_codes == chrom_idx].sum(axis=0)
        for chrom_idx in range(len(unique_chroms))
    ])
    keep = (counts.sum(axis=0) >= 5) & (counts > 0).all(axis=0) & (counts.min(axis=0) >= ngene_chr)
    return keep


def copykat(rawmat, id_type="S", cell_line="no", ngene_chr=5, min_gene_per_cell=200,
            LOW_DR=0.05, UP_DR=0.1, win_size=25, norm_cell_names="",
            KS_cut=0.1, sam_name="", distance="euclidean", output_seg=False,
            plot_genes=True, genome="hg20", n_cores=1, pca_components=None,
            meta_csv=None, row_split_col=None):
    """Run CopyKAT analysis: infer copy number profiles from scRNA-seq data.
    
    Parameters
    ----------
    rawmat : pd.DataFrame, np.ndarray, scipy.sparse, or str
        UMI count matrix (genes in rows, cells in columns).
        If str, path to .mtx, .csv, or .tsv file.
    id_type : str
        Gene ID type: "S" for Symbol, "E" for Ensembl.
    cell_line : str
        "yes" for pure cell line data, "no" for tumor/normal mixture.
    ngene_chr : int
        Minimum number of genes per chromosome for cell filtering.
    min_gene_per_cell : int
        Minimum genes detected per cell.
    LOW_DR : float
        Minimum gene detection rate for smoothing.
    UP_DR : float
        Minimum gene detection rate for segmentation.
    win_size : int
        Window size for MCMC segmentation.
    norm_cell_names : str or list
        Known normal cell barcodes ("" for auto-detection).
    KS_cut : float
        KS test cutoff for breakpoint detection (0 to 1).
    sam_name : str
        Sample name prefix for output files.
    distance : str
        Distance metric: "euclidean", "pearson", or "spearman".
    output_seg : bool
        Whether to output .seg file for IGV.
    plot_genes : bool
        Whether to plot gene-level heatmap.
    genome : str
        "hg20" or "mm10".
    n_cores : int
        Number of CPU cores for parallel computation.
    pca_components : int or None
        Adaptive PCA component cap for large clustering steps. When omitted,
        CopyKAT-Py uses the built-in rule: 256 PCs for fewer than 50,000
        input cells, otherwise 128 PCs.
    meta_csv : str or None
        Path to a per-cell annotation CSV for the annotated heatmap.
        First column = cell name; remaining columns become coloured annotation
        sidebars.  When provided (and ``plot_genes=True``), an annotated
        heatmap is saved as ``{sam_name}annotated_heatmap.png`` in addition to
        the standard heatmap.  Header row is auto-detected.
    row_split_col : str or None
        Column in ``meta_csv`` used to split and label heatmap rows.
        Defaults to the second column when ``None``.

    Returns
    -------
    dict with keys:
        'prediction': pd.DataFrame with columns [cell.names, copykat.pred]
        'CNAmat': pd.DataFrame with CNA results
        'hclustering': linkage matrix or cluster labels
    """
    start_time = time.perf_counter()
    np.random.seed(1234)
    sample_name = f"{sam_name}_copykat_"
    runtime_info = {
        "sample_name": sam_name,
        "requested_cores": int(n_cores),
        "available_cores": int(os.cpu_count() or 1),
        "steps": [],
    }
    
    print("running copykat-py v1.0.0")
    
    # =========================================================================
    # Step 1: Read and filter data
    # =========================================================================
    print("step 1: read and filter data ...")
    step_start = time.perf_counter()
    rawmat, original_cell_names, prep_stats = _prepare_input_dataframe(rawmat, min_gene_per_cell, LOW_DR)
    input_cell_count = int(len(original_cell_names))
    selected_pca_components = resolve_adaptive_pca_components(input_cell_count, pca_components=pca_components)
    runtime_info["pca_components"] = int(selected_pca_components)
    runtime_info["pca_selection_mode"] = "manual" if pca_components is not None else "auto_by_input_cell_count"
    runtime_info["pca_selection_input_cells"] = input_cell_count
    runtime_info["pca_selection_cutoff"] = AUTO_PCA_CELL_COUNT_CUTOFF
    runtime_info["pca_selection_small_sample"] = AUTO_PCA_SMALL_SAMPLE
    runtime_info["pca_selection_large_sample"] = AUTO_PCA_LARGE_SAMPLE
    print(f"  {rawmat.shape[0]} genes, {rawmat.shape[1]} cells in raw data")
    print(
        f"  adaptive PCA components: {selected_pca_components} "
        f"({'manual override' if pca_components is not None else f'auto from input cell count {input_cell_count}'})"
    )
    if prep_stats["filtered_cells"] > 0:
        print(f"  filtered out {prep_stats['filtered_cells']} cells with <= {min_gene_per_cell} genes; remaining {rawmat.shape[1]} cells")
    print(f"  {rawmat.shape[0]} genes past LOW_DR filtering")
    
    WNS1 = "data quality is ok"
    if rawmat.shape[0] < 7000:
        WNS1 = "low data quality"
        UP_DR = LOW_DR
        print("  WARNING: low data quality; assigned LOW_DR to UP_DR...")
    elapsed = _record_step(runtime_info, "read_and_filter", step_start, extra=prep_stats)
    print(f"  step 1 runtime: {_format_seconds(elapsed)}")
    
    # =========================================================================
    # Step 2: Annotate gene coordinates
    # =========================================================================
    print("step 2: annotating gene coordinates ...")
    step_start = time.perf_counter()
    anno_mat = annotate_genes(rawmat, id_type=id_type, genome=genome)
    
    # =========================================================================
    # Step 3: Remove cell cycle genes and HLA genes (hg20 only)
    # =========================================================================
    if genome == "hg20":
        symbol_col = "hgnc_symbol"
        cyclegenes = load_cyclegenes()
        hla_genes = anno_mat[symbol_col][anno_mat[symbol_col].str.startswith("HLA-", na=False)].tolist()
        genes_to_remove = set(cyclegenes) | set(hla_genes)
        anno_mat = anno_mat[~anno_mat[symbol_col].isin(genes_to_remove)].reset_index(drop=True)
    else:
        symbol_col = "mgi_symbol"
    elapsed = _record_step(runtime_info, "annotate_genes", step_start, extra={"genes_after_annotation": int(anno_mat.shape[0])})
    print(f"  step 2 runtime: {_format_seconds(elapsed)}")
    
    # Secondary cell filtering: ensure each cell has genes across chromosomes
    anno_cols = ["abspos", "chromosome_name", "start_position", "end_position",
                 "ensembl_gene_id", symbol_col, "band"]
    cell_cols = [c for c in anno_mat.columns if c not in anno_cols]
    step_start = time.perf_counter()
    expr_values = anno_mat[cell_cols].to_numpy()
    keep_cells = _keep_cells_by_chr_coverage(expr_values, anno_mat["chromosome_name"].values, ngene_chr)
    if keep_cells.sum() == 0:
        raise ValueError("All cells are filtered out")
    if not np.all(keep_cells):
        cell_cols = [cell_cols[i] for i in np.where(keep_cells)[0]]
        anno_mat = pd.concat([anno_mat[anno_cols], anno_mat[cell_cols]], axis=1)
    rawmat3 = anno_mat[cell_cols].to_numpy(dtype=np.float64, copy=False)
    _record_step(runtime_info, "cell_filter_pre_smoothing", step_start, extra={"cells_after_filter": int(len(cell_cols))})
    
    # Freeman-Tukey transformation: log(sqrt(x) + sqrt(x+1))
    step_start = time.perf_counter()
    norm_mat = np.log(np.sqrt(rawmat3) + np.sqrt(rawmat3 + 1))
    # Center each cell
    norm_mat = norm_mat - norm_mat.mean(axis=0, keepdims=True)
    _record_step(runtime_info, "freeman_tukey_transform", step_start, extra={"matrix_shape": list(norm_mat.shape)})
    
    print(f"  {norm_mat.shape[0]} genes, {norm_mat.shape[1]} cells after preprocessing")
    
    # =========================================================================
    # Step 3: DLM smoothing
    # =========================================================================
    print("step 3: smoothing data with DLM ...")
    step_start = time.perf_counter()
    norm_mat_smooth = dlm_smooth(norm_mat, n_cores=n_cores)
    dlm_info = get_last_dlm_smooth_info()
    elapsed = _record_step(runtime_info, "dlm_smoothing", step_start, parallel_info=dlm_info)
    print(
        f"  smoothing runtime: {_format_seconds(elapsed)} "
        f"(parallel={dlm_info['parallel']}, cores={dlm_info['effective_cores']})"
    )
    
    # =========================================================================
    # Step 4: Measure baselines
    # =========================================================================
    print("step 4: measuring baselines ...")
    step_start = time.perf_counter()
    
    cell_name_list = cell_cols
    
    if cell_line == "yes":
        print("  running pure cell line mode")
        relt = baseline_synthetic(
            norm_mat_smooth,
            min_cells=10,
            n_cores=n_cores,
            pca_components=selected_pca_components,
        )
        norm_mat_relat = relt["expr_relat"]
        CL = relt["cl"]
        WNS = "run with cell line mode"
        preN = None
    elif isinstance(norm_cell_names, list) and len(norm_cell_names) > 1:
        # Known normal cells provided
        known_normal_mask = np.array([c in norm_cell_names for c in cell_name_list])
        NNN = known_normal_mask.sum()
        print(f"  {NNN} known normal cells found in dataset")
        
        if NNN == 0:
            raise ValueError("Known normal cells provided but none found in dataset")
        
        print("  run with known normal...")
        basel = np.median(norm_mat_smooth[:, known_normal_mask], axis=1)
        
        # Cluster all cells
        data_t = norm_mat_smooth.T
        step4_reduce = data_t.shape[0] > FULL_CLUSTER_MAX_CELLS
        km = 6
        CL, Z = _hierarchical_cluster(
            data_t,
            km,
            method="ward",
            metric="euclidean",
            n_cores=n_cores,
            reduce=step4_reduce,
            pca_components=selected_pca_components,
        )
        
        while not all(np.bincount(CL)[np.bincount(CL) > 0] > 5):
            km -= 1
            if Z is not None:
                CL = fcluster(Z, t=km, criterion="maxclust")
            else:
                CL, Z = _hierarchical_cluster(
                    data_t,
                    km,
                    method="ward",
                    metric="euclidean",
                    n_cores=n_cores,
                    reduce=step4_reduce,
                    pca_components=selected_pca_components,
                )
            if km == 2:
                break
        
        WNS = "run with known normal"
        preN = np.asarray(cell_name_list, dtype=object)[known_normal_mask].tolist()
        norm_mat_relat = norm_mat_smooth - basel[:, np.newaxis]
    else:
        # Auto-detect normal cells
        basa = baseline_norm_cl(
            norm_mat_smooth,
            min_cells=5,
            n_cores=n_cores,
            cell_names=cell_name_list,
            pca_components=selected_pca_components,
        )
        basel = basa["basel"]
        WNS = basa["WNS"]
        preN = basa["preN"]
        CL = basa["cl"]
        
        if WNS == "unclassified.prediction":
            cluster_preN = list(preN) if preN is not None else []
            keep_cluster_anchor = (
                WNS1 == "low data quality"
                and len(cluster_preN) >= max(50, int(0.05 * len(cell_name_list)))
            )
            if keep_cluster_anchor:
                print("  low-data-quality mode: keeping cluster-based normal anchor")
            else:
                basa = baseline_gmm(
                    norm_mat_smooth,
                    cell_name_list,
                    max_normal=5,
                    mu_cut=0.05,
                    Nfraq_cut=0.99,
                    RE_before=basa,
                    n_cores=n_cores,
                    pca_components=selected_pca_components,
                )
                basel = basa["basel"]
                WNS = basa["WNS"]
                preN = basa["preN"]
        
        norm_mat_relat = norm_mat_smooth - basel[:, np.newaxis]
    baseline_cluster_info = get_last_cluster_info()
    elapsed = _record_step(runtime_info, "baseline_estimation", step_start, parallel_info=baseline_cluster_info, extra={"warning": WNS})
    print(
        f"  baseline runtime: {_format_seconds(elapsed)} "
        f"(parallel={baseline_cluster_info['parallel']}, cores={baseline_cluster_info['effective_cores']}, "
        f"engine={baseline_cluster_info.get('engine', 'n/a')})"
    )
    
    # =========================================================================
    # Apply stricter gene filtering for segmentation
    # =========================================================================
    step_start = time.perf_counter()
    DR2 = (rawmat3 > 0).sum(axis=1) / rawmat3.shape[1]
    seg_mask = DR2 >= UP_DR
    norm_mat_relat = norm_mat_relat[seg_mask, :]
    anno_mat2 = anno_mat.iloc[seg_mask].reset_index(drop=True)
    
    # Filter cells again with the reduced gene set
    keep_cells2 = _keep_cells_by_chr_coverage(anno_mat2[cell_cols].to_numpy(), anno_mat2["chromosome_name"].values, ngene_chr)
    if keep_cells2.sum() == 0:
        raise ValueError("All cells are filtered out after UP_DR filtering")
    if not np.all(keep_cells2):
        norm_mat_relat = norm_mat_relat[:, keep_cells2]
        cell_cols_seg = [cell_cols[i] for i in np.where(keep_cells2)[0]]
        CL_filtered = CL[keep_cells2] if len(CL) == len(cell_cols) else CL
    else:
        cell_cols_seg = cell_cols
        CL_filtered = CL
    _record_step(runtime_info, "cell_filter_pre_segmentation", step_start, extra={"cells_after_filter": int(len(cell_cols_seg)), "genes_after_filter": int(norm_mat_relat.shape[0])})
    
    # Ensure CL alignment
    if len(CL_filtered) != norm_mat_relat.shape[1]:
        # Recompute if shape mismatch
        data_t = norm_mat_relat.T
        step4_reduce = data_t.shape[0] > FULL_CLUSTER_MAX_CELLS
        CL_filtered, _ = _hierarchical_cluster(
            data_t,
            min(6, norm_mat_relat.shape[1]),
            method="ward",
            metric="euclidean",
            n_cores=n_cores,
            reduce=step4_reduce,
            pca_components=selected_pca_components,
        )
    
    # =========================================================================
    # Step 5: Segmentation
    # =========================================================================
    print("step 5: segmentation ...")
    step_start = time.perf_counter()
    results = cna_mcmc(CL_filtered, norm_mat_relat, bins=win_size, cut_cor=KS_cut, n_cores=n_cores)
    
    if len(results["breaks"]) < 25:
        print("  too few breakpoints; decreased KS_cut to 50%")
        results = cna_mcmc(CL_filtered, norm_mat_relat, bins=win_size, cut_cor=0.5 * KS_cut, n_cores=n_cores)
    
    if len(results["breaks"]) < 25:
        print("  too few breakpoints; decreased KS_cut to 25%")
        results = cna_mcmc(CL_filtered, norm_mat_relat, bins=win_size, cut_cor=0.25 * KS_cut, n_cores=n_cores)
    
    if len(results["breaks"]) < 25:
        raise ValueError("Too few segments; try decreasing KS_cut or improving data quality")
    seg_info = get_last_cna_mcmc_info()
    elapsed = _record_step(runtime_info, "segmentation", step_start, parallel_info=seg_info, extra={"breakpoints": int(len(results["breaks"]))})
    print(
        f"  segmentation runtime: {_format_seconds(elapsed)} "
        f"(parallel={seg_info['parallel']}, cores={seg_info['effective_cores']}, engine={seg_info.get('engine', 'n/a')})"
    )
    
    results_com = results["logCNA"]
    # Center each cell
    results_com = results_com - results_com.mean(axis=0, keepdims=True)
    
    # Save gene-by-cell CNA results
    RNA_copycat = anno_mat2[anno_cols].copy()
    cna_df = pd.DataFrame(results_com, columns=cell_cols_seg)
    RNA_copycat = pd.concat([RNA_copycat.reset_index(drop=True), cna_df], axis=1)
    
    step_start = time.perf_counter()
    RNA_copycat.to_csv(f"{sample_name}CNA_raw_results_gene_by_cell.txt", sep="\t", index=False)
    _record_step(runtime_info, "write_gene_level_output", step_start, extra={"rows": int(RNA_copycat.shape[0]), "cols": int(RNA_copycat.shape[1])})
    
    # =========================================================================
    # Step 6: Convert to genomic bins (hg20 only)
    # =========================================================================
    if genome == "hg20":
        print("step 6: convert to genomic bins ...")
        step_start = time.perf_counter()
        Aj = convert_to_bins(RNA_copycat, genome=genome, n_cores=n_cores)
        convert_info = get_last_convert_bins_info()
        elapsed = _record_step(runtime_info, "convert_to_bins", step_start, parallel_info=convert_info)
        print(
            f"  bin conversion runtime: {_format_seconds(elapsed)} "
            f"(parallel={convert_info['parallel']}, cores={convert_info['effective_cores']})"
        )
        
        uber_mat_adj = Aj["RNA_adj"].iloc[:, 3:].values  # skip chrom, chrompos, abspos
        chrom_info = Aj["DNA_adj"]["chrom"].values
        
        print("step 7: adjust baseline ...")
        step_start = time.perf_counter()
        step7_reduce = uber_mat_adj.shape[1] > FULL_CLUSTER_MAX_CELLS
        
        if cell_line == "yes":
            mat_adj = uber_mat_adj.copy()
        else:
            # First hierarchical clustering for initial prediction
            labels, Z = _hierarchical_cluster(
                uber_mat_adj.T,
                2,
                method="ward",
                metric="euclidean",
                n_cores=n_cores,
                reduce=step7_reduce,
                pca_components=selected_pca_components,
            )
            hc_umap = labels
            
            # Determine which cluster is normal based on preN enrichment
            if preN is not None and len(preN) > 0:
                preN_names = _preN_to_names(preN, cell_name_list)
                cl_ID = []
                for cl_val in sorted(set(hc_umap)):
                    cli_names = [cell_cols_seg[j] for j in range(len(cell_cols_seg)) if hc_umap[j] == cl_val]
                    pid = len(set(cli_names) & preN_names) / max(len(cli_names), 1)
                    cl_ID.append(pid)
                com_pred = _assign_binary_labels(hc_umap, cl_ID, "diploid", "aneuploid")
            else:
                # If no preN, assign based on total CNA magnitude
                cl_mag = []
                for cl_val in sorted(set(hc_umap)):
                    mask = hc_umap == cl_val
                    cl_mag.append(np.mean(np.abs(uber_mat_adj[:, mask])))
                com_pred = _assign_binary_labels(hc_umap, -np.asarray(cl_mag, dtype=float), "diploid", "aneuploid")
            
            # Baseline adjustment: subtract diploid mean, then denoise
            diploid_mask = com_pred == "diploid"
            if diploid_mask.sum() > 0:
                results_com_rat = uber_mat_adj - uber_mat_adj[:, diploid_mask].mean(axis=1, keepdims=True)
                results_com_rat = results_com_rat - results_com_rat.mean(axis=0, keepdims=True)
                
                results_com_rat_norm = results_com_rat[:, diploid_mask]
                cf_h = np.std(results_com_rat_norm, axis=1)
                base = np.mean(results_com_rat_norm, axis=1)
                noise_mask = np.abs(results_com_rat - base[:, np.newaxis]) <= (0.25 * cf_h)[:, np.newaxis]
                cell_means = results_com_rat.mean(axis=0, keepdims=True)
                adj_results = np.where(noise_mask, cell_means, results_com_rat)
                
                mat_adj = adj_results - adj_results.mean(axis=0, keepdims=True)
            else:
                mat_adj = uber_mat_adj.copy()
        cluster_info = get_last_cluster_info()
        elapsed = _record_step(runtime_info, "baseline_adjustment", step_start, parallel_info=cluster_info, extra={"warning": WNS})
        print(
            f"  step 7 runtime: {_format_seconds(elapsed)} "
            f"(parallel={cluster_info['parallel']}, cores={cluster_info['effective_cores']}, engine={cluster_info.get('engine', 'n/a')})"
        )

        # =========================================================================
        # Step 8: Final prediction
        # =========================================================================
        print("step 8: final prediction ...")
        step_start = time.perf_counter()
        step8_reduce = mat_adj.shape[1] > FULL_CLUSTER_MAX_CELLS
        if cell_line != "yes":
            labels_final, Z_final = _hierarchical_cluster(
                mat_adj.T,
                2,
                method="ward",
                metric="euclidean",
                n_cores=n_cores,
                reduce=step8_reduce,
                pca_components=selected_pca_components,
            )
            hc_final = labels_final
            
            if preN is not None and len(preN) > 0:
                preN_names = _preN_to_names(preN, cell_name_list)
                cl_ID_final = []
                for cl_val in sorted(set(hc_final)):
                    cli_names = [cell_cols_seg[j] for j in range(len(cell_cols_seg)) if hc_final[j] == cl_val]
                    pid = len(set(cli_names) & preN_names) / max(len(cli_names), 1)
                    cl_ID_final.append(pid)
                com_preN = _assign_binary_labels(hc_final, cl_ID_final, "diploid", "aneuploid")
            else:
                cl_mag = []
                for cl_val in sorted(set(hc_final)):
                    mask = hc_final == cl_val
                    cl_mag.append(np.mean(np.abs(mat_adj[:, mask])))
                com_preN = _assign_binary_labels(hc_final, -np.asarray(cl_mag, dtype=float), "diploid", "aneuploid")
            
            if WNS == "unclassified.prediction":
                com_preN = np.where(com_preN == "diploid", "c1:diploid:low.conf", com_preN)
                com_preN = np.where(com_preN == "aneuploid", "c2:aneuploid:low.conf", com_preN)
        else:
            labels, Z = _hierarchical_cluster(
                mat_adj.T,
                2,
                method="ward",
                metric="euclidean",
                n_cores=n_cores,
                reduce=step8_reduce,
                pca_components=selected_pca_components,
            )
            labels_final, Z_final = labels, Z
        cluster_info = get_last_cluster_info()
        elapsed = _record_step(runtime_info, "final_prediction", step_start, parallel_info=cluster_info, extra={"warning": WNS})
        print(
            f"  step 8 runtime: {_format_seconds(elapsed)} "
            f"(parallel={cluster_info['parallel']}, cores={cluster_info['effective_cores']}, engine={cluster_info.get('engine', 'n/a')})"
        )
        
        # =========================================================================
        # Step 9: Save results
        # =========================================================================
        pred_dict = None
        if cell_line != "yes":
            pred_dict = {cell_cols_seg[i]: com_preN[i] for i in range(len(cell_cols_seg))}
            for cell in original_cell_names:
                if cell not in pred_dict:
                    pred_dict[cell] = "not.defined"

        print("step 9: saving results ...")
        step_start = time.perf_counter()
        
        if cell_line != "yes":
            res = pd.DataFrame({
                "cell.names": list(pred_dict.keys()),
                "copykat.pred": list(pred_dict.values()),
            })
            res.to_csv(f"{sample_name}prediction.txt", sep="\t", index=False)
        
        # Save CNA results
        cna_out = Aj["RNA_adj"].copy()
        cna_out.iloc[:, 3:] = mat_adj
        _write_cna_csv(cna_out, f"{sample_name}CNA_results.txt")
        
        # Save clustering
        clustering_data = {"labels": labels_final if cell_line != "yes" else labels,
                          "Z": Z_final if cell_line != "yes" else Z}
        with open(f"{sample_name}clustering_results.pkl", "wb") as f:
            pickle.dump(clustering_data, f)
        elapsed = _record_step(runtime_info, "write_final_outputs", step_start, extra={"bins": int(cna_out.shape[0]), "cells": int(cna_out.shape[1] - 3)})
        print(f"  step 9 runtime: {_format_seconds(elapsed)}")
        
        # =========================================================================
        # Step 10: Plot heatmap
        # =========================================================================
        if plot_genes:
            print("step 10: plotting heatmap ...")
            plot_step_start = time.perf_counter()
            predictions = pred_dict if cell_line != "yes" else None
            _run_plot_heatmap(
                mat_adj,
                chrom_info,
                predictions,
                sample_name,
                distance,
                n_cores,
                WNS1,
                WNS,
                f"{sample_name}heatmap.png",
            )
            elapsed = _record_step(runtime_info, "plot_heatmap", plot_step_start)
            print(f"  step 10 runtime: {_format_seconds(elapsed)}")

        if plot_genes and meta_csv is not None:
            print("step 10b: plotting annotated heatmap ...")
            step_ann = time.perf_counter()
            from copykat_py.plotting import plot_heatmap_annotated
            meta_pred_path = _meta_with_pred(meta_csv, pred_dict, sample_name)
            plot_heatmap_annotated(
                mat=mat_adj,
                cell_names=cna_out.columns[3:].tolist(),
                chrom_info=chrom_info,
                meta_csv=meta_pred_path,
                row_split_col=row_split_col,
                sample_name=sample_name,
                distance=distance,
                n_cores=n_cores,
                output_path=f"{sample_name}annotated_heatmap.png",
            )
            elapsed = _record_step(runtime_info, "plot_annotated_heatmap", step_ann)
            print(f"  step 10b runtime: {_format_seconds(elapsed)}")

        # =========================================================================
        # Output SEG file
        # =========================================================================
        if output_seg:
            print("  generating seg files for IGV viewer")
            _write_seg_file(Aj["RNA_adj"], mat_adj, cell_cols_seg, sample_name)
        runtime_info["total_seconds"] = round(time.perf_counter() - start_time, 4)
        with open(f"{sample_name}runtime.json", "w", encoding="utf-8") as f:
            json.dump(runtime_info, f, indent=2)
        print(f"Done. Elapsed time: {_format_seconds(runtime_info['total_seconds'])}")
        print(f"Runtime report saved to: {sample_name}runtime.json")
        
        if cell_line == "yes":
            return {
                "CNAmat": cna_out,
                "hclustering": clustering_data,
                "runtime": runtime_info,
            }
        else:
            return {
                "prediction": res,
                "CNAmat": cna_out,
                "hclustering": clustering_data,
                "runtime": runtime_info,
            }
    
    else:
        # mm10: no bin conversion, use gene-level results directly
        uber_mat_adj = results_com.copy()
        chrom_info = anno_mat2["chromosome_name"].values
        
        print("step 7: adjust baseline ...")
        step_start = time.perf_counter()
        step7_reduce = uber_mat_adj.shape[1] > FULL_CLUSTER_MAX_CELLS
        # Same prediction logic as hg20 (mirroring the R code mm10 section)
        labels, Z = _hierarchical_cluster(
            uber_mat_adj.T,
            2,
            method="ward",
            metric="euclidean",
            n_cores=n_cores,
            reduce=step7_reduce,
            pca_components=selected_pca_components,
        )
        hc_umap = labels
        
        if preN is not None and len(preN) > 0:
            preN_names = _preN_to_names(preN, cell_name_list)
            
            cl_ID = []
            for cl_val in sorted(set(hc_umap)):
                cli_names = [cell_cols_seg[j] for j in range(len(cell_cols_seg)) if hc_umap[j] == cl_val]
                pid = len(set(cli_names) & preN_names) / max(len(cli_names), 1)
                cl_ID.append(pid)
        else:
            cl_ID = []
            for cl_val in sorted(set(hc_umap)):
                mask = hc_umap == cl_val
                cl_ID.append(np.mean(np.abs(uber_mat_adj[:, mask])))
        
        if preN is not None and len(preN) > 0:
            com_pred = _assign_binary_labels(hc_umap, cl_ID, "diploid", "aneuploid")
        else:
            com_pred = _assign_binary_labels(hc_umap, -np.asarray(cl_ID, dtype=float), "diploid", "aneuploid")
        
        # Baseline adjustment
        diploid_mask = com_pred == "diploid"
        if diploid_mask.sum() > 0:
            results_com_rat = uber_mat_adj - uber_mat_adj[:, diploid_mask].mean(axis=1, keepdims=True)
            results_com_rat = results_com_rat - results_com_rat.mean(axis=0, keepdims=True)
            results_com_rat_norm = results_com_rat[:, diploid_mask]
            cf_h = np.std(results_com_rat_norm, axis=1)
            base = np.mean(results_com_rat_norm, axis=1)
            noise_mask = np.abs(results_com_rat - base[:, np.newaxis]) <= (0.25 * cf_h)[:, np.newaxis]
            cell_means = results_com_rat.mean(axis=0, keepdims=True)
            adj_results = np.where(noise_mask, cell_means, results_com_rat)
            mat_adj = adj_results - adj_results.mean(axis=0, keepdims=True)
        else:
            mat_adj = uber_mat_adj.copy()
        cluster_info = get_last_cluster_info()
        elapsed = _record_step(runtime_info, "baseline_adjustment", step_start, parallel_info=cluster_info, extra={"warning": WNS})
        print(
            f"  step 7 runtime: {_format_seconds(elapsed)} "
            f"(parallel={cluster_info['parallel']}, cores={cluster_info['effective_cores']}, engine={cluster_info.get('engine', 'n/a')})"
        )
        
        # Final prediction
        print("step 8: final prediction ...")
        step_start = time.perf_counter()
        step8_reduce = mat_adj.shape[1] > FULL_CLUSTER_MAX_CELLS
        labels_final, Z_final = _hierarchical_cluster(
            mat_adj.T,
            2,
            method="ward",
            metric="euclidean",
            n_cores=n_cores,
            reduce=step8_reduce,
            pca_components=selected_pca_components,
        )
        hc_final = labels_final
        
        if preN is not None and len(preN) > 0:
            preN_names = _preN_to_names(preN, cell_name_list)
            cl_ID_final = []
            for cl_val in sorted(set(hc_final)):
                cli_names = [cell_cols_seg[j] for j in range(len(cell_cols_seg)) if hc_final[j] == cl_val]
                pid = len(set(cli_names) & preN_names) / max(len(cli_names), 1)
                cl_ID_final.append(pid)
            com_preN = _assign_binary_labels(hc_final, cl_ID_final, "diploid", "aneuploid")
        else:
            cl_mag = []
            for cl_val in sorted(set(hc_final)):
                mask = hc_final == cl_val
                cl_mag.append(np.mean(np.abs(mat_adj[:, mask])))
            com_preN = _assign_binary_labels(hc_final, -np.asarray(cl_mag, dtype=float), "diploid", "aneuploid")
        
        if WNS == "unclassified.prediction":
            com_preN = np.where(com_preN == "diploid", "c1:diploid:low.conf", com_preN)
            com_preN = np.where(com_preN == "aneuploid", "c2:aneuploid:low.conf", com_preN)
        cluster_info = get_last_cluster_info()
        elapsed = _record_step(runtime_info, "final_prediction", step_start, parallel_info=cluster_info, extra={"warning": WNS})
        print(
            f"  step 8 runtime: {_format_seconds(elapsed)} "
            f"(parallel={cluster_info['parallel']}, cores={cluster_info['effective_cores']}, engine={cluster_info.get('engine', 'n/a')})"
        )
        
        # Save
        print("step 9: saving results ...")
        step_start = time.perf_counter()
        pred_dict = {cell_cols_seg[i]: com_preN[i] for i in range(len(cell_cols_seg))}
        for cell in original_cell_names:
            if cell not in pred_dict:
                pred_dict[cell] = "not.defined"
        
        res = pd.DataFrame({
            "cell.names": list(pred_dict.keys()),
            "copykat.pred": list(pred_dict.values()),
        })
        res.to_csv(f"{sample_name}prediction.txt", sep="\t", index=False)
        
        cna_df = pd.DataFrame(mat_adj, columns=cell_cols_seg)
        cna_out = pd.concat([anno_mat2[anno_cols].reset_index(drop=True), cna_df], axis=1)
        _write_cna_csv(cna_out, f"{sample_name}CNA_results.txt")
        
        clustering_data = {"labels": labels_final, "Z": Z_final}
        with open(f"{sample_name}clustering_results.pkl", "wb") as f:
            pickle.dump(clustering_data, f)
        elapsed = _record_step(runtime_info, "write_final_outputs", step_start, extra={"bins": int(cna_out.shape[0]), "cells": int(len(cell_cols_seg))})
        print(f"  step 9 runtime: {_format_seconds(elapsed)}")
        
        chrom_numeric = pd.to_numeric(anno_mat2["chromosome_name"], errors="coerce").fillna(0).values
        if plot_genes:
            print("step 10: plotting heatmap ...")
            step_start = time.perf_counter()
            from copykat_py.plotting import plot_heatmap
            plot_heatmap(
                mat_adj, chrom_numeric,
                predictions=pred_dict,
                sample_name=sample_name,
                distance=distance,
                n_cores=n_cores,
                WNS1=WNS1, WNS=WNS,
                output_path=f"{sample_name}heatmap.png"
            )
            elapsed = _record_step(runtime_info, "plot_heatmap", step_start)
            print(f"  step 10 runtime: {_format_seconds(elapsed)}")

        if plot_genes and meta_csv is not None:
            print("step 10b: plotting annotated heatmap ...")
            step_ann = time.perf_counter()
            from copykat_py.plotting import plot_heatmap_annotated
            meta_pred_path = _meta_with_pred(meta_csv, pred_dict, sample_name)
            plot_heatmap_annotated(
                mat=mat_adj,
                cell_names=cna_out.columns[3:].tolist(),
                chrom_info=chrom_numeric,
                meta_csv=meta_pred_path,
                row_split_col=row_split_col,
                sample_name=sample_name,
                distance=distance,
                n_cores=n_cores,
                output_path=f"{sample_name}annotated_heatmap.png",
            )
            elapsed = _record_step(runtime_info, "plot_annotated_heatmap", step_ann)
            print(f"  step 10b runtime: {_format_seconds(elapsed)}")
        runtime_info["total_seconds"] = round(time.perf_counter() - start_time, 4)
        with open(f"{sample_name}runtime.json", "w", encoding="utf-8") as f:
            json.dump(runtime_info, f, indent=2)
        print(f"Done. Elapsed time: {_format_seconds(runtime_info['total_seconds'])}")
        print(f"Runtime report saved to: {sample_name}runtime.json")
        
        return {
            "prediction": res,
            "CNAmat": cna_out,
            "hclustering": clustering_data,
            "runtime": runtime_info,
        }


def _write_seg_file(RNA_adj_df, mat_adj, cell_cols, sample_name):
    """Write .seg file for IGV visualization."""
    rows = []
    chroms = RNA_adj_df["chrom"].values
    chrompos = RNA_adj_df["chrompos"].values
    unique_chroms = np.unique(chroms)
    
    for ci, cell in enumerate(cell_cols):
        vals = mat_adj[:, ci]
        for chrom_val in unique_chroms:
            mask = chroms == chrom_val
            sub_vals = vals[mask]
            sub_pos = chrompos[mask]
            
            # RLE encoding
            if len(sub_vals) == 0:
                continue
            
            changes = np.where(np.diff(sub_vals) != 0)[0] + 1
            starts = np.concatenate([[0], changes])
            ends = np.concatenate([changes, [len(sub_vals)]])
            
            for s, e in zip(starts, ends):
                rows.append({
                    "ID": cell,
                    "chrom": chrom_val,
                    "loc.start": sub_pos[s],
                    "loc.end": sub_pos[e - 1],
                    "num.mark": e - s,
                    "seg.mean": sub_vals[s],
                })
    
    seg_df = pd.DataFrame(rows)
    seg_df.to_csv(f"{sample_name}CNA_results.seg", sep="\t", index=False)
