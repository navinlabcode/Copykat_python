"""Command line interfaces for CopyKAT-Py wrappers."""

import argparse
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy import sparse as sp
from scipy.io import mmread


class TeeStream:
    """Write stream output to both terminal and file."""

    def __init__(self, stream, logfile_handle):
        self.stream = stream
        self.logfile_handle = logfile_handle

    def write(self, data):
        self.stream.write(data)
        self.logfile_handle.write(data)
        if "\n" in data or "\r" in data:
            self.flush()

    def flush(self):
        self.stream.flush()
        self.logfile_handle.flush()


def _add_common_copykat_args(parser):
    """Attach CopyKAT runtime arguments shared by matrix and Python wrappers."""
    parser.add_argument(
        "--id-type",
        default="S",
        choices=["S", "E"],
        help="[optional] Gene ID type: S=Symbol, E=Ensembl (default: S)",
    )
    parser.add_argument(
        "--cell-line",
        default="no",
        choices=["yes", "no"],
        help="[optional] Pure cell line mode (default: no)",
    )
    parser.add_argument(
        "--ngene-chr",
        type=int,
        default=5,
        help="[optional] Min genes per chromosome (default: 5)",
    )
    parser.add_argument(
        "--min-genes",
        type=int,
        default=200,
        help="[optional] Min genes per cell (default: 200)",
    )
    parser.add_argument(
        "--low-dr",
        type=float,
        default=0.05,
        help="[optional] Min detection rate for smoothing (default: 0.05)",
    )
    parser.add_argument(
        "--up-dr",
        type=float,
        default=0.1,
        help="[optional] Min detection rate for segmentation (default: 0.1)",
    )
    parser.add_argument(
        "--win-size",
        type=int,
        default=25,
        help="[optional] Window size for segmentation (default: 25)",
    )
    parser.add_argument(
        "--norm-cells",
        default="",
        help="[optional] File with known normal cell barcodes (one per line)",
    )
    parser.add_argument(
        "--ks-cut",
        type=float,
        default=0.1,
        help="[optional] KS test cutoff for breakpoints (default: 0.1)",
    )
    parser.add_argument(
        "--sample-name",
        default="",
        help="[optional] Sample name prefix for output files",
    )
    parser.add_argument(
        "--distance",
        default="euclidean",
        choices=["euclidean", "pearson", "spearman"],
        help="[optional] Distance metric for clustering (default: euclidean)",
    )
    parser.add_argument(
        "--output-seg",
        action="store_true",
        help="[optional] Output .seg file for IGV",
    )
    parser.add_argument(
        "--plot-genes",
        dest="plot_genes",
        action="store_true",
        default=True,
        help="[optional] Generate the heatmap plot (default: enabled)",
    )
    parser.add_argument(
        "--no-plot-genes",
        dest="plot_genes",
        action="store_false",
        help="[optional] Skip heatmap plotting for faster runs",
    )
    parser.add_argument(
        "--genome",
        default="hg20",
        choices=["hg20", "mm10"],
        help="[optional] Genome assembly (default: hg20)",
    )
    parser.add_argument(
        "--n-cores",
        type=int,
        default=1,
        help="[optional] Number of CPU cores (default: 1)",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=None,
        help="[optional] Adaptive PCA component cap for large clustering steps "
             "(default: automatic by cell count)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=".",
        help="[optional] Output directory (default: current)",
    )


def _add_matrix_metadata_args(parser):
    """Attach metadata CSV arguments used by the matrix-style wrappers."""
    parser.add_argument(
        "--meta",
        default=None,
        metavar="CSV",
        help="[optional] Per-cell annotation CSV for the annotated heatmap. "
             "First column = cell name; remaining columns are drawn as "
             "coloured sidebars.",
    )
    parser.add_argument(
        "--row-split",
        default=None,
        metavar="COLUMN",
        help="[optional] Column in --meta used to split and label heatmap rows. "
             "Defaults to the second column of the CSV when not given.",
    )


def _build_main_parser():
    parser = argparse.ArgumentParser(
        prog="copykat-py",
        description="CopyKAT-Py: Inference of genomic copy number from single cell RNA-seq data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="[required] Input UMI count matrix (.csv, .tsv, .txt, or .mtx)",
    )
    parser.add_argument(
        "--genes",
        default=None,
        help="[optional] Gene names file (required for .mtx input)",
    )
    parser.add_argument(
        "--barcodes",
        default=None,
        help="[optional] Barcode names file (required for .mtx input)",
    )
    _add_common_copykat_args(parser)
    _add_matrix_metadata_args(parser)
    return parser


def _build_matrix_parser():
    parser = argparse.ArgumentParser(
        prog="copykat_matrix",
        description=(
            "Matrix-focused CopyKAT-Py wrapper for raw count matrices plus an "
            "optional metadata CSV, designed for R and 10X-style outputs."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="[required] Input UMI count matrix (.csv, .tsv, .txt, or .mtx)",
    )
    parser.add_argument(
        "--genes",
        default=None,
        help="[optional] Gene names file (required for .mtx input)",
    )
    parser.add_argument(
        "--barcodes",
        default=None,
        help="[optional] Barcode names file (required for .mtx input)",
    )
    _add_common_copykat_args(parser)
    _add_matrix_metadata_args(parser)
    return parser


def _normalize_selected_meta(values):
    """Return selected AnnData obs columns, handling CSV-style input too."""
    if not values:
        return []

    columns = []
    seen = set()
    for raw_value in values:
        for value in str(raw_value).split(","):
            column = value.strip()
            if not column or column in seen:
                continue
            seen.add(column)
            columns.append(column)
    return columns


def _load_matrix_input(input_path, genes_path=None, barcodes_path=None):
    """Load raw matrix input from file paths used by CLI wrappers."""
    input_path = str(input_path)
    if input_path.endswith(".mtx") or input_path.endswith(".mtx.gz"):
        mat = mmread(input_path)

        if genes_path is None or barcodes_path is None:
            input_dir = os.path.dirname(input_path)
            if genes_path is None:
                for gf in ["genes.tsv", "genes.txt", "features.tsv", "features.tsv.gz"]:
                    candidate = os.path.join(input_dir, gf)
                    if os.path.exists(candidate):
                        genes_path = candidate
                        break
            if barcodes_path is None:
                for bf in ["barcodes.tsv", "barcodes.txt"]:
                    candidate = os.path.join(input_dir, bf)
                    if os.path.exists(candidate):
                        barcodes_path = candidate
                        break

        if genes_path:
            genes = pd.read_csv(genes_path, sep="\t", header=None)
            gene_names = genes.iloc[:, -1].values if genes.shape[1] > 1 else genes.iloc[:, 0].values
        else:
            gene_names = [f"gene_{i}" for i in range(mat.shape[0])]

        if barcodes_path:
            barcodes = pd.read_csv(barcodes_path, sep="\t", header=None).iloc[:, 0].values
        else:
            barcodes = np.array([f"cell_{i}" for i in range(mat.shape[1])], dtype=object)

        return {
            "matrix": mat,
            "genes": gene_names,
            "barcodes": barcodes,
        }

    return input_path


def _load_normal_cells(norm_cells_path):
    """Read known-normal barcode names from a text file when provided."""
    if not norm_cells_path or not os.path.exists(norm_cells_path):
        return ""

    with open(norm_cells_path, encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _prepare_output_dir(output_dir):
    """Create and activate the output directory used by a run."""
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    os.chdir(output_path)
    mpl_dir = output_path / ".mplconfig"
    cache_dir = output_path / ".cache"
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    mpl_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    return output_path


def _sample_stub(sample_name, default_name):
    """Build a readable file stem for wrapper-created helper files."""
    cleaned = str(sample_name).strip()
    return cleaned if cleaned else default_name


def _anndata_to_rawmat(adata, layer=None, use_raw=False):
    """Convert an AnnData object to CopyKAT's rawmat structure."""
    if layer and use_raw:
        raise ValueError("--layer and --use-raw cannot be used together")

    if use_raw:
        if adata.raw is None:
            raise ValueError("The supplied AnnData object does not contain adata.raw")
        matrix = adata.raw.X
        gene_names = adata.raw.var_names.astype(str).to_numpy()
        matrix_label = "adata.raw.X"
    elif layer:
        if layer not in adata.layers:
            raise ValueError(
                f"Layer '{layer}' not found in adata.layers. "
                f"Available layers: {list(adata.layers.keys())}"
            )
        matrix = adata.layers[layer]
        gene_names = adata.var_names.astype(str).to_numpy()
        matrix_label = f"adata.layers['{layer}']"
    else:
        matrix = adata.X
        gene_names = adata.var_names.astype(str).to_numpy()
        matrix_label = "adata.X"

    if sp.issparse(matrix):
        matrix_t = matrix.T.tocsc(copy=True).astype(np.float32)
    else:
        matrix_t = sp.csc_matrix(np.asarray(matrix, dtype=np.float32).T)

    rawmat = {
        "matrix": matrix_t,
        "genes": gene_names,
        "barcodes": adata.obs_names.astype(str).to_numpy(),
    }
    return adata, rawmat, matrix_label


def _write_selected_obs_meta_csv(adata, selecting_meta, output_dir, sample_name):
    """Persist selected AnnData obs columns as the metadata CSV expected by copykat()."""
    columns = _normalize_selected_meta(selecting_meta)
    if not columns:
        return None, []

    obs_columns = [str(col) for col in adata.obs.columns.tolist()]
    missing = [column for column in columns if column not in obs_columns]
    if missing:
        raise ValueError(
            f"Requested obs columns not found: {missing}. "
            f"Available obs columns: {obs_columns}"
        )

    meta_df = adata.obs.loc[:, columns].copy()
    meta_df.index = meta_df.index.astype(str)
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    meta_path = output_path / f"{_sample_stub(sample_name, 'copykat_anndata')}_selected_obs_meta.csv"
    meta_df.to_csv(meta_path, index=True, index_label="cell_name")
    return str(meta_path), columns


def _run_copykat_analysis(
    args,
    rawmat,
    *,
    meta_csv=None,
    row_split_col=None,
    input_label=None,
    post_plot_meta=None,
):
    """Run copykat() with consistent logging and output-directory setup."""
    norm_cells_path = os.path.abspath(args.norm_cells) if args.norm_cells else ""
    meta_csv = os.path.abspath(meta_csv) if meta_csv is not None else None
    post_plot_meta = os.path.abspath(post_plot_meta) if post_plot_meta is not None else None

    norm_cell_names = _load_normal_cells(norm_cells_path)
    output_dir = _prepare_output_dir(args.output_dir)

    from copykat_py.copykat import copykat

    log_path = output_dir / "copykat_run.log"
    log_handle = open(log_path, "a", encoding="utf-8")
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = TeeStream(sys.stdout, log_handle)
    sys.stderr = TeeStream(sys.stderr, log_handle)

    print("=" * 80)
    print(f"CopyKAT-Py run started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Working directory: {output_dir}")
    print(f"Input: {input_label or getattr(args, 'input', '<in-memory>')}")
    print(f"Requested cores: {args.n_cores}")
    if args.pca_components is not None:
        print(f"Requested adaptive PCA components: {args.pca_components}")
    if meta_csv is not None:
        print(f"Metadata source: {meta_csv}")
    if row_split_col is not None:
        print(f"Row split column: {row_split_col}")

    try:
        result = copykat(
            rawmat=rawmat,
            id_type=args.id_type,
            cell_line=args.cell_line,
            ngene_chr=args.ngene_chr,
            min_gene_per_cell=args.min_genes,
            LOW_DR=args.low_dr,
            UP_DR=args.up_dr,
            win_size=args.win_size,
            norm_cell_names=norm_cell_names,
            KS_cut=args.ks_cut,
            sam_name=args.sample_name,
            distance=args.distance,
            output_seg=args.output_seg,
            plot_genes=args.plot_genes,
            genome=args.genome,
            n_cores=args.n_cores,
            pca_components=args.pca_components,
            meta_csv=meta_csv,
            row_split_col=row_split_col,
        )

        print("CopyKAT-Py analysis complete.")
        if "prediction" in result:
            pred = result["prediction"]["copykat.pred"].value_counts()
            for key, value in pred.items():
                print(f"  {key}: {value} cells")

        if post_plot_meta and args.plot_genes and "CNAmat" in result:
            print("\nGenerating annotated heatmap...")
            from copykat_py.plotting import plot_heatmap_annotated

            cna_df = result["CNAmat"]
            ann_mat = cna_df.iloc[:, 3:].values.astype(np.float32)
            ann_cell_names = cna_df.columns[3:].tolist()
            ann_chrom_info = cna_df.iloc[:, 0].values
            ann_output = f"{args.sample_name}_copykat_annotated_heatmap.png"
            plot_heatmap_annotated(
                mat=ann_mat,
                cell_names=ann_cell_names,
                chrom_info=ann_chrom_info,
                meta_csv=post_plot_meta,
                row_split_col=args.row_split,
                sample_name=args.sample_name,
                distance=args.distance,
                n_cores=args.n_cores,
                output_path=ann_output,
            )
        return result
    finally:
        print(f"CopyKAT-Py run finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Detailed log saved to: {log_path}")
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        log_handle.close()


def main():
    """Entry point for the legacy matrix-focused ``copykat-py`` CLI."""
    parser = _build_main_parser()
    args = parser.parse_args()
    rawmat = _load_matrix_input(args.input, args.genes, args.barcodes)
    return _run_copykat_analysis(
        args,
        rawmat,
        meta_csv=None,
        row_split_col=None,
        input_label=args.input,
        post_plot_meta=args.meta,
    )


def matrix_main():
    """Entry point for ``copykat_matrix``."""
    parser = _build_matrix_parser()
    args = parser.parse_args()
    rawmat = _load_matrix_input(args.input, args.genes, args.barcodes)
    return _run_copykat_analysis(
        args,
        rawmat,
        meta_csv=args.meta,
        row_split_col=args.row_split,
        input_label=args.input,
        post_plot_meta=None,
    )


def copykat_anndata(
    adata,
    *,
    selecting_meta=None,
    row_split=None,
    sample_name="",
    distance="euclidean",
    genome="hg20",
    n_cores=1,
    output_dir=".",
    layer=None,
    use_raw=False,
    id_type="S",
    cell_line="no",
    ngene_chr=5,
    min_genes=200,
    low_dr=0.05,
    up_dr=0.1,
    win_size=25,
    norm_cells="",
    ks_cut=0.1,
    output_seg=False,
    plot_genes=True,
    pca_components=None,
):
    """Python-friendly AnnData wrapper that accepts an in-memory AnnData object."""
    _, rawmat, matrix_label = _anndata_to_rawmat(
        adata,
        layer=layer,
        use_raw=use_raw,
    )
    meta_csv, selected_meta = _write_selected_obs_meta_csv(
        adata,
        selecting_meta,
        output_dir,
        sample_name,
    )

    if row_split and meta_csv is None:
        raise ValueError("row_split requires selecting_meta")

    row_split_col = row_split
    if meta_csv is not None and row_split_col is None:
        row_split_col = selected_meta[0]

    if selected_meta:
        print(f"Selected obs columns: {selected_meta}")

    args = argparse.Namespace(
        input="<AnnData object>",
        layer=layer,
        use_raw=use_raw,
        selecting_meta=selecting_meta,
        row_split=row_split,
        id_type=id_type,
        cell_line=cell_line,
        ngene_chr=ngene_chr,
        min_genes=min_genes,
        low_dr=low_dr,
        up_dr=up_dr,
        win_size=win_size,
        norm_cells=str(norm_cells) if norm_cells else "",
        ks_cut=ks_cut,
        sample_name=sample_name,
        distance=distance,
        output_seg=output_seg,
        plot_genes=plot_genes,
        genome=genome,
        n_cores=n_cores,
        pca_components=pca_components,
        output_dir=str(output_dir),
    )
    return _run_copykat_analysis(
        args,
        rawmat,
        meta_csv=meta_csv,
        row_split_col=row_split_col,
        input_label=f"<AnnData object> ({matrix_label})",
        post_plot_meta=None,
    )


def plot_main():
    """Entry point for ``copykat-py-plot``: annotated heatmap from CNA results."""
    parser = argparse.ArgumentParser(
        prog="copykat-py-plot",
        description=(
            "Draw an annotated CNA heatmap from a copykat-py CNA results file "
            "and a per-cell metadata CSV. Rows are split and colour-labelled by "
            "a chosen metadata column; cells within each group are ordered by "
            "hierarchical or K-means clustering so intra-group CNA structure is "
            "preserved."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Use second column (leiden_cluster) as row-split (default):
  copykat-py-plot \\
      --cna  sample_copykat_CNA_results.txt \\
      --meta meta.csv

  # Explicitly split rows by inferred_CellType and also annotate leiden_cluster:
  copykat-py-plot \\
      --cna  sample_copykat_CNA_results.txt \\
      --meta xenium_ft_full_meta_celltype_leiden.csv \\
      --row-split inferred_CellType \\
      --sample-name xenium_all_cells \\
      --n-cores 40 \\
      --output xenium_annotated_heatmap.png

Meta CSV format
---------------
  First column : cell name (must match column names in the CNA results file)
  Remaining    : any metadata (cell type, cluster, condition, ...)

  With header:    cell_name,leiden_cluster,inferred_CellType
  Without header: aaaajgij-1,5,lumhr   (first row treated as data)
""",
    )

    parser.add_argument(
        "--cna",
        "-c",
        required=True,
        help="[required] CNA results file produced by copykat-py "
             "(*_copykat_CNA_results.txt, tab-separated). "
             "First three columns must be chrom / chrompos / abspos; "
             "remaining columns are cells.",
    )
    parser.add_argument(
        "--meta",
        "-m",
        required=True,
        help="[required] Annotation CSV. First column = cell name; remaining columns are "
             "drawn as coloured sidebars. Header row is auto-detected.",
    )
    parser.add_argument(
        "--row-split",
        default=None,
        metavar="COLUMN",
        help="[optional] Metadata column used to split and label rows. "
             "Defaults to the second column of the CSV when not supplied.",
    )
    parser.add_argument(
        "--sample-name",
        default="",
        help="[optional] Label shown in the figure title and used as the output filename "
             "prefix when --output is not given.",
    )
    parser.add_argument(
        "--distance",
        default="euclidean",
        choices=["euclidean", "pearson", "spearman"],
        help="[optional] Distance metric for within-group cell clustering (default: euclidean).",
    )
    parser.add_argument(
        "--n-cores",
        type=int,
        default=1,
        help="[optional] Parallel threads for clustering (default: 1).",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        metavar="PATH",
        help="[optional] Output PNG path. Defaults to "
             "{sample_name}_copykat_annotated_heatmap.png.",
    )

    args = parser.parse_args()

    print(f"Loading CNA results: {args.cna}")
    cna_df = pd.read_csv(args.cna, sep="\t", index_col=False)
    cell_names = cna_df.columns[3:].tolist()
    chrom_info = cna_df.iloc[:, 0].values
    mat = cna_df.iloc[:, 3:].values.astype(np.float32)
    print(f"  {mat.shape[1]} cells x {mat.shape[0]} bins")

    from copykat_py.plotting import plot_heatmap_annotated

    plot_heatmap_annotated(
        mat=mat,
        cell_names=cell_names,
        chrom_info=chrom_info,
        meta_csv=args.meta,
        row_split_col=args.row_split,
        sample_name=args.sample_name,
        distance=args.distance,
        n_cores=args.n_cores,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
