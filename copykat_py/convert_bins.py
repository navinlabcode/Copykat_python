"""Convert gene-by-cell CNA matrix to genomic-bin-by-cell matrix.

Mirrors convert.all.bins.hg20.R from the R package.
Maps gene-level copy number values into 220KB variable genomic bins.
"""

import numpy as np
import pandas as pd
import os
from joblib import Parallel, delayed
from copykat_py.data_loader import load_full_anno, load_dna_bins

_LAST_PAR_INFO = {
    "step": "convert_to_bins",
    "parallel": False,
    "requested_cores": 1,
    "effective_cores": 1,
    "tasks": 0,
    "chunk_size": 0,
}


def get_last_convert_bins_info():
    return dict(_LAST_PAR_INFO)


def convert_to_bins(RNA_mat, genome="hg20", n_cores=1):
    """Convert gene-by-cell CNA results to 220KB genomic bins.
    
    Parameters
    ----------
    RNA_mat : pd.DataFrame
        CNA results with columns: abspos, chromosome_name, start_position, 
        end_position, ensembl_gene_id, hgnc_symbol (or mgi_symbol), band, cell1, cell2, ...
    genome : str
        "hg20" (bins conversion only supported for hg20).
    n_cores : int
        Number of parallel workers.
    
    Returns
    -------
    dict with keys:
        'DNA_adj': pd.DataFrame - DNA bin coordinates
        'RNA_adj': pd.DataFrame - CNA values at genomic bins (chrom, chrompos, abspos, cell1, ...)
    """
    if genome != "hg20":
        # For mm10, return gene-level results (no bin conversion, same as R)
        return None
    
    full_anno = load_full_anno("hg20")
    DNA_mat = load_dna_bins("hg20")
    
    # Remove chromosome 24 (Y)
    DNA = DNA_mat[DNA_mat["chrom"] != 24].copy().reset_index(drop=True)
    
    end = DNA["chrompos"].values
    start = np.concatenate([[0], end[:-1]])
    
    # Determine gene symbol column
    if "hgnc_symbol" in RNA_mat.columns:
        symbol_col = "hgnc_symbol"
    elif "mgi_symbol" in RNA_mat.columns:
        symbol_col = "mgi_symbol"
    else:
        raise ValueError("Cannot find gene symbol column in RNA_mat")
    
    # Cell data columns (after the 7 annotation columns)
    anno_cols = ["abspos", "chromosome_name", "start_position", "end_position",
                 "ensembl_gene_id", symbol_col, "band"]
    cell_cols = [c for c in RNA_mat.columns if c not in anno_cols]
    RNA_values = RNA_mat[cell_cols].values.astype(np.float32, copy=False)  # shape: (n_genes, n_cells)
    gene_symbols = RNA_mat[symbol_col].values
    
    # Map genes in RNA_mat to indices for fast lookup.
    gene_to_idx = {}
    for i, g in enumerate(gene_symbols):
        if g not in gene_to_idx:
            gene_to_idx[g] = []
        gene_to_idx[g].append(i)

    # Pre-compute which RNA rows belong to each DNA bin while preserving
    # the original R logic of using the full annotation table for bin membership.
    bin_gene_indices = [[] for _ in range(len(DNA))]
    chrom_values = DNA["chrom"].values

    for chrom_id in np.unique(chrom_values):
        dna_idx = np.where(chrom_values == chrom_id)[0]
        if len(dna_idx) == 0:
            continue

        sub = full_anno[full_anno["chromosome_name"] == chrom_id].copy()
        if sub.empty:
            continue

        sub["center"] = 0.5 * (sub["start_position"] + sub["end_position"])
        centers = sub["center"].to_numpy()
        symbols = sub["hgnc_symbol"].to_numpy()
        order = np.argsort(centers, kind="mergesort")
        centers = centers[order]
        symbols = symbols[order]

        left_edges = start[dna_idx]
        right_edges = end[dna_idx]
        left_idx = np.searchsorted(centers, left_edges, side="left")
        right_idx = np.searchsorted(centers, right_edges, side="right")

        for local_i, bin_i in enumerate(dna_idx):
            if right_idx[local_i] <= left_idx[local_i]:
                continue
            row_indices = []
            for symbol in symbols[left_idx[local_i]:right_idx[local_i]]:
                row_indices.extend(gene_to_idx.get(symbol, ()))
            if row_indices:
                bin_gene_indices[bin_i] = row_indices

    def _process_bin(i):
        row_idx = bin_gene_indices[i]
        if row_idx:
            return np.median(RNA_values[row_idx, :], axis=0)
        return None
    
    n_bins = len(DNA)
    max_cores = int(os.getenv("COPYKAT_MAX_CORES", str(os.cpu_count() or 1)))
    n_jobs = max(1, min(int(n_cores), max_cores, n_bins))
    chunk_size = max(1, n_bins // (n_jobs * 4))
    _LAST_PAR_INFO.update({
        "parallel": n_jobs > 1,
        "requested_cores": int(n_cores),
        "effective_cores": int(n_jobs),
        "tasks": int(n_bins),
        "chunk_size": int(chunk_size),
    })

    def _process_chunk(start, end):
        return [_process_bin(i) for i in range(start, end)]

    ranges = [(s, min(s + chunk_size, n_bins)) for s in range(0, n_bins, chunk_size)]
    chunk_results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_process_chunk)(s, e) for s, e in ranges
    )
    results = [r for chunk in chunk_results for r in chunk]
    
    # Separate bins with data from empty bins
    n_cells = len(cell_cols)
    RNA_adj = np.zeros((len(DNA), n_cells))
    valid_mask = np.zeros(len(DNA), dtype=bool)
    
    for i, r in enumerate(results):
        if r is not None:
            RNA_adj[i, :] = r
            valid_mask[i] = True
    
    # Fill empty bins with nearest valid bin
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) > 0 and len(valid_indices) < len(DNA):
        empty_indices = np.where(~valid_mask)[0]
        for ei in empty_indices:
            distances = np.abs(valid_indices - ei)
            nearest = valid_indices[np.argmin(distances)]
            RNA_adj[ei, :] = RNA_adj[nearest, :]
    
    # Build output DataFrame
    RNA_adj_df = pd.DataFrame(RNA_adj, columns=cell_cols)
    RNA_adj_df = pd.concat([DNA[["chrom", "chrompos", "abspos"]].reset_index(drop=True), RNA_adj_df], axis=1)
    
    return {
        "DNA_adj": DNA,
        "RNA_adj": RNA_adj_df,
    }
