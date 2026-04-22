"""Load reference data (gene annotations, DNA bins, cycle genes) and example data."""

import os
import pandas as pd
import numpy as np
from scipy.io import mmread

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _load_csv(filename):
    return pd.read_csv(os.path.join(_DATA_DIR, filename))


def load_full_anno(genome="hg20"):
    """Load gene annotation DataFrame (abspos, chromosome_name, start_position, end_position, ensembl_gene_id, hgnc_symbol/mgi_symbol, band)."""
    if genome == "hg20":
        return _load_csv("full_anno_hg20.csv")
    elif genome == "mm10":
        return _load_csv("full_anno_mm10.csv")
    else:
        raise ValueError(f"Unsupported genome: {genome}")


def load_dna_bins(genome="hg20"):
    """Load 220KB variable genomic bins (chrom, chrompos, abspos)."""
    if genome == "hg20":
        return _load_csv("DNA_hg20.csv")
    else:
        raise ValueError(f"DNA bins only available for hg20, got {genome}")


def load_cyclegenes():
    """Load cell-cycle gene list."""
    df = _load_csv("cyclegenes.csv")
    return df["gene"].tolist()


def load_example_data():
    """Load the built-in breast tumor example dataset (302 cells, 33694 genes).
    
    Returns
    -------
    pd.DataFrame
        UMI count matrix with genes as rows and cells as columns.
    """
    mtx = mmread(os.path.join(_DATA_DIR, "exp_rawdata_sparse.mtx"))
    genes = open(os.path.join(_DATA_DIR, "exp_rawdata_genes.txt")).read().strip().split("\n")
    barcodes = open(os.path.join(_DATA_DIR, "exp_rawdata_barcodes.txt")).read().strip().split("\n")
    
    dense = np.array(mtx.todense())
    return pd.DataFrame(dense, index=genes, columns=barcodes)
