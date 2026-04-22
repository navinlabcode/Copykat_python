"""Gene annotation: map genes to genomic coordinates, mirroring annotateGenes.hg20.R / annotateGenes.mm10.R."""

import numpy as np
import pandas as pd
from copykat_py.data_loader import load_full_anno


def annotate_genes(mat, id_type="S", genome="hg20"):
    """Annotate gene expression matrix with genomic coordinates.

    Parameters
    ----------
    mat : pd.DataFrame
        Gene expression matrix, genes in rows, cells in columns.
    id_type : str
        "S" for gene Symbol, "E" for Ensembl ID.
    genome : str
        "hg20" or "mm10".

    Returns
    -------
    pd.DataFrame
        Combined annotation + expression, sorted by abspos.
        Columns: abspos, chromosome_name, start_position, end_position,
                 ensembl_gene_id, hgnc_symbol (or mgi_symbol), band, <cell1>, <cell2>, ...
    """
    print("  start annotation ...")
    full_anno = load_full_anno(genome)

    if genome == "mm10":
        symbol_col = "mgi_symbol"
    else:
        symbol_col = "hgnc_symbol"

    if id_type.upper().startswith("E"):
        id_col = "ensembl_gene_id"
    else:
        id_col = symbol_col

    # Intersect genes
    shared = list(set(mat.index) & set(full_anno[id_col]))
    if len(shared) == 0:
        raise ValueError("No shared genes found between input matrix and annotation.")

    mat = mat.loc[mat.index.isin(shared)]
    anno = full_anno[full_anno[id_col].isin(shared)].copy()
    anno = anno.drop_duplicates(subset=id_col)
    # Align annotation to matrix order
    anno = anno.set_index(id_col).reindex(mat.index)
    anno.index.name = id_col
    anno = anno.reset_index()

    # Combine annotation + expression
    data = pd.concat([anno.reset_index(drop=True), mat.reset_index(drop=True)], axis=1)

    # R's order(abspos) preserves tied gene order; this matters for mm10, where
    # abspos is chromosome-level and smoothing/segmentation are order-sensitive.
    data = data.sort_values("abspos", kind="mergesort").reset_index(drop=True)

    # Re-derive 'chrom' as integer for downstream compatibility
    # chromosome_name can be 1..22, X=23, Y=24
    data["chromosome_name"] = data["chromosome_name"].astype(str)
    
    print(f"  {len(data)} genes annotated")
    return data
