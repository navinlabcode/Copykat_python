# CopyKAT-Py: Python Implementation of CopyKAT

A robust, high-efficiency Python implementation of CopyKAT (Copynumber Karyotyping of Aneuploid Tumors) 
for inferring genomic copy number profiles from single-cell RNA-seq data.

## Key Improvements over R version

- **No cell limit**: Overcomes the R `hclust` 65,536 cell barrier using mini-batch and approximate methods
- **Faster execution**: Leverages NumPy vectorization, sparse matrices, and multiprocessing
- **Same algorithm**: Faithfully reimplements the Bayesian segmentation approach (FTT → DLM smoothing → GMM baseline → MCMC/KS segmentation)

## Installation

```bash
conda env create -f environment.yml
conda activate copykat_py
pip install -e .
```

## Quick Start

```python
import copykat_py

results = copykat_py.copykat(
    rawmat="path/to/matrix.mtx",     # or a pandas DataFrame / scipy sparse matrix
    id_type="S",                      # "S" for Symbol, "E" for Ensembl
    genome="hg20",
    n_cores=8,
    sam_name="test",
)

# Access results
predictions = results["prediction"]       # DataFrame: cell.names, copykat.pred
cna_mat = results["CNAmat"]               # DataFrame: chrom, chrompos, abspos, cell1, cell2, ...
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| rawmat | required | UMI count matrix (genes×cells), DataFrame, sparse matrix, or path |
| id_type | "S" | Gene ID type: "S" (Symbol) or "E" (Ensembl) |
| cell_line | "no" | "yes" for pure cell line data |
| ngene_chr | 5 | Min genes per chromosome for cell filtering |
| LOW_DR | 0.05 | Min gene detection rate for smoothing |
| UP_DR | 0.1 | Min gene detection rate for segmentation |
| win_size | 25 | Window size for segmentation |
| norm_cell_names | "" | Known normal cell barcodes (list or "") |
| KS_cut | 0.1 | KS test cutoff for breakpoint calling |
| sam_name | "" | Sample name prefix for output files |
| distance | "euclidean" | Distance metric: "euclidean", "pearson", "spearman" |
| n_cores | 1 | Number of CPU cores |
| genome | "hg20" | Genome: "hg20" or "mm10" |
| output_seg | False | Output .seg file for IGV |
| plot_genes | True | Plot gene-level heatmap |
| min_gene_per_cell | 200 | Minimum genes per cell |

## Reference

Gao, R., et al. "Delineating copy number and clonal substructure in human tumors from single-cell transcriptomes." 
Nature Biotechnology 39, 599–608 (2021).
