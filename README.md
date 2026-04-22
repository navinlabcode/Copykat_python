# CopyKAT-Python

CopyKAT-Python is a Python implementation of the CopyKAT workflow for inferring large-scale copy number alterations from single-cell RNA-seq data. It is designed for Python-based single-cell analysis pipelines and aims to improve usability, scalability, and integration with modern `AnnData`/`Scanpy` workflows.

## Why CopyKAT-Python?

The original CopyKAT-R package is widely used for distinguishing aneuploid tumor cells from diploid normal cells using scRNA-seq data. However, users have reported practical limitations when applying CopyKAT-R to large modern datasets.

Open GitHub issues in the original CopyKAT repository highlight several recurring needs:

- Long runtime, including reports of >1 hour for ~8,000 cells.
- Difficulty running very large datasets, including hundreds of thousands to millions of cells.

CopyKAT-Python was developed to address these practical issues while preserving the main biological idea of CopyKAT: large-scale chromosomal expression patterns can be used to infer copy number profiles and separate malignant from non-malignant cells.

## What Is Improved?

Compared with CopyKAT-R, CopyKAT-Python focuses on:

- Native Python workflow support.
- Easier integration with `AnnData`, `Scanpy`, and Python pipelines.
- Improved handling of large datasets.
- More transparent intermediate outputs.
- Clearer confidence reporting for uncertain cells.
- More flexible downstream use of CNV matrices and cell-level annotations.
- Better reproducibility through Python package management and scripted workflows.

CopyKAT-Python is not intended to be a line-by-line clone of CopyKAT-R. It is a Python reimplementation designed to reproduce the core CopyKAT strategy while improving scalability and usability.

## Confidence of Results

CopyKAT-Python reports tumor/normal predictions together with confidence-related outputs. High-confidence results usually show clear chromosome-arm or whole-chromosome CNV patterns, consistent CNV profiles within clusters, and strong separation between inferred diploid and aneuploid cells.

Lower-confidence results may occur in samples with weak CNV signal, low sequencing depth, few normal reference cells, strong batch effects, or tumors with near-diploid genomes.

## Why Results May Differ from CopyKAT-R

CopyKAT-Python results may not be identical to CopyKAT-R because of differences in:

- Gene annotation versions.
- Filtering and preprocessing.
- Numerical implementation.
- Smoothing and segmentation details.
- Clustering behavior and random seeds.
- Handling of uncertain or `not.defined` cells.

These differences are expected for an independent Python implementation.

## Figures and Tables to Add

### Figure 1. Workflow overview

Input expression matrix → gene genomic ordering → smoothing → CNV inference → clustering → tumor/normal prediction.

### Figure 2. Example CNV heatmap

Show inferred CNV profiles across chromosomes with cells grouped by predicted tumor/normal status.

### Figure 3. CopyKAT-R vs CopyKAT-Python comparison

Show side-by-side CNV heatmaps or classification agreement on the same dataset.

### Table 1. Runtime and scalability benchmark

| Dataset | Cells | Genes | CopyKAT-R runtime | CopyKAT-Python runtime | Notes |
|---|---:|---:|---:|---:|---|
| TODO | TODO | TODO | TODO | TODO | TODO |

### Table 2. Classification concordance

| Dataset | Tumor/normal agreement | Aneuploid agreement | Diploid agreement | Uncertain cells | Notes |
|---|---:|---:|---:|---:|---|
| TODO | TODO | TODO | TODO | TODO | TODO |

## Installation

```bash
git clone https://github.com/NavinLab/copykat-python.git
cd copykat-python
pip install -e .
