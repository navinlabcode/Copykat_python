# CopyKAT-Py — Singularity Container Guide

## Overview

This guide covers building and running the CopyKAT-Py Singularity container.  
The image bundles Python 3.11 + all required libraries in an isolated conda environment, so no local Python installation is needed on the compute node.

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| Singularity ≥ 3.8 (or Apptainer ≥ 1.0) | Available on most HPC clusters; check with `singularity --version` |
| Root / `fakeroot` or a build node | Required only for **building**; running needs no special privilege |
| ~4 GB free disk space | For the `.sif` image |

---

## 1. Build the Container

Run from the `copykat_py/` directory (where `copykat_py.def` lives):

```bash
cd /path/to/copykat_py

# with root (local workstation)
sudo singularity build copykat_py.sif copykat_py.def

# without root — fakeroot (many HPC clusters)
singularity build --fakeroot copykat_py.sif copykat_py.def
```

The build copies the local package source into the image (`%files` section) and installs it via `pip`, so **no internet access is needed at run time**.

> **Tip:** If you build on a login node that restricts root, transfer the source to a build node first, or ask your sysadmin to pre-build it.

---

## 2. Verify the Image

```bash
singularity exec copykat_py.sif copykat-py --help
```

---

## 3. Running CopyKAT-Py

### Basic syntax

```bash
singularity run [singularity-flags] copykat_py.sif [copykat-py-flags]
# equivalent to:
singularity exec copykat_py.sif copykat-py [copykat-py-flags]
```

### Input formats

| Format | Description |
|--------|-------------|
| `.mtx` / `.mtx.gz` | 10x Genomics sparse matrix; `genes.tsv` and `barcodes.tsv` are auto-detected from the same directory |
| `.csv` / `.tsv` / `.txt` | Dense count matrix, genes × cells, row names = gene symbols |

---

## 4. Usage Examples

### 4a. 10x MTX input (auto-detect gene/barcode files)

```bash
singularity run copykat_py.sif \
    -i /data/sample1/matrix.mtx \
    -o /results/sample1/ \
    --n-cores 8
```

### 4b. CSV/TSV count matrix

```bash
singularity run copykat_py.sif \
    -i /data/counts.csv \
    -o /results/sample1/ \
    --sample-name sample1
```

### 4c. With explicit gene / barcode files (MTX)

```bash
singularity run copykat_py.sif \
    -i /data/matrix.mtx \
    --genes  /data/genes.tsv \
    --barcodes /data/barcodes.tsv \
    -o /results/sample1/
```

### 4d. Provide known normal-cell barcodes

```bash
singularity run copykat_py.sif \
    -i /data/matrix.mtx \
    --norm-cells /data/normal_barcodes.txt \
    -o /results/sample1/
```

### 4e. Mouse genome + IGV .seg output

```bash
singularity run copykat_py.sif \
    -i /data/matrix.mtx \
    --genome mm10 \
    --output-seg \
    -o /results/sample1/
```

### 4f. Full parameter set

```bash
singularity run copykat_py.sif \
    -i /data/matrix.mtx \
    -o /results/sample1/ \
    --sample-name sample1 \
    --genome hg20 \
    --id-type S \
    --cell-line no \
    --ngene-chr 5 \
    --min-genes 200 \
    --low-dr 0.05 \
    --up-dr 0.1 \
    --win-size 25 \
    --ks-cut 0.1 \
    --distance euclidean \
    --n-cores 16 \
    --output-seg
```

---

## 5. Binding Host Paths

By default Singularity only mounts `$HOME` and `$CWD`. For data elsewhere, bind explicitly:

```bash
singularity run \
    --bind /scratch/mydata:/data \
    --bind /scratch/results:/results \
    copykat_py.sif \
    -i /data/matrix.mtx \
    -o /results/sample1/
```

---

## 6. SLURM Job Script Template

```bash
#!/usr/bin/env bash
#SBATCH --job-name=copykat_py
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out

SIF=/path/to/copykat_py.sif
INPUT=/scratch/$USER/data/matrix.mtx
OUTDIR=/scratch/$USER/results/sample1

singularity run \
    --bind /scratch/$USER:/scratch/$USER \
    "$SIF" \
    -i "$INPUT" \
    -o "$OUTDIR" \
    --n-cores "$SLURM_CPUS_PER_TASK" \
    --sample-name sample1
```

---

## 7. All CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `-i / --input` | *(required)* | Input matrix file (`.mtx`, `.csv`, `.tsv`, `.txt`) |
| `-o / --output-dir` | `.` | Output directory (created if absent) |
| `--genes` | auto-detect | Gene names file (for `.mtx` input) |
| `--barcodes` | auto-detect | Barcode names file (for `.mtx` input) |
| `--sample-name` | `""` | Prefix for all output files |
| `--genome` | `hg20` | Reference genome: `hg20` or `mm10` |
| `--id-type` | `S` | Gene ID type: `S` = symbol, `E` = Ensembl |
| `--cell-line` | `no` | Pure cell-line mode: `yes` / `no` |
| `--ngene-chr` | `5` | Minimum genes per chromosome to keep |
| `--min-genes` | `200` | Minimum genes expressed per cell |
| `--low-dr` | `0.05` | Min detection rate for smoothing window |
| `--up-dr` | `0.1` | Min detection rate for segmentation |
| `--win-size` | `25` | Window size for CBS segmentation |
| `--ks-cut` | `0.1` | KS-test p-value cutoff for breakpoints |
| `--distance` | `euclidean` | Distance metric: `euclidean`, `pearson`, `spearman` |
| `--norm-cells` | `""` | File with known normal-cell barcodes (one per line) |
| `--output-seg` | off | Emit `.seg` file compatible with IGV |
| `--n-cores` | `1` | CPU cores for parallel steps |

---

## 8. Output Files

| File | Description |
|------|-------------|
| `<sample>_copykat_CNA_results.csv` | Per-cell CNA matrix (genes × cells) |
| `<sample>_copykat_prediction.txt` | Aneuploid / diploid prediction per cell |
| `<sample>_copykat_heatmap.png` | CNA heatmap with dendrogram |
| `<sample>_copykat_CNA_raw_results.csv` | Raw (un-binned) CNA values |
| `<sample>.seg` | IGV-compatible segment file *(if `--output-seg`)* |

---

## Troubleshooting

**`FATAL: container creation failed`** — ensure Singularity ≥ 3.8 and that `--fakeroot` is supported on your cluster, or build with sudo on a workstation.

**`ModuleNotFoundError: numba`** — numba is installed via conda (not pip) in the definition file to ensure LLVM compatibility; rebuilding the image should resolve it.

**`ModuleNotFoundError: fastcluster`** — fastcluster is installed via conda-forge; if it fails to resolve during build, replace the conda line with `pip install fastcluster>=1.3.0` after the conda block.

**Blank / missing plots** — `MPLBACKEND=Agg` is set in the container so matplotlib writes files without a display. If you still see Qt/Tk errors, add `--env MPLBACKEND=Agg` to your `singularity run` call.

**Out of memory** — reduce `--n-cores` or request more RAM in your scheduler job; the DLM smoothing step scales with `n_cells × n_genes`.
