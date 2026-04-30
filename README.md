# CopyKAT-Python

CopyKAT-Python is a Python reimplementation of the [CopyKAT](https://github.com/navinlabcode/copykat) workflow for inferring large-scale copy number alterations (CNAs) from single-cell RNA-seq data. It reproduces the core CopyKAT strategy while improving scalability, usability, and integration with modern `AnnData`/`Scanpy` pipelines.

## Why CopyKAT-Python?

The original CopyKAT-R package is widely used for distinguishing aneuploid tumor cells from diploid normal cells using scRNA-seq data. Recurring practical limitations include:

- Long runtimes, with reports of >1 hour for ~8,000 cells
- Inability to handle very large datasets (hundreds of thousands to millions of cells) due to hierarchical clustering limits
  
**Highlights:**

- Identical core parameters as CopyKAT-R with convenient Python improvements
- Handles datasets from thousands to hundreds of thousands of cells with significantly faster speed
- Annotated CNA heatmaps with per-cell metadata sidebars (cell type, cluster labels, etc.)
- Pre-built Singularity container for reproducible deployment
- Validated across 16 human and mouse 10X datasets and a 170k-cell Xenium whole-transcript dataset
  
---

## Installation

**From source:**

Installs `copykat-py` into your current environment

```bash
git clone https://github.com/navinlabcode/Copykat_python.git
cd Copykat_python
pip install -e .
```

**From `environment.yml` with conda:**

Creates a fresh conda environment for `copykat-py` with all required packages

```bash
git clone https://github.com/navinlabcode/Copykat_python.git
cd Copykat_python
conda env create -f environment.yml
conda activate copykit_py

## After activation, confirm the commands are available
copykat_matrix --help
copykat_anndata --help
```

**Singularity container** (recommended for HPC environments):

```bash
wget https://github.com/navinlabcode/Copykat_python/releases/download/v1.0.0/copykat_py.sif
singularity exec copykat_py.sif copykat-py --help
```

---

## How to Run

CopyKAT-Python supports two main entry points:

| Entry point | When to use |
| --- | --- |
| `copykat_matrix` / `copykat-py` | Input is a `.csv`, `.tsv`, or `.mtx` matrix file on disk in linux |
| `copykat_anndata()` | Input is an already-loaded `AnnData` object in python|

<img width="1183" height="566" alt="image" src="https://github.com/user-attachments/assets/2e8b1ae0-ef1c-4cc1-9e3d-f8cf6f9a88b5" />

### Terminal — `copykat_matrix` / `copykat-py`

**CSV or TSV matrix:**

```bash
copykat_matrix \
    --input sample_counts.csv \
    --sample-name sample1 \
    --genome hg20 \
    --n-cores 24 \
    --output-dir results/sample1
```

**10X matrix market input:**

```bash
copykat_matrix \
    --input filtered_feature_bc_matrix/matrix.mtx.gz \
    --genes filtered_feature_bc_matrix/features.tsv.gz \
    --barcodes filtered_feature_bc_matrix/barcodes.tsv.gz \
    --sample-name sample1 \
    --genome hg20 \
    --n-cores 24 \
    --output-dir results/sample1
```

Pass `--meta` (and optionally `--row-split`) to produce an annotated heatmap alongside the standard output. See [Annotated Heatmap with Metadata](#annotated-heatmap-with-metadata).

### Python API — `copykat()`

```python
import pandas as pd
from copykat_py import copykat

counts = pd.read_csv("sample_counts.csv", index_col=0)

result = copykat(
    rawmat=counts,
    id_type="S",
    sam_name="sample1",
    genome="hg20",
    distance="euclidean",
    n_cores=24,
)

print(result["prediction"].head())
```

`rawmat` can also be a dict with keys `matrix`, `genes`, and `barcodes` for sparse matrices.

### Python API — `copykat_anndata()`

```python
import anndata as ad
from copykat_py import copykat_anndata

adata = ad.read_h5ad("sample.h5ad")

result = copykat_anndata(
    adata=adata,
    selecting_meta=["CellType", "copykat_pred", "seurat_clusters"],
    row_split="CellType",
    sample_name="sample1",
    genome="hg20",
    distance="euclidean",
    n_cores=24,
    output_dir="results/sample1_anndata",
)

print(result["prediction"]["copykat.pred"].value_counts())
```

Useful options: `layer` (use `adata.layers[...]`), `use_raw` (use `adata.raw`), `selecting_meta` (export obs columns for annotated heatmaps), `row_split` (column defining row groups).

### Output files

All entry points produce the same outputs as copykat-R:

- `*_copykat_CNA_results.txt`
- `*_copykat_prediction.txt`
- `*_copykat_heatmap.png`
- `copykat_run.log`

When metadata is supplied, an additional annotated heatmap PNG is produced. AnnData workflows also write `*_selected_obs_meta.csv`.

---

## Annotated Heatmap with Metadata

Produce a CNA heatmap with per-cell metadata annotations (cell type, cluster labels, etc.). Rows are split into labelled groups by a chosen metadata column and ordered by hierarchical or K-means clustering within each group.

### CLI — standalone re-plot (`copykat-py-plot`)

Re-plot from an existing CNA results file without re-running the full analysis:

```bash
copykat-py-plot \
    --cna  sample_copykat_CNA_results.txt \
    --meta xenium_ft_full_meta_celltype_leiden.csv \
    --row-split inferred_CellType \
    --sample-name xenium_all_cells \
    --n-cores 40 \
    --output xenium_annotated_heatmap.png
```

| Flag | Default | Description |
| --- | --- | --- |
| `--cna` / `-c` | *(required)* | `*_copykat_CNA_results.txt` from a copykat-py run |
| `--meta` / `-m` | *(required)* | Annotation CSV — first column = cell name, rest = metadata |
| `--row-split` | second column | Column used to split and label row groups |

**Meta CSV format** — header row is auto-detected:

```csv
cell_name,leiden_cluster,inferred_CellType
aaaajgij-1,5,lumhr
aaaandia-1,5,lumhr
...
```

Cells present in the CNA results but absent from the CSV are labelled `"unknown"` and shown in grey. All remaining metadata columns are drawn as coloured annotation sidebars.

### Python API — `plot_heatmap_annotated`

```python
import pandas as pd
from copykat_py.plotting import plot_heatmap_annotated

cna = pd.read_csv("sample_copykat_CNA_results.txt", sep="\t")
plot_heatmap_annotated(
    mat           = cna.iloc[:, 3:].values.astype("float32"),
    cell_names    = cna.columns[3:].tolist(),
    chrom_info    = cna.iloc[:, 0].values,
    meta_csv      = "xenium_ft_full_meta_celltype_leiden.csv",
    row_split_col = "inferred_CellType",
    sample_name   = "xenium_all_cells",
    n_cores       = 40,
    output_path   = "xenium_annotated_heatmap.png",
)
```

---

## Benchmarking and Validation

### 10X Dataset Validation

Both CopyKAT-R and CopyKAT-Python were tested on raw datasets (no QC filtering) using 24 cores. A total of 20 datasets spanning human and mouse tissues across multiple 10X platforms were used for validation.

<details>
<summary><b>10x Genomics Benchmark Datasets</b></summary>

<br>

| Sample | Species | Tissue | Assay | Reported Cells |
| :--- | :--- | :--- | :--- | ---: |
| [human_pbmc_10k_3pv3](https://cf.10xgenomics.com/samples/cell-exp/3.0.0/pbmc_10k_v3/pbmc_10k_v3_web_summary.html) | human | PBMC healthy control | Universal 3' v3 | 11,769 |
| [human_nsclc_5pv1](https://www.10xgenomics.com/datasets/nsclc-tumor-5-gene-expression-1-standard-2-2-0) | human | NSCLC tumor | Universal 5' v1 | 7,802 |
| [human_ovarian_flex](https://www.10xgenomics.com/datasets/17k-human-ovarian-cancer-scFFPE) | human | Ovarian cancer FFPE | Flex | 17,553 |
| [human_kidney_gemx_flex](https://www.10xgenomics.com/datasets/Human_Kidney_4k_GEM-X_Flex) | human | Kidney nuclei control | GEM-X Flex | 4,633 |
| [human_hodgkins_3pv31](https://www.10xgenomics.com/datasets/hodgkins-lymphoma-dissociated-tumor-whole-transcriptome-analysis-3-1-standard-4-0-0) | human | Hodgkin's lymphoma | Universal 3' v3.1 | 3,394 |
| [human_glioblastoma_3pv3](https://www.10xgenomics.com/datasets/human-glioblastoma-multiforme-3-v-3-whole-transcriptome-analysis-3-standard-4-0-0) | human | Glioblastoma multiforme | Universal 3' v3 | 5,604 |
| [human_breast_idc_7p5k_3pv31](https://www.10xgenomics.com/jp/datasets/7-5-k-sorted-cells-from-human-invasive-ductal-carcinoma-3-v-3-1-3-1-standard-6-0-0) | human | Invasive ductal carcinoma | Universal 3' v3.1 | 5,680 |
| [human_breast_idc_750_lt_3pv31](https://www.10xgenomics.com/datasets/750-sorted-cells-from-human-invasive-ductal-carcinoma-3-lt-v-3-1-3-1-low-6-0-0) | human | Invasive ductal carcinoma | Universal 3' LT v3.1 | 687 |
| [human_melanoma_3p_gemx](https://www.10xgenomics.com/datasets/10k-human-dtc-melanoma-GEM-X) | human | Melanoma dissociated tumor cells | Universal 3' GEM-X | 10,645 |
| [human_melanoma_5p_nextgem](https://www.10xgenomics.com/datasets/10k-human-dtc-melanoma-NextGEM-5p) | human | Melanoma dissociated tumor cells | Universal 5' NextGEM | 6,704 |
| [mouse_brain_neurons_2k_v21](https://cf.10xgenomics.com/samples/cell-exp/2.1.0/neurons_2000/neurons_2000_web_summary.html) | mouse | E18 brain neurons | Universal 3' v2.1 | 2,022 |
| [mouse_brain_neurons_10k_v3](https://cf.10xgenomics.com/samples/cell-exp/3.0.0/neuron_10k_v3/neuron_10k_v3_web_summary.html) | mouse | Brain neurons | Universal 3' v3 | 11,843 |
| [mouse_heart_1k_v3](https://cf.10xgenomics.com/samples/cell-exp/3.0.0/heart_1k_v3/heart_1k_v3_web_summary.html) | mouse | Heart E18 | Universal 3' v3 | 1,011 |
| [mouse_heart_10k_v3](https://cf.10xgenomics.com/samples/cell-exp/3.0.0/heart_10k_v3/heart_10k_v3_web_summary.html) | mouse | Heart E18 | Universal 3' v3 | 7,713 |
| [mouse_heart_1k_v2](https://cf.10xgenomics.com/samples/cell-exp/3.0.0/heart_1k_v2/heart_1k_v2_web_summary.html) | mouse | Heart E18 | Universal 3' v2 | 712 |
| [mouse_brain_e18_10k_si_3pv31](https://www.10xgenomics.com/datasets/10-k-mouse-e-18-combined-cortex-hippocampus-and-subventricular-zone-cells-single-indexed-3-1-standard-4-0-0) | mouse | E18 cortex hippocampus SVZ | Universal 3' v3.1 SI | 11,316 |
| [mouse_kidney_nuclei_1k_3pv31](https://www.10xgenomics.com/cn/datasets/1k-mouse-kidney-nuclei-isolated-with-chromium-nuclei-isolation-kit-3-1-standard) | mouse | Adult kidney nuclei | Universal 3' v3.1 | 1,385 |
| [mouse_liver_nuclei_5k_3pv31](https://www.10xgenomics.com/datasets/5k-adult-mouse-liver-nuclei-isolated-with-chromium-nuclei-isolation-kit-3-1-standard) | mouse | Adult liver nuclei | Universal 3' v3.1 | 6,311 |
| [mouse_lung_nuclei_5k_3pv31](https://www.10xgenomics.com/datasets/5k-adult-mouse-lung-nuclei-isolated-with-chromium-nuclei-isolation-kit-3-1-standard) | mouse | Adult lung nuclei | Universal 3' v3.1 | 7,788 |
| [mouse_brain_gemx](https://www.10xgenomics.com/datasets/10k-Mouse-Neurons-3p-gemx) | mouse | E18 brain neurons | Universal 3' GEM-X | 12,441 |

</details>


#### Side-by-Side Comparison: CopyKAT-R vs CopyKAT-Python

**human_pbmc_10k_3pv3**
<img width="1209" height="559" alt="image" src="https://github.com/user-attachments/assets/9a7f6b5f-884d-41d6-805b-59933e8daf1b" />

**human_breast_idc_7p5k_3pv31**
<img width="1216" height="587" alt="image" src="https://github.com/user-attachments/assets/22b0ffd1-075e-495e-856a-3a1c80a9b510" />

**mouse_brain_e18_10k_si_3pv31**
<img width="1207" height="568" alt="image" src="https://github.com/user-attachments/assets/aef52f54-aa80-430a-b719-7781fa91b792" />

**Key Metrics Comparison**

<img width="2700" height="1650" alt="image" src="https://github.com/user-attachments/assets/a4cd6b97-e448-419b-b28b-3293bc9501fa" />

---

### Large-Scale Testing: Xenium Atera Dataset

The full [FFPE Human Breast Cancer](https://www.10xgenomics.com/datasets/atera-wta-ffpe-human-breast-cancer) Xenium (Atera) dataset was subsetted to 50k, 100k, and full (~170k cells) to evaluate scalability.

**Runtime Comparison**
<img width="1014" height="677" alt="image" src="https://github.com/user-attachments/assets/f534aca8-2d7b-40c6-acb1-dafb1775e224" />

**CNV Heatmap with Annotation**

<img width="1150" height="874" alt="image" src="https://github.com/user-attachments/assets/6fdd0ee4-2398-4723-886e-17e1ab49b703" />


**CNV comparison**
<img width="1296" height="720" alt="image" src="https://github.com/user-attachments/assets/d21e99ce-2dcf-4d64-813f-2de59ef8641a" />

## Why Results May Differ from CopyKAT-R

From the above comparison of the final prediction, the Seurat cluster 4 was called diploid by CopyKAT-R but aneuploid by CopyKAT-Py.
The copykat-py call was confirmed correct through the corresponding H&E cell morphology in this case. 

The key difference is in the final prediction step (step 8), where both implementations perform hierarchical clustering on the adjusted CNA matrix and cut the tree at k=2. R's copykat explicitly uses method = "ward.D" in hclust(), while CopyKAT-Python uses scipy/fastcluster's "ward", which implements the mathematically correct ward.D2 criterion.
For cells cluster (like  Seurat cluster 4 here,) with subtle CNV profiles that sit near the boundary of the diploid/aneuploid split, the two linkage variants produce different dendrogram topologies, causing the binary label assignment to flip. 

CopyKAT-Python results may not be identical to CopyKAT-R due to differences in:

**High-confidence results** typically show:
- Clear chromosome-arm or whole-chromosome CNV patterns
- Consistent CNV profiles within clusters
- Strong separation between inferred diploid and aneuploid cells

**Lower-confidence results** may occur in samples with:
- Weak CNV signal or low sequencing depth
- Few normal reference cells
- Strong batch effects
- Near-diploid tumor genomes


**Disclaimer:** 
CopyKAT-Python is an independent reimplementation focused on scalability and usability, while faithfully reproducing the core CopyKAT analytical strategy.

- Gene annotation versions
- Filtering and preprocessing steps
- Numerical implementation details
- Smoothing and segmentation algorithms
- Clustering behavior (parDist + hcluster vs. PCA + fastcluster)
