# CopyKAT-Python

CopyKAT-Python is a Python reimplementation of the [CopyKAT](https://github.com/navinlabcode/copykat) workflow for inferring large-scale copy number alterations (CNAs) from single-cell RNA-seq data. It is designed to reproduce the core CopyKAT strategy while improving scalability, usability, and integration with modern `AnnData`/`Scanpy` pipelines.

**Highlights:**
- Handles datasets from thousands to hundreds of thousands of cells
- Identical core parameters as CopyKAT-R with convenient Python improvements
- Pre-built Singularity container for reproducible deployment
- Validated across 16 human and mouse 10X datasets and a 170k-cell Xenium dataset

---

## Why CopyKAT-Python?

The original CopyKAT-R package is widely used for distinguishing aneuploid tumor cells from diploid normal cells using scRNA-seq data. However, users have reported practical limitations when applying it to large modern datasets. Recurring issues include:

- Long runtimes, with reports of >1 hour for ~8,000 cells
- Inability to handle very large datasets (hundreds of thousands to millions of cells) due to hierarchical clustering limits

CopyKAT-Python is not a line-by-line clone of CopyKAT-R. It is an independent Python reimplementation focused on scalability and usability, while faithfully reproducing the core CopyKAT analytical strategy.

---

## Installation

**From source:**
```bash
git clone https://github.com/navinlabcode/Copykat_python.git
cd Copykat_python
pip install -e .
```

**Singularity container** (recommended for HPC environments):
```bash
wget https://github.com/navinlabcode/Copykat_python/releases/download/v1.0.0/copykat_py.sif
singularity exec copykat_py.sif copykat-py --help
```

---

## How to Run

CopyKAT-Python shares identical parameters with CopyKAT-R, with a few convenient additions.

<img width="956" height="508" alt="image" src="https://github.com/user-attachments/assets/144da3e7-d856-4f75-91fa-e89883e98213" />

---

## Benchmarking and Validation

### 10X Dataset Validation

Both CopyKAT-R and CopyKAT-Python were tested on raw datasets (no QC filtering) using 24 cores. A total of 20 datasets spanning human and mouse tissues across multiple 10X platforms were used for validation.

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

**CNV Heatmap — Full Dataset (~170k cells)**
<img width="2601" height="1825" alt="xenium_all_cells_copykat_heatmap" src="https://github.com/user-attachments/assets/2d604c37-2959-4bfb-be0e-38f3999b14d4" />

---

## Interpreting Results

CopyKAT-Python reports tumor/normal predictions alongside confidence-related outputs.

**High-confidence results** typically show:
- Clear chromosome-arm or whole-chromosome CNV patterns
- Consistent CNV profiles within clusters
- Strong separation between inferred diploid and aneuploid cells

**Lower-confidence results** may occur in samples with:
- Weak CNV signal or low sequencing depth
- Few normal reference cells
- Strong batch effects
- Near-diploid tumor genomes

---

## Why Results May Differ from CopyKAT-R

CopyKAT-Python results may not be identical to CopyKAT-R due to differences in:

- Gene annotation versions
- Filtering and preprocessing steps
- Numerical implementation details
- Smoothing and segmentation algorithms
- Clustering behavior and random seed handling
- Treatment of uncertain (`not.defined`) cells
