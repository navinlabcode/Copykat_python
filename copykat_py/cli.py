"""Command line interface for CopyKAT-Py."""

import argparse
import sys
import os
import time
import pandas as pd
import numpy as np
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


def main():
    parser = argparse.ArgumentParser(
        description="CopyKAT-Py: Inference of genomic copy number from single cell RNA-seq data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Input
    parser.add_argument("--input", "-i", required=True,
                        help="[required] Input UMI count matrix (.csv, .tsv, .txt, or .mtx)")
    parser.add_argument("--genes", default=None,
                        help="[optional] Gene names file (required for .mtx input)")
    parser.add_argument("--barcodes", default=None,
                        help="[optional] Barcode names file (required for .mtx input)")

    # Parameters
    parser.add_argument("--id-type", default="S", choices=["S", "E"],
                        help="[optional] Gene ID type: S=Symbol, E=Ensembl (default: S)")
    parser.add_argument("--cell-line", default="no", choices=["yes", "no"],
                        help="[optional] Pure cell line mode (default: no)")
    parser.add_argument("--ngene-chr", type=int, default=5,
                        help="[optional] Min genes per chromosome (default: 5)")
    parser.add_argument("--min-genes", type=int, default=200,
                        help="[optional] Min genes per cell (default: 200)")
    parser.add_argument("--low-dr", type=float, default=0.05,
                        help="[optional] Min detection rate for smoothing (default: 0.05)")
    parser.add_argument("--up-dr", type=float, default=0.1,
                        help="[optional] Min detection rate for segmentation (default: 0.1)")
    parser.add_argument("--win-size", type=int, default=25,
                        help="[optional] Window size for segmentation (default: 25)")
    parser.add_argument("--norm-cells", default="",
                        help="[optional] File with known normal cell barcodes (one per line)")
    parser.add_argument("--ks-cut", type=float, default=0.1,
                        help="[optional] KS test cutoff for breakpoints (default: 0.1)")
    parser.add_argument("--sample-name", default="",
                        help="[optional] Sample name prefix for output files")
    parser.add_argument("--distance", default="euclidean",
                        choices=["euclidean", "pearson", "spearman"],
                        help="[optional] Distance metric for clustering (default: euclidean)")
    parser.add_argument("--output-seg", action="store_true",
                        help="[optional] Output .seg file for IGV")
    parser.add_argument("--plot-genes", dest="plot_genes", action="store_true", default=True,
                        help="[optional] Generate the heatmap plot (default: enabled)")
    parser.add_argument("--no-plot-genes", dest="plot_genes", action="store_false",
                        help="[optional] Skip heatmap plotting for faster runs")
    parser.add_argument("--genome", default="hg20", choices=["hg20", "mm10"],
                        help="[optional] Genome assembly (default: hg20)")
    parser.add_argument("--n-cores", type=int, default=1,
                        help="[optional] Number of CPU cores (default: 1)")
    parser.add_argument("--pca-components", type=int, default=None,
                        help="[optional] Adaptive PCA component cap for large clustering steps (default: automatic by cell count)")
    parser.add_argument("--output-dir", "-o", default=".",
                        help="[optional] Output directory (default: current)")

    # Annotated heatmap options
    parser.add_argument("--meta", default=None,
                        metavar="CSV",
                        help="[optional] Per-cell annotation CSV for the annotated heatmap. "
                             "First column = cell name; remaining columns are drawn "
                             "as coloured sidebars. Header is auto-detected. "
                             "When provided, an additional annotated heatmap is saved "
                             "alongside the standard one.")
    parser.add_argument("--row-split", default=None,
                        metavar="COLUMN",
                        help="[optional] Column in --meta used to split and label heatmap rows. "
                             "Defaults to the second column of the CSV when not given.")
    
    args = parser.parse_args()
    
    # Load input
    input_path = args.input
    if input_path.endswith(".mtx") or input_path.endswith(".mtx.gz"):
        mat = mmread(input_path)
        
        if args.genes is None or args.barcodes is None:
            # Try to find genes.tsv and barcodes.tsv in same dir
            input_dir = os.path.dirname(input_path)
            for gf in ["genes.tsv", "genes.txt", "features.tsv", "features.tsv.gz"]:
                gp = os.path.join(input_dir, gf)
                if os.path.exists(gp):
                    args.genes = gp
                    break
            for bf in ["barcodes.tsv", "barcodes.txt"]:
                bp = os.path.join(input_dir, bf)
                if os.path.exists(bp):
                    args.barcodes = bp
                    break
        
        if args.genes:
            genes = pd.read_csv(args.genes, sep="\t", header=None)
            gene_names = genes.iloc[:, -1].values if genes.shape[1] > 1 else genes.iloc[:, 0].values
        else:
            gene_names = [f"gene_{i}" for i in range(mat.shape[0])]
        
        if args.barcodes:
            barcodes = pd.read_csv(args.barcodes, sep="\t", header=None).iloc[:, 0].values
        else:
            barcodes = np.array([f"cell_{i}" for i in range(mat.shape[1])], dtype=object)

        # Keep sparse inputs sparse and let copykat decide when to densify.
        rawmat = {
            "matrix": mat,
            "genes": gene_names,
            "barcodes": barcodes,
        }
    else:
        rawmat = input_path  # Let copykat handle loading
    
    # Normal cells
    norm_cell_names = ""
    if args.norm_cells and os.path.exists(args.norm_cells):
        with open(args.norm_cells) as f:
            norm_cell_names = [line.strip() for line in f if line.strip()]
    
    # Change to output dir
    os.makedirs(args.output_dir, exist_ok=True)
    os.chdir(args.output_dir)
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(args.output_dir, ".mplconfig"))
    os.environ.setdefault("XDG_CACHE_HOME", os.path.join(args.output_dir, ".cache"))
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
    os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)

    from copykat_py.copykat import copykat

    log_path = os.path.join(args.output_dir, "copykat_run.log")
    log_handle = open(log_path, "a", encoding="utf-8")
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = TeeStream(sys.stdout, log_handle)
    sys.stderr = TeeStream(sys.stderr, log_handle)

    print("=" * 80)
    print(f"CopyKAT-Py run started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Working directory: {args.output_dir}")
    print(f"Input: {args.input}")
    print(f"Requested cores: {args.n_cores}")
    if args.pca_components is not None:
        print(f"Requested adaptive PCA components: {args.pca_components}")
    
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
        )
        
        print("CopyKAT-Py analysis complete.")
        if "prediction" in result:
            pred = result["prediction"]["copykat.pred"].value_counts()
            for k, v in pred.items():
                print(f"  {k}: {v} cells")

        if args.meta and args.plot_genes and "CNAmat" in result:
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
                meta_csv=args.meta,
                row_split_col=args.row_split,
                sample_name=args.sample_name,
                distance=args.distance,
                n_cores=args.n_cores,
                output_path=ann_output,
            )
    finally:
        print(f"CopyKAT-Py run finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Detailed log saved to: {log_path}")
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        log_handle.close()


def plot_main():
    """Entry point for ``copykat-py-plot``: annotated heatmap from CNA results."""
    parser = argparse.ArgumentParser(
        prog="copykat-py-plot",
        description=(
            "Draw an annotated CNA heatmap from a copykat-py CNA results file "
            "and a per-cell metadata CSV.  Rows are split and colour-labelled by "
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
  Remaining    : any metadata (cell type, cluster, condition, …)

  With header:    cell_name,leiden_cluster,inferred_CellType
  Without header: aaaajgij-1,5,lumhr   (first row treated as data)
""",
    )

    parser.add_argument(
        "--cna", "-c", required=True,
        help="[required] CNA results file produced by copykat-py "
             "(*_copykat_CNA_results.txt, tab-separated). "
             "First three columns must be chrom / chrompos / abspos; "
             "remaining columns are cells.",
    )
    parser.add_argument(
        "--meta", "-m", required=True,
        help="[required] Annotation CSV.  First column = cell name; remaining columns are "
             "drawn as coloured sidebars.  Header row is auto-detected.",
    )
    parser.add_argument(
        "--row-split", default=None,
        metavar="COLUMN",
        help="[optional] Metadata column used to split and label rows.  "
             "Defaults to the second column of the CSV when not supplied.",
    )
    parser.add_argument(
        "--sample-name", default="",
        help="[optional] Label shown in the figure title and used as the output filename "
             "prefix when --output is not given.",
    )
    parser.add_argument(
        "--distance", default="euclidean",
        choices=["euclidean", "pearson", "spearman"],
        help="[optional] Distance metric for within-group cell clustering (default: euclidean).",
    )
    parser.add_argument(
        "--n-cores", type=int, default=1,
        help="[optional] Parallel threads for clustering (default: 1).",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        metavar="PATH",
        help="[optional] Output PNG path.  Defaults to "
             "{sample_name}_copykat_annotated_heatmap.png.",
    )

    args = parser.parse_args()

    print(f"Loading CNA results: {args.cna}")
    cna_df = pd.read_csv(args.cna, sep="\t", index_col=False)
    cell_names = cna_df.columns[3:].tolist()
    chrom_info = cna_df.iloc[:, 0].values
    mat = cna_df.iloc[:, 3:].values.astype(np.float32)
    print(f"  {mat.shape[1]} cells × {mat.shape[0]} bins")

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
