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
                        help="Input UMI count matrix (.csv, .tsv, .txt, or .mtx)")
    parser.add_argument("--genes", default=None,
                        help="Gene names file (required for .mtx input)")
    parser.add_argument("--barcodes", default=None,
                        help="Barcode names file (required for .mtx input)")
    
    # Parameters
    parser.add_argument("--id-type", default="S", choices=["S", "E"],
                        help="Gene ID type: S=Symbol, E=Ensembl (default: S)")
    parser.add_argument("--cell-line", default="no", choices=["yes", "no"],
                        help="Pure cell line mode (default: no)")
    parser.add_argument("--ngene-chr", type=int, default=5,
                        help="Min genes per chromosome (default: 5)")
    parser.add_argument("--min-genes", type=int, default=200,
                        help="Min genes per cell (default: 200)")
    parser.add_argument("--low-dr", type=float, default=0.05,
                        help="Min detection rate for smoothing (default: 0.05)")
    parser.add_argument("--up-dr", type=float, default=0.1,
                        help="Min detection rate for segmentation (default: 0.1)")
    parser.add_argument("--win-size", type=int, default=25,
                        help="Window size for segmentation (default: 25)")
    parser.add_argument("--norm-cells", default="",
                        help="File with known normal cell barcodes (one per line)")
    parser.add_argument("--ks-cut", type=float, default=0.1,
                        help="KS test cutoff for breakpoints (default: 0.1)")
    parser.add_argument("--sample-name", default="",
                        help="Sample name prefix for output files")
    parser.add_argument("--distance", default="euclidean",
                        choices=["euclidean", "pearson", "spearman"],
                        help="Distance metric for clustering (default: euclidean)")
    parser.add_argument("--output-seg", action="store_true",
                        help="Output .seg file for IGV")
    parser.add_argument("--plot-genes", dest="plot_genes", action="store_true", default=True,
                        help="Generate the heatmap plot (default: enabled)")
    parser.add_argument("--no-plot-genes", dest="plot_genes", action="store_false",
                        help="Skip heatmap plotting for faster runs")
    parser.add_argument("--genome", default="hg20", choices=["hg20", "mm10"],
                        help="Genome assembly (default: hg20)")
    parser.add_argument("--n-cores", type=int, default=1,
                        help="Number of CPU cores (default: 1)")
    parser.add_argument("--output-dir", "-o", default=".",
                        help="Output directory (default: current)")
    
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
        )
        
        print("CopyKAT-Py analysis complete.")
        if "prediction" in result:
            pred = result["prediction"]["copykat.pred"].value_counts()
            for k, v in pred.items():
                print(f"  {k}: {v} cells")
    finally:
        print(f"CopyKAT-Py run finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Detailed log saved to: {log_path}")
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        log_handle.close()


if __name__ == "__main__":
    main()
