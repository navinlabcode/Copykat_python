#!/usr/bin/env bash
#===============================================================================
# run_copykat_py.sh — Wrapper to run CopyKAT-Py in the copykit_py conda env
#
# Usage:
#   ./run_copykat_py.sh -i <input> [options]
#
# Examples:
#   # 10x .mtx format (auto-detects genes.tsv / barcodes.tsv in same dir)
#   ./run_copykat_py.sh -i /path/to/matrix.mtx -o results/ --n-cores 8
#
#   # CSV/TSV count matrix (genes x cells, row names = gene symbols)
#   ./run_copykat_py.sh -i counts.csv -o results/ --sample-name sample1
#
#   # With known normal cells
#   ./run_copykat_py.sh -i matrix.mtx --norm-cells normals.txt -o results/
#
# Run with --help to see all options.
#===============================================================================
set -euo pipefail

# ---- Activate environment ----------------------------------------------------
# shellcheck source=/dev/null
source "/volumes/USR/ctang4/opt/Conda/etc/profile.d/conda.sh"
conda activate copykit_py

# ---- Run CopyKAT-Py ---------------------------------------------------------
exec copykat-py "$@"
