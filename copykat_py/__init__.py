"""CopyKAT-Py: Python implementation of CopyKAT for scRNA-seq copy number inference."""

from copykat_py.copykat import copykat
from copykat_py.data_loader import load_example_data

__version__ = "1.0.0"
__all__ = ["copykat", "load_example_data"]
