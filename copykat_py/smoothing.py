"""Dynamic Linear Model (DLM) smoothing, mirroring the R dlm package's dlmModPoly + dlmSmooth.

Implements a first-order polynomial DLM (local level model):
    Observation:  y_t = theta_t + v_t,   v_t ~ N(0, V=0.16)
    State:        theta_t = theta_{t-1} + w_t,  w_t ~ N(0, W=0.001)

The Kalman smoother produces smoothed state estimates.
"""

import numpy as np
import os
import numba
from numba import njit, prange

_LAST_PAR_INFO = {
    "step": "dlm_smooth",
    "parallel": False,
    "requested_cores": 1,
    "effective_cores": 1,
    "tasks": 0,
    "chunk_size": 0,
    "engine": "numba_shared_gains",
}


def get_last_dlm_smooth_info():
    return dict(_LAST_PAR_INFO)


def _dlm_gains(n, dV=0.16, dW=0.001):
    """Kalman filter and RTS smoother gains for the local level model.

    For a time-invariant model with a fixed prior, the gains depend only on
    (n, dV, dW) and never on the observations, so they are identical for every
    cell and can be computed once.

    Returns
    -------
    K : np.ndarray, shape (n,)
        Filter gains: m_{t+1} = m_t + K_t * (y_t - m_t).
    B : np.ndarray, shape (n,)
        Smoother gains: s_t = m_t + B_t * (s_{t+1} - m_t).
    """
    C = np.empty(n + 1)  # filtered state variances
    R = np.empty(n)  # prior state variances
    K = np.empty(n)
    C[0] = 1e7  # diffuse prior
    for t in range(n):
        R[t] = C[t] + dW
        Q = R[t] + dV  # forecast variance (F=1)
        K[t] = R[t] / Q
        C[t + 1] = R[t] * (1 - K[t])
    # R's dlm convention: B_t = C_t / R_{t+1}; in our indexing R[t] is R's R_{t+1}
    B = np.zeros(n)
    np.divide(C[:-1], R, out=B, where=R > 0)
    return K, B


_BLOCK_CELLS = 64


@njit(parallel=True)
def _dlm_smooth_blocks(y, K, B, out):
    """Apply the shared filter/smoother gains to every column of ``y``.

    Columns are processed in blocks of ``_BLOCK_CELLS`` so the inner loop walks
    contiguous memory; blocks run in parallel threads.
    """
    n, n_cells = y.shape
    n_blocks = (n_cells + _BLOCK_CELLS - 1) // _BLOCK_CELLS
    for b in prange(n_blocks):
        c0 = b * _BLOCK_CELLS
        w = min(c0 + _BLOCK_CELLS, n_cells) - c0
        m = np.empty((n + 1, w))  # filtered means, then smoothed means in place
        for j in range(w):
            m[0, j] = 0.0
        # Forward Kalman filter
        for t in range(n):
            for j in range(w):
                m[t + 1, j] = m[t, j] + K[t] * (y[t, c0 + j] - m[t, j])
        # Backward smoother (Rauch-Tung-Striebel); m[n] is already s[n]
        for t in range(n - 1, -1, -1):
            for j in range(w):
                m[t, j] = m[t, j] + B[t] * (m[t + 1, j] - m[t, j])
        # Return smoothed states (skip s[0] which is the prior), centered
        for j in range(w):
            total = 0.0
            for t in range(1, n + 1):
                total += m[t, j]
            mean = total / n
            for t in range(n):
                out[t, c0 + j] = m[t + 1, j] - mean


def dlm_smooth(norm_mat, n_cores=1):
    """Apply DLM smoothing to all cells in parallel.
    
    Parameters
    ----------
    norm_mat : np.ndarray, shape (n_genes, n_cells)
        Normalized gene expression matrix.
    n_cores : int
        Number of parallel threads.
    
    Returns
    -------
    np.ndarray, shape (n_genes, n_cells)
        Smoothed expression matrix (float32).
    """
    n_genes, n_cells = norm_mat.shape
    max_cores = int(os.getenv("COPYKAT_MAX_CORES", str(os.cpu_count() or 1)))
    n_blocks = -(-n_cells // _BLOCK_CELLS)
    n_jobs = max(1, min(int(n_cores), max_cores, n_blocks, numba.config.NUMBA_NUM_THREADS))
    _LAST_PAR_INFO.update({
        "parallel": n_jobs > 1,
        "requested_cores": int(n_cores),
        "effective_cores": int(n_jobs),
        "tasks": int(n_cells),
        "chunk_size": int(_BLOCK_CELLS),
        "engine": "numba_shared_gains",
    })

    K, B = _dlm_gains(n_genes)
    y = np.asarray(norm_mat, dtype=np.float64)
    smoothed = np.empty((n_genes, n_cells), dtype=np.float32)
    previous_threads = numba.get_num_threads()
    numba.set_num_threads(n_jobs)
    try:
        _dlm_smooth_blocks(y, K, B, smoothed)
    finally:
        numba.set_num_threads(previous_threads)
    return smoothed
