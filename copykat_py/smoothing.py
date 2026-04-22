"""Dynamic Linear Model (DLM) smoothing, mirroring the R dlm package's dlmModPoly + dlmSmooth.

Implements a first-order polynomial DLM (local level model):
    Observation:  y_t = theta_t + v_t,   v_t ~ N(0, V=0.16)
    State:        theta_t = theta_{t-1} + w_t,  w_t ~ N(0, W=0.001)

The Kalman smoother produces smoothed state estimates.
"""

import numpy as np
import os
from joblib import Parallel, delayed
from numba import jit

_LAST_PAR_INFO = {
    "step": "dlm_smooth",
    "parallel": False,
    "requested_cores": 1,
    "effective_cores": 1,
    "tasks": 0,
    "chunk_size": 0,
}


def get_last_dlm_smooth_info():
    return dict(_LAST_PAR_INFO)


@jit(nopython=True)
def _dlm_smooth_single(y, dV=0.16, dW=0.001):
    """Kalman filter + backward smoother for a single cell's gene expression vector.
    
    Parameters
    ----------
    y : np.ndarray, shape (n_genes,)
        Normalized gene expression values.
    dV : float
        Observation variance.
    dW : float
        State evolution variance.
    
    Returns
    -------
    np.ndarray, shape (n_genes,)
        Smoothed and centered expression values.
    """
    n = len(y)
    
    # Forward Kalman filter
    # Initialize: diffuse prior
    m = np.zeros(n + 1)  # filtered state means
    C = np.zeros(n + 1)  # filtered state variances
    m[0] = 0.0
    C[0] = 1e7  # diffuse prior
    
    a = np.zeros(n)  # prior state means
    R = np.zeros(n)  # prior state variances
    
    for t in range(n):
        # Predict
        a[t] = m[t]
        R[t] = C[t] + dW
        
        # Update
        f = a[t]  # forecast mean (F=1)
        Q = R[t] + dV  # forecast variance
        e = y[t] - f  # forecast error
        K = R[t] / Q  # Kalman gain
        
        m[t + 1] = a[t] + K * e
        C[t + 1] = R[t] * (1 - K)
    
    # Backward smoother (Rauch-Tung-Striebel)
    # R's dlm convention: B_t = C_t / R_{t+1}, s_t = m_t + B_t*(s_{t+1} - a_{t+1})
    # In our indexing: my R[t] = R's R_{t+1}, my C[t] = R's C_t, my m[t] = R's m_t
    s = np.zeros(n + 1)  # smoothed state means
    S = np.zeros(n + 1)  # smoothed state variances
    s[n] = m[n]
    S[n] = C[n]
    
    for t in range(n - 1, -1, -1):
        B = C[t] / R[t] if R[t] > 0 else 0  # smoother gain
        s[t] = m[t] + B * (s[t + 1] - m[t])
        S[t] = C[t] + B * B * (S[t + 1] - R[t])
    
    # Return smoothed states (skip s[0] which is the prior)
    smoothed = s[1:]
    smoothed = smoothed - np.mean(smoothed)
    return smoothed


def dlm_smooth(norm_mat, n_cores=1):
    """Apply DLM smoothing to all cells in parallel.
    
    Parameters
    ----------
    norm_mat : np.ndarray, shape (n_genes, n_cells)
        Normalized gene expression matrix.
    n_cores : int
        Number of parallel workers.
    
    Returns
    -------
    np.ndarray, shape (n_genes, n_cells)
        Smoothed expression matrix (float32).
    """
    n_cells = norm_mat.shape[1]
    max_cores = int(os.getenv("COPYKAT_MAX_CORES", str(os.cpu_count() or 1)))
    n_jobs = max(1, min(int(n_cores), max_cores, n_cells))
    
    # Adaptive chunk size: larger chunks for better parallelization efficiency
    chunk_size = max(1, min(n_cells // n_jobs, max(4, n_cells // (n_jobs * 2))))
    _LAST_PAR_INFO.update({
        "parallel": n_jobs > 1,
        "requested_cores": int(n_cores),
        "effective_cores": int(n_jobs),
        "tasks": int(n_cells),
        "chunk_size": int(chunk_size),
    })

    def _smooth_chunk(start, end):
        return np.column_stack([_dlm_smooth_single(norm_mat[:, c]) for c in range(start, end)])

    ranges = [(s, min(s + chunk_size, n_cells)) for s in range(0, n_cells, chunk_size)]
    chunks = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_smooth_chunk)(s, e) for s, e in ranges
    )
    smoothed = np.hstack(chunks).astype(np.float32, copy=False)
    return smoothed
