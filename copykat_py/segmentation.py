"""MCMC segmentation: Poisson-Gamma posterior estimation with KS-test breakpoint detection.

Mirrors CNA.MCMC.R from the R package.

Algorithm:
1. Compute cluster consensus profiles (median per cluster)
2. For each cluster consensus, find breakpoints by:
   a. Divide genome into windows of `bins` genes
   b. Sample Poisson-Gamma posteriors for adjacent windows (MCpoissongamma)
   c. KS-test between posteriors; if D > cut_cor, call breakpoint
3. Union all breakpoints across clusters
4. For each cell, compute posterior means per segment
"""

import numpy as np
import os
from scipy.stats import ks_2samp, gamma as gamma_dist
from joblib import Parallel, delayed
from numba import jit

_LAST_PAR_INFO = {
    "step": "cna_mcmc",
    "parallel": False,
    "requested_cores": 1,
    "effective_cores": 1,
    "tasks": 0,
    "chunk_size": 0,
    "mc_samples": 1000,
    "engine": "posterior_sampling",
}


def get_last_cna_mcmc_info():
    return dict(_LAST_PAR_INFO)


@jit(nopython=True)
def _mc_poisson_gamma_numba(data, alpha, beta=1.0, mc=1000, seed=42):
    """Sample from the posterior of a Poisson-Gamma model using Numba.
    
    Prior: lambda ~ Gamma(alpha, beta)
    Likelihood: X_i ~ Poisson(lambda)
    Posterior: lambda | X ~ Gamma(alpha + sum(X), beta + n)
    
    Parameters
    ----------
    data : np.ndarray
        Observed counts (back-transformed expression values).
    alpha : float
        Prior shape parameter (set to the mean of data).
    beta : float
        Prior rate parameter.
    mc : int
        Number of Monte Carlo samples.
    seed : int
        Random seed.
    
    Returns
    -------
    np.ndarray, shape (mc,)
        Posterior samples.
    """
    np.random.seed(seed)
    n = len(data)
    post_shape = alpha + np.sum(data)
    post_rate = beta + n
    # Gamma distribution: shape and scale=1/rate
    samples = np.random.gamma(post_shape, 1.0 / post_rate, mc)
    return samples


def _mc_poisson_gamma(data, alpha, beta=1.0, mc=1000, rng=None):
    """Sample from the posterior of a Poisson-Gamma model.
    
    Prior: lambda ~ Gamma(alpha, beta)
    Likelihood: X_i ~ Poisson(lambda)
    Posterior: lambda | X ~ Gamma(alpha + sum(X), beta + n)
    
    Parameters
    ----------
    data : np.ndarray
        Observed counts (back-transformed expression values).
    alpha : float
        Prior shape parameter (set to the mean of data).
    beta : float
        Prior rate parameter.
    mc : int
        Number of Monte Carlo samples.
    rng : np.random.RandomState or None
        Random state.
    
    Returns
    -------
    np.ndarray, shape (mc,)
        Posterior samples.
    """
    if rng is None:
        rng = np.random.RandomState()
    
    n = len(data)
    post_shape = alpha + np.sum(data)
    post_rate = beta + n
    # Gamma distribution: scipy uses shape and scale=1/rate
    samples = rng.gamma(shape=post_shape, scale=1.0 / post_rate, size=mc)
    return samples


def _find_breakpoints_for_cluster(consensus, bins, cut_cor, rng_seed=42, mc_samples=1000):
    """Find breakpoints in a cluster consensus profile.
    
    Parameters
    ----------
    consensus : np.ndarray, shape (n_genes,)
        Exponentiated consensus profile.
    bins : int
        Window size.
    cut_cor : float
        KS test cutoff.
    rng_seed : int
        Random seed.
    mc_samples : int
        Number of MCMC samples.
    
    Returns
    -------
    list
        List of breakpoint indices.
    """
    rng = np.random.RandomState(rng_seed)
    n = len(consensus)
    
    # Create bin boundaries
    breks = list(range(0, (n // bins - 1) * bins, bins)) + [n - 1]
    
    bre = []
    for i in range(len(breks) - 2):
        seg1 = consensus[breks[i]:breks[i + 1] + 1]
        a1 = max(np.mean(seg1), 0.001)
        posterior1 = _mc_poisson_gamma_numba(seg1, a1, 1.0, mc=mc_samples, seed=rng.randint(0, 2**31))
        
        seg2 = consensus[breks[i + 1] + 1:breks[i + 2] + 1]
        a2 = max(np.mean(seg2), 0.001)
        posterior2 = _mc_poisson_gamma_numba(seg2, a2, 1.0, mc=mc_samples, seed=rng.randint(0, 2**31))
        
        ks_stat, _ = ks_2samp(posterior1, posterior2)
        if ks_stat > cut_cor:
            bre.append(breks[i + 1])
    
    return bre


def cna_mcmc(clu, fttmat, bins=25, cut_cor=0.1, n_cores=1, mc_samples=None):
    """MCMC segmentation of copy number data.
    
    Parameters
    ----------
    clu : np.ndarray, shape (n_cells,)
        Cluster assignments for cells (1-indexed).
    fttmat : np.ndarray, shape (n_genes, n_cells)
        Relative expression matrix (log-scale).
    bins : int
        Window size for segmentation.
    cut_cor : float
        KS test cutoff for breakpoint calling.
    n_cores : int
        Number of parallel workers.
    mc_samples : int or None
        Number of MCMC samples (default: 1000); reduce for speed on small datasets.
    
    Returns
    -------
    dict with keys:
        'logCNA': np.ndarray, shape (n_genes, n_cells) - segmented CNA values
        'breaks': list - breakpoint positions
    """
    n_genes, n_cells = fttmat.shape
    
    # Adaptive MC sample size: fewer samples for small datasets (speed optimization)
    if mc_samples is None:
        mc_samples = 1000 if n_cells > 500 else min(1000, max(500, n_cells * 2))
    
    # Step 1: Compute cluster consensus profiles (median per cluster)
    unique_clusters = sorted(set(clu))
    CON = []
    for cl_id in unique_clusters:
        mask = clu == cl_id
        consensus = np.median(fttmat[:, mask], axis=1)
        CON.append(consensus)
    CON = np.column_stack(CON)
    
    # Back-transform: exp()
    norm_mat_sm = np.exp(CON)
    
    # Step 2: Find breakpoints for each cluster consensus
    BR = set()
    for c in range(norm_mat_sm.shape[1]):
        bre = _find_breakpoints_for_cluster(norm_mat_sm[:, c], bins, cut_cor, rng_seed=42 + c, mc_samples=mc_samples)
        bre_full = sorted(set([0] + bre + [n_genes - 1]))
        BR.update(bre_full)
    
    BR = sorted(BR)
    
    # Step 3: For each cell, compute segment posterior means.
    # With alpha initialized to the segment mean and beta fixed to 1,
    # the Poisson-Gamma posterior mean reduces exactly to the segment mean.
    norm_mat_all = np.exp(fttmat)
    cumsum = np.vstack([
        np.zeros((1, n_cells), dtype=norm_mat_all.dtype),
        np.cumsum(norm_mat_all, axis=0),
    ])

    _LAST_PAR_INFO.update({
        "parallel": False,
        "requested_cores": int(n_cores),
        "effective_cores": 1,
        "tasks": int(len(BR) - 1),
        "chunk_size": 0,
        "mc_samples": int(mc_samples),
        "engine": "closed_form_segment_mean",
    })

    logCNA = np.empty((n_genes, n_cells), dtype=np.float32)
    for left, right in zip(BR[:-1], BR[1:]):
        seg_sum = cumsum[right + 1] - cumsum[left]
        seg_len = max(1, right - left + 1)
        seg_mean = np.maximum(seg_sum / seg_len, 1e-300)
        logCNA[left:right + 1, :] = np.log(seg_mean).astype(np.float32, copy=False)
    
    return {"logCNA": logCNA, "breaks": BR}
