"""Baseline estimation: find diploid normal cells and compute copy-number baseline.

Mirrors baseline.norm.cl.R, baseline.GMM.R, and baseline.synthetic.R from the R package.
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from joblib import Parallel, delayed

try:
    import fastcluster
    HAS_FASTCLUSTER = True
except ImportError:
    fastcluster = None
    HAS_FASTCLUSTER = False

_LAST_CLUSTER_INFO = {
    "step": "hierarchical_cluster",
    "parallel": False,
    "requested_cores": 1,
    "effective_cores": 1,
    "tasks": 0,
    "approximate": False,
    "engine": "scipy.linkage",
}

FULL_CLUSTER_MAX_CELLS = 2000
ADAPTIVE_PCA_COMPONENTS = 512


def get_last_cluster_info():
    return dict(_LAST_CLUSTER_INFO)


def _cluster_sizes(labels):
    _, counts = np.unique(labels, return_counts=True)
    return counts


def _reduce_for_clustering(data, max_components=64):
    n_samples, n_features = data.shape
    if n_samples <= FULL_CLUSTER_MAX_CELLS or n_features <= 256:
        return data, None

    n_components = min(max_components, n_samples - 1, n_features)
    if n_components < 8:
        return data, None

    reducer = PCA(n_components=n_components, svd_solver="randomized", random_state=1234)
    return reducer.fit_transform(data), n_components


def _hierarchical_cluster(
    data,
    n_clusters,
    method="ward",
    metric="euclidean",
    max_cells=65536,
    n_cores=1,
    reduce=True,
    pca_components=64,
):
    """Hierarchical clustering with fastcluster-first execution.

    For Ward + Euclidean clustering, prefer ``fastcluster.linkage_vector``
    regardless of cell count so the main pipeline stays on the same exact
    hierarchical engine for both small and large inputs. The ``max_cells``
    argument is kept for compatibility but is no longer used to switch away
    from fastcluster.
    
    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_features)
        Data matrix (cells × genomic bins).
    n_clusters : int
        Number of clusters to cut.
    method : str
        Linkage method.
    metric : str
        Distance metric.
    max_cells : int
        Retained for backward compatibility; no longer used as a fastcluster
        cutoff.
    reduce : bool
        Whether to apply PCA reduction for large Ward/Euclidean clustering.
        Set to False for R-compatibility paths that should cluster the full
        cells × features matrix through an explicit distance object.
    pca_components : int
        Maximum number of PCA components to retain when reduction is enabled.
    
    Returns
    -------
    labels : np.ndarray
        Cluster labels (1-indexed to match R convention).
    linkage_matrix : np.ndarray or None
        Linkage matrix if available.
    """
    n_samples = data.shape[0]
    _LAST_CLUSTER_INFO.update({
        "requested_cores": int(n_cores),
        "effective_cores": 1,
        "tasks": int(n_samples),
        "approximate": False,
        "engine": "scipy.linkage",
    })
    
    if not reduce:
        if metric == "euclidean" and method.startswith("ward") and HAS_FASTCLUSTER:
            Z = fastcluster.linkage_vector(data, method="ward", metric="euclidean")
            _LAST_CLUSTER_INFO["engine"] = "full_matrix+fastcluster.linkage_vector"
        else:
            dist = pdist(data, metric=metric)
            if HAS_FASTCLUSTER:
                Z = fastcluster.linkage(dist, method=method, preserve_input=True)
                _LAST_CLUSTER_INFO["engine"] = "full_pdist+fastcluster.linkage"
            else:
                Z = linkage(dist, method=method)
                _LAST_CLUSTER_INFO["engine"] = "full_pdist+scipy.linkage"
        labels = fcluster(Z, t=n_clusters, criterion="maxclust")
        return labels, Z

    # Keep Ward + Euclidean on the vectorized full/PCA matrix path.
    if metric == "euclidean" and method.startswith("ward"):
        cluster_data, n_components = _reduce_for_clustering(data, max_components=pca_components)
        if HAS_FASTCLUSTER:
            Z = fastcluster.linkage_vector(cluster_data, method="ward", metric="euclidean")
            _LAST_CLUSTER_INFO["engine"] = "fastcluster.linkage_vector"
        else:
            dist = pdist(cluster_data, metric="euclidean")
            Z = linkage(dist, method=method)
            _LAST_CLUSTER_INFO["engine"] = "scipy.linkage"
        if n_components is not None:
            _LAST_CLUSTER_INFO["approximate"] = True
            engine = "fastcluster.linkage_vector" if HAS_FASTCLUSTER else "scipy.linkage"
            _LAST_CLUSTER_INFO["engine"] = f"pca{n_components}+{engine}"
        labels = fcluster(Z, t=n_clusters, criterion="maxclust")
        return labels, Z

    if metric == "euclidean":
        dist = pdist(data, metric="euclidean")
    else:
        dist = pdist(data, metric=metric)

    if HAS_FASTCLUSTER:
        Z = fastcluster.linkage(dist, method=method, preserve_input=True)
        _LAST_CLUSTER_INFO["engine"] = "fastcluster.linkage"
    else:
        Z = linkage(dist, method=method)

    labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    return labels, Z


def _fit_gmm_3component(data, mu_init=None, sigma_init=None, max_iter=500, tol=1e-8):
    """Fit a 3-component Gaussian Mixture Model (gain, neutral, loss).
    
    Parameters
    ----------
    data : np.ndarray, shape (n,)
        Expression values.
    mu_init : array-like or None
        Initial means for 3 components.
    sigma_init : float or None
        Initial shared standard deviation.
    
    Returns
    -------
    means : np.ndarray, shape (3,)
        Fitted means.
    weights : np.ndarray, shape (3,)
        Fitted weights (lambdas).
    sigma : float
        Fitted shared sigma.
    """
    if sigma_init is None:
        sigma_init = max(0.05, 0.5 * np.std(data))
    if mu_init is None:
        mu_init = [-0.2, 0.0, 0.2]
    
    x = np.asarray(data, dtype=np.float64).ravel()
    means = np.asarray(mu_init, dtype=np.float64).copy()
    weights = np.full(3, 1.0 / 3.0, dtype=np.float64)
    sigma = float(max(sigma_init, 1e-8))
    prev_loglik = -np.inf

    for _ in range(max_iter):
        z = (x[:, None] - means[None, :]) / sigma
        density = np.exp(-0.5 * z * z) / (sigma * np.sqrt(2.0 * np.pi))
        weighted = density * weights[None, :]
        row_sums = weighted.sum(axis=1, keepdims=True)
        row_sums[row_sums <= 0] = np.finfo(np.float64).tiny
        resp = weighted / row_sums

        nk = resp.sum(axis=0)
        nk[nk <= 0] = np.finfo(np.float64).tiny
        weights = nk / len(x)
        means = (resp * x[:, None]).sum(axis=0) / nk
        var = (resp * (x[:, None] - means[None, :]) ** 2).sum() / len(x)
        sigma = float(np.sqrt(max(var, 1e-12)))

        loglik = float(np.log(row_sums.ravel()).sum())
        if abs(loglik - prev_loglik) < tol * (abs(prev_loglik) + tol):
            break
        prev_loglik = loglik
    
    return means, weights, sigma


def baseline_norm_cl(norm_mat_smooth, min_cells=5, n_cores=1, cell_names=None):
    """Find a cluster of diploid cells using integrative clustering + GMM variance test.
    
    Mirrors baseline.norm.cl.R: 
    1. Hierarchical clustering into 6 groups
    2. GMM on each cluster consensus to estimate variance
    3. Cluster with minimum variance is 'confident normal'
    4. Silhouette + F-test to validate
    
    Parameters
    ----------
    norm_mat_smooth : np.ndarray, shape (n_genes, n_cells)
        Smoothed expression matrix.
    min_cells : int
        Minimum cells per cluster.
    n_cores : int
        Number of cores.
    
    Returns
    -------
    dict with keys: 'basel', 'WNS', 'preN', 'cl'
    """
    n_genes, n_cells = norm_mat_smooth.shape
    
    # Hierarchical clustering
    data_t = norm_mat_smooth.T  # cells × genes
    step4_reduce = n_cells > FULL_CLUSTER_MAX_CELLS
    km = 6
    labels, Z = _hierarchical_cluster(
        data_t,
        km,
        method="ward",
        metric="euclidean",
        n_cores=n_cores,
        reduce=step4_reduce,
        pca_components=ADAPTIVE_PCA_COMPONENTS,
    )
    
    # Reduce clusters until all have > min_cells
    while not np.all(_cluster_sizes(labels) > min_cells):
        km -= 1
        if Z is not None:
            labels = fcluster(Z, t=km, criterion="maxclust")
        else:
            labels, Z = _hierarchical_cluster(
                data_t,
                km,
                method="ward",
                metric="euclidean",
                n_cores=n_cores,
                reduce=step4_reduce,
                pca_components=ADAPTIVE_PCA_COMPONENTS,
            )
        if km == 2:
            break
    
    # GMM on each cluster consensus - parallelized for speed
    SDM = []
    SSD = []
    unique_clusters = sorted(set(labels))
    
    def fit_gmm_for_cluster(cl_id):
        """Fit GMM for a single cluster (for parallel execution)."""
        mask = labels == cl_id
        cluster_consensus = np.median(norm_mat_smooth[:, mask], axis=1)
        sx = max(0.05, 0.5 * np.std(cluster_consensus))
        means, weights, sigma = _fit_gmm_3component(cluster_consensus, sigma_init=sx, max_iter=5000)
        return sigma, np.std(cluster_consensus)
    
    # Parallel GMM fitting
    results = Parallel(n_jobs=n_cores)(
        delayed(fit_gmm_for_cluster)(cl_id) for cl_id in unique_clusters
    )
    
    SDM = np.array([r[0] for r in results])
    SSD = np.array([r[1] for r in results])
    
    # Silhouette width for 2-cluster separation
    if Z is not None:
        labels_2 = fcluster(Z, t=2, criterion="maxclust")
    else:
        labels_2, _ = _hierarchical_cluster(
            data_t,
            2,
            method="ward",
            metric="euclidean",
            n_cores=n_cores,
            reduce=step4_reduce,
            pca_components=ADAPTIVE_PCA_COMPONENTS,
        )
    
    # Compute silhouette on a stratified subsample for medium/large datasets.
    # Stratify by labels_2 so each cluster (diploid/aneuploid) is proportionally
    # represented; floor at 200 per cluster to protect rare populations.
    if n_cells > 3000:
        rng = np.random.RandomState(1234)
        target = max(3000, min(int(0.20 * n_cells), 20000))
        idx = []
        for cl in np.unique(labels_2):
            cl_idx = np.where(labels_2 == cl)[0]
            n_take = max(200, int(target * len(cl_idx) / n_cells))
            n_take = min(n_take, len(cl_idx))
            idx.append(rng.choice(cl_idx, size=n_take, replace=False))
        idx = np.concatenate(idx)
        wn = silhouette_score(data_t[idx], labels_2[idx], metric="euclidean")
    else:
        wn = silhouette_score(data_t, labels_2, metric="euclidean")
    
    # F-test: compare max variance to min variance
    from scipy.stats import f as f_dist
    f_stat = max(SDM) ** 2 / min(SDM) ** 2
    PDt = f_dist.sf(f_stat, n_genes, n_genes)
    
    if wn <= 0.15 or not np.all(_cluster_sizes(labels) > min_cells) or PDt > 0.05:
        WNS = "unclassified.prediction"
        print("  low confidence in classification")
    else:
        WNS = ""
    
    # Cluster with minimum sigma is the 'confident normal' cluster
    min_sigma_idx = np.argmin(SDM)
    normal_cluster_id = unique_clusters[min_sigma_idx]
    
    normal_mask = labels == normal_cluster_id
    basel = np.median(norm_mat_smooth[:, normal_mask], axis=1)
    preN_indices = np.where(normal_mask)[0]
    if cell_names is not None:
        cell_names = np.asarray(cell_names, dtype=object)
        preN = cell_names[preN_indices].tolist()
    else:
        preN = preN_indices

    return {
        "basel": basel,
        "WNS": WNS,
        "preN": preN,
        "cl": labels,
    }


def baseline_gmm(CNA_mat, cell_names, max_normal=5, mu_cut=0.05, Nfraq_cut=0.99,
                  RE_before=None, n_cores=1):
    """Identify diploid cells one-by-one using GMM (fallback when clustering is uncertain).
    
    Mirrors baseline.GMM.R.
    
    Parameters
    ----------
    CNA_mat : np.ndarray, shape (n_genes, n_cells)
        Smoothed expression matrix.
    cell_names : list
        Cell names corresponding to columns.
    max_normal : int
        Max number of normal cells to find before stopping.
    mu_cut : float
        Threshold for neutral mean.
    Nfraq_cut : float
        Min fraction of genes in neutral state.
    RE_before : dict or None
        Previous baseline result to fall back on.
    n_cores : int
        Number of cores.
    
    Returns
    -------
    dict with keys: 'basel', 'WNS', 'preN', 'cl'
    """
    n_genes, n_cells = CNA_mat.shape
    N_normal = []
    N_normal_labels = []
    
    for m in range(n_cells):
        sam = CNA_mat[:, m]
        sg = max(0.05, 0.5 * np.std(sam))
        
        means, weights, sigma = _fit_gmm_3component(sam, sigma_init=sg, max_iter=500)
        
        # Check if any component mean is near zero (neutral)
        neutral_mask = np.abs(means) <= mu_cut
        s = np.sum(neutral_mask)
        
        if s >= 1:
            frq = np.sum(weights[neutral_mask])
            if frq > Nfraq_cut:
                pred = "diploid"
            else:
                pred = "aneuploid"
        else:
            pred = "aneuploid"
        
        N_normal_labels.append(pred)
        
        if pred == "diploid":
            N_normal.append(cell_names[m])
        
        if len(N_normal) >= max_normal:
            break
    
    # Hierarchical clustering for the full dataset
    data_t = CNA_mat.T
    step4_reduce = n_cells > FULL_CLUSTER_MAX_CELLS
    km = 6
    labels, Z = _hierarchical_cluster(
        data_t,
        km,
        method="ward",
        metric="euclidean",
        n_cores=n_cores,
        reduce=step4_reduce,
        pca_components=ADAPTIVE_PCA_COMPONENTS,
    )
    
    if len(N_normal) > 2:
        WNS = ""
        preN = N_normal
        normal_mask = np.isin(np.asarray(cell_names, dtype=object), np.asarray(preN, dtype=object))
        basel = np.mean(CNA_mat[:, normal_mask], axis=1)
        return {"basel": basel, "WNS": WNS, "preN": preN, "cl": labels}
    else:
        if RE_before is not None:
            return RE_before
        else:
            # Fallback: use the full dataset median as baseline
            WNS = "unclassified.prediction"
            return {"basel": np.median(CNA_mat, axis=1), "WNS": WNS, 
                    "preN": N_normal, "cl": labels}


def baseline_synthetic(norm_mat, min_cells=10, n_cores=1):
    """Estimate baseline using synthetic normal profiles (for cell line data).
    
    Mirrors baseline.synthetic.R.
    
    Parameters
    ----------
    norm_mat : np.ndarray, shape (n_genes, n_cells)
        Smoothed expression matrix.
    min_cells : int
        Min cells per cluster.
    n_cores : int
        Number of cores.
    
    Returns
    -------
    dict with keys: 'expr_relat', 'cl'
    """
    n_genes, n_cells = norm_mat.shape
    data_t = norm_mat.T
    
    km = 6
    labels, Z = _hierarchical_cluster(data_t, km, method="ward", metric="euclidean")
    
    while not all(np.bincount(labels)[np.bincount(labels) > 0] > min_cells):
        km -= 1
        if Z is not None:
            labels = fcluster(Z, t=km, criterion="maxclust")
        else:
            labels, Z = _hierarchical_cluster(data_t, km, method="ward", metric="euclidean")
        if km == 2:
            break
    
    rng = np.random.RandomState(123)
    expr_relat_parts = []
    unique_clusters = sorted(set(labels))
    
    for cl_id in unique_clusters:
        mask = labels == cl_id
        data_c = norm_mat[:, mask]
        sd_per_gene = np.std(data_c, axis=1)
        syn_norm = rng.normal(0, sd_per_gene)
        relat = data_c - syn_norm[:, np.newaxis]
        expr_relat_parts.append(relat)
    
    expr_relat = np.hstack(expr_relat_parts)
    
    # Reorder to match original cell order
    cluster_order = []
    for cl_id in unique_clusters:
        cluster_order.extend(np.where(labels == cl_id)[0])
    
    # Create inverse permutation
    inv_perm = np.argsort(cluster_order)
    expr_relat = expr_relat[:, inv_perm]
    
    return {"expr_relat": expr_relat, "cl": labels}
