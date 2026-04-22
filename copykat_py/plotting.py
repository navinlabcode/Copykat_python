"""Heatmap plotting for CNA results, mirroring heatmap.3.R visualizations."""

import sys
import time
from contextlib import contextmanager
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist

# Try to use fastcluster for faster linkage computation on large datasets
try:
    import fastcluster
    from fastcluster import linkage as fastcluster_linkage
    HAS_FASTCLUSTER = True
except ImportError:
    fastcluster = None
    HAS_FASTCLUSTER = False

try:
    from sklearn.cluster import MiniBatchKMeans, KMeans
    HAS_SKLEARN_CLUSTER = True
except ImportError:
    MiniBatchKMeans = None
    KMeans = None
    HAS_SKLEARN_CLUSTER = False

try:
    from sklearn.decomposition import TruncatedSVD
    HAS_SKLEARN_DECOMP = True
except ImportError:
    TruncatedSVD = None
    HAS_SKLEARN_DECOMP = False

try:
    from threadpoolctl import threadpool_limits
    HAS_THREADPOOLCTL = True
except ImportError:
    threadpool_limits = None
    HAS_THREADPOOLCTL = False


@contextmanager
def _thread_limited_numeric_ops(max_threads=1):
    """Limit BLAS/OpenMP thread fan-out during plotting-time clustering."""
    if not HAS_THREADPOOLCTL:
        yield
        return

    try:
        with threadpool_limits(limits=max_threads, user_api="blas"):
            with threadpool_limits(limits=max_threads, user_api="openmp"):
                yield
    except ValueError:
        with threadpool_limits(limits=max_threads):
            yield


def _build_plot_embedding(mat, random_state=1234):
    """Return a float32 embedding optimized for large-cell heatmap ordering."""
    data = np.asarray(mat.T, dtype=np.float32, order="C")
    n_cells, n_features = data.shape
    if (
        n_cells < 12000
        or n_features <= 64
        or not HAS_SKLEARN_DECOMP
    ):
        return data

    n_components = min(48, n_features - 1)
    if n_components < 8:
        return data

    with _thread_limited_numeric_ops(1):
        reducer = TruncatedSVD(n_components=n_components, random_state=random_state)
        reduced = reducer.fit_transform(data)
    return np.asarray(reduced, dtype=np.float32, order="C")


def _simple_cell_order(mat, predictions=None):
    """Cheap fallback ordering used when clustering is unavailable or unsafe."""
    n_cells = mat.shape[1]
    if predictions is not None:
        pred_list = list(predictions.values())
        pred_rank = np.array(
            [
                0 if "aneuploid" in str(p)
                else 1 if "diploid" in str(p)
                else 2
                for p in pred_list
            ],
            dtype=np.int16,
        )
        cna_magnitude = np.sum(np.abs(mat), axis=0)
        return np.lexsort((-cna_magnitude, pred_rank))

    cna_magnitude = np.sum(np.abs(mat), axis=0)
    return np.argsort(cna_magnitude)[::-1]


def _compute_distance(mat, distance="euclidean", n_cores=1):
    """Compute distance matrix for cells.
    
    Parameters
    ----------
    mat : np.ndarray, shape (n_bins, n_cells)
        CNA matrix.
    distance : str
        "euclidean", "pearson", or "spearman".
    
    Returns
    -------
    dist : np.ndarray
        Condensed distance matrix.
    """
    if distance == "euclidean":
        return pdist(mat.T, metric="euclidean")
    elif distance == "pearson":
        corr = np.corrcoef(mat.T)
        corr = np.clip(corr, -1, 1)
        return pdist(1 - corr)
    elif distance == "spearman":
        from scipy.stats import spearmanr
        corr, _ = spearmanr(mat, axis=0)
        corr = np.clip(corr, -1, 1)
        return pdist(1 - corr)
    else:
        return pdist(mat.T, metric=distance)


def _safe_linkage(mat, distance="euclidean", method="ward", n_cores=1, max_cells=65536):
    """Compute linkage with fastcluster-first execution.

    For Ward + Euclidean linkage, keep the full matrix on fastcluster
    regardless of cell count so plotting matches the main clustering path.
    The ``max_cells`` argument is retained for compatibility.
    """
    n_cells = mat.shape[1]

    if HAS_FASTCLUSTER and distance == "euclidean" and method.startswith("ward"):
        return fastcluster.linkage_vector(mat.T, method="ward", metric="euclidean")

    dist = _compute_distance(mat, distance, n_cores)
    if HAS_FASTCLUSTER:
        return fastcluster_linkage(dist, method=method)
    return linkage(dist, method=method if distance == "euclidean" else "ward")


def _clustered_block_layout(mat, n_clusters=128, random_state=1234):
    """Fast cell ordering plus a cluster-level dendrogram layout."""
    n_cells = mat.shape[1]
    if not HAS_SKLEARN_CLUSTER or n_cells <= 1:
        return {
            "cell_order": np.arange(n_cells, dtype=int),
            "labels": np.zeros(n_cells, dtype=int),
            "cluster_order": [0],
            "cluster_sizes": np.array([n_cells], dtype=int),
            "centroid_linkage": None,
        }

    n_clusters = max(2, min(int(n_clusters), n_cells))
    data = _build_plot_embedding(mat, random_state=random_state)

    if n_cells > 4000:
        model = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            batch_size=min(2048, n_cells),
            n_init=1 if n_cells >= 20000 else 3,
        )
    else:
        model = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10,
        )

    with _thread_limited_numeric_ops(1):
        labels = model.fit_predict(data)
    centroids = model.cluster_centers_
    present_clusters = sorted(np.unique(labels).tolist())

    if len(present_clusters) > 1:
        centroid_mat = centroids[present_clusters]
        with _thread_limited_numeric_ops(1):
            if HAS_FASTCLUSTER:
                centroid_linkage = fastcluster.linkage_vector(centroid_mat, method="ward", metric="euclidean")
            else:
                centroid_linkage = linkage(pdist(centroid_mat, metric="euclidean"), method="ward")
        cluster_order_local = dendrogram(centroid_linkage, no_plot=True)["leaves"]
        cluster_order = [present_clusters[i] for i in cluster_order_local]
    else:
        centroid_linkage = None
        cluster_order = [0]

    ordered_cells = []
    cluster_sizes = []
    for cluster_id in cluster_order:
        idx = np.where(labels == cluster_id)[0]
        if len(idx) == 0:
            continue
        cluster_sizes.append(len(idx))
        centroid = centroids[cluster_id]
        denom = np.linalg.norm(centroid)
        if denom > 0:
            scores = data[idx] @ centroid / denom
            idx = idx[np.argsort(scores)]
        ordered_cells.extend(idx.tolist())

    if len(ordered_cells) != n_cells:
        missing = sorted(set(range(n_cells)) - set(ordered_cells))
        ordered_cells.extend(missing)
    return {
        "cell_order": np.asarray(ordered_cells, dtype=int),
        "labels": labels,
        "cluster_order": cluster_order,
        "cluster_sizes": np.asarray(cluster_sizes, dtype=int),
        "centroid_linkage": centroid_linkage,
    }


def _clustered_block_order(mat, n_clusters=128, random_state=1234):
    return _clustered_block_layout(mat, n_clusters=n_clusters, random_state=random_state)["cell_order"]


def _draw_cluster_dendrogram(ax, centroid_linkage, cluster_sizes):
    """Render a cluster-level dendrogram aligned to block heights in the heatmap."""
    if centroid_linkage is None or len(cluster_sizes) <= 1:
        ax.text(0.5, 0.5, "cluster\ndendrogram\nunavailable", ha="center", va="center", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    dendro = dendrogram(centroid_linkage, no_plot=True)
    block_edges = np.concatenate([[0.0], np.cumsum(cluster_sizes, dtype=float)])
    block_centers = 0.5 * (block_edges[:-1] + block_edges[1:]) - 0.5

    src_centers = np.array([5.0 + 10.0 * i for i in range(len(cluster_sizes))], dtype=float)

    def _map_y(yvals):
        return np.interp(yvals, src_centers, block_centers)

    segments = []
    max_height = 0.0
    for icoord, dcoord in zip(dendro["icoord"], dendro["dcoord"]):
        ys = _map_y(np.asarray(icoord, dtype=float))
        xs = np.asarray(dcoord, dtype=float)
        max_height = max(max_height, float(xs.max()))
        points = np.column_stack([xs, ys])
        segments.extend([
            [points[0], points[1]],
            [points[1], points[2]],
            [points[2], points[3]],
        ])

    lc = LineCollection(segments, colors="black", linewidths=0.8)
    ax.add_collection(lc)
    ax.set_xlim(max_height * 1.05 if max_height > 0 else 1.0, 0.0)
    ax.set_ylim(block_edges[-1] - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])


def _safe_dendrogram_with_recursion_management(Z, ax, n_cells):
    """Safely plot dendrogram with increased recursion limit.
    
    Parameters
    ----------
    Z : np.ndarray
        Linkage matrix from scipy.cluster.hierarchy.linkage.
    ax : matplotlib.axes.Axes
        Axes object to plot on.
    n_cells : int
        Number of cells (for estimating required recursion depth).
    """
    # Estimate required recursion depth based on number of cells
    # Rough estimate: depth ~ log2(n_cells) * 2
    estimated_depth = max(1000, int(np.log2(n_cells + 1) * 10 + 500))
    old_limit = sys.getrecursionlimit()
    
    try:
        # Temporarily increase recursion limit
        sys.setrecursionlimit(min(estimated_depth, 1000000))  # Cap at 1M to avoid stack overflow
        dendrogram(Z, orientation="left", ax=ax, no_labels=True,
                   color_threshold=0, above_threshold_color="black",
                   link_color_func=lambda _: "black")
    except RecursionError:
        # Still failed, draw placeholder
        ax.text(0.5, 0.5, "dendrogram\nskipped\n(recursion)", ha="center", va="center", fontsize=8)
    finally:
        # Restore original recursion limit
        sys.setrecursionlimit(old_limit)


def plot_heatmap(mat, chrom_info, predictions=None, sample_name="",
                 distance="euclidean", n_cores=1, WNS1="", WNS="",
                 output_path=None):
    """Plot CNA heatmap with hierarchical clustering dendrogram.

    Layout mirrors R copykat heatmap.3:
      Row 0 (thin):  [empty] [empty] [chr bar]
      Row 1 (main):  [dendrogram] [pred sidebar] [heatmap]
    
    For datasets > 200k cells, uses optimized approximate clustering.

    Parameters
    ----------
    mat : np.ndarray, shape (n_bins, n_cells)
        CNA values (bins x cells).
    chrom_info : np.ndarray
        Chromosome IDs for each bin.
    predictions : dict or None
        Cell name -> "aneuploid"/"diploid" predictions.
    sample_name : str
        Sample name for title.
    distance : str
        Distance metric.
    n_cores : int
        Number of cores.
    WNS1 : str
        Data quality warning.
    WNS : str
        Classification warning.
    output_path : str or None
        Path to save figure.
    """
    if output_path is None:
        output_path = f"{sample_name}_copykat_heatmap.png"

    plot_start = time.perf_counter()
    n_bins, n_cells = mat.shape

    # Determine cell ordering strategy based on dataset size
    # Strategy 1: Standard hierarchical clustering (up to 20k cells)
    # Strategy 2: K-means clustering (20k-200k cells)
    # Strategy 3: Simple ordering by prediction/CNA (>200k cells)
    
    max_dendro_cells = 3000
    max_kmeans_cells = 200000
    Z = None
    Z_summary = None
    cell_order = None
    skip_dendrogram = False
    cluster_sizes = None
    
    if n_cells <= max_dendro_cells:
        # Full hierarchical clustering with dendrogram
        print(f"  Step 10a: Computing dendrogram for {n_cells} cells...")
        try:
            Z = _safe_linkage(mat, distance, "ward", n_cores)
            # Use safe dendrogram with recursion management
            dn_temp = {}
            try:
                old_limit = sys.getrecursionlimit()
                estimated_depth = max(1000, int(np.log2(n_cells + 1) * 10 + 500))
                sys.setrecursionlimit(min(estimated_depth, 1000000))
                dn_temp = dendrogram(Z, no_plot=True)
                sys.setrecursionlimit(old_limit)
            except RecursionError:
                sys.setrecursionlimit(old_limit)
                print("  WARNING: dendrogram recursion limit reached; using K-means ordering.")
                dn_temp = {}
            
            if dn_temp:
                cell_order = dn_temp["leaves"]
            else:
                # Fallback to fast block ordering
                cell_order = _clustered_block_order(mat, n_clusters=min(96, max(24, n_cells // 40)))
        except Exception as e:
            print(f"  WARNING: dendrogram computation failed ({e}); using fast ordering.")
            cell_order = _clustered_block_order(mat, n_clusters=min(96, max(24, n_cells // 40)))
    
    elif n_cells <= max_kmeans_cells:
        # Fast clustered ordering for large datasets with a summarized dendrogram.
        print(f"  Step 10a: Computing fast clustered ordering for {n_cells} cells...")
        try:
            layout = _clustered_block_layout(mat, n_clusters=min(128, max(32, n_cells // 160)))
            cell_order = layout["cell_order"]
            Z_summary = layout["centroid_linkage"]
            cluster_sizes = layout["cluster_sizes"]
        except Exception as e:
            print(f"  WARNING: fast clustered ordering failed ({e}); using simple ordering.")
            skip_dendrogram = True
            cell_order = _simple_cell_order(mat, predictions=predictions)
    
    else:
        # For very large datasets (>200k cells), use simple ordering
        print(f"  Step 10a: Using simple ordering for {n_cells} cells (too large for clustering).")
        skip_dendrogram = True
        cell_order = _simple_cell_order(mat, predictions=predictions)

    # Ensure cell_order is valid
    if cell_order is None:
        cell_order = np.arange(n_cells)
    
    # Reorder matrix
    mat_ordered = mat[:, cell_order]

    # --- Figure & GridSpec ------------------------------------------------
    h = 10 if n_cells < 3000 else 15
    has_pred = predictions is not None

    # Columns: dendrogram | heatmap | pred sidebar | colorbar | legend
    # Rows:    chr bar (thin) | main
    n_cols = 5 if has_pred else 3
    if has_pred:
        width_ratios = [8, 50, 1.4, 1.2, 5.5]
        col_dendro, col_heat, col_pred, col_cbar, col_legend = 0, 1, 2, 3, 4
    else:
        width_ratios = [8, 50, 1.2]
        col_dendro, col_heat, col_cbar = 0, 1, 2

    gs = GridSpec(2, n_cols, height_ratios=[1, 50], width_ratios=width_ratios,
                  hspace=0.02, wspace=0.02)
    fig = plt.figure(figsize=(22, h))

    # --- Chromosome bar (top, above heatmap only) -------------------------
    ax_chr = fig.add_subplot(gs[0, col_heat])
    chr_colors = (chrom_info.astype(int) % 2).astype(float)
    ax_chr.imshow(chr_colors.reshape(1, -1), aspect="auto", cmap="binary",
                  interpolation="nearest")
    ax_chr.set_xticks([])
    ax_chr.set_yticks([])
    ax_chr.set_ylabel("Chr", fontsize=8)

    # --- Main heatmap -----------------------------------------------------
    ax_heat = fig.add_subplot(gs[1, col_heat])

    vmin, vmax = -0.5, 0.5
    cmap = plt.cm.RdBu_r
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    im = ax_heat.imshow(mat_ordered.T, aspect="auto", cmap=cmap, norm=norm,
                        interpolation="nearest")
    ax_heat.set_xlabel("Genomic position")
    ax_heat.set_yticks([])
    title_parts = [p for p in (WNS1, WNS) if p]
    ax_heat.set_title("; ".join(title_parts) if title_parts else "", fontsize=12)

    # Chromosome boundaries
    chrom_changes = np.where(np.diff(chrom_info.astype(int)))[0]
    for pos in chrom_changes:
        ax_heat.axvline(x=pos, color="gray", linewidth=0.3, alpha=0.5)

    # --- Dendrogram (left) ------------------------------------------------
    ax_dendro = fig.add_subplot(gs[1, col_dendro])
    if Z is not None and not skip_dendrogram:
        print(f"  Step 10b: Rendering dendrogram...")
        _safe_dendrogram_with_recursion_management(Z, ax_dendro, n_cells)
    elif Z_summary is not None and cluster_sizes is not None:
        print(f"  Step 10b: Rendering cluster dendrogram...")
        _draw_cluster_dendrogram(ax_dendro, Z_summary, cluster_sizes)
    else:
        msg = "fast order\n(no dendrogram)" if skip_dendrogram else "dendrogram\nskipped"
        ax_dendro.text(0.5, 0.5, msg, ha="center", va="center", fontsize=9)
    ax_dendro.set_xticks([])
    ax_dendro.set_yticks([])
    if Z is not None and not skip_dendrogram:
        # Invert so first leaf is at top (matching imshow origin='upper')
        ax_dendro.invert_yaxis()
        # Tighten y-limits to leaf range so leaves align with heatmap rows
        leaf_step = 10  # scipy default leaf spacing
        leaf_min = 5
        leaf_max = 5 + (n_cells - 1) * leaf_step
        ax_dendro.set_ylim(leaf_max + leaf_step / 2, leaf_min - leaf_step / 2)
    for spine in ax_dendro.spines.values():
        spine.set_visible(False)

    # --- Prediction sidebar (outside heatmap area, on the right) ----------
    if has_pred:
        ax_pred = fig.add_subplot(gs[1, col_pred], sharey=ax_heat)
        pred_colors = np.zeros(n_cells)
        pred_list = list(predictions.values())
        pred_ordered = [pred_list[i] if i < len(pred_list) else "not.defined"
                        for i in cell_order]
        for i, p in enumerate(pred_ordered):
            if "aneuploid" in str(p):
                pred_colors[i] = 1.0
            elif "diploid" in str(p):
                pred_colors[i] = 0.0
            else:
                pred_colors[i] = 0.5

        sidebar_cmap = mcolors.ListedColormap(["#1B9E77", "#7570B3", "#D95F02"])
        sidebar_norm = mcolors.BoundaryNorm([0, 0.3, 0.7, 1.0], 3)
        ax_pred.imshow(pred_colors.reshape(-1, 1), aspect="auto",
                       cmap=sidebar_cmap, norm=sidebar_norm,
                       interpolation="nearest")
        ax_pred.set_xticks([])
        ax_pred.set_yticks([])
        ax_pred.set_title("Pred", fontsize=8, pad=4)
        for spine in ax_pred.spines.values():
            spine.set_linewidth(0.6)

    # Hide empty top-left cells
    for c in range(n_cols):
        if c == col_heat:
            continue
        ax_empty = fig.add_subplot(gs[0, c])
        ax_empty.axis("off")

    # --- Legend for predictions -------------------------------------------
    if has_pred:
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#D95F02", label="pred aneuploid"),
            Patch(facecolor="#1B9E77", label="pred diploid"),
            Patch(facecolor="#7570B3", label="pred not.defined"),
        ]
        ax_legend = fig.add_subplot(gs[1, col_legend])
        ax_legend.axis("off")
        ax_legend.legend(
            handles=legend_elements,
            loc="upper left",
            bbox_to_anchor=(0.0, 1.0),
            fontsize=9,
            frameon=True,
            fancybox=False,
            edgecolor="black",
            borderaxespad=0.0,
        )

    # --- Colorbar (vertical, right side) ----------------------------------
    ax_cbar = fig.add_subplot(gs[1, col_cbar])
    cbar = fig.colorbar(im, cax=ax_cbar, orientation="vertical")
    cbar.set_label("Relative CNA")

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Heatmap saved to {output_path} in {time.perf_counter() - plot_start:.2f}s")
