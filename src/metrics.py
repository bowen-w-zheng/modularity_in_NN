"""
Modularity metrics from Johnston et al.

Implements:
1. Contextual Fraction (explicit modularity)
2. Subspace Specialization (implicit modularity)
3. Optional: Clustering via BIC
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from typing import Dict, List, Tuple


def contextual_fraction(act_by_ctx: Dict[int, np.ndarray], thresh: float = 0.01) -> float:
    """
    Compute Contextual Fraction: fraction of units selective for exactly one context.

    From Johnston et al. Methods M7:
    "For each context, we computed the mean activity of each hidden unit across 1000 stimuli.
    A unit was considered context-selective if its mean activity was above threshold (0.01)
    in exactly one context."

    Args:
        act_by_ctx: Dictionary mapping context_id -> (n_samples, n_hidden) activations
        thresh: Threshold for determining if a unit is "active" in a context

    Returns:
        cf: Contextual fraction (between 0 and 1)
    """
    # Get number of contexts and hidden units
    n_contexts = len(act_by_ctx)
    context_ids = sorted(act_by_ctx.keys())

    # Get hidden dimension from first context
    n_hidden = act_by_ctx[context_ids[0]].shape[1]

    # Compute mean activity per context per unit
    mean_activity = np.zeros((n_contexts, n_hidden))
    for i, ctx in enumerate(context_ids):
        mean_activity[i, :] = np.mean(act_by_ctx[ctx], axis=0)

    # For each unit, count how many contexts it's active in
    is_active = mean_activity > thresh  # (n_contexts, n_hidden)
    n_active_contexts = np.sum(is_active, axis=0)  # (n_hidden,)

    # Count units active in exactly one context
    n_selective = np.sum(n_active_contexts == 1)

    # Contextual fraction
    cf = n_selective / n_hidden

    return cf


def alignment_index(U: np.ndarray, V: np.ndarray) -> float:
    """
    Compute alignment index between two subspaces.

    A(U, V) = (1/d) * ||U^T V||_F^2

    where U and V are orthonormal bases (columns are basis vectors),
    and d is the dimension.

    Args:
        U: (n_features, d_U) orthonormal basis
        V: (n_features, d_V) orthonormal basis

    Returns:
        alignment: Alignment index
    """
    # Compute U^T V
    UTdotV = U.T @ V  # (d_U, d_V)

    # Frobenius norm squared
    frob_sq = np.sum(UTdotV ** 2)

    # Dimension (use minimum of the two)
    d = min(U.shape[1], V.shape[1])

    if d == 0:
        return 0.0

    alignment = frob_sq / d

    return alignment


def compute_pca_subspace(activations: np.ndarray, var_threshold: float = 1e-10) -> np.ndarray:
    """
    Compute PCA subspace, keeping components with variance > threshold.

    Args:
        activations: (n_samples, n_features) activations
        var_threshold: Minimum variance to keep a component

    Returns:
        basis: (n_features, n_components) orthonormal basis vectors
    """
    if activations.shape[0] < 2:
        # Not enough samples for PCA
        return np.zeros((activations.shape[1], 0))

    n_samples, n_features = activations.shape

    # PCA with automatic centering
    # Use 'full' solver for deterministic results and better accuracy
    n_components = min(n_samples - 1, n_features)  # Max components for full SVD

    # Use full SVD for small/medium datasets (more accurate than randomized)
    if n_samples <= 1000:
        pca = PCA(n_components=n_components, svd_solver='full', random_state=42)
    else:
        # Use randomized for large datasets
        pca = PCA(n_components=n_components, svd_solver='randomized', random_state=42)

    pca.fit(activations)

    # Keep components with explained variance > threshold
    explained_var = pca.explained_variance_
    keep_components = explained_var > var_threshold

    if not np.any(keep_components):
        # No components above threshold
        return np.zeros((n_features, 0))

    # Get basis vectors
    # Components are rows of pca.components_
    basis = pca.components_[keep_components, :].T  # (n_features, n_components)

    return basis


def subspace_specialization(
    acts: np.ndarray,
    latents: Dict,
    ctx_index: np.ndarray,
    var_threshold: float = 1e-10
) -> float:
    """
    Compute Subspace Specialization metric.

    From Johnston et al. Methods M8:
    "For each latent variable, we split the dataset by that variable's value (0 or 1).
    We computed PCA subspaces for each split, keeping components with explained variance > 1e-10.
    We computed the alignment index between subspaces for context vs non-context variables.
    Subspace specialization = max(0, log(mean_alignment_nonctx / mean_alignment_ctx))."

    Args:
        acts: (N, n_hidden) hidden activations for all samples
        latents: Dictionary with z_dec, z_ctx, z_irr, z_all
        ctx_index: (N,) context index for each sample
        var_threshold: Variance threshold for PCA

    Returns:
        S: Subspace specialization index
    """
    z_all = latents['z_all']  # (N, D+C+R)
    z_ctx = latents['z_ctx']  # (N, C)

    N, n_latents = z_all.shape
    _, C = z_ctx.shape

    # Identify which latent variables are context variables
    # z_all = [z_dec, z_ctx, z_irr]
    D = latents['z_dec'].shape[1]
    R = latents['z_irr'].shape[1]

    # Context variable indices in z_all
    ctx_var_indices = list(range(D, D + C))
    # Non-context variable indices
    nonctx_var_indices = list(range(D)) + list(range(D + C, D + C + R))

    # Compute alignments for context pairs
    ctx_alignments = []
    for var_idx in ctx_var_indices:
        # Split by this variable
        mask_0 = z_all[:, var_idx] == 0
        mask_1 = z_all[:, var_idx] == 1

        acts_0 = acts[mask_0]
        acts_1 = acts[mask_1]

        if acts_0.shape[0] > 1 and acts_1.shape[0] > 1:
            # Compute PCA subspaces
            basis_0 = compute_pca_subspace(acts_0, var_threshold)
            basis_1 = compute_pca_subspace(acts_1, var_threshold)

            if basis_0.shape[1] > 0 and basis_1.shape[1] > 0:
                align = alignment_index(basis_0, basis_1)
                ctx_alignments.append(align)

    # Compute alignments for non-context pairs
    nonctx_alignments = []
    for var_idx in nonctx_var_indices:
        mask_0 = z_all[:, var_idx] == 0
        mask_1 = z_all[:, var_idx] == 1

        acts_0 = acts[mask_0]
        acts_1 = acts[mask_1]

        if acts_0.shape[0] > 1 and acts_1.shape[0] > 1:
            basis_0 = compute_pca_subspace(acts_0, var_threshold)
            basis_1 = compute_pca_subspace(acts_1, var_threshold)

            if basis_0.shape[1] > 0 and basis_1.shape[1] > 0:
                align = alignment_index(basis_0, basis_1)
                nonctx_alignments.append(align)

    # Mean alignments
    if len(ctx_alignments) == 0 or len(nonctx_alignments) == 0:
        return 0.0

    mean_ctx = np.mean(ctx_alignments)
    mean_nonctx = np.mean(nonctx_alignments)

    # Avoid division by zero
    if mean_ctx <= 0:
        return 0.0

    # Rectified log ratio
    S = max(0.0, np.log(mean_nonctx / mean_ctx))

    return S


def clustering_bic(mean_activity: np.ndarray, max_k: int = 10) -> Tuple[int, np.ndarray]:
    """
    Cluster units based on mean activity across contexts using Gaussian Mixture Model.
    Select number of clusters via BIC.

    From Johnston et al. Methods M6.

    Args:
        mean_activity: (n_hidden, n_contexts) mean activity per unit per context
        max_k: Maximum number of clusters to try

    Returns:
        best_k: Number of clusters with highest BIC
        labels: (n_hidden,) cluster assignments
    """
    n_hidden, n_contexts = mean_activity.shape

    # Try different numbers of clusters
    bic_scores = []
    models = []

    for k in range(1, min(max_k + 1, n_hidden + 1)):
        gmm = GaussianMixture(n_components=k, random_state=0, covariance_type='full')
        gmm.fit(mean_activity)
        bic = gmm.bic(mean_activity)
        bic_scores.append(bic)
        models.append(gmm)

    # BIC: lower is better, but we want max
    # Actually, BIC is defined such that higher BIC is better in some conventions
    # sklearn's BIC is lower is better, so we want minimum
    best_idx = np.argmin(bic_scores)
    best_k = best_idx + 1
    labels = models[best_idx].predict(mean_activity)

    return best_k, labels


def compute_all_metrics(
    model,
    X: np.ndarray,
    ctx_index: np.ndarray,
    latents: Dict,
    device: str = 'cpu',
    n_samples_per_ctx: int = 1000
) -> Dict:
    """
    Compute all modularity metrics.

    Args:
        model: Trained ContextMLP
        X: (N, in_dim) inputs
        ctx_index: (N,) context indices
        latents: Latent variables dictionary
        device: Device for computation
        n_samples_per_ctx: Number of samples to use per context for metrics

    Returns:
        metrics: Dictionary with all computed metrics
    """
    from src.model import get_hidden_activations
    import torch

    # Get hidden activations
    X_torch = torch.from_numpy(X).float()
    acts = get_hidden_activations(model, X_torch, device)

    # Ensure no NaN or Inf in activations
    if np.any(np.isnan(acts)) or np.any(np.isinf(acts)):
        print("Warning: NaN or Inf in activations, replacing with zeros")
        acts = np.nan_to_num(acts, nan=0.0, posinf=0.0, neginf=0.0)

    # Sample n_samples_per_ctx per context for contextual fraction
    n_contexts = len(np.unique(ctx_index))
    act_by_ctx = {}

    for ctx in range(n_contexts):
        mask = ctx_index == ctx
        indices = np.where(mask)[0]

        if len(indices) == 0:
            print(f"Warning: No samples for context {ctx}")
            continue

        # Sample up to n_samples_per_ctx
        n_to_sample = min(len(indices), n_samples_per_ctx)
        if len(indices) > n_samples_per_ctx:
            sampled_indices = np.random.choice(indices, n_to_sample, replace=False)
        else:
            sampled_indices = indices

        act_by_ctx[ctx] = acts[sampled_indices]

    # Compute Contextual Fraction
    try:
        cf = contextual_fraction(act_by_ctx, thresh=0.01)
    except Exception as e:
        print(f"Warning: CF computation failed: {e}")
        cf = 0.0

    # Compute Subspace Specialization (use all data)
    try:
        ss = subspace_specialization(acts, latents, ctx_index, var_threshold=1e-10)
    except Exception as e:
        print(f"Warning: SS computation failed: {e}")
        ss = 0.0

    # Optional: Clustering BIC
    # Compute mean activity matrix for clustering
    mean_activity_matrix = np.zeros((acts.shape[1], n_contexts))
    for ctx in range(n_contexts):
        if ctx in act_by_ctx:
            mean_activity_matrix[:, ctx] = np.mean(act_by_ctx[ctx], axis=0)

    try:
        best_k, cluster_labels = clustering_bic(mean_activity_matrix, max_k=n_contexts)
    except Exception as e:
        print(f"Warning: Clustering failed: {e}")
        best_k = 1
        cluster_labels = np.zeros(acts.shape[1], dtype=int)

    metrics = {
        'contextual_fraction': float(cf),
        'subspace_specialization': float(ss),
        'best_k_clusters': int(best_k),
        'cluster_labels': cluster_labels
    }

    return metrics
