"""Task sampling and target generation.

Implements context-dependent linear classification tasks.
"""
import numpy as np
from typing import Dict


def sample_tasks(T: int, D: int, C: int, class_balance: str, rng: np.random.Generator) -> Dict:
    """
    Sample T linear classification tasks, each context-dependent.

    For each task t and context c, generate a random linear separator:
        y_{t,c} = 1{w_{t,c}^T z_dec > tau_{t,c}}

    Args:
        T: Number of tasks
        D: Number of decision variables
        C: Number of context variables (contexts = 2^C)
        class_balance: 'median' or 'auto' thresholding
        rng: Numpy random generator

    Returns:
        Dictionary with:
            - W: (T, n_contexts, D) weight vectors in {-1, 0, 1}
            - tau: (T, n_contexts) thresholds
    """
    n_contexts = 2 ** C

    # Sample random weights from {-1, 0, 1}^D (not all zero)
    W = rng.integers(-1, 2, size=(T, n_contexts, D))

    # Ensure not all zero (re-sample if needed)
    for t in range(T):
        for c in range(n_contexts):
            while np.all(W[t, c] == 0):
                W[t, c] = rng.integers(-1, 2, size=D)

    # Initialize thresholds
    tau = np.zeros((T, n_contexts))

    if class_balance == 'median':
        # Will compute median after seeing data
        # For now, use middle of range
        for t in range(T):
            for c in range(n_contexts):
                w = W[t, c]
                min_val = np.sum(np.minimum(w, 0))
                max_val = np.sum(np.maximum(w, 0))
                tau[t, c] = (min_val + max_val) / 2.0
    else:  # 'auto'
        # Grid search to keep class ratio in [0.35, 0.65]
        # Will be computed when building targets with actual data
        for t in range(T):
            for c in range(n_contexts):
                w = W[t, c]
                min_val = np.sum(np.minimum(w, 0))
                max_val = np.sum(np.maximum(w, 0))
                # Start with middle range
                tau[t, c] = min_val + 0.5 * (max_val - min_val)

    return {
        'W': W,
        'tau': tau,
        'class_balance': class_balance
    }


def build_targets(latents: Dict, tasks: Dict) -> np.ndarray:
    """
    Build target labels for all samples and tasks.

    Args:
        latents: Dictionary from sample_latents with:
            - z_dec: (N, D) decision variables
            - ctx_index: (N,) context index
            - N: total samples
        tasks: Dictionary from sample_tasks with:
            - W: (T, n_contexts, D) weights
            - tau: (T, n_contexts) thresholds
            - class_balance: thresholding method

    Returns:
        Y: (N, T) binary targets
    """
    z_dec = latents['z_dec']
    ctx_index = latents['ctx_index']
    N = latents['N']

    W = tasks['W']
    tau = tasks['tau']
    T = W.shape[0]
    n_contexts = W.shape[1]

    class_balance = tasks.get('class_balance', 'median')

    # If using 'median' or 'auto', adjust thresholds based on actual data
    if class_balance == 'median':
        # Compute median score per task per context
        for t in range(T):
            for c in range(n_contexts):
                ctx_mask = (ctx_index == c)
                if np.any(ctx_mask):
                    z_dec_ctx = z_dec[ctx_mask]
                    scores_ctx = z_dec_ctx @ W[t, c]
                    tau[t, c] = np.median(scores_ctx)

    elif class_balance == 'auto':
        # Grid search to keep class ratio in [0.35, 0.65]
        for t in range(T):
            for c in range(n_contexts):
                ctx_mask = (ctx_index == c)
                if np.any(ctx_mask):
                    z_dec_ctx = z_dec[ctx_mask]
                    scores_ctx = z_dec_ctx @ W[t, c]

                    # Try different percentiles to find good balance
                    best_thresh = np.median(scores_ctx)
                    best_ratio = 0.5

                    for percentile in np.linspace(30, 70, 9):
                        thresh_candidate = np.percentile(scores_ctx, percentile)
                        pos_ratio = np.mean(scores_ctx > thresh_candidate)

                        # Check if ratio is in target range [0.35, 0.65]
                        if 0.35 <= pos_ratio <= 0.65:
                            if abs(pos_ratio - 0.5) < abs(best_ratio - 0.5):
                                best_thresh = thresh_candidate
                                best_ratio = pos_ratio

                    tau[t, c] = best_thresh

    # Build targets
    Y = np.zeros((N, T), dtype=np.float32)

    for i in range(N):
        ctx = ctx_index[i]
        z = z_dec[i]
        for t in range(T):
            score = np.dot(W[t, ctx], z)
            Y[i, t] = 1.0 if score > tau[t, ctx] else 0.0

    return Y
