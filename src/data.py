"""Data generation: latent variables, input models, and task sampling.

Implements the input model from Johnston et al., interpolating between
disentangled and unstructured representations.
"""
import numpy as np
from typing import Dict, Tuple
from itertools import combinations_with_replacement
from src.utils import normalize_rows


def sample_latents(n_per_ctx: int, D: int, C: int, R: int, rng: np.random.Generator) -> Dict:
    """
    Sample binary latent variables for each context.

    Args:
        n_per_ctx: Number of samples per context
        D: Number of decision-relevant variables
        C: Number of context variables
        R: Number of irrelevant variables
        rng: Numpy random generator

    Returns:
        Dictionary with:
            - z_dec: (N, D) decision-relevant latents
            - z_ctx: (N, C) context latents
            - z_irr: (N, R) irrelevant latents
            - ctx_index: (N,) context index for each sample
            - z_all: (N, D+C+R) all latents concatenated
    """
    n_contexts = 2 ** C
    N = n_per_ctx * n_contexts

    # Sample all latents uniformly from {0, 1}
    z_dec = rng.integers(0, 2, size=(N, D))
    z_ctx = rng.integers(0, 2, size=(N, C))
    z_irr = rng.integers(0, 2, size=(N, R)) if R > 0 else np.zeros((N, 0))

    # Assign context indices (which context each sample belongs to)
    # For simplicity, cycle through contexts
    ctx_index = np.repeat(np.arange(n_contexts), n_per_ctx)

    # Override z_ctx to match context index (binary representation)
    for i in range(N):
        ctx_binary = format(ctx_index[i], f'0{C}b')  # Binary string
        z_ctx[i] = [int(b) for b in ctx_binary]

    z_all = np.concatenate([z_dec, z_ctx, z_irr], axis=1)

    return {
        'z_dec': z_dec,
        'z_ctx': z_ctx,
        'z_irr': z_irr,
        'ctx_index': ctx_index,
        'z_all': z_all,
        'N': N
    }


def build_disentangled_representation(z_all: np.ndarray, M_dis: int, rng: np.random.Generator) -> np.ndarray:
    """
    Build disentangled representation: each latent variable maps to separate dimensions.

    Args:
        z_all: (N, D+C+R) all latent variables
        M_dis: Target dimension for disentangled representation
        rng: Numpy random generator

    Returns:
        phi_dis: (N, M_dis) disentangled representation
    """
    N, n_latents = z_all.shape

    # Simple approach: one-hot-like encoding where each latent gets its own dimensions
    # For binary variables, we can just replicate or use scaled versions
    # To reach M_dis dimensions, we'll create a simple projection

    # Create a block-diagonal-ish structure
    # Each latent variable gets M_dis // n_latents dimensions
    dims_per_latent = max(1, M_dis // n_latents)

    # Simple one-hot style: replicate each latent value
    phi_parts = []
    for i in range(n_latents):
        latent_vals = z_all[:, i:i+1]  # (N, 1)
        # Replicate to dims_per_latent dimensions
        replicated = np.tile(latent_vals, (1, dims_per_latent))
        phi_parts.append(replicated)

    phi_dis = np.concatenate(phi_parts, axis=1)

    # Pad or truncate to exactly M_dis
    if phi_dis.shape[1] < M_dis:
        padding = np.zeros((N, M_dis - phi_dis.shape[1]))
        phi_dis = np.concatenate([phi_dis, padding], axis=1)
    else:
        phi_dis = phi_dis[:, :M_dis]

    return phi_dis.astype(np.float32)


def build_unstructured_representation(z_all: np.ndarray, M_un: int, q: int, rng: np.random.Generator) -> np.ndarray:
    """
    Build unstructured representation via random projection of interaction terms.

    Args:
        z_all: (N, D+C+R) all latent variables
        M_un: Target dimension for unstructured representation
        q: Order of interaction terms (e.g., q=2 for pairwise)
        rng: Numpy random generator

    Returns:
        phi_un: (N, M_un) unstructured representation
    """
    N, n_latents = z_all.shape

    # Generate monomial features up to order q
    # For q=2, include singles and pairwise products
    monomials = [z_all]  # Order 1

    # Add interaction terms
    if q >= 2:
        # Pairwise products
        n_pairs = n_latents * (n_latents + 1) // 2
        pairwise = np.zeros((N, n_pairs))
        idx = 0
        for i in range(n_latents):
            for j in range(i, n_latents):
                pairwise[:, idx] = z_all[:, i] * z_all[:, j]
                idx += 1
        monomials.append(pairwise)

    # For higher orders, can extend similarly
    # For now, just do q=2

    # Concatenate all monomials
    psi = np.concatenate(monomials, axis=1)  # (N, n_monomials)

    # Random projection to M_un dimensions
    n_monomials = psi.shape[1]
    W = rng.normal(0, 1/np.sqrt(M_un), size=(M_un, n_monomials))

    phi_un = psi @ W.T  # (N, M_un)

    # Optional: ReLU nonlinearity (paper mentions this as optional)
    # phi_un = np.maximum(0, phi_un)

    return phi_un.astype(np.float32)


def build_input_representation(
    latents: Dict,
    M_dis: int,
    M_un: int,
    q: int,
    s: float,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Build input representation interpolating between disentangled and unstructured.

    Args:
        latents: Dictionary from sample_latents
        M_dis: Dimension for disentangled part
        M_un: Dimension for unstructured part
        q: Interaction order for unstructured
        s: Interpolation parameter in [0,1]
           s=1: fully disentangled
           s=0: fully unstructured
        rng: Numpy random generator

    Returns:
        X: (N, M_eff) input features, L2-normalized per row
    """
    z_all = latents['z_all']

    # Build both representations
    phi_dis = build_disentangled_representation(z_all, M_dis, rng)
    phi_un = build_unstructured_representation(z_all, M_un, q, rng)

    # Interpolate and concatenate
    # x = concat(sqrt(s) * phi_dis, sqrt(1-s) * phi_un)
    phi_dis_scaled = np.sqrt(s) * phi_dis
    phi_un_scaled = np.sqrt(1 - s) * phi_un

    X = np.concatenate([phi_dis_scaled, phi_un_scaled], axis=1)

    # L2 normalize each row
    X = normalize_rows(X)

    return X


def sample_tasks(T: int, D: int, C: int, rng: np.random.Generator) -> Dict:
    """
    Sample T linear classification tasks, each context-dependent.

    For each task t and context c, generate a random linear separator:
        y_{t,c} = 1{w_{t,c}^T z_dec > tau_{t,c}}

    Args:
        T: Number of tasks
        D: Number of decision variables
        C: Number of context variables
        rng: Numpy random generator

    Returns:
        Dictionary with:
            - W: (T, n_contexts, D) weight vectors
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

    # Sample thresholds to avoid severe class imbalance
    # For binary variables in {0,1}, w^T z is in range [sum of negative weights, sum of positive weights]
    tau = np.zeros((T, n_contexts))
    for t in range(T):
        for c in range(n_contexts):
            w = W[t, c]
            min_val = np.sum(np.minimum(w, 0))
            max_val = np.sum(np.maximum(w, 0))
            # Choose threshold in the middle range
            tau[t, c] = rng.uniform(min_val + 0.2 * (max_val - min_val),
                                   min_val + 0.8 * (max_val - min_val))

    return {
        'W': W,
        'tau': tau
    }


def build_targets(latents: Dict, tasks: Dict) -> np.ndarray:
    """
    Build target labels for all samples and tasks.

    Args:
        latents: Dictionary from sample_latents
        tasks: Dictionary from sample_tasks

    Returns:
        Y: (N, T) binary targets
    """
    z_dec = latents['z_dec']
    ctx_index = latents['ctx_index']
    N = latents['N']

    W = tasks['W']
    tau = tasks['tau']
    T = W.shape[0]

    Y = np.zeros((N, T), dtype=np.float32)

    for i in range(N):
        ctx = ctx_index[i]
        z = z_dec[i]
        for t in range(T):
            score = np.dot(W[t, ctx], z)
            Y[i, t] = 1.0 if score > tau[t, ctx] else 0.0

    return Y


def generate_dataset(
    n_per_ctx: int,
    D: int,
    C: int,
    R: int,
    T: int,
    M_dis: int,
    M_un: int,
    q: int,
    s: float,
    seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Generate a complete dataset with inputs and targets.

    Args:
        n_per_ctx: Samples per context
        D, C, R: Latent variable dimensions
        T: Number of tasks
        M_dis, M_un: Representation dimensions
        q: Interaction order
        s: Structure parameter
        seed: Random seed

    Returns:
        X: (N, M_dis + M_un) inputs with context one-hot appended
        Y: (N, T) targets
        ctx_index: (N,) context indices
        metadata: Dictionary with latents and tasks
    """
    rng = np.random.default_rng(seed)

    # Sample latents
    latents = sample_latents(n_per_ctx, D, C, R, rng)

    # Build input representation (without context one-hot yet)
    X_features = build_input_representation(latents, M_dis, M_un, q, s, rng)

    # Append context one-hot
    n_contexts = 2 ** C
    N = latents['N']
    ctx_onehot = np.eye(n_contexts)[latents['ctx_index']]
    X = np.concatenate([X_features, ctx_onehot], axis=1).astype(np.float32)

    # Sample tasks and build targets
    tasks = sample_tasks(T, D, C, rng)
    Y = build_targets(latents, tasks)

    metadata = {
        'latents': latents,
        'tasks': tasks,
        'n_contexts': n_contexts
    }

    return X, Y, latents['ctx_index'], metadata
