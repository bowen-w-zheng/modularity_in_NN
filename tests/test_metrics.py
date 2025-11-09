"""Tests for modularity metrics."""
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics import contextual_fraction, alignment_index, compute_pca_subspace, subspace_specialization


def test_contextual_fraction_perfect():
    """Test contextual fraction with perfect selectivity."""
    print("Testing contextual_fraction with perfect selectivity...")

    # Create synthetic activations: 100 units, 4 contexts
    # First 25 units selective to context 0, next 25 to context 1, etc.
    n_hidden = 100
    n_contexts = 4
    n_samples = 1000

    act_by_ctx = {}
    for c in range(n_contexts):
        acts = np.zeros((n_samples, n_hidden))
        # Units c*25:(c+1)*25 are active (mean > 0.01) only in context c
        start = c * 25
        end = (c + 1) * 25
        acts[:, start:end] = np.random.rand(n_samples, 25) + 0.1  # Mean ~ 0.6
        act_by_ctx[c] = acts

    cf = contextual_fraction(act_by_ctx, thresh=0.01)

    # All 100 units should be selective
    assert cf == 1.0, f"Expected CF=1.0, got {cf}"

    print("  ✓ Perfect selectivity gives CF=1.0")


def test_contextual_fraction_zero():
    """Test contextual fraction with no selectivity."""
    print("\nTesting contextual_fraction with no selectivity...")

    # All units active in all contexts
    n_hidden = 100
    n_contexts = 4
    n_samples = 1000

    act_by_ctx = {}
    for c in range(n_contexts):
        acts = np.random.rand(n_samples, n_hidden) + 0.1  # All active
        act_by_ctx[c] = acts

    cf = contextual_fraction(act_by_ctx, thresh=0.01)

    # No units should be selective (active in exactly one context)
    assert cf == 0.0, f"Expected CF=0.0, got {cf}"

    print("  ✓ No selectivity gives CF=0.0")


def test_alignment_index():
    """Test alignment index computation."""
    print("\nTesting alignment_index...")

    # Test 1: Identical subspaces
    U = np.eye(5)[:, :3]  # First 3 standard basis vectors
    V = np.eye(5)[:, :3]  # Same
    align = alignment_index(U, V)
    assert np.isclose(align, 1.0, atol=1e-6), f"Expected align=1.0 for identical subspaces, got {align}"

    print("  ✓ Identical subspaces give alignment=1.0")

    # Test 2: Orthogonal subspaces
    U = np.eye(5)[:, :2]  # First 2 basis vectors
    V = np.eye(5)[:, 2:4]  # Last 2 basis vectors
    align = alignment_index(U, V)
    assert np.isclose(align, 0.0, atol=1e-6), f"Expected align=0.0 for orthogonal subspaces, got {align}"

    print("  ✓ Orthogonal subspaces give alignment=0.0")


def test_compute_pca_subspace():
    """Test PCA subspace computation."""
    print("\nTesting compute_pca_subspace...")

    # Generate data lying in a 2D subspace of 5D space
    rng = np.random.default_rng(42)
    n_samples = 100

    # True subspace: span of first two basis vectors
    coeffs = rng.normal(size=(n_samples, 2))
    basis_true = np.eye(5)[:, :2]
    data = coeffs @ basis_true.T  # (n_samples, 5)

    # Add small noise
    data += rng.normal(scale=0.01, size=data.shape)

    # Compute PCA subspace
    basis_pca = compute_pca_subspace(data, var_threshold=1e-3)

    # Should recover approximately 2 components
    assert basis_pca.shape[1] >= 2, f"Expected at least 2 components, got {basis_pca.shape[1]}"

    # Alignment with true subspace should be high
    align = alignment_index(basis_true, basis_pca[:, :2])
    assert align > 0.9, f"Expected high alignment with true subspace, got {align}"

    print("  ✓ PCA subspace computed correctly")


def test_subspace_specialization():
    """Test subspace specialization metric."""
    print("\nTesting subspace_specialization...")

    # Create synthetic scenario:
    # - Context variables create separated subspaces
    # - Non-context variables don't

    rng = np.random.default_rng(42)
    n_samples = 500
    n_hidden = 50

    # Latents: 2 decision, 1 context, 1 irrelevant
    D, C, R = 2, 1, 1

    z_dec = rng.integers(0, 2, size=(n_samples, D))
    z_ctx = rng.integers(0, 2, size=(n_samples, C))
    z_irr = rng.integers(0, 2, size=(n_samples, R))
    z_all = np.concatenate([z_dec, z_ctx, z_irr], axis=1)

    ctx_index = z_ctx[:, 0]

    # Create activations that depend strongly on context
    acts = np.zeros((n_samples, n_hidden))
    for i in range(n_samples):
        if ctx_index[i] == 0:
            # Context 0: activations in first half of neurons
            acts[i, :25] = rng.normal(loc=1.0, scale=0.1, size=25)
        else:
            # Context 1: activations in second half of neurons
            acts[i, 25:] = rng.normal(loc=1.0, scale=0.1, size=25)

    latents = {
        'z_dec': z_dec,
        'z_ctx': z_ctx,
        'z_irr': z_irr,
        'z_all': z_all,
        'N': n_samples
    }

    ss = subspace_specialization(acts, latents, ctx_index, var_threshold=1e-10)

    # Should be positive (context creates specialization)
    assert ss > 0, f"Expected positive SS, got {ss}"

    print(f"  ✓ Subspace specialization computed: {ss:.3f}")


if __name__ == '__main__':
    print("="*60)
    print("Running Metrics Tests")
    print("="*60)

    test_contextual_fraction_perfect()
    test_contextual_fraction_zero()
    test_alignment_index()
    test_compute_pca_subspace()
    test_subspace_specialization()

    print("\n" + "="*60)
    print("All Metrics Tests Passed! ✓")
    print("="*60)
