"""Tests for data generation."""
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import (
    sample_latents,
    build_disentangled_representation,
    build_unstructured_representation,
    build_input_representation,
    generate_dataset
)


def test_sample_latents():
    """Test latent variable sampling."""
    print("Testing sample_latents...")

    rng = np.random.default_rng(42)
    latents = sample_latents(n_per_ctx=100, D=3, C=2, R=2, rng=rng)

    # Check shapes
    n_contexts = 2 ** 2  # C=2
    N = 100 * n_contexts
    assert latents['z_dec'].shape == (N, 3), f"Wrong z_dec shape: {latents['z_dec'].shape}"
    assert latents['z_ctx'].shape == (N, 2), f"Wrong z_ctx shape: {latents['z_ctx'].shape}"
    assert latents['z_irr'].shape == (N, 2), f"Wrong z_irr shape: {latents['z_irr'].shape}"
    assert latents['ctx_index'].shape == (N,), f"Wrong ctx_index shape: {latents['ctx_index'].shape}"
    assert latents['z_all'].shape == (N, 7), f"Wrong z_all shape: {latents['z_all'].shape}"

    # Check binary values
    assert np.all((latents['z_dec'] == 0) | (latents['z_dec'] == 1)), "z_dec not binary"
    assert np.all((latents['z_ctx'] == 0) | (latents['z_ctx'] == 1)), "z_ctx not binary"

    # Check context assignment
    assert len(np.unique(latents['ctx_index'])) == n_contexts, "Wrong number of contexts"

    print("  ✓ Shapes correct")
    print("  ✓ Binary values correct")
    print("  ✓ Context assignment correct")


def test_input_representation():
    """Test input representation building."""
    print("\nTesting input representation...")

    rng = np.random.default_rng(42)
    latents = sample_latents(n_per_ctx=50, D=3, C=2, R=2, rng=rng)

    # Test disentangled
    X_dis = build_disentangled_representation(latents['z_all'], M_dis=32, rng=rng)
    assert X_dis.shape[1] == 32, f"Wrong disentangled dimension: {X_dis.shape[1]}"

    # Test unstructured
    X_un = build_unstructured_representation(latents['z_all'], M_un=64, q=2, rng=rng)
    assert X_un.shape[1] == 64, f"Wrong unstructured dimension: {X_un.shape[1]}"

    # Test interpolated
    X = build_input_representation(latents, M_dis=32, M_un=64, q=2, s=0.5, rng=rng)
    assert X.shape[1] == 96, f"Wrong interpolated dimension: {X.shape[1]}"

    # Check L2 normalization
    norms = np.linalg.norm(X, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), "Rows not L2-normalized"

    print("  ✓ Disentangled representation correct")
    print("  ✓ Unstructured representation correct")
    print("  ✓ Interpolation correct")
    print("  ✓ L2 normalization correct")


def test_generate_dataset():
    """Test full dataset generation."""
    print("\nTesting generate_dataset...")

    X, Y, ctx_index, metadata = generate_dataset(
        n_per_ctx=50,
        D=3,
        C=2,
        R=2,
        T=5,
        M_dis=32,
        M_un=64,
        q=2,
        s=0.5,
        seed=42,
        class_balance='median'
    )

    n_contexts = 2 ** 2
    N = 50 * n_contexts

    # Check shapes
    assert X.shape == (N, 96 + n_contexts), f"Wrong X shape: {X.shape}"
    assert Y.shape == (N, 5), f"Wrong Y shape: {Y.shape}"
    assert ctx_index.shape == (N,), f"Wrong ctx_index shape: {ctx_index.shape}"

    # Check binary targets
    assert np.all((Y == 0) | (Y == 1)), "Y not binary"

    # Check context one-hot appended
    ctx_onehot = X[:, -n_contexts:]
    assert np.allclose(ctx_onehot.sum(axis=1), 1.0), "Context one-hot not correct"

    # Check class balance (should be roughly balanced)
    for t in range(5):
        pos_ratio = Y[:, t].mean()
        assert 0.2 <= pos_ratio <= 0.8, f"Task {t} severely imbalanced: {pos_ratio}"

    print("  ✓ Dataset shapes correct")
    print("  ✓ Binary targets correct")
    print("  ✓ Context one-hot appended")
    print("  ✓ Class balance reasonable")


if __name__ == '__main__':
    print("="*60)
    print("Running Data Tests")
    print("="*60)

    test_sample_latents()
    test_input_representation()
    test_generate_dataset()

    print("\n" + "="*60)
    print("All Data Tests Passed! ✓")
    print("="*60)
