"""Tests for task generation."""
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import sample_latents
from src.tasks import sample_tasks, build_targets


def test_sample_tasks():
    """Test task sampling."""
    print("Testing sample_tasks...")

    rng = np.random.default_rng(42)
    tasks = sample_tasks(T=5, D=3, C=2, class_balance='median', rng=rng)

    n_contexts = 2 ** 2

    # Check shapes
    assert tasks['W'].shape == (5, n_contexts, 3), f"Wrong W shape: {tasks['W'].shape}"
    assert tasks['tau'].shape == (5, n_contexts), f"Wrong tau shape: {tasks['tau'].shape}"

    # Check weights in {-1, 0, 1}
    assert np.all(np.isin(tasks['W'], [-1, 0, 1])), "W contains values not in {-1, 0, 1}"

    # Check no all-zero weight vectors
    for t in range(5):
        for c in range(n_contexts):
            assert not np.all(tasks['W'][t, c] == 0), f"Task {t}, context {c} has all-zero weights"

    print("  ✓ Task shapes correct")
    print("  ✓ Weights in {-1, 0, 1}")
    print("  ✓ No all-zero weight vectors")


def test_build_targets():
    """Test target building."""
    print("\nTesting build_targets...")

    rng = np.random.default_rng(42)

    # Sample latents
    latents = sample_latents(n_per_ctx=100, D=3, C=2, R=0, rng=rng)

    # Sample tasks
    tasks = sample_tasks(T=5, D=3, C=2, class_balance='median', rng=rng)

    # Build targets
    Y = build_targets(latents, tasks)

    N = latents['N']

    # Check shape
    assert Y.shape == (N, 5), f"Wrong Y shape: {Y.shape}"

    # Check binary
    assert np.all((Y == 0) | (Y == 1)), "Y not binary"

    # Check class balance with median thresholding
    for t in range(5):
        pos_ratio = Y[:, t].mean()
        # With median, should be close to 0.5
        assert 0.3 <= pos_ratio <= 0.7, f"Task {t} has poor balance: {pos_ratio}"

    print("  ✓ Target shapes correct")
    print("  ✓ Binary values correct")
    print("  ✓ Class balance reasonable")


def test_class_balance_auto():
    """Test auto class balancing."""
    print("\nTesting class_balance='auto'...")

    rng = np.random.default_rng(42)

    # Sample latents
    latents = sample_latents(n_per_ctx=100, D=3, C=2, R=0, rng=rng)

    # Sample tasks with auto balancing
    tasks = sample_tasks(T=5, D=3, C=2, class_balance='auto', rng=rng)

    # Build targets
    Y = build_targets(latents, tasks)

    # Check class balance - should be in [0.35, 0.65] range
    for t in range(5):
        pos_ratio = Y[:, t].mean()
        assert 0.3 <= pos_ratio <= 0.7, f"Task {t} has poor balance with auto: {pos_ratio}"

    print("  ✓ Auto class balancing works")


if __name__ == '__main__':
    print("="*60)
    print("Running Task Tests")
    print("="*60)

    test_sample_tasks()
    test_build_targets()
    test_class_balance_auto()

    print("\n" + "="*60)
    print("All Task Tests Passed! ✓")
    print("="*60)
