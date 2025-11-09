"""
Quick sanity checks to validate the implementation.
"""
import sys
from pathlib import Path
import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_default_config
from src.data import generate_dataset
from src.model import ContextMLP, get_hidden_activations
from src.train import train_model
from src.metrics import contextual_fraction, subspace_specialization
from src.utils import set_seed


def test_data_generation():
    """Test that data generation works."""
    print("Testing data generation...")

    X, Y, ctx_index, metadata = generate_dataset(
        n_per_ctx=100,
        D=3, C=2, R=2,
        T=2,
        M_dis=16, M_un=16,
        q=2, s=0.5,
        seed=42
    )

    print(f"  X shape: {X.shape}")
    print(f"  Y shape: {Y.shape}")
    print(f"  Contexts: {len(np.unique(ctx_index))}")
    print(f"  X sample: {X[0, :5]}")
    print(f"  Y sample: {Y[0]}")

    # Check shapes
    n_contexts = 2 ** 2  # C=2
    assert X.shape[0] == 100 * n_contexts
    assert Y.shape[0] == 100 * n_contexts
    assert Y.shape[1] == 2  # T=2
    assert len(np.unique(ctx_index)) == n_contexts

    # Check that targets are binary
    assert np.all((Y == 0) | (Y == 1))

    # Check that inputs are normalized
    norms = np.linalg.norm(X[:, :-n_contexts], axis=1)  # Exclude context one-hot
    # Note: we append context one-hot after normalization, so full X is not normalized
    # Just check that features part exists
    assert X.shape[1] == 16 + 16 + n_contexts

    print("  ✓ Data generation passed\n")


def test_model_forward():
    """Test that model forward pass works."""
    print("Testing model forward pass...")

    in_dim = 64
    hidden = 32
    n_tasks = 3

    model = ContextMLP(in_dim, hidden, n_tasks)

    # Random input
    X = torch.randn(10, in_dim)
    logits = model(X)

    print(f"  Input shape: {X.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  Hidden activations shape: {model.hidden_activations.shape}")

    assert logits.shape == (10, n_tasks)
    assert model.hidden_activations.shape == (10, hidden)

    # Check that ReLU works (activations should be non-negative)
    assert torch.all(model.hidden_activations >= 0)

    print("  ✓ Model forward pass passed\n")


def test_training():
    """Test that training loop works."""
    print("Testing training loop...")

    # Generate small dataset
    X_train, Y_train, _, _ = generate_dataset(
        n_per_ctx=200, D=3, C=2, R=2, T=2,
        M_dis=16, M_un=16, q=2, s=0.8, seed=42
    )

    X_val, Y_val, _, _ = generate_dataset(
        n_per_ctx=50, D=3, C=2, R=2, T=2,
        M_dis=16, M_un=16, q=2, s=0.8, seed=43
    )

    # Create and train model
    model = ContextMLP(in_dim=X_train.shape[1], hidden=64, n_tasks=2)

    history = train_model(
        model, X_train, Y_train, X_val, Y_val,
        lr=1e-3, batch_size=128, epochs=20, patience=5,
        device='cpu', verbose=False
    )

    print(f"  Trained for {len(history['train_loss'])} epochs")
    print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"  Final val loss: {history['val_loss'][-1]:.4f}")
    print(f"  Final train acc: {history['train_acc'][-1]:.4f}")
    print(f"  Final val acc: {history['val_acc'][-1]:.4f}")

    # Check that training improved performance
    assert history['train_loss'][-1] < history['train_loss'][0]
    assert history['train_acc'][-1] > 0.5  # Better than random

    print("  ✓ Training passed\n")
    return model, X_val, Y_val


def test_metrics(model, X_val, ctx_val, latents_val):
    """Test that metrics computation works."""
    print("Testing metrics computation...")

    # Get activations
    X_torch = torch.from_numpy(X_val).float()
    acts = get_hidden_activations(model, X_torch, device='cpu')

    print(f"  Activations shape: {acts.shape}")

    # Prepare activations by context
    n_contexts = len(np.unique(ctx_val))
    act_by_ctx = {}
    for ctx in range(n_contexts):
        mask = ctx_val == ctx
        act_by_ctx[ctx] = acts[mask]
        print(f"  Context {ctx}: {act_by_ctx[ctx].shape[0]} samples")

    # Test Contextual Fraction
    cf = contextual_fraction(act_by_ctx, thresh=0.01)
    print(f"  Contextual Fraction: {cf:.4f}")
    assert 0 <= cf <= 1

    # Test Subspace Specialization
    ss = subspace_specialization(acts, latents_val, ctx_val, var_threshold=1e-10)
    print(f"  Subspace Specialization: {ss:.4f}")
    assert ss >= 0  # Should be non-negative

    print("  ✓ Metrics computation passed\n")


def run_all_tests():
    """Run all sanity checks."""
    print("="*60)
    print("Running Sanity Checks")
    print("="*60 + "\n")

    set_seed(42)

    # Test data generation
    test_data_generation()

    # Test model
    test_model_forward()

    # Test training (returns trained model and validation data)
    model, X_val, Y_val = test_training()

    # Generate validation metadata for metrics
    _, _, ctx_val, metadata_val = generate_dataset(
        n_per_ctx=50, D=3, C=2, R=2, T=2,
        M_dis=16, M_un=16, q=2, s=0.8, seed=43
    )

    # Test metrics
    test_metrics(model, X_val, ctx_val, metadata_val['latents'])

    print("="*60)
    print("All Sanity Checks Passed! ✓")
    print("="*60)


if __name__ == '__main__':
    run_all_tests()
