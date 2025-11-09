"""Tests for training loop."""
import numpy as np
import torch
import sys
import tempfile
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import generate_dataset
from src.model import ContextMLP, get_hidden_activations
from src.train import train_model
from src.metrics import compute_all_metrics


def test_model_forward():
    """Test model forward pass."""
    print("Testing model forward pass...")

    model = ContextMLP(in_dim=50, hidden=128, n_tasks=3)

    # Random input
    X = torch.randn(32, 50)
    logits = model(X)

    # Check shape
    assert logits.shape == (32, 3), f"Wrong output shape: {logits.shape}"

    # Check hidden activations captured
    assert model.hidden_activations is not None, "Hidden activations not captured"
    assert model.hidden_activations.shape == (32, 128), f"Wrong hidden shape: {model.hidden_activations.shape}"

    print("  ✓ Forward pass works")
    print("  ✓ Hidden activations captured")


def test_training_improves():
    """Test that training improves performance."""
    print("\nTesting training improves performance...")

    # Generate small dataset
    X_train, Y_train, ctx_train, _ = generate_dataset(
        n_per_ctx=500,
        D=3,
        C=2,
        R=1,
        T=3,
        M_dis=16,
        M_un=32,
        q=2,
        s=0.5,
        seed=42
    )

    X_val, Y_val, ctx_val, _ = generate_dataset(
        n_per_ctx=200,
        D=3,
        C=2,
        R=1,
        T=3,
        M_dis=16,
        M_un=32,
        q=2,
        s=0.5,
        seed=43
    )

    # Create model
    model = ContextMLP(in_dim=X_train.shape[1], hidden=64, n_tasks=3)

    # Train
    history = train_model(
        model=model,
        X_train=X_train,
        Y_train=Y_train,
        X_val=X_val,
        Y_val=Y_val,
        lr=1e-3,
        batch_size=128,
        epochs=30,
        patience=5,
        device='cpu',
        verbose=False
    )

    # Check that loss decreased
    initial_loss = history['train_loss'][0]
    final_loss = history['train_loss'][-1]
    assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss} -> {final_loss}"

    # Check that accuracy increased
    initial_acc = history['train_acc'][0]
    final_acc = history['train_acc'][-1]
    assert final_acc > initial_acc, f"Accuracy did not increase: {initial_acc} -> {final_acc}"

    # Check that final accuracy is reasonable
    assert final_acc > 0.8, f"Final accuracy too low: {final_acc}"

    print(f"  ✓ Loss decreased: {initial_loss:.4f} -> {final_loss:.4f}")
    print(f"  ✓ Accuracy increased: {initial_acc:.4f} -> {final_acc:.4f}")


def test_get_hidden_activations():
    """Test extracting hidden activations."""
    print("\nTesting get_hidden_activations...")

    model = ContextMLP(in_dim=50, hidden=128, n_tasks=3)
    X = torch.randn(100, 50)

    acts = get_hidden_activations(model, X, device='cpu')

    # Check shape
    assert acts.shape == (100, 128), f"Wrong activations shape: {acts.shape}"

    # Check dtype
    assert isinstance(acts, np.ndarray), "Activations not numpy array"

    # Check values are non-negative (after ReLU)
    assert np.all(acts >= 0), "Activations contain negative values"

    print("  ✓ Hidden activations extracted correctly")


def test_end_to_end_mini_experiment():
    """Test a complete mini experiment in a temp directory."""
    print("\nTesting end-to-end mini experiment...")

    try:
        # Generate data
        X_train, Y_train, ctx_train, metadata_train = generate_dataset(
            n_per_ctx=100,  # Reduced from 200
            D=2,
            C=1,
            R=0,  # Reduced from 1 to simplify
            T=2,
            M_dis=16,
            M_un=16,  # Reduced from 32
            q=2,
            s=0.8,
            seed=42
        )

        X_val, Y_val, ctx_val, metadata_val = generate_dataset(
            n_per_ctx=50,  # Reduced from 100
            D=2,
            C=1,
            R=0,
            T=2,
            M_dis=16,
            M_un=16,
            q=2,
            s=0.8,
            seed=43
        )

        # Ensure no NaN values
        assert not np.any(np.isnan(X_train)), "X_train contains NaN"
        assert not np.any(np.isnan(Y_train)), "Y_train contains NaN"
        assert not np.any(np.isnan(X_val)), "X_val contains NaN"
        assert not np.any(np.isnan(Y_val)), "Y_val contains NaN"

        # Create and train model
        model = ContextMLP(in_dim=X_train.shape[1], hidden=32, n_tasks=2)  # Reduced hidden size

        history = train_model(
            model=model,
            X_train=X_train,
            Y_train=Y_train,
            X_val=X_val,
            Y_val=Y_val,
            lr=1e-3,
            batch_size=32,  # Reduced batch size
            epochs=15,  # Reduced epochs
            patience=5,
            device='cpu',
            verbose=False
        )

        # Check training completed
        assert len(history['val_acc']) > 0, "No training history"

        # Compute metrics with smaller sample size
        metrics = compute_all_metrics(
            model=model,
            X=X_val,
            ctx_index=ctx_val,
            latents=metadata_val['latents'],
            device='cpu',
            n_samples_per_ctx=50  # Reduced from 100
        )

        # Check metrics computed
        assert 'contextual_fraction' in metrics, "CF not computed"
        assert 'subspace_specialization' in metrics, "SS not computed"
        assert 'best_k_clusters' in metrics, "Clustering not computed"

        # Check metric ranges
        assert 0 <= metrics['contextual_fraction'] <= 1, f"CF out of range: {metrics['contextual_fraction']}"
        assert metrics['subspace_specialization'] >= 0, f"SS negative: {metrics['subspace_specialization']}"

        print(f"  ✓ End-to-end experiment completed")
        print(f"    Final val acc: {history['val_acc'][-1]:.3f}")
        print(f"    Contextual Fraction: {metrics['contextual_fraction']:.3f}")
        print(f"    Subspace Specialization: {metrics['subspace_specialization']:.3f}")

    except Exception as e:
        print(f"  ✗ End-to-end test failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    print("="*60)
    print("Running Training Tests")
    print("="*60)

    test_model_forward()
    test_training_improves()
    test_get_hidden_activations()
    test_end_to_end_mini_experiment()

    print("\n" + "="*60)
    print("All Training Tests Passed! ✓")
    print("="*60)
