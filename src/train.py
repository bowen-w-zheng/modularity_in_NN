"""Training loop with early stopping."""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from typing import Dict, Tuple
from tqdm import tqdm


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: str
) -> float:
    """
    Train for one epoch.

    Args:
        model: Neural network
        dataloader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on

    Returns:
        avg_loss: Average loss over epoch
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for X_batch, Y_batch in dataloader:
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)

        optimizer.zero_grad()

        # Forward pass
        logits = model(X_batch)

        # Compute loss
        loss = criterion(logits, Y_batch)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / n_batches
    return avg_loss


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str
) -> Tuple[float, float]:
    """
    Evaluate model on dataset.

    Args:
        model: Neural network
        dataloader: Data loader
        criterion: Loss function
        device: Device to evaluate on

    Returns:
        avg_loss: Average loss
        accuracy: Average accuracy across all tasks
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_correct = []
    all_total = []

    with torch.no_grad():
        for X_batch, Y_batch in dataloader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            # Forward pass
            logits = model(X_batch)

            # Compute loss
            loss = criterion(logits, Y_batch)
            total_loss += loss.item()
            n_batches += 1

            # Compute accuracy
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct = (preds == Y_batch).float()
            all_correct.append(correct.cpu().numpy())
            all_total.append(np.ones_like(correct.cpu().numpy()))

    avg_loss = total_loss / n_batches

    # Compute accuracy (average over all samples and tasks)
    all_correct = np.concatenate(all_correct, axis=0)
    accuracy = np.mean(all_correct)

    return avg_loss, accuracy


def train_model(
    model: nn.Module,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    lr: float = 1e-3,
    batch_size: int = 512,
    epochs: int = 80,
    patience: int = 10,
    device: str = 'cpu',
    verbose: bool = True
) -> Dict:
    """
    Train model with early stopping.

    Args:
        model: Neural network to train
        X_train, Y_train: Training data
        X_val, Y_val: Validation data
        lr: Learning rate
        batch_size: Batch size
        epochs: Maximum number of epochs
        patience: Early stopping patience
        device: Device to train on
        verbose: Print progress

    Returns:
        history: Dictionary with training history
    """
    model.to(device)

    # Create data loaders
    train_dataset = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(Y_train).float()
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val).float(),
        torch.from_numpy(Y_val).float()
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Setup optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'best_epoch': 0,
        'best_val_loss': float('inf')
    }

    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    # Training loop
    epoch_iter = tqdm(range(epochs), desc="Training") if verbose else range(epochs)

    for epoch in epoch_iter:
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # Evaluate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        train_loss_eval, train_acc = evaluate(model, train_loader, criterion, device)

        # Record history
        history['train_loss'].append(train_loss_eval)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            history['best_epoch'] = epoch
            history['best_val_loss'] = best_val_loss
        else:
            patience_counter += 1

        if verbose and epoch % 10 == 0:
            tqdm.write(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                      f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")

        # Early stopping
        if patience_counter >= patience:
            if verbose:
                tqdm.write(f"Early stopping at epoch {epoch}")
            break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return history
