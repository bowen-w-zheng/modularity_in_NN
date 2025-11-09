"""Neural network model: 1-hidden-layer MLP with ReLU activation."""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict


class ContextMLP(nn.Module):
    """
    Single-hidden-layer MLP for contextual multi-task learning.

    Architecture:
        input -> Linear -> ReLU -> Linear -> Sigmoid (per task)
    """

    def __init__(self, in_dim: int, hidden: int, n_tasks: int):
        """
        Initialize the MLP.

        Args:
            in_dim: Input dimension (features + context one-hot)
            hidden: Hidden layer width
            n_tasks: Number of output tasks
        """
        super().__init__()
        self.in_dim = in_dim
        self.hidden = hidden
        self.n_tasks = n_tasks

        # Hidden layer
        self.fc1 = nn.Linear(in_dim, hidden)
        self.relu = nn.ReLU()

        # Output layer
        self.fc2 = nn.Linear(hidden, n_tasks)

        # Store activations for analysis
        self.hidden_activations = None

        # Register hook to capture hidden activations
        self.fc1.register_forward_hook(self._save_hidden_activations)

    def _save_hidden_activations(self, module, input, output):
        """Hook to save hidden layer activations after ReLU."""
        # Output here is before ReLU, we want after ReLU
        # So we'll capture in forward pass instead
        pass

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: (batch, in_dim) input tensor

        Returns:
            logits: (batch, n_tasks) output logits (before sigmoid)
        """
        # Hidden layer with ReLU
        h = self.fc1(x)
        h = self.relu(h)

        # Store activations for metrics
        self.hidden_activations = h.detach()

        # Output layer
        logits = self.fc2(h)

        return logits

    def predict_proba(self, x):
        """Get probabilities via sigmoid."""
        logits = self.forward(x)
        return torch.sigmoid(logits)

    def predict(self, x):
        """Get binary predictions."""
        proba = self.predict_proba(x)
        return (proba > 0.5).float()


def get_hidden_activations(model: ContextMLP, X: torch.Tensor, device: str = 'cpu') -> np.ndarray:
    """
    Extract hidden layer activations for a batch of inputs.

    Args:
        model: Trained ContextMLP
        X: (N, in_dim) input tensor
        device: Device to run on

    Returns:
        activations: (N, hidden) numpy array of ReLU activations
    """
    model.eval()
    model.to(device)
    X = X.to(device)

    activations_list = []

    with torch.no_grad():
        # Process in batches to avoid memory issues
        batch_size = 1024
        for i in range(0, X.shape[0], batch_size):
            X_batch = X[i:i + batch_size]
            _ = model(X_batch)  # Forward pass populates hidden_activations
            activations_list.append(model.hidden_activations.cpu().numpy())

    activations = np.concatenate(activations_list, axis=0)
    return activations
