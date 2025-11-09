"""Utility functions for seeding, tensor operations, and I/O."""
import numpy as np
import torch
import random
import os
from pathlib import Path


def set_seed(seed: int):
    """Set random seed for reproducibility across numpy, torch, and python random."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Make CuDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def normalize_rows(X: np.ndarray) -> np.ndarray:
    """L2-normalize each row of X."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms == 0, 1, norms)
    return X / norms


def to_numpy(tensor):
    """Convert torch tensor to numpy array."""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor


def to_torch(array, device='cpu'):
    """Convert numpy array to torch tensor."""
    if isinstance(array, np.ndarray):
        return torch.from_numpy(array).float().to(device)
    return array.to(device)
