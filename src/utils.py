"""Utility functions for seeding, tensor operations, and I/O."""
import numpy as np
import torch
import random
import os
import json
import hashlib
from pathlib import Path
from typing import Any, Dict
import pandas as pd


def set_global_seed(seed: int) -> None:
    """
    Set random seed for reproducibility across numpy, torch, and python random.

    Args:
        seed: Random seed integer
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Make CuDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Alias for backward compatibility
set_seed = set_global_seed


def make_run_dir(base_dir: str, cfg_hash: str) -> str:
    """
    Create a unique run directory based on config hash.

    Args:
        base_dir: Base directory for runs
        cfg_hash: Short hash of configuration

    Returns:
        path: Full path to created run directory
    """
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"{timestamp}_{cfg_hash}")
    ensure_dir(run_dir)
    return run_dir


def save_json(path: str, obj: dict) -> None:
    """
    Save dictionary to JSON file.

    Args:
        path: File path
        obj: Dictionary to save
    """
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)


def save_csv(path: str, df: pd.DataFrame) -> None:
    """
    Save DataFrame to CSV file.

    Args:
        path: File path
        df: DataFrame to save
    """
    df.to_csv(path, index=False)


def save_npz(path: str, **arrays) -> None:
    """
    Save multiple numpy arrays to npz file.

    Args:
        path: File path
        **arrays: Named arrays to save
    """
    np.savez(path, **arrays)


def hash_config(obj: Any) -> str:
    """
    Create a stable short hash for configuration object.

    Args:
        obj: Configuration object (dict or dataclass)

    Returns:
        hash_str: 8-character hex hash
    """
    # Convert to JSON string
    if hasattr(obj, '__dict__'):
        obj_dict = obj.__dict__
    elif isinstance(obj, dict):
        obj_dict = obj
    else:
        obj_dict = {'value': str(obj)}

    # Create stable JSON representation
    json_str = json.dumps(obj_dict, sort_keys=True)

    # Hash
    hash_obj = hashlib.md5(json_str.encode('utf-8'))
    return hash_obj.hexdigest()[:8]


def to_device(tensor, device):
    """
    Move tensor to specified device.

    Args:
        tensor: PyTorch tensor
        device: Device string ('cpu' or 'cuda')

    Returns:
        tensor: Tensor on specified device
    """
    return tensor.to(device)


def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def normalize_rows(X: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    L2-normalize each row of X.

    Args:
        X: (N, D) array
        eps: Small epsilon to avoid division by zero

    Returns:
        X_norm: (N, D) array with unit-norm rows
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    # Avoid division by zero: clip norms to minimum of eps
    norms_safe = np.maximum(norms, eps)
    return (X / norms_safe).astype(X.dtype)


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
