"""Configuration dataclasses for all hyperparameters."""
from dataclasses import dataclass, asdict
from typing import List
import json


@dataclass
class DatasetConfig:
    """Configuration for dataset generation (combines data and latents)."""
    D: int = 3  # Decision-relevant binary latents
    C: int = 2  # Number of contexts
    R: int = 0  # Irrelevant latents
    n_train_per_ctx: int = 20000  # Samples per context for training
    n_val_per_ctx: int = 5000  # Samples per context for validation
    q: int = 2  # Unstructured interaction order
    M_dis: int = 0  # Dimensionality of disentangled part (auto if 0)
    M_un: int = 256  # Dimensionality of unstructured projection
    s: float = 0.5  # Structure interpolation in [0,1]
    seed: int = 1


@dataclass
class TaskConfig:
    """Configuration for task generation."""
    T: int = 5  # Number of tasks
    class_balance: str = 'median'  # 'median' or 'auto' thresholding
    seed: int = 1


@dataclass
class ModelConfig:
    """Configuration for neural network architecture."""
    in_dim: int = -1  # Computed automatically (set -1 to auto)
    hidden: int = 256  # Hidden layer size
    out_dim: int = -1  # Equals T (set -1 to auto)
    activation: str = 'relu'  # Activation function
    dropout: float = 0.0  # Dropout rate
    seed: int = 1


@dataclass
class TrainConfig:
    """Configuration for training."""
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 80
    patience: int = 10  # Early stop patience
    device: str = 'cpu'
    seed: int = 1


@dataclass
class SweepConfig:
    """Configuration for parameter sweeps."""
    s_values: List[float] = None
    T_values: List[int] = None
    seeds: List[int] = None

    def __post_init__(self):
        if self.s_values is None:
            self.s_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        if self.T_values is None:
            self.T_values = [1, 3, 5, 10]
        if self.seeds is None:
            self.seeds = [1, 2, 3]


# Legacy configs for backward compatibility
@dataclass
class LatentConfig:
    """Configuration for latent variables."""
    D: int = 3  # Decision-relevant variables
    C: int = 2  # Context variables
    R: int = 4  # Irrelevant variables


@dataclass
class InputModelConfig:
    """Configuration for input representation."""
    M_dis: int = 32  # Dimension for disentangled representation
    M_un: int = 64   # Dimension for unstructured representation
    q: int = 2       # Order of interaction terms for unstructured
    s: float = 0.5   # Interpolation parameter [0,1]: 0=unstructured, 1=disentangled


@dataclass
class DataConfig:
    """Configuration for dataset generation."""
    train_per_ctx: int = 20000
    val_per_ctx: int = 5000


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""
    latent: LatentConfig
    input_model: InputModelConfig
    task: TaskConfig
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    seed: int = 1

    def to_dict(self):
        """Convert to dictionary."""
        return {
            'latent': asdict(self.latent),
            'input_model': asdict(self.input_model),
            'task': asdict(self.task),
            'data': asdict(self.data),
            'model': asdict(self.model),
            'train': asdict(self.train),
            'seed': self.seed
        }

    def save(self, path: str):
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str):
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(
            latent=LatentConfig(**data['latent']),
            input_model=InputModelConfig(**data['input_model']),
            task=TaskConfig(**data['task']),
            data=DataConfig(**data['data']),
            model=ModelConfig(**data['model']),
            train=TrainConfig(**data['train']),
            seed=data['seed']
        )


def get_default_config() -> ExperimentConfig:
    """Return default configuration matching the paper."""
    return ExperimentConfig(
        latent=LatentConfig(D=3, C=2, R=4),
        input_model=InputModelConfig(M_dis=32, M_un=64, q=2, s=0.5),
        task=TaskConfig(T=5),
        data=DataConfig(train_per_ctx=20000, val_per_ctx=5000),
        model=ModelConfig(hidden=256),
        train=TrainConfig(lr=1e-3, batch_size=512, epochs=80, patience=10),
        seed=1
    )
