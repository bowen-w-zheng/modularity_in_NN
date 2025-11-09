"""Configuration dataclasses for all hyperparameters."""
from dataclasses import dataclass, asdict
from typing import List
import json


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
class TaskConfig:
    """Configuration for task generation."""
    T: int = 5       # Number of tasks per context


@dataclass
class DataConfig:
    """Configuration for dataset generation."""
    train_per_ctx: int = 20000
    val_per_ctx: int = 5000


@dataclass
class ModelConfig:
    """Configuration for neural network architecture."""
    hidden: int = 256


@dataclass
class TrainConfig:
    """Configuration for training."""
    lr: float = 1e-3
    batch_size: int = 512
    epochs: int = 80
    patience: int = 10  # Early stopping patience
    device: str = "cpu"


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
