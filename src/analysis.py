"""Analysis functions for running experiments and aggregating results.

Implements experiment orchestration and result loading.
"""
import os
import json
import pandas as pd
import torch
from typing import Dict, List
from src.config import DatasetConfig, TaskConfig, ModelConfig, TrainConfig, SweepConfig
from src.data import generate_dataset
from src.model import ContextMLP
from src.train import train_model
from src.metrics import compute_all_metrics
from src.utils import set_global_seed, ensure_dir, save_json, hash_config


def run_single_experiment(
    dataset_cfg: DatasetConfig,
    task_cfg: TaskConfig,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    out_root: str
) -> Dict:
    """
    Run a single experiment with given configuration.

    Args:
        dataset_cfg: Dataset configuration
        task_cfg: Task configuration
        model_cfg: Model configuration
        train_cfg: Training configuration
        out_root: Root directory for outputs

    Returns:
        results: Dictionary with metrics, history, and paths
    """
    # Set seeds
    set_global_seed(dataset_cfg.seed)

    # Create run directory
    cfg_dict = {
        'dataset': dataset_cfg.__dict__,
        'task': task_cfg.__dict__,
        'model': model_cfg.__dict__,
        'train': train_cfg.__dict__
    }
    cfg_hash = hash_config(cfg_dict)
    run_dir = os.path.join(out_root, f"s{dataset_cfg.s:.2f}_T{task_cfg.T}_seed{dataset_cfg.seed}_{cfg_hash}")
    ensure_dir(run_dir)

    # Generate training data
    X_train, Y_train, ctx_train, metadata_train = generate_dataset(
        n_per_ctx=dataset_cfg.n_train_per_ctx,
        D=dataset_cfg.D,
        C=dataset_cfg.C,
        R=dataset_cfg.R,
        T=task_cfg.T,
        M_dis=dataset_cfg.M_dis if dataset_cfg.M_dis > 0 else 32,
        M_un=dataset_cfg.M_un,
        q=dataset_cfg.q,
        s=dataset_cfg.s,
        seed=dataset_cfg.seed,
        class_balance=task_cfg.class_balance
    )

    # Generate validation data (different seed)
    X_val, Y_val, ctx_val, metadata_val = generate_dataset(
        n_per_ctx=dataset_cfg.n_val_per_ctx,
        D=dataset_cfg.D,
        C=dataset_cfg.C,
        R=dataset_cfg.R,
        T=task_cfg.T,
        M_dis=dataset_cfg.M_dis if dataset_cfg.M_dis > 0 else 32,
        M_un=dataset_cfg.M_un,
        q=dataset_cfg.q,
        s=dataset_cfg.s,
        seed=dataset_cfg.seed + 10000,
        class_balance=task_cfg.class_balance
    )

    # Create model
    in_dim = X_train.shape[1]
    n_tasks = Y_train.shape[1]

    if model_cfg.in_dim == -1:
        model_cfg.in_dim = in_dim
    if model_cfg.out_dim == -1:
        model_cfg.out_dim = n_tasks

    model = ContextMLP(
        in_dim=in_dim,
        hidden=model_cfg.hidden,
        n_tasks=n_tasks
    )

    # Train model
    history = train_model(
        model=model,
        X_train=X_train,
        Y_train=Y_train,
        X_val=X_val,
        Y_val=Y_val,
        lr=train_cfg.lr,
        batch_size=train_cfg.batch_size,
        epochs=train_cfg.epochs,
        patience=train_cfg.patience,
        device=train_cfg.device,
        verbose=True
    )

    # Compute metrics
    metrics = compute_all_metrics(
        model=model,
        X=X_val,
        ctx_index=ctx_val,
        latents=metadata_val['latents'],
        device=train_cfg.device,
        n_samples_per_ctx=1000
    )

    # Save results
    results = {
        'config': cfg_dict,
        'history': {
            'train_loss': [float(x) for x in history['train_loss']],
            'val_loss': [float(x) for x in history['val_loss']],
            'train_acc': [float(x) for x in history['train_acc']],
            'val_acc': [float(x) for x in history['val_acc']],
            'best_epoch': int(history['best_epoch']),
            'best_val_loss': float(history['best_val_loss'])
        },
        'metrics': {
            'contextual_fraction': float(metrics['contextual_fraction']),
            'subspace_specialization': float(metrics['subspace_specialization']),
            'best_k_clusters': int(metrics['best_k_clusters'])
        },
        'paths': {
            'run_dir': run_dir,
            'config': os.path.join(run_dir, 'config.json'),
            'final_metrics': os.path.join(run_dir, 'final_metrics.json'),
            'model_state': os.path.join(run_dir, 'model_state.pt')
        }
    }

    # Save artifacts
    save_json(os.path.join(run_dir, 'config.json'), cfg_dict)
    save_json(os.path.join(run_dir, 'final_metrics.json'), results['metrics'])
    torch.save(model.state_dict(), os.path.join(run_dir, 'model_state.pt'))

    return results


def run_sweep(
    sweep_cfg: SweepConfig,
    dataset_cfg: DatasetConfig,
    task_cfg: TaskConfig,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    out_root: str
) -> None:
    """
    Run a parameter sweep over s and T values with multiple seeds.

    Args:
        sweep_cfg: Sweep configuration
        dataset_cfg: Base dataset configuration
        task_cfg: Base task configuration
        model_cfg: Base model configuration
        train_cfg: Base training configuration
        out_root: Root directory for outputs
    """
    all_results = []

    total_runs = len(sweep_cfg.s_values) * len(sweep_cfg.T_values) * len(sweep_cfg.seeds)
    run_idx = 0

    for s in sweep_cfg.s_values:
        for T in sweep_cfg.T_values:
            for seed in sweep_cfg.seeds:
                run_idx += 1
                print(f"\n{'='*60}")
                print(f"Run {run_idx}/{total_runs}: s={s}, T={T}, seed={seed}")
                print(f"{'='*60}")

                # Update configs for this run
                dataset_cfg_run = DatasetConfig(
                    D=dataset_cfg.D,
                    C=dataset_cfg.C,
                    R=dataset_cfg.R,
                    n_train_per_ctx=dataset_cfg.n_train_per_ctx,
                    n_val_per_ctx=dataset_cfg.n_val_per_ctx,
                    q=dataset_cfg.q,
                    M_dis=dataset_cfg.M_dis,
                    M_un=dataset_cfg.M_un,
                    s=s,  # Sweep parameter
                    seed=seed  # Sweep parameter
                )

                task_cfg_run = TaskConfig(
                    T=T,  # Sweep parameter
                    class_balance=task_cfg.class_balance,
                    seed=seed
                )

                model_cfg_run = ModelConfig(
                    in_dim=model_cfg.in_dim,
                    hidden=model_cfg.hidden,
                    out_dim=model_cfg.out_dim,
                    activation=model_cfg.activation,
                    dropout=model_cfg.dropout,
                    seed=seed
                )

                train_cfg_run = TrainConfig(
                    batch_size=train_cfg.batch_size,
                    lr=train_cfg.lr,
                    weight_decay=train_cfg.weight_decay,
                    epochs=train_cfg.epochs,
                    patience=train_cfg.patience,
                    device=train_cfg.device,
                    seed=seed
                )

                # Run experiment
                results = run_single_experiment(
                    dataset_cfg_run,
                    task_cfg_run,
                    model_cfg_run,
                    train_cfg_run,
                    out_root
                )
                all_results.append(results)

    # Save aggregate CSV
    summary_data = []
    for res in all_results:
        cfg = res['config']
        summary_data.append({
            's': cfg['dataset']['s'],
            'T': cfg['task']['T'],
            'seed': cfg['dataset']['seed'],
            'D': cfg['dataset']['D'],
            'C': cfg['dataset']['C'],
            'R': cfg['dataset']['R'],
            'contextual_fraction': res['metrics']['contextual_fraction'],
            'subspace_specialization': res['metrics']['subspace_specialization'],
            'best_k_clusters': res['metrics']['best_k_clusters'],
            'best_val_loss': res['history']['best_val_loss'],
            'final_train_acc': res['history']['train_acc'][-1],
            'final_val_acc': res['history']['val_acc'][-1]
        })

    df = pd.DataFrame(summary_data)
    aggregate_path = os.path.join(out_root, 'aggregate.csv')
    df.to_csv(aggregate_path, index=False)

    print(f"\n{'='*60}")
    print(f"Sweep complete! Aggregate results saved to {aggregate_path}")
    print(f"{'='*60}\n")


def load_results(out_root: str) -> pd.DataFrame:
    """
    Load aggregated results from a sweep.

    Args:
        out_root: Root directory with results

    Returns:
        df: DataFrame with all results
    """
    aggregate_path = os.path.join(out_root, 'aggregate.csv')
    if not os.path.exists(aggregate_path):
        # Try loading from summary.csv (legacy)
        aggregate_path = os.path.join(out_root, 'summary.csv')

    if not os.path.exists(aggregate_path):
        raise FileNotFoundError(f"No aggregate results found in {out_root}")

    return pd.read_csv(aggregate_path)
