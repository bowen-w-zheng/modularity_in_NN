"""
Script to run parameter sweeps over structure parameter s and number of tasks T.
"""
import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import ExperimentConfig, LatentConfig, InputModelConfig, TaskConfig, DataConfig, ModelConfig, TrainConfig
from src.data import generate_dataset
from src.model import ContextMLP
from src.train import train_model
from src.metrics import compute_all_metrics
from src.utils import set_seed, ensure_dir


def run_single_experiment(config: ExperimentConfig, output_dir: str, verbose: bool = True) -> dict:
    """
    Run a single experiment with given configuration.

    Args:
        config: Experiment configuration
        output_dir: Directory to save results
        verbose: Print progress

    Returns:
        results: Dictionary with metrics and config
    """
    # Set seed
    set_seed(config.seed)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Running experiment:")
        print(f"  s={config.input_model.s}, T={config.task.T}, seed={config.seed}")
        print(f"  D={config.latent.D}, C={config.latent.C}, R={config.latent.R}")
        print(f"{'='*60}\n")

    # Generate training data
    X_train, Y_train, ctx_train, metadata_train = generate_dataset(
        n_per_ctx=config.data.train_per_ctx,
        D=config.latent.D,
        C=config.latent.C,
        R=config.latent.R,
        T=config.task.T,
        M_dis=config.input_model.M_dis,
        M_un=config.input_model.M_un,
        q=config.input_model.q,
        s=config.input_model.s,
        seed=config.seed
    )

    # Generate validation data (use different seed)
    X_val, Y_val, ctx_val, metadata_val = generate_dataset(
        n_per_ctx=config.data.val_per_ctx,
        D=config.latent.D,
        C=config.latent.C,
        R=config.latent.R,
        T=config.task.T,
        M_dis=config.input_model.M_dis,
        M_un=config.input_model.M_un,
        q=config.input_model.q,
        s=config.input_model.s,
        seed=config.seed + 10000  # Different seed for validation
    )

    if verbose:
        print(f"Generated dataset:")
        print(f"  Train: X={X_train.shape}, Y={Y_train.shape}")
        print(f"  Val:   X={X_val.shape}, Y={Y_val.shape}")
        print(f"  Input dim: {X_train.shape[1]}, Tasks: {Y_train.shape[1]}\n")

    # Create model
    in_dim = X_train.shape[1]
    n_tasks = Y_train.shape[1]
    model = ContextMLP(in_dim=in_dim, hidden=config.model.hidden, n_tasks=n_tasks)

    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model: {n_params} parameters\n")

    # Train model
    history = train_model(
        model=model,
        X_train=X_train,
        Y_train=Y_train,
        X_val=X_val,
        Y_val=Y_val,
        lr=config.train.lr,
        batch_size=config.train.batch_size,
        epochs=config.train.epochs,
        patience=config.train.patience,
        device=config.train.device,
        verbose=verbose
    )

    if verbose:
        print(f"\nTraining complete:")
        print(f"  Best epoch: {history['best_epoch']}")
        print(f"  Best val loss: {history['best_val_loss']:.4f}")
        print(f"  Final train acc: {history['train_acc'][-1]:.4f}")
        print(f"  Final val acc: {history['val_acc'][-1]:.4f}\n")

    # Compute metrics
    if verbose:
        print("Computing modularity metrics...")

    metrics = compute_all_metrics(
        model=model,
        X=X_val,
        ctx_index=ctx_val,
        latents=metadata_val['latents'],
        device=config.train.device,
        n_samples_per_ctx=1000
    )

    if verbose:
        print(f"  Contextual Fraction: {metrics['contextual_fraction']:.4f}")
        print(f"  Subspace Specialization: {metrics['subspace_specialization']:.4f}")
        print(f"  Best K clusters: {metrics['best_k_clusters']}")

    # Prepare results
    results = {
        'config': config.to_dict(),
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
        }
    }

    # Save results
    run_name = f"s{config.input_model.s:.2f}_T{config.task.T}_seed{config.seed}"
    run_dir = os.path.join(output_dir, run_name)
    ensure_dir(run_dir)

    # Save JSON
    with open(os.path.join(run_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Save model checkpoint
    torch.save(model.state_dict(), os.path.join(run_dir, 'model.pt'))

    if verbose:
        print(f"\nResults saved to {run_dir}\n")

    return results


def run_sweep(
    s_values,
    T_values,
    seeds,
    D=3, C=2, R=4,
    M_dis=32, M_un=64, q=2,
    train_per_ctx=20000, val_per_ctx=5000,
    hidden=256,
    lr=1e-3, batch_size=512, epochs=80, patience=10,
    device='cpu',
    output_dir='results/runs',
    verbose=True
):
    """
    Run a sweep over s and T values with multiple seeds.

    Args:
        s_values: List of structure parameters to sweep
        T_values: List of task counts to sweep
        seeds: List of random seeds
        ... (other parameters with defaults)

    Returns:
        all_results: List of all result dictionaries
    """
    all_results = []

    total_runs = len(s_values) * len(T_values) * len(seeds)
    run_idx = 0

    for s in s_values:
        for T in T_values:
            for seed in seeds:
                run_idx += 1
                print(f"\n{'#'*60}")
                print(f"Run {run_idx}/{total_runs}")
                print(f"{'#'*60}")

                config = ExperimentConfig(
                    latent=LatentConfig(D=D, C=C, R=R),
                    input_model=InputModelConfig(M_dis=M_dis, M_un=M_un, q=q, s=s),
                    task=TaskConfig(T=T),
                    data=DataConfig(train_per_ctx=train_per_ctx, val_per_ctx=val_per_ctx),
                    model=ModelConfig(hidden=hidden),
                    train=TrainConfig(lr=lr, batch_size=batch_size, epochs=epochs, patience=patience, device=device),
                    seed=seed
                )

                results = run_single_experiment(config, output_dir, verbose=verbose)
                all_results.append(results)

    # Save summary CSV
    summary_data = []
    for res in all_results:
        cfg = res['config']
        summary_data.append({
            's': cfg['input_model']['s'],
            'T': cfg['task']['T'],
            'seed': cfg['seed'],
            'D': cfg['latent']['D'],
            'C': cfg['latent']['C'],
            'R': cfg['latent']['R'],
            'contextual_fraction': res['metrics']['contextual_fraction'],
            'subspace_specialization': res['metrics']['subspace_specialization'],
            'best_k_clusters': res['metrics']['best_k_clusters'],
            'best_val_loss': res['history']['best_val_loss'],
            'final_train_acc': res['history']['train_acc'][-1],
            'final_val_acc': res['history']['val_acc'][-1]
        })

    df = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, 'summary.csv')
    df.to_csv(summary_path, index=False)
    print(f"\n{'='*60}")
    print(f"Sweep complete! Summary saved to {summary_path}")
    print(f"{'='*60}\n")

    return all_results


def main():
    parser = argparse.ArgumentParser(description='Run modularity reproduction experiments')

    # Sweep parameters
    parser.add_argument('--s_values', type=float, nargs='+', default=[0.0, 0.25, 0.5, 0.75, 1.0],
                       help='Structure parameter values to sweep')
    parser.add_argument('--T_values', type=int, nargs='+', default=[1, 3, 5, 10],
                       help='Number of tasks to sweep')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3],
                       help='Random seeds')

    # Latent config
    parser.add_argument('--D', type=int, default=3, help='Decision variables')
    parser.add_argument('--C', type=int, default=2, help='Context variables')
    parser.add_argument('--R', type=int, default=4, help='Irrelevant variables')

    # Input model config
    parser.add_argument('--M_dis', type=int, default=32, help='Disentangled dimension')
    parser.add_argument('--M_un', type=int, default=64, help='Unstructured dimension')
    parser.add_argument('--q', type=int, default=2, help='Interaction order')

    # Data config
    parser.add_argument('--train_per_ctx', type=int, default=20000, help='Training samples per context')
    parser.add_argument('--val_per_ctx', type=int, default=5000, help='Validation samples per context')

    # Model config
    parser.add_argument('--hidden', type=int, default=256, help='Hidden layer size')

    # Training config
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--epochs', type=int, default=80, help='Max epochs')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu or cuda)')

    # Output
    parser.add_argument('--output_dir', type=str, default='results/runs', help='Output directory')
    parser.add_argument('--quiet', action='store_true', help='Quiet mode')

    args = parser.parse_args()

    # Run sweep
    run_sweep(
        s_values=args.s_values,
        T_values=args.T_values,
        seeds=args.seeds,
        D=args.D, C=args.C, R=args.R,
        M_dis=args.M_dis, M_un=args.M_un, q=args.q,
        train_per_ctx=args.train_per_ctx, val_per_ctx=args.val_per_ctx,
        hidden=args.hidden,
        lr=args.lr, batch_size=args.batch_size, epochs=args.epochs, patience=args.patience,
        device=args.device,
        output_dir=args.output_dir,
        verbose=not args.quiet
    )


if __name__ == '__main__':
    main()
