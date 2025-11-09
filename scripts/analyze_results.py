"""
Script to aggregate results and generate figures reproducing Johnston et al.
"""
import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import ensure_dir


def load_summary(results_dir: str) -> pd.DataFrame:
    """Load summary CSV from results directory."""
    summary_path = os.path.join(results_dir, 'summary.csv')
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Summary file not found: {summary_path}")
    return pd.read_csv(summary_path)


def plot_heatmap(df: pd.DataFrame, metric: str, output_path: str, title: str = None):
    """
    Plot heatmap of metric vs s and T.

    Args:
        df: DataFrame with columns s, T, and metric
        metric: Name of metric column to plot
        output_path: Path to save figure
        title: Optional custom title
    """
    # Aggregate over seeds (mean)
    df_agg = df.groupby(['s', 'T'])[metric].mean().reset_index()

    # Pivot for heatmap
    pivot = df_agg.pivot(index='T', columns='s', values=metric)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot heatmap
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='viridis', ax=ax,
                cbar_kws={'label': metric})

    # Labels
    ax.set_xlabel('Structure parameter s', fontsize=12)
    ax.set_ylabel('Number of tasks T', fontsize=12)
    if title is None:
        title = f'{metric} vs s and T'
    ax.set_title(title, fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved heatmap to {output_path}")


def plot_metric_by_s(df: pd.DataFrame, metric: str, output_path: str, title: str = None):
    """
    Plot metric vs s for different T values.

    Args:
        df: DataFrame with columns s, T, and metric
        metric: Name of metric column to plot
        output_path: Path to save figure
        title: Optional custom title
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot for each T value
    T_values = sorted(df['T'].unique())
    for T in T_values:
        df_T = df[df['T'] == T]
        # Aggregate over seeds
        df_agg = df_T.groupby('s')[metric].agg(['mean', 'std']).reset_index()

        ax.plot(df_agg['s'], df_agg['mean'], marker='o', label=f'T={T}')
        ax.fill_between(df_agg['s'],
                        df_agg['mean'] - df_agg['std'],
                        df_agg['mean'] + df_agg['std'],
                        alpha=0.2)

    ax.set_xlabel('Structure parameter s', fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    if title is None:
        title = f'{metric} vs structure parameter s'
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved line plot to {output_path}")


def plot_metric_by_T(df: pd.DataFrame, metric: str, output_path: str, title: str = None):
    """
    Plot metric vs T for different s values.

    Args:
        df: DataFrame with columns s, T, and metric
        metric: Name of metric column to plot
        output_path: Path to save figure
        title: Optional custom title
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot for each s value
    s_values = sorted(df['s'].unique())
    for s in s_values:
        df_s = df[df['s'] == s]
        # Aggregate over seeds
        df_agg = df_s.groupby('T')[metric].agg(['mean', 'std']).reset_index()

        ax.plot(df_agg['T'], df_agg['mean'], marker='o', label=f's={s:.2f}')
        ax.fill_between(df_agg['T'],
                        df_agg['mean'] - df_agg['std'],
                        df_agg['mean'] + df_agg['std'],
                        alpha=0.2)

    ax.set_xlabel('Number of tasks T', fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    if title is None:
        title = f'{metric} vs number of tasks T'
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved line plot to {output_path}")


def generate_all_figures(results_dir: str, output_dir: str):
    """
    Generate all figures from results.

    Creates:
    - Heatmaps of Contextual Fraction and Subspace Specialization
    - Line plots showing trends
    """
    # Load data
    print(f"Loading results from {results_dir}...")
    df = load_summary(results_dir)

    print(f"Loaded {len(df)} runs")
    print(f"  s values: {sorted(df['s'].unique())}")
    print(f"  T values: {sorted(df['T'].unique())}")
    print(f"  seeds: {sorted(df['seed'].unique())}")
    print()

    # Ensure output directory exists
    ensure_dir(output_dir)

    # Contextual Fraction heatmap
    plot_heatmap(
        df, 'contextual_fraction',
        os.path.join(output_dir, 'contextual_fraction_heatmap.png'),
        title='Contextual Fraction (Explicit Modularity)'
    )

    # Subspace Specialization heatmap
    plot_heatmap(
        df, 'subspace_specialization',
        os.path.join(output_dir, 'subspace_specialization_heatmap.png'),
        title='Subspace Specialization (Implicit Modularity)'
    )

    # Contextual Fraction vs s
    plot_metric_by_s(
        df, 'contextual_fraction',
        os.path.join(output_dir, 'contextual_fraction_vs_s.png'),
        title='Contextual Fraction vs Structure Parameter'
    )

    # Subspace Specialization vs s
    plot_metric_by_s(
        df, 'subspace_specialization',
        os.path.join(output_dir, 'subspace_specialization_vs_s.png'),
        title='Subspace Specialization vs Structure Parameter'
    )

    # Contextual Fraction vs T
    plot_metric_by_T(
        df, 'contextual_fraction',
        os.path.join(output_dir, 'contextual_fraction_vs_T.png'),
        title='Contextual Fraction vs Number of Tasks'
    )

    # Subspace Specialization vs T
    plot_metric_by_T(
        df, 'subspace_specialization',
        os.path.join(output_dir, 'subspace_specialization_vs_T.png'),
        title='Subspace Specialization vs Number of Tasks'
    )

    # Summary statistics
    print("\n" + "="*60)
    print("Summary Statistics")
    print("="*60)

    print("\nContextual Fraction by s and T:")
    print(df.groupby(['s', 'T'])['contextual_fraction'].mean().unstack())

    print("\nSubspace Specialization by s and T:")
    print(df.groupby(['s', 'T'])['subspace_specialization'].mean().unstack())

    print("\nValidation Accuracy by s and T:")
    print(df.groupby(['s', 'T'])['final_val_acc'].mean().unstack())

    print(f"\nAll figures saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Analyze modularity reproduction results')
    parser.add_argument('--results_dir', type=str, default='results/runs',
                       help='Directory with run results')
    parser.add_argument('--output_dir', type=str, default='results/figures',
                       help='Directory to save figures')

    args = parser.parse_args()

    generate_all_figures(args.results_dir, args.output_dir)


if __name__ == '__main__':
    main()
