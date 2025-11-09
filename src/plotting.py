"""Plotting functions for visualizing modularity metrics.

Generates heatmaps and line plots for analysis.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional
from src.utils import ensure_dir


def plot_heatmap(
    df: pd.DataFrame,
    x: str = 's',
    y: str = 'T',
    value: str = 'contextual_fraction',
    out_path: str = 'heatmap.png',
    title: Optional[str] = None,
    cmap: str = 'viridis',
    figsize: tuple = (8, 6)
) -> None:
    """
    Plot heatmap of metric vs two parameters.

    Args:
        df: DataFrame with columns x, y, and value
        x: Column name for x-axis (default: 's')
        y: Column name for y-axis (default: 'T')
        value: Column name for heatmap values
        out_path: Path to save figure
        title: Optional custom title
        cmap: Colormap name
        figsize: Figure size (width, height)
    """
    # Aggregate over seeds (mean)
    df_agg = df.groupby([x, y])[value].mean().reset_index()

    # Pivot for heatmap
    pivot = df_agg.pivot(index=y, columns=x, values=value)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap
    sns.heatmap(
        pivot,
        annot=True,
        fmt='.3f',
        cmap=cmap,
        ax=ax,
        cbar_kws={'label': value}
    )

    # Labels
    ax.set_xlabel(f'{x}', fontsize=12)
    ax.set_ylabel(f'{y}', fontsize=12)

    if title is None:
        title = f'{value} vs {x} and {y}'
    ax.set_title(title, fontsize=14)

    plt.tight_layout()
    ensure_dir(os.path.dirname(out_path))
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved heatmap to {out_path}")


def plot_scatter_means(
    A: np.ndarray,
    out_path: str,
    title: str = 'Mean Activity per Unit per Context'
) -> None:
    """
    Scatter plot of mean activity matrix.

    Args:
        A: (n_hidden, n_contexts) mean activity per unit per context
        out_path: Path to save figure
        title: Plot title
    """
    n_hidden, n_contexts = A.shape

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot each context as a scatter
    for c in range(n_contexts):
        ax.scatter(
            np.arange(n_hidden),
            A[:, c],
            alpha=0.6,
            s=10,
            label=f'Context {c}'
        )

    ax.set_xlabel('Hidden Unit Index', fontsize=12)
    ax.set_ylabel('Mean Activity', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    ensure_dir(os.path.dirname(out_path))
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved scatter plot to {out_path}")


def plot_metric_by_s(
    df: pd.DataFrame,
    metric: str,
    out_path: str,
    title: Optional[str] = None,
    figsize: tuple = (8, 6)
) -> None:
    """
    Plot metric vs structure parameter s for different T values.

    Args:
        df: DataFrame with columns s, T, and metric
        metric: Name of metric column to plot
        out_path: Path to save figure
        title: Optional custom title
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot for each T value
    T_values = sorted(df['T'].unique())
    for T in T_values:
        df_T = df[df['T'] == T]
        # Aggregate over seeds
        df_agg = df_T.groupby('s')[metric].agg(['mean', 'std']).reset_index()

        ax.plot(df_agg['s'], df_agg['mean'], marker='o', label=f'T={T}', linewidth=2)
        ax.fill_between(
            df_agg['s'],
            df_agg['mean'] - df_agg['std'],
            df_agg['mean'] + df_agg['std'],
            alpha=0.2
        )

    ax.set_xlabel('Structure parameter s', fontsize=12)
    ax.set_ylabel(metric, fontsize=12)

    if title is None:
        title = f'{metric} vs structure parameter s'
    ax.set_title(title, fontsize=14)

    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    ensure_dir(os.path.dirname(out_path))
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved line plot to {out_path}")


def plot_metric_by_T(
    df: pd.DataFrame,
    metric: str,
    out_path: str,
    title: Optional[str] = None,
    figsize: tuple = (8, 6)
) -> None:
    """
    Plot metric vs number of tasks T for different s values.

    Args:
        df: DataFrame with columns s, T, and metric
        metric: Name of metric column to plot
        out_path: Path to save figure
        title: Optional custom title
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot for each s value
    s_values = sorted(df['s'].unique())
    for s in s_values:
        df_s = df[df['s'] == s]
        # Aggregate over seeds
        df_agg = df_s.groupby('T')[metric].agg(['mean', 'std']).reset_index()

        ax.plot(df_agg['T'], df_agg['mean'], marker='o', label=f's={s:.2f}', linewidth=2)
        ax.fill_between(
            df_agg['T'],
            df_agg['mean'] - df_agg['std'],
            df_agg['mean'] + df_agg['std'],
            alpha=0.2
        )

    ax.set_xlabel('Number of tasks T', fontsize=12)
    ax.set_ylabel(metric, fontsize=12)

    if title is None:
        title = f'{metric} vs number of tasks T'
    ax.set_title(title, fontsize=14)

    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    ensure_dir(os.path.dirname(out_path))
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved line plot to {out_path}")


def generate_all_plots(
    df: pd.DataFrame,
    out_dir: str,
    metrics: list = None
) -> None:
    """
    Generate all standard plots for analysis.

    Args:
        df: DataFrame with results
        out_dir: Directory to save plots
        metrics: List of metrics to plot (default: ['contextual_fraction', 'subspace_specialization'])
    """
    ensure_dir(out_dir)

    if metrics is None:
        metrics = ['contextual_fraction', 'subspace_specialization']

    for metric in metrics:
        # Heatmap
        plot_heatmap(
            df,
            x='s',
            y='T',
            value=metric,
            out_path=os.path.join(out_dir, f'{metric}_heatmap.png'),
            title=f'{metric.replace("_", " ").title()}'
        )

        # Line plot vs s
        plot_metric_by_s(
            df,
            metric=metric,
            out_path=os.path.join(out_dir, f'{metric}_vs_s.png'),
            title=f'{metric.replace("_", " ").title()} vs Structure Parameter'
        )

        # Line plot vs T
        plot_metric_by_T(
            df,
            metric=metric,
            out_path=os.path.join(out_dir, f'{metric}_vs_T.png'),
            title=f'{metric.replace("_", " ").title()} vs Number of Tasks'
        )

    print(f"\nAll plots saved to {out_dir}")
