# Modularity in Neural Networks - Johnston et al. Reproduction

This repository reproduces the key findings from:

**Johnston et al., "Modular representations emerge in neural networks trained to perform context-dependent tasks"**

## Overview

The paper demonstrates two forms of modularity in neural networks trained on context-dependent multi-task learning:

1. **Explicit Modularity (Contextual Fraction)**: Individual neurons selective to specific contexts
2. **Implicit Modularity (Subspace Specialization)**: Distributed representations organized into context-specific subspaces

Key findings reproduced:
- High input structure (disentangled) → high Contextual Fraction, independent of #tasks
- Low input structure (unstructured) → low Contextual Fraction, but Subspace Specialization increases with #tasks

## Repository Structure

```
modularity_in_NN/
├── src/
│   ├── config.py          # Configuration dataclasses
│   ├── data.py            # Data generation (latents, tasks, input models)
│   ├── model.py           # 1-hidden-layer MLP
│   ├── train.py           # Training loop with early stopping
│   ├── metrics.py         # Modularity metrics
│   └── utils.py           # Utilities (seeding, normalization)
├── scripts/
│   ├── run_sweep.py       # Run parameter sweeps
│   └── analyze_results.py # Generate figures from results
├── results/
│   ├── runs/              # Individual run outputs
│   └── figures/           # Generated plots
├── pyproject.toml
└── README.md
```

## Installation

```bash
# Install dependencies
pip install -e .

# Or install required packages directly
pip install numpy torch scikit-learn matplotlib seaborn pandas tqdm
```

## Quick Start

### 1. Run a Small Validation Experiment

Test the implementation with a quick run:

```bash
python scripts/run_sweep.py \
  --s_values 0.0 1.0 \
  --T_values 1 5 \
  --seeds 1 \
  --train_per_ctx 5000 \
  --val_per_ctx 1000 \
  --epochs 40 \
  --output_dir results/runs
```

### 2. Run Full Reproduction Sweep

Reproduce the paper's main results (this will take longer):

```bash
python scripts/run_sweep.py \
  --s_values 0.0 0.25 0.5 0.75 1.0 \
  --T_values 1 3 5 10 \
  --seeds 1 2 3 \
  --train_per_ctx 20000 \
  --val_per_ctx 5000 \
  --epochs 80 \
  --output_dir results/runs
```

### 3. Generate Figures

After running experiments:

```bash
python scripts/analyze_results.py \
  --results_dir results/runs \
  --output_dir results/figures
```

This creates:
- `contextual_fraction_heatmap.png`
- `subspace_specialization_heatmap.png`
- Line plots showing trends vs s and T

## Key Parameters

### Latent Variables
- `D`: Decision-relevant variables (default: 3)
- `C`: Context variables (default: 2)
- `R`: Irrelevant variables (default: 4)

### Input Model
- `s`: Structure parameter [0,1]
  - `s=1`: Fully disentangled (each latent → separate dimensions)
  - `s=0`: Fully unstructured (random projection of interaction terms)
- `M_dis`: Disentangled representation dimension (default: 32)
- `M_un`: Unstructured representation dimension (default: 64)
- `q`: Interaction order for unstructured (default: 2, pairwise)

### Tasks
- `T`: Number of tasks per context (sweep: 1, 3, 5, 10)
- Each task is a linear binary classifier over decision variables, context-dependent

### Model & Training
- `hidden`: Hidden layer size (default: 256)
- `lr`: Learning rate (default: 1e-3)
- `batch_size`: Batch size (default: 512)
- `epochs`: Max epochs (default: 80)
- `patience`: Early stopping patience (default: 10)

## Expected Results

### Contextual Fraction
- **High s (disentangled)**: CF > 0.05, relatively independent of T
- **Low s (unstructured)**: CF ≈ 0 for all T

### Subspace Specialization
- **High s**: Low values, decreases with T
- **Low s**: Increases with T (implicit modularity emerges)

### Validation Accuracy
- Should achieve >95% accuracy across all conditions (tasks are linearly separable)

## Metrics Implementation

### Contextual Fraction (src/metrics.py:contextual_fraction)
From Johnston et al. Methods M7:
1. Sample 1000 stimuli per context
2. Compute mean hidden activation per unit per context
3. Count units with mean activity > 0.01 in exactly one context
4. CF = fraction of such units

### Subspace Specialization (src/metrics.py:subspace_specialization)
From Johnston et al. Methods M8:
1. For each latent variable, split data by its value (0/1)
2. Compute PCA subspaces (keep components with variance > 1e-10)
3. Compute alignment index between subspaces
4. SS = max(0, log(mean_align_nonctx / mean_align_ctx))

## Sanity Checks

The implementation includes several validation checks:

1. **Task solvability**: Linear probes should achieve >99% AUC
2. **Metric sanity**:
   - Shuffled context labels → CF ≈ 0, SS ≈ 0
   - s=1 (disentangled) → CF increases, GMM selects k=C clusters
   - s=0, increasing T → SS increases

## Reproducibility

All experiments use fixed random seeds for reproducibility:
- Data generation uses the specified seed
- Validation data uses seed + 10000
- PyTorch, NumPy, and Python random modules are seeded
- CuDNN deterministic mode enabled

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{johnston2023modular,
  title={Modular representations emerge in neural networks trained to perform context-dependent tasks},
  author={Johnston, W. Jeffrey and Fusi, Stefano},
  journal={Nature Communications},
  year={2023},
  publisher={Nature Publishing Group}
}
```

## Next Steps

After reproducing Johnston et al.:

1. **Analyze NTK spectrum**: Compute and visualize the Neural Tangent Kernel eigenspectrum
2. **Curriculum learning**: Implement slow α(t) sweeps between tasks
3. **Mode locking theory**: Connect discrete modules to NTK spectral peaks
4. **Plateau width derivation**: Derive and validate plateau width formulas

See the project description for the full theoretical framework connecting modularity to NTK peak selection under curriculum learning.

## Troubleshooting

### Low validation accuracy
- Check task generation (weights should not be all zeros)
- Increase training epochs or decrease learning rate
- Verify input normalization

### Metrics always zero
- Ensure model is trained (not random initialization)
- Check that contexts are properly encoded in data
- Verify sufficient samples per context for metrics

### Memory issues
- Reduce `train_per_ctx` and `val_per_ctx`
- Decrease `batch_size`
- Process activations in smaller batches in metrics

## Contact

For questions about this implementation or the original paper, please open an issue on GitHub.
