# Quick Start Guide

## Installation

```bash
cd /home/user/modularity_in_NN

# Install dependencies
pip install -r requirements.txt
# or
pip install numpy torch scikit-learn matplotlib seaborn pandas tqdm
```

## Verify Installation

```bash
python scripts/test_sanity.py
```

Expected output:
```
==============================================================
Running Sanity Checks
==============================================================

Testing data generation...
  X shape: (400, 36)
  Y shape: (400, 2)
  Contexts: 4
  ✓ Data generation passed

Testing model forward pass...
  Input shape: torch.Size([10, 64])
  Output shape: torch.Size([10, 3])
  ✓ Model forward pass passed

Testing training loop...
  Trained for N epochs
  Final train acc: >0.90
  ✓ Training passed

Testing metrics computation...
  Contextual Fraction: 0.XX
  Subspace Specialization: 0.XX
  ✓ Metrics computation passed

==============================================================
All Sanity Checks Passed! ✓
==============================================================
```

## Quick Experiment (5-10 minutes)

Test the implementation with a minimal sweep:

```bash
python scripts/run_sweep.py \
  --s_values 0.0 1.0 \
  --T_values 1 5 \
  --seeds 1 \
  --train_per_ctx 2000 \
  --val_per_ctx 500 \
  --epochs 30 \
  --output_dir results/quick_test
```

This runs 4 experiments (2 s-values × 2 T-values × 1 seed).

## Generate Test Figures

```bash
python scripts/analyze_results.py \
  --results_dir results/quick_test \
  --output_dir results/quick_test/figures
```

Inspect the heatmaps:
```bash
ls results/quick_test/figures/
```

Expected files:
- `contextual_fraction_heatmap.png`
- `subspace_specialization_heatmap.png`
- Line plots showing trends

## Full Reproduction (~2-3 hours)

Reproduce Johnston et al. main results:

```bash
python scripts/run_sweep.py \
  --s_values 0.0 0.25 0.5 0.75 1.0 \
  --T_values 1 3 5 10 \
  --seeds 1 2 3 \
  --output_dir results/full_reproduction
```

This runs 60 experiments (5 s × 4 T × 3 seeds).

Generate figures:

```bash
python scripts/analyze_results.py \
  --results_dir results/full_reproduction \
  --output_dir results/full_reproduction/figures
```

## Expected Results

### Contextual Fraction (Explicit Modularity)
| s \\ T |  1   |  3   |  5   |  10  |
|--------|------|------|------|------|
| 0.00   | ~0.0 | ~0.0 | ~0.0 | ~0.0 |
| 0.25   | low  | low  | low  | low  |
| 0.50   | med  | med  | med  | med  |
| 0.75   | high | high | high | high |
| 1.00   | high | high | high | high |

**Pattern**: CF increases with s (structure), relatively independent of T.

### Subspace Specialization (Implicit Modularity)
| s \\ T |  1   |  3   |  5   |  10  |
|--------|------|------|------|------|
| 0.00   | low  | med  | med  | high |
| 0.25   | low  | low  | med  | med  |
| 0.50   | low  | low  | low  | med  |
| 0.75   | low  | low  | low  | low  |
| 1.00   | ~0.0 | ~0.0 | ~0.0 | ~0.0 |

**Pattern**: SS increases with T (tasks) when s is low (unstructured).

## Troubleshooting

### Dependencies not installing
```bash
# Try with explicit versions
pip install numpy==2.3.4 torch==2.9.0 scikit-learn==1.7.2 matplotlib==3.10.7
```

### Out of memory
```bash
# Reduce dataset size
python scripts/run_sweep.py --train_per_ctx 5000 --val_per_ctx 1000
```

### Sanity checks fail
```bash
# Check Python version (need >=3.8)
python --version

# Re-install dependencies
pip install --upgrade --force-reinstall -r requirements.txt
```

## File Structure After Running

```
modularity_in_NN/
├── results/
│   ├── runs/
│   │   ├── s0.00_T1_seed1/
│   │   │   ├── results.json      # Metrics + config
│   │   │   └── model.pt          # Trained weights
│   │   ├── s0.00_T5_seed1/
│   │   ├── ...
│   │   └── summary.csv           # All results aggregated
│   └── figures/
│       ├── contextual_fraction_heatmap.png
│       ├── subspace_specialization_heatmap.png
│       └── ...
```

## Next Steps

1. **Inspect results**: `cat results/runs/summary.csv | column -t -s,`
2. **View figures**: Open PNG files in `results/figures/`
3. **Single run analysis**: `cat results/runs/s1.00_T5_seed1/results.json | jq`
4. **Extend to NTK**: Implement kernel computation, curriculum learning
5. **Theory derivation**: Match plateau widths to spectral gaps

## Common Commands

```bash
# List all completed runs
ls results/runs/*/results.json

# Count total runs
ls -d results/runs/s* | wc -l

# View summary statistics
python -c "import pandas as pd; df = pd.read_csv('results/runs/summary.csv'); print(df.groupby(['s','T']).mean()[['contextual_fraction', 'subspace_specialization']])"

# Run single experiment
python scripts/run_sweep.py --s_values 0.5 --T_values 3 --seeds 42 --output_dir results/single
```

## Performance Notes

- **Quick test (4 runs)**: ~5-10 minutes
- **Full reproduction (60 runs)**: ~2-3 hours
- **Per-run time**: ~2-3 minutes (depends on early stopping)
- **Bottleneck**: Training loop (can parallelize across runs)

## Citation

If using this code:

```bibtex
@article{johnston2023modular,
  title={Modular representations emerge in neural networks trained to perform context-dependent tasks},
  author={Johnston, W. Jeffrey and Fusi, Stefano},
  journal={Nature Communications},
  year={2023}
}
```
