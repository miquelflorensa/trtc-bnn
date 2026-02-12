# Repository Structure

This repository contains the implementation and experiments for the TRTC-BNN paper.

## Directory Structure

```
.
├── src/                          # Source code
│   ├── methods/                  # Analytical methods (LL-Softmax, MM-Softmax, MM-Remax, etc.)
│   ├── models/                   # Neural network models
│   ├── training/                 # Training utilities
│   ├── metrics/                  # Evaluation metrics
│   └── utils/                    # Helper functions
│
├── experiments/                  # Experiment scripts
│   ├── train_all_methods_multiseed.py   # Multi-seed training experiments
│   ├── run_exhaustive_experiments.py    # Exhaustive scenario testing
│   └── plot_exhaustive_results.py       # Generate heatmap figures
│
├── scripts/                      # Utility scripts
│   ├── train.py                          # Training script
│   ├── evaluate.py                       # Evaluation script
│   ├── create_bnn_cycle_figure.py       # Generate BNN cycle diagram
│   ├── create_individual_plots.py       # Generate individual cycle components
│   ├── create_mm_softmax_procedure_figure.py  # Generate procedure figure
│   ├── plot_variance_delta_grid.py      # Generate variance-delta grids
│   └── plot_method_comparison.py        # Generate method comparison plots
│
├── figures/                      # Paper figures
│   ├── bnn_cycle_components/            # Individual BNN cycle components
│   ├── mm_softmax_procedure_compact.pdf # Compact procedure figure
│   ├── softmax_variance_delta_grid.pdf  # Variance-delta grid
│   ├── methods_performance_remax_mae.pdf # Method performance comparison
│   └── exhaustive_experiments/
│       └── mae_combined_heatmap.pdf     # Combined MAE heatmap
│
├── results/                      # Experiment results
│   ├── exhaustive_experiments/          # Exhaustive experiment data
│   └── method_comparison/               # Method comparison data
│
├── checkpoints/                  # Model checkpoints
├── configs/                      # Configuration files
├── data/                         # Dataset storage
└── tests/                        # Unit tests
```

## Key Files

### Figures for Paper
- **BNN Cycle Components**: `figures/bnn_cycle_components/*.pdf`
- **MM-Softmax Procedure**: `figures/mm_softmax_procedure_compact.pdf`
- **Variance-Delta Grid**: `figures/softmax_variance_delta_grid.pdf`
- **MAE Heatmap**: `figures/exhaustive_experiments/mae_combined_heatmap.pdf`
- **Method Performance**: `figures/methods_performance_remax_mae.pdf`

### Experiment Scripts
- **Multi-seed Training**: `experiments/train_all_methods_multiseed.py`
- **Exhaustive Experiments**: `experiments/run_exhaustive_experiments.py`
- **Plot Exhaustive Results**: `experiments/plot_exhaustive_results.py`

### Figure Generation Scripts
- **BNN Cycle Diagram**: `scripts/create_bnn_cycle_figure.py`
- **Individual Components**: `scripts/create_individual_plots.py`
- **Procedure Figure**: `scripts/create_mm_softmax_procedure_figure.py`
- **Variance-Delta Grid**: `scripts/plot_variance_delta_grid.py`
- **Method Comparison**: `scripts/plot_method_comparison.py`

## Reproducing Results

### 1. Run Exhaustive Experiments
```bash
python experiments/run_exhaustive_experiments.py
```

### 2. Generate MAE Heatmap
```bash
python experiments/plot_exhaustive_results.py
```

### 3. Run Multi-seed Training
```bash
python experiments/train_all_methods_multiseed.py --seeds 5
```

### 4. Generate Method Performance Figure
```bash
python scripts/plot_method_comparison.py
```

### 5. Generate BNN Cycle Figures
```bash
# Full cycle diagram
python scripts/create_bnn_cycle_figure.py

# Individual components
python scripts/create_individual_plots.py
```

### 6. Generate Procedure Figure
```bash
python scripts/create_mm_softmax_procedure_figure.py
```

### 7. Generate Variance-Delta Grid
```bash
python scripts/plot_variance_delta_grid.py --method mm_softmax
```

## Backup

All removed files have been backed up to the `backup_YYYYMMDD_HHMMSS/` directory.
