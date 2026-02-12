# Repository Cleanup Summary

## Date: February 12, 2026

## What was cleaned:

### 1. Root Directory
- ✅ Removed temporary output files: `ll-softmax.out`, `mm-softmax.out`, `nohup.out`
- ✅ Removed test file: `test_normalized_softmax.py`
- ✅ Archived old documentation: `CUTAGI_MODIFICATIONS_SUMMARY.md`, `cuTAGI_modifications.md`

### 2. Figures Directory
Kept only **5 essential figures** for the paper:

#### ✅ Kept:
1. **BNN Cycle Components** (`figures/bnn_cycle_components/`)
   - 1_prior_logits.pdf/.png
   - 2_prediction.pdf/.png
   - 3_ground_truth.pdf/.png
   - 4_posterior_logits.pdf/.png

2. **MM-Softmax Procedure** (`figures/mm_softmax_procedure_compact.pdf`)

3. **Variance-Delta Grid** (`figures/softmax_variance_delta_grid.pdf`)
   - Renamed from `mm_softmax_variance_delta_grid.pdf`

4. **MAE Combined Heatmap** (`figures/exhaustive_experiments/mae_combined_heatmap.pdf`)

5. **Methods Performance** (`figures/methods_performance_remax_mae.pdf`)
   - Copied from `results/method_comparison/figures/`

#### ❌ Removed (moved to backup):
- all_methods_comparison.pdf/.png
- bnn_cycle_diagram.pdf/.png
- ece_plot.png
- example_*.png (12 files)
- ll_softmax_variance_delta_grid.pdf
- mm_remax_*.pdf/.png (6 files)
- mm_softmax_procedure.pdf/.png
- mm_softmax_procedure_2col.pdf
- ood_entropy_histogram.png
- rejection_curves.png
- test_*.png (4 files)
- 1074_cat.png
- And other figures from exhaustive_experiments/

### 3. Scripts Directory
Kept only **7 essential scripts** needed for experiments and figure generation:

#### ✅ Kept:
1. `train.py` - Training script
2. `evaluate.py` - Evaluation script
3. `create_bnn_cycle_figure.py` - Generate BNN cycle diagram
4. `create_individual_plots.py` - Generate individual cycle components
5. `create_mm_softmax_procedure_figure.py` - Generate procedure figure
6. `plot_variance_delta_grid.py` - Generate variance-delta grids
7. `plot_method_comparison.py` - Generate method comparison plots

#### ❌ Removed (moved to backup):
- compare_all_methods.py
- compare_methods.py
- compare_mm_remax_vs_mc.py
- compare_mm_softmax_vs_mc.py
- create_framework_figure.py
- create_framework_figure_v2.py
- generate_figures.py
- generate_summary_table.py
- plot_all_methods.py
- plot_combined_mae_and_accuracy.py
- plot_crash_timeline.py
- plot_final_comparison_with_crashes.py
- plot_mae_vs_stability.py
- plot_softmax_comparison.py
- plot_stability_analysis.py
- run_all_experiments.py

### 4. Experiments Directory
Kept only **5 essential files** needed for reproducing results:

#### ✅ Kept:
1. `train_all_methods_multiseed.py` - Multi-seed training experiments
2. `run_exhaustive_experiments.py` - Exhaustive scenario testing
3. `plot_exhaustive_results.py` - Generate heatmap figures
4. `EXPERIMENTS_USAGE.md` - Documentation
5. `RESULTS_SUMMARY.md` - Results documentation

#### ❌ Removed (moved to backup):
- BUGFIX_JAX_REMAX.md
- JAX_ACCELERATION.md
- JAX_IMPLEMENTATION_SUMMARY.md
- MAE_COMPARISON_PLOT_GUIDE.md
- MC_OPTIMIZATION.md
- PLOTTING_GUIDE.md
- QUICK_REFERENCE.md
- README_MULTISEED.md
- test_jax_mc.py

### 5. Results Directory
Reorganized to keep only essential experiment data:

#### ✅ Kept:
- `results/exhaustive_experiments/` - All exhaustive experiment results
- `results/method_comparison/all_seeds_summary.npz` - Summary data

#### ❌ Removed (moved to backup):
- Various intermediate files and old results
- `3cnn_remax_history.npz`
- `3cnn_remax_results.txt`
- `sigma_v_schedules.pdf/.png`

### 6. Checkpoints Directory
Kept only **1 best checkpoint**:

#### ✅ Kept:
- `3cnn_remax_best_1_1_01_005.bin`

#### ❌ Removed (moved to backup):
- `3cnn_remax_best_1_1_01.bin`
- `3cnn_remax_best_1_1_036.bin`
- `3cnn_remax_comparison.bin`
- `3cnn_softmax_comparison.bin`

## Summary Statistics:

- **Figures**: 5 kept, ~40 removed
- **Scripts**: 7 kept, ~16 removed
- **Experiments**: 5 kept, ~9 removed
- **Checkpoints**: 1 kept, 4 removed
- **Root files**: 4 removed

## Backup Location:

All removed files are backed up in: `backup_20260212_210647/`

### Backup Structure:
```
backup_20260212_210647/
├── old_docs/
│   ├── CUTAGI_MODIFICATIONS_SUMMARY.md
│   └── cuTAGI_modifications.md
├── figures_original/      # All original figures
├── scripts_removed/       # Removed scripts
├── experiments_removed/   # Removed experiment files
├── results_original/      # Original results directory
└── checkpoints_removed/   # Removed checkpoints
```

## Note on Missing Figure:

The requested figure `combined_error_charts.pdf` was not found in the repository. It may need to be generated from existing scripts or might be a different name for one of the existing figures.

## How to Restore:

If you need any removed file, you can copy it back from the backup directory:

```bash
# Example: Restore a removed script
cp backup_20260212_210647/scripts_removed/compare_methods.py scripts/

# Example: Restore a removed figure
cp backup_20260212_210647/figures_original/bnn_cycle_diagram.pdf figures/
```

## Next Steps:

1. Test that all essential scripts still work correctly
2. Verify that all paper figures can be regenerated
3. Confirm that experiment results can be reproduced
4. Consider adding the backup directory to `.gitignore` if not committing it
5. Update any documentation that references removed files
