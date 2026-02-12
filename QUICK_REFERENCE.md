# Quick Reference - trtc-bnn Repository

## 📚 Documentation Files

- **README.md** - Main repository documentation
- **REPOSITORY_STRUCTURE.md** - Complete structure and how to reproduce results
- **CLEANUP_SUMMARY.md** - Detailed cleanup report (what was removed/kept)
- **FIGURES_STATUS.md** - Status of all paper figures and generation commands
- **CHANGELOG.md** - Version history

## 🎨 Paper Figures (5 essential figures kept)

1. **BNN Cycle Components** (8 files)
   - Location: `figures/bnn_cycle_components/`
   - Generate: `python scripts/create_individual_plots.py`

2. **MM-Softmax Procedure Compact**
   - Location: `figures/mm_softmax_procedure_compact.pdf`
   - Generate: `python scripts/create_mm_softmax_procedure_figure.py`

3. **Softmax Variance-Delta Grid**
   - Location: `figures/softmax_variance_delta_grid.pdf`
   - Generate: `python scripts/plot_variance_delta_grid.py --method mm_softmax`

4. **MAE Combined Heatmap**
   - Location: `figures/exhaustive_experiments/mae_combined_heatmap.pdf`
   - Generate: 
     ```bash
     python experiments/run_exhaustive_experiments.py
     python experiments/plot_exhaustive_results.py
     ```

5. **Methods Performance (Remax MAE)**
   - Location: `figures/methods_performance_remax_mae.pdf`
   - Generate:
     ```bash
     python experiments/train_all_methods_multiseed.py --seeds 5
     python scripts/plot_method_comparison.py
     ```

## 🔬 Running Experiments

### Exhaustive Experiments
```bash
python experiments/run_exhaustive_experiments.py
python experiments/plot_exhaustive_results.py
```

### Multi-seed Training
```bash
python experiments/train_all_methods_multiseed.py --seeds 5
python scripts/plot_method_comparison.py
```

## 📜 Essential Scripts (7 kept)

| Script | Purpose |
|--------|---------|
| `train.py` | Training script |
| `evaluate.py` | Evaluation script |
| `create_bnn_cycle_figure.py` | Generate BNN cycle diagram |
| `create_individual_plots.py` | Generate individual cycle components |
| `create_mm_softmax_procedure_figure.py` | Generate procedure figure |
| `plot_variance_delta_grid.py` | Generate variance-delta grids |
| `plot_method_comparison.py` | Generate method comparison plots |

## 💾 Backup

All removed files are in: `backup_20260212_210647/`

To restore a file:
```bash
# Example: Restore a removed script
cp backup_20260212_210647/scripts_removed/SCRIPT_NAME.py scripts/

# Example: Restore a removed figure
cp backup_20260212_210647/figures_original/FIGURE_NAME.pdf figures/
```

## ⚠️ Missing Figure

The requested `combined_error_charts.pdf` was not found in the repository. Please check `FIGURES_STATUS.md` for details.

## 📊 Cleanup Statistics

- **Figures**: 5 kept, ~40 removed
- **Scripts**: 7 kept, ~16 removed
- **Experiments**: 5 kept, ~9 removed
- **Checkpoints**: 1 kept, 4 removed
- **Root files**: 4 removed

## 🚀 Quick Commands

### Regenerate all paper figures
```bash
# Components
python scripts/create_individual_plots.py

# Procedure
python scripts/create_mm_softmax_procedure_figure.py

# Variance-Delta Grid
python scripts/plot_variance_delta_grid.py --method mm_softmax

# Heatmap (requires experiment results)
python experiments/plot_exhaustive_results.py

# Performance (requires training results)
python scripts/plot_method_comparison.py
```

### View current structure
```bash
# Figures
find figures -type f -name "*.pdf" | sort

# Scripts
ls scripts/*.py

# Experiments
ls experiments/*.py experiments/*.md
```

---

**Last Updated**: February 12, 2026  
**Cleanup Performed**: See `CLEANUP_SUMMARY.md`  
**Backup Location**: `backup_20260212_210647/`
