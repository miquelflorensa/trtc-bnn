# Changelog

## [Unreleased]

### Added
- **Sigma_v Scheduler** for observation noise annealing during training
  - Supports 4 schedule types: constant, linear, cosine, exponential
  - Decays from `sigma_v_max` (start) to `sigma_v_min` (end)
  - Command-line arguments: `--sigma-v-min`, `--sigma-v-max`, `--sigma-v-schedule`
  - Tracks sigma_v in training history
  - Prints current sigma_v value for each epoch
  
- **Documentation** for sigma_v scheduler (`docs/SIGMA_V_SCHEDULER.md`)
  - Complete usage guide with examples
  - Mathematical equations for each schedule type
  - Best practices and tips
  - Backward compatibility notes

- **Test script** for visualizing scheduler behavior (`scripts/test_scheduler.py`)
  - Generates plots comparing all 4 schedule types
  - Saves to `results/sigma_v_schedules.pdf` and `.png`

### Changed
- Updated `train_3cnn_cifar10.py` to use scheduler instead of fixed sigma_v
- Modified default sigma_v values: starts at 0.1, can decay to 0.001
- Updated training output to show current sigma_v each epoch

### Fixed
- Corrected scheduler direction: now properly decays from max to min (high to low noise)

## Behavior

**Before**: Fixed observation noise throughout training via `--sigma-v` argument

**After**: Dynamic observation noise with scheduling:
```bash
# Constant (backward compatible)
python scripts/train_3cnn_cifar10.py --sigma-v-max 0.001

# With annealing (recommended)
python scripts/train_3cnn_cifar10.py \
    --sigma-v-min 0.001 \
    --sigma-v-max 0.1 \
    --sigma-v-schedule cosine
```

The scheduler follows the principle:
- **High noise at start** (sigma_v_max) → Regularization, prevents overfitting
- **Low noise at end** (sigma_v_min) → Precision, better convergence
