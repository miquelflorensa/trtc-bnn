# Sigma_v Scheduler for Training

## Overview

The training script now includes a flexible scheduler for the observation noise parameter `sigma_v` (σ_v). This allows you to dynamically adjust the observation noise throughout training, which can help with:

- **Regularization**: Start with higher noise to prevent overfitting early in training
- **Fine-tuning**: Reduce noise later in training for better convergence
- **Noise annealing**: Gradually decrease noise as the model becomes more confident

**Important**: The scheduler **starts at `sigma_v_max` and decays to `sigma_v_min`**. This follows the common practice of beginning training with high noise (regularization) and reducing it over time (precision).

## Usage

The scheduler is controlled by three command-line arguments:

```bash
python scripts/train_3cnn_cifar10.py \
    --sigma-v-min 0.001 \
    --sigma-v-max 0.1 \
    --sigma-v-schedule cosine
```

### Arguments

- `--sigma-v-min`: Minimum (ending) σ_v value (default: 0.001)
- `--sigma-v-max`: Maximum (starting) σ_v value (default: 0.001)
- `--sigma-v-schedule`: Schedule type (default: 'constant')

### Schedule Types

#### 1. Constant (default)
```bash
--sigma-v-schedule constant
```
- σ_v stays at `sigma_v_max` throughout training
- Use when you know the optimal noise level
- For backward compatibility, set both min and max to the same value

**Equation**: σ_v(t) = σ_v_max

#### 2. Linear
```bash
--sigma-v-schedule linear
```
- Linear decay from `sigma_v_max` to `sigma_v_min`
- Smooth, predictable decrease
- Good for gradual noise reduction

**Equation**: σ_v(t) = σ_v_max - (σ_v_max - σ_v_min) × (t / T)

#### 3. Cosine
```bash
--sigma-v-schedule cosine
```
- Cosine annealing from `sigma_v_max` to `sigma_v_min`
- Slow decay at start, faster in middle, slow at end
- Popular in learning rate scheduling, works well for noise too

**Equation**: σ_v(t) = σ_v_min + (σ_v_max - σ_v_min) × [1 + cos(π × t/T)] / 2

#### 4. Exponential
```bash
--sigma-v-schedule exponential
```
- Exponential decay from `sigma_v_max` to `sigma_v_min`
- Fast decay initially, slower later
- Good for exploring log-scale noise ranges

**Equation**: σ_v(t) = σ_v_max × (σ_v_min / σ_v_max)^(t/T)

## Examples

### Example 1: Constant Noise (No Scheduling)
```bash
python scripts/train_3cnn_cifar10.py \
    --epochs 50 \
    --batch-size 128 \
    --sigma-v-max 0.001 \
    --sigma-v-min 0.001 \
    --sigma-v-schedule constant
```
Or simply:
```bash
python scripts/train_3cnn_cifar10.py --sigma-v-max 0.001
```

### Example 2: Linear Noise Annealing (Recommended)
Start with high noise (0.1) for regularization, linearly decay to low noise (0.001):
```bash
python scripts/train_3cnn_cifar10.py \
    --epochs 50 \
    --batch-size 128 \
    --sigma-v-min 0.001 \
    --sigma-v-max 0.1 \
    --sigma-v-schedule linear
```

### Example 3: Cosine Annealing (Best Practice)
Start with high noise, smoothly decay with cosine schedule:
```bash
python scripts/train_3cnn_cifar10.py \
    --epochs 50 \
    --batch-size 128 \
    --sigma-v-min 0.001 \
    --sigma-v-max 0.1 \
    --sigma-v-schedule cosine
```

### Example 4: Exponential Decay
Fast decay initially, slower later (log-scale from 10^-1 to 10^-3):
```bash
python scripts/train_3cnn_cifar10.py \
    --epochs 50 \
    --batch-size 128 \
    --sigma-v-min 0.001 \
    --sigma-v-max 0.1 \
    --sigma-v-schedule exponential
```

## Output

The training script now:
1. **Prints** the σ_v schedule configuration at the start
2. **Tracks** σ_v in the training history (`history['sigma_v']`)
3. **Displays** the current σ_v value for each epoch in the progress output
4. **Saves** σ_v values in the `.npz` history file

Example output:
```
Observation noise (sigma_v) schedule:
  Min: 0.0010, Max: 0.1000
  Schedule: cosine
  Initial sigma_v: 0.1000

Starting training for 50 epochs...
----------------------------------------------------------------------
Epoch   1/50 (12.3s) | σ_v: 0.1000 | Train NLL: 2.3025, Err: 90.00% | ...
Epoch   2/50 (12.1s) | σ_v: 0.0985 | Train NLL: 2.1234, Err: 85.23% | ...
Epoch  25/50 (12.0s) | σ_v: 0.0505 | Train NLL: 1.5234, Err: 45.12% | ...
Epoch  50/50 (11.9s) | σ_v: 0.0010 | Train NLL: 0.8234, Err: 18.45% | ...
```

## Implementation Details

The scheduler is implemented as a `SigmaVScheduler` class in `scripts/train_3cnn_cifar10.py`:

```python
class SigmaVScheduler:
    def __init__(self, sigma_v_min, sigma_v_max, total_epochs, schedule_type):
        ...
    
    def get_sigma_v(self, epoch: int) -> float:
        """Get sigma_v value for the current epoch (0-indexed)."""
        ...
```

The variance for the output layer is updated each epoch:
```python
for epoch in range(epochs):
    # Update sigma_v for this epoch
    sigma_v = scheduler.get_sigma_v(epoch)
    var_y = np.full((batch_size * 10,), sigma_v**2, dtype=np.float32)
    
    # Training with updated variance
    train_metrics = train_epoch(model, out_updater, train_loader, var_y, epoch)
```

## Visualization

Run the test script to visualize all schedule types:
```bash
python scripts/test_scheduler.py
```

This generates plots showing how σ_v evolves over 50 epochs for each schedule type, saved to:
- `results/sigma_v_schedules.pdf`
- `results/sigma_v_schedules.png`

## Backward Compatibility

To maintain backward compatibility with the old `--sigma-v` argument behavior:
- Use `--sigma-v-max` with your desired constant value
- The default schedule is 'constant', so it will stay at the max value

Example (equivalent to old `--sigma-v 0.001`):
```bash
python scripts/train_3cnn_cifar10.py --sigma-v-max 0.001
```

## Tips

1. **Start with Annealing**: Use `cosine` or `linear` schedule starting from 0.1 down to 0.001
2. **Monitor Validation**: Watch the effect on validation NLL - excessive noise hurts performance
3. **Log Scale for Large Ranges**: For ranges like 0.001 to 1.0, use `exponential` schedule
4. **Typical Range**: Most experiments work well with σ_v from 0.1 (start) to 0.001 (end)
5. **Don't Over-regularize**: If validation performance plateaus early, reduce sigma_v_max
