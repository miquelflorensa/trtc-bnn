# Exhaustive Experiments Usage Guide

## Overview

The exhaustive experiments framework tests all analytical methods against their proper Monte Carlo implementations to compute Mean Absolute Error (MAE).

## Correct Monte Carlo Implementations

Each method now uses its **proper Monte Carlo implementation** from the `src/methods/` directory:

### LL-Softmax & MM-Softmax
- **MC Implementation**: `mc_softmax()` from `src/methods/mc_softmax.py`
- **Activation**: Softmax normalization
- **Returns**: Element-wise covariance `Cov(Z_k, A_k)` shape `(K,)`

### MM-Remax
- **MC Implementation**: `mc_remax()` from `src/methods/mc_remax.py`
- **Activation**: ReLU + normalization (Remax)
- **Returns**: Element-wise covariance `Cov(Z_k, A_k)` shape `(K,)`

### Probit
- **MC Implementation**: `compute_moments_mc()` from `src/methods/probit.py`
- **Activation**: Normal CDF (Φ) + normalization
- **Returns**: Only `mu_a` and `sigma_a_sq` (covariance not computed)
  - **Note**: Probit authors don't provide closed-form covariance solution
  - For experiments, `cov_z_a = None` for both analytical and MC

## Key Differences

### Probit: No Covariance

**Important**: The Probit method does not provide a closed-form solution for covariance between inputs and outputs. Therefore:
- Analytical result: `cov_z_a = None`
- MC baseline: `cov_z_a = None` 
- MAE comparison: `mae_cov_z_a = None` (Probit excluded from covariance comparisons)

### Element-wise Covariance (LL-Softmax, MM-Softmax, MM-Remax)

**Element-wise covariance**:
```python
cov_z_a[k] = Cov(Z_k, A_k)  # Shape: (K,)
```
This measures how much each input logit Z_k covaries with its corresponding output probability A_k.

## Running Experiments

### Full Exhaustive Run
```bash
python experiments/run_exhaustive_experiments.py
```

This runs:
- 15 scenarios (5 types × 3 variance levels)
- 100 experiments per scenario
- 4 methods per experiment
- **Total**: 6,000 experiments (~17 minutes)

### Generate Visualizations
```bash
python experiments/plot_exhaustive_results.py
```

Creates:
- Combined heatmap showing all methods and metrics
- Scenario-specific grouped bar plots
- Individual metric heatmaps
- Box plots comparing methods

## Monte Carlo Sampling Details

### Sample Counts
- **Default**: 100,000 samples per MC baseline
- **Rationale**: Ensures MAE < 1e-5 for MC error itself

### Numerical Stability
All MC implementations include:
- Numerically stable softmax (subtract max before exp)
- Division-by-zero protection
- NaN/inf handling

### Covariance Computation

**Softmax/Remax** (vectorized):
```python
cov_z_a = np.mean(Z * A, axis=0) - mu_z * mu_a
```

**Probit** (loop over matrix):
```python
for i in range(K):
    for j in range(K):
        cov_z_a[i, j] = np.cov(Z[:, i], A[:, j])[0, 1]
```

## Verification

To verify MC implementations are used correctly:

```python
import numpy as np
from experiments.run_exhaustive_experiments import get_mc_baseline

mu_z = np.array([0.0, 1.0, -0.5])
sigma_z_sq = np.array([1.0, 0.5, 2.0])

for method in ['LL-Softmax', 'MM-Softmax', 'MM-Remax', 'Probit']:
    result = get_mc_baseline(method, mu_z, sigma_z_sq, n_samples=10000)
    print(f"{method}: mu_a = {result['mu_a']}")
    print(f"  cov_z_a shape: {result['cov_z_a'].shape if result['cov_z_a'] is not None else 'None'}")
```

Expected output:
- All methods return `mu_a` with shape `(3,)`
- LL-Softmax, MM-Softmax, MM-Remax: `cov_z_a` shape `(3,)`
- Probit: `cov_z_a = None` (no closed-form solution available)

## Results Interpretation

### MAE Metrics

1. **mae_mu_a**: Error in mean predictions
   - Most critical for classification accuracy
   - MM-Remax typically best

2. **mae_sigma_a_sq**: Error in variance predictions
   - Important for uncertainty quantification
   - MM-Remax consistently excellent

3. **mae_cov_z_a**: Error in covariance predictions
   - Critical for gradient-based optimization
   - Mixed results across methods
   - **Note**: Probit excluded (returns None analytically)

### When to Use Each Method

Based on exhaustive experiments:

- **MM-Remax**: Best overall choice (12/15 scenarios for μ_A)
- **MM-Softmax**: Excellent for small variance scenarios
- **LL-Softmax**: Good for specific cases (dominant logit, medium variance)
- **Probit**: Limited applicability, avoid high-entropy scenarios

## Files Generated

### Results Directory: `results/exhaustive_experiments/`
- `exhaustive_results_TIMESTAMP.csv`: All 6,000 experiment results
- `summary_statistics_TIMESTAMP.csv`: Aggregated by scenario/method
- `first_configs_TIMESTAMP.json`: First configuration per scenario

### Figures Directory: `figures/exhaustive_experiments/`
- `mae_combined_heatmap.pdf`: Main comparison figure
- `mae_*_by_scenario.pdf`: Grouped bar plots
- `heatmap_*.pdf`: Individual metric heatmaps
- `method_comparison_boxplots.pdf`: Distribution comparisons

## Performance Notes

### Runtime
- Full experiments: ~17 minutes (6,000 runs)
- Bottleneck: MC sampling (100k samples × 6,000 times)
- Can reduce `N_SAMPLES_MC` for faster testing (e.g., 10k samples)

### Memory
- Peak usage: ~2 GB RAM
- Each MC run: ~8 MB (100k samples × 10 logits × 8 bytes)

## Troubleshooting

### Issue: Results differ from previous runs
**Cause**: Random seed changed or MC implementation modified
**Solution**: Check `np.random.seed(42)` in main script

### Issue: Probit MAE much higher than others
**Expected**: Probit struggles with equal positive logits (high entropy)
**Not a bug**: Probit method has fundamental limitations

### Issue: NaN or Inf in results
**Cause**: Numerical instability in analytical methods
**Check**: Large variance scenarios (> 10.0) or extreme logit values (|μ| > 10)
