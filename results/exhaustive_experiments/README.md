# Exhaustive Experiments Results

## Overview

This directory contains results from exhaustive experiments comparing all methods (LL-Softmax, MM-Softmax, MM-Remax, Probit) across different scenarios.

## Experiment Configuration

- **Number of logits (K)**: 10
- **Monte Carlo samples**: 100,000
- **Experiments per configuration**: 100
- **Total experiments**: 6,000 (15 scenarios × 100 repetitions × 4 methods)

## Scenarios Tested

Each scenario is tested with three variance levels: Small (0.001-0.1), Medium (0.5-3.0), Large (5.0-10.0)

1. **Random/Initialization**: Logits sampled from N(0, 0.5)
2. **One Dominant Logit**: One positive logit (2-4), others negative (-4 to -1)
3. **Multiple Competing Logits**: 2-3 positive logits, rest negative
4. **All Negative Logits**: All logits negative (-4 to -1)
5. **Equal Positive Logits**: All logits approximately equal (2-4)

## Metrics Computed

For each method, we compute the Mean Absolute Error (MAE) between analytical and Monte Carlo results for:

1. **μ_A**: Mean of output probabilities
2. **σ²_A**: Variance of output probabilities
3. **Cov(Z,A)**: Covariance between logits and outputs
   - **Note**: Not available for Probit (authors don't provide closed-form solution)
   - Probit is excluded from covariance comparisons

## Key Findings

### Overall Best Method by Metric

#### MAE μ_A (Mean Predictions)
- **MM-Remax wins**: 12/15 scenarios
- **MM-Softmax wins**: 2/15 scenarios (All Negative - Small Var, Random - Small Var)
- **LL-Softmax wins**: 1/15 scenarios (All Negative - Medium Var)

#### MAE σ²_A (Variance Predictions)
- **MM-Remax wins**: 13/15 scenarios
- **MM-Softmax wins**: 2/15 scenarios (All Negative - Small Var, Random - Small Var)
- **Probit wins**: 0/15 scenarios (but only 1/15 if compared: All Negative - Medium Var)

#### MAE Cov(Z,A) (Covariance)
- **MM-Remax wins**: 8/12 scenarios (Probit excluded)
- **MM-Softmax wins**: 3/12 scenarios
- **LL-Softmax wins**: 1/12 scenarios

### Method Performance by Variance Level

#### Small Variance (0.001-0.1)
- **MM-Softmax** and **MM-Remax** both perform excellently
- MAE typically < 1e-4 for μ_A
- **MM-Remax** often achieves near-perfect accuracy (MAE < 1e-7)

#### Medium Variance (0.5-3.0)
- **MM-Remax** consistently outperforms others
- **LL-Softmax** competitive in some scenarios
- MAE typically 1e-3 to 1e-2

#### Large Variance (5.0-10.0)
- **MM-Remax** maintains best performance
- **LL-Softmax** shows significant degradation
- **MM-Softmax** struggles with variance predictions (MAE > 1e2)
- MAE can reach 1e-2 to 1e-1

### Scenario-Specific Insights

1. **Dominant Logit**: MM-Remax achieves near-perfect accuracy across all variance levels
2. **Equal Positive**: Most challenging for all methods, especially Probit (MAE > 0.7)
3. **All Negative**: Probit performs poorly; MM-Softmax best for small variance
4. **Competing Logits**: MM-Remax consistently best
5. **Random**: Mixed performance; MM-Remax generally best

## Probit Method Notes

- **Does not compute covariance** analytically (authors don't provide closed-form solution)
  - Both analytical and MC return `cov_z_a = None`
  - Excluded from all covariance comparisons and plots
- **Worst performance** in Equal Positive scenarios (MAE > 0.8)
- **Best suited** for: scenarios with negative logits and small-to-medium variance
- **Not recommended** for high-entropy or equal probability scenarios

## Recommendations

### For Production Use:
1. **MM-Remax**: Best all-around choice
   - Most accurate across scenarios
   - Handles large variance well
   - Computes all three metrics

2. **MM-Softmax**: Good alternative when:
   - Variance is small
   - Computational efficiency is critical
   - Softmax normalization is required

3. **LL-Softmax**: Use when:
   - Medium variance with dominant logit
   - Gradient computations are needed
   - Covariance accuracy is critical

4. **Probit**: Limited use cases
   - Only when covariance is not needed
   - Small-to-medium variance scenarios
   - Avoid for equal probability scenarios

## Files in This Directory

- `exhaustive_results_YYYYMMDD_HHMMSS.csv`: Raw results for all experiments
- `summary_statistics_YYYYMMDD_HHMMSS.csv`: Aggregated statistics by scenario and method
- `first_configs_YYYYMMDD_HHMMSS.json`: First configuration of each scenario for reproducibility

## Visualizations

See `figures/exhaustive_experiments/` for:
- `mae_combined_heatmap.pdf`: Single heatmap showing all methods and metrics
- `mae_*_by_scenario.pdf`: Grouped bar plots by scenario type and variance
- `heatmap_*.pdf`: Individual metric heatmaps
- `method_comparison_boxplots.pdf`: Box plots comparing methods across all scenarios
