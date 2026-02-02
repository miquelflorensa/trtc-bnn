# TRTC-BNN: Analytically Tractable Real-to-Categorical Transformations for Bayesian Neural Networks

This repository contains the code for reproducing the experiments and results from the paper:

**"Analytically Tractable Real-to-Categorical Transformations for Bayesian Neural Networks"**

## Overview

We propose analytically tractable methods for real-to-categorical transformations in Bayesian Neural Networks (BNNs), enabling efficient uncertainty quantification in classification tasks without requiring Monte Carlo sampling.

## Repository Structure

```
trtc-bnn/
├── src/                              # Source code (library)
│   ├── models/                       # BNN model architectures
│   │   ├── bnn_mnist.py              # BNN for MNIST
│   │   ├── bnn_cifar.py              # BNN for CIFAR
│   │   └── remax_head.py             # Remax classification head
│   ├── methods/                      # Proposed analytical methods
│   │   ├── mm_remax.py               # Moment-matching for Remax
│   │   ├── mm_softmax.py             # Moment-matching for Softmax
│   │   ├── ll_softmax.py             # Log-linear Softmax approximation
│   │   ├── mc_remax.py               # Monte Carlo Remax (baseline)
│   │   └── mc_softmax.py             # Monte Carlo Softmax (baseline)
│   ├── metrics/                      # Evaluation metrics
│   │   ├── nll.py                    # Negative Log-Likelihood
│   │   ├── ece.py                    # Expected Calibration Error
│   │   └── accuracy.py               # Accuracy metrics
│   ├── data/                         # Data loading utilities
│   │   ├── mnist_loader.py           # MNIST data loader
│   │   └── cifar_loader.py           # CIFAR data loader
│   ├── training/                     # Training utilities
│   │   └── trainer.py                # BNN trainer using cuTAGI
│   └── utils/                        # Utility functions
│       ├── config.py                 # Configuration utilities
│       └── visualization.py          # Plotting utilities
├── scripts/                          # Runnable scripts
│   ├── train.py                      # Unified training script
│   ├── evaluate.py                   # Evaluate trained models
│   ├── compare_methods.py            # Compare methods with MC sampling
│   ├── compare_mm_remax_vs_mc.py     # Detailed Remax comparison
│   ├── compare_mm_softmax_vs_mc.py   # Detailed Softmax comparison
│   ├── generate_figures.py           # Generate paper figures
│   ├── plot_method_comparison.py     # Plot comparison results
│   ├── plot_softmax_comparison.py    # Plot softmax comparison
│   ├── plot_variance_delta_grid.py   # Plot variance-delta grids
│   └── run_all_experiments.py        # Run all paper experiments
├── tests/                            # Unit tests and development scripts
│   ├── test_methods.py               # Tests for analytical methods
│   ├── test_metrics.py               # Tests for metrics
│   ├── test_scheduler.py             # Tests for sigma_v scheduler
│   └── test_qualitative_plots.py     # Tests for plotting functions
├── configs/                          # Configuration files
│   ├── mnist_default.yaml            # MNIST experiment config
│   ├── cifar10_default.yaml          # CIFAR-10 experiment config
│   └── cifar100_default.yaml         # CIFAR-100 experiment config
├── docs/                             # Documentation
│   └── SIGMA_V_SCHEDULER.md          # Observation noise scheduler docs
├── checkpoints/                      # Model checkpoints
├── data/                             # Dataset storage (gitignored)
├── results/                          # Experiment results
├── figures/                          # Generated figures
├── requirements.txt                  # Python dependencies
├── setup.py                          # Package setup
├── CHANGELOG.md                      # Version history
└── README.md                         # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/miquelflorensa/trtc-bnn.git
cd trtc-bnn
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install cuTAGI (follow instructions at https://github.com/lhnguyen102/cuTAGI):
```bash
pip install pytagi
```

## Usage

### Training BNN

The unified training script supports multiple datasets and classification heads:

```bash
# Train on CIFAR-10 with Remax head (default)
python scripts/train.py --dataset cifar10 --head remax --epochs 50

# Train on MNIST with Softmax head
python scripts/train.py --dataset mnist --head softmax --epochs 50

# Train with custom hyperparameters
python scripts/train.py --dataset cifar10 --head remax \
    --epochs 100 --batch-size 64 --sigma-v-schedule cosine
```

#### Observation Noise Scheduling

The training script supports dynamic scheduling of the observation noise parameter (σ_v):

```bash
# Constant noise (default)
python scripts/train.py --sigma-v-max 0.1 --sigma-v-schedule constant

# Linear decay from max to min
python scripts/train.py --sigma-v-min 0.001 --sigma-v-max 0.1 --sigma-v-schedule linear

# Cosine annealing
python scripts/train.py --sigma-v-min 0.001 --sigma-v-max 0.1 --sigma-v-schedule cosine

# Exponential decay
python scripts/train.py --sigma-v-min 0.001 --sigma-v-max 0.1 --sigma-v-schedule exponential
```

See [docs/SIGMA_V_SCHEDULER.md](docs/SIGMA_V_SCHEDULER.md) for detailed documentation.

### Evaluating a Trained Model

```bash
python scripts/evaluate.py --checkpoint checkpoints/cifar10_remax_best.bin --dataset cifar10
```

### Comparing Analytical Methods with MC Sampling

```bash
# Compare MM-Remax vs MC-Remax
python scripts/compare_mm_remax_vs_mc.py --epochs 50 --mc-samples 10000

# Compare MM-Softmax vs MC-Softmax
python scripts/compare_mm_softmax_vs_mc.py --epochs 50 --mc-samples 10000
```

### Running All Experiments

```bash
# Run all experiments
python scripts/run_all_experiments.py

# Dry run (print commands without executing)
python scripts/run_all_experiments.py --dry-run

# Run specific experiments
python scripts/run_all_experiments.py --experiments cifar10_remax comparison_remax
```

### Generating Paper Figures

```bash
python scripts/generate_figures.py --results-dir results --output-dir figures
python scripts/plot_method_comparison.py --results results/method_comparison/mm_remax_comparison.npz
python scripts/plot_variance_delta_grid.py --method mm_remax --output-dir figures
```

## Proposed Methods

We propose analytically tractable methods for real-to-categorical transformations:

| Method | Description |
|--------|-------------|
| **MM-Remax** | Moment-matching approximation for the Remax transformation |
| **MM-Softmax** | Moment-matching approximation for the Softmax transformation |
| **LL-Softmax** | Log-linear approximation for Softmax |

All analytical methods are compared against Monte Carlo sampling baselines (MC-Remax, MC-Softmax).

## Remax Classification Head

The Remax classification head provides an analytically tractable alternative to softmax for BNNs, enabling closed-form computation of predictive distributions.

## Metrics

We evaluate our methods using:
- **Accuracy**: Classification accuracy
- **NLL**: Negative Log-Likelihood  
- **ECE**: Expected Calibration Error
- **MAE**: Mean Absolute Error (for method comparison)

## Citation

If you use this code in your research, please cite:

```bibtex
@article{author2024trtc,
  title={Analytically Tractable Real-to-Categorical Transformations for Bayesian Neural Networks},
  author={Author, Name},
  journal={Journal/Conference},
  year={2024}
}
```

## License

MIT License

## Acknowledgments

This work uses [cuTAGI](https://github.com/lhnguyen102/cuTAGI) for Bayesian Neural Network training.
