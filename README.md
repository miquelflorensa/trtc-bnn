# TRTC-BNN: Analytically Tractable Real-to-Categorical Transformations for Bayesian Neural Networks

This repository contains the code for reproducing the experiments and results from the paper:

**"Analytically Tractable Real-to-Categorical Transformations for Bayesian Neural Networks"**

## Overview

We propose analytically tractable methods for real-to-categorical transformations in Bayesian Neural Networks (BNNs), enabling efficient uncertainty quantification in classification tasks without requiring Monte Carlo sampling.

## Repository Structure

```
trtc-bnn/
├── src/                          # Source code
│   ├── models/                   # BNN model architectures
│   │   ├── bnn_mnist.py          # BNN for MNIST
│   │   ├── bnn_cifar.py          # BNN for CIFAR
│   │   └── remax_head.py         # Remax classification head
│   ├── methods/                  # Proposed methods
│   │   ├── method_1.py           # Method 1 implementation
│   │   ├── method_2.py           # Method 2 implementation
│   │   ├── method_3.py           # Method 3 implementation
│   │   ├── mc_samples.py         # MC sampling baseline
│   │   └── comparison.py         # Method comparison utilities
│   ├── metrics/                  # Evaluation metrics
│   │   ├── nll.py                # Negative Log-Likelihood
│   │   ├── ece.py                # Expected Calibration Error
│   │   └── accuracy.py           # Accuracy metrics
│   ├── data/                     # Data loading utilities
│   │   ├── mnist_loader.py       # MNIST data loader
│   │   └── cifar_loader.py       # CIFAR data loader
│   ├── training/                 # Training utilities
│   │   └── trainer.py            # BNN trainer using cuTAGI
│   └── utils/                    # Utility functions
│       ├── config.py             # Configuration utilities
│       └── visualization.py      # Plotting utilities
├── scripts/                      # Runnable scripts
│   ├── train_mnist.py            # Train BNN on MNIST
│   ├── train_cifar.py            # Train BNN on CIFAR
│   ├── evaluate.py               # Evaluate trained models
│   ├── compare_methods.py        # Compare methods with MC sampling
│   └── generate_plots.py         # Generate paper figures
├── configs/                      # Configuration files
│   ├── mnist_default.yaml        # MNIST experiment config
│   ├── cifar10_default.yaml      # CIFAR-10 experiment config
│   └── cifar100_default.yaml     # CIFAR-100 experiment config
├── experiments/                  # Experiment runners
│   └── run_all_experiments.py    # Run all paper experiments
├── tests/                        # Unit tests
│   ├── test_metrics.py           # Tests for metrics
│   └── test_methods.py           # Tests for methods
├── data/                         # Dataset storage (gitignored)
├── results/                      # Experiment results (gitignored)
├── figures/                      # Generated figures (gitignored)
├── checkpoints/                  # Model checkpoints (gitignored)
├── requirements.txt              # Python dependencies
├── setup.py                      # Package setup
└── README.md                     # This file
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

### Training BNN on MNIST

```bash
python scripts/train_mnist.py --config configs/mnist_default.yaml
```

### Training BNN on CIFAR-10

```bash
python scripts/train_cifar.py --config configs/cifar10_default.yaml --cifar-version 10
```

### Training BNN on CIFAR-100

```bash
python scripts/train_cifar.py --config configs/cifar100_default.yaml --cifar-version 100
```

### Evaluating a Trained Model

```bash
python scripts/evaluate.py --model-path checkpoints/model.pt --dataset mnist
```

### Comparing Methods with MC Sampling

```bash
python scripts/compare_methods.py --mc-samples 10000
```

### Running All Experiments

```bash
python experiments/run_all_experiments.py
```

### Generating Paper Figures

```bash
python scripts/generate_plots.py --results-dir results --output-dir figures
```

## Proposed Methods

We propose three analytically tractable methods for real-to-categorical transformations:

1. **Method 1**: [Brief description]
2. **Method 2**: [Brief description]
3. **Method 3**: [Brief description]

All methods are compared against Monte Carlo sampling baselines.

## Remax Classification Head

The Remax classification head provides an analytically tractable alternative to softmax for BNNs, enabling closed-form computation of predictive distributions.

## Metrics

We evaluate our methods using:
- **Accuracy**: Classification accuracy
- **NLL**: Negative Log-Likelihood
- **ECE**: Expected Calibration Error

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
