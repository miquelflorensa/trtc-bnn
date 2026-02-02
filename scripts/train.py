"""
Unified training script for BNN models with various classification heads.

Supports:
- MNIST and CIFAR-10/100 datasets
- Remax and Softmax classification heads
- Multiple architectures (3-block CNN, etc.)
- Sigma_v scheduling strategies

Usage:
    python scripts/train.py --dataset cifar10 --head remax --epochs 50
    python scripts/train.py --dataset mnist --head softmax --epochs 100
"""

import os
import sys
import time
import argparse
import numpy as np
from pathlib import Path

# pytagi imports
import pytagi
from pytagi.nn import (
    Sequential,
    Conv2d,
    Linear,
    MixtureReLU,
    Remax,
    ClosedFormSoftmax,
    OutputUpdater,
)

import torch
import torchvision
import torchvision.transforms as transforms


# ============================================================================
# Configuration
# ============================================================================

# Default hyperparameters
DEFAULT_CONFIG = {
    "epochs": 50,
    "batch_size": 128,
    "validation_split": 0.1,
    "gain_w": 1.0,
    "gain_b": 1.0,
    "sigma_v_min": 0.001,
    "sigma_v_max": 0.1,
    "sigma_v_schedule": "constant",
    "device": "cuda",
}

# Paths
DATA_DIR = "./data"
SAVE_DIR = "./checkpoints"
RESULTS_DIR = "./results"


# ============================================================================
# Sigma_v Scheduler
# ============================================================================

class SigmaVScheduler:
    """
    Scheduler for observation noise (sigma_v) during training.
    
    Supports different scheduling strategies:
    - 'constant': No change (sigma_v stays at sigma_v_max)
    - 'linear': Linear interpolation from sigma_v_max to sigma_v_min
    - 'cosine': Cosine annealing from sigma_v_max to sigma_v_min
    - 'exponential': Exponential decay from sigma_v_max to sigma_v_min
    """
    
    def __init__(self, sigma_v_min: float, sigma_v_max: float, 
                 total_epochs: int, schedule_type: str = 'constant'):
        self.sigma_v_min = sigma_v_min
        self.sigma_v_max = sigma_v_max
        self.total_epochs = total_epochs
        self.schedule_type = schedule_type
        
    def get_sigma_v(self, epoch: int) -> float:
        """Get sigma_v value for the current epoch."""
        if self.schedule_type == 'constant':
            return self.sigma_v_max
            
        progress = epoch / max(self.total_epochs - 1, 1)
        
        if self.schedule_type == 'linear':
            return self.sigma_v_max - (self.sigma_v_max - self.sigma_v_min) * progress
        elif self.schedule_type == 'cosine':
            return self.sigma_v_min + (self.sigma_v_max - self.sigma_v_min) * \
                   (1 + np.cos(progress * np.pi)) / 2
        elif self.schedule_type == 'exponential':
            if self.sigma_v_min > 0 and self.sigma_v_max > 0:
                ratio = self.sigma_v_min / self.sigma_v_max
                return self.sigma_v_max * (ratio ** progress)
            else:
                return self.sigma_v_max - (self.sigma_v_max - self.sigma_v_min) * progress
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")


# ============================================================================
# Model Definitions
# ============================================================================

def build_3cnn_model(num_classes: int = 10, head: str = "remax",
                     gain_w: float = 1.0, gain_b: float = 1.0,
                     input_channels: int = 3, input_size: int = 32):
    """
    Build the 3-block CNN with specified classification head.
    
    Args:
        num_classes: Number of output classes
        head: Classification head type ('remax' or 'softmax')
        gain_w: Weight initialization gain
        gain_b: Bias initialization gain
        input_channels: Number of input channels (3 for CIFAR, 1 for MNIST)
        input_size: Input image size (32 for CIFAR, 28 for MNIST)
    """
    # Determine output size after conv blocks
    if input_size == 32:
        final_size = 4  # 32 -> 16 -> 8 -> 4
    elif input_size == 28:
        final_size = 3  # 28 -> 14 -> 7 -> 3
    else:
        raise ValueError(f"Unsupported input size: {input_size}")
    
    layers = [
        # Block 1
        Conv2d(input_channels, 64, 3, bias=True, padding=1, 
               in_width=input_size, in_height=input_size, 
               gain_weight=gain_w, gain_bias=gain_b),
        MixtureReLU(),
        Conv2d(64, 64, 4, bias=True, padding=1, stride=2, 
               gain_weight=gain_w, gain_bias=gain_b),
        MixtureReLU(),
        
        # Block 2
        Conv2d(64, 128, 3, bias=True, padding=1, 
               gain_weight=gain_w, gain_bias=gain_b),
        MixtureReLU(),
        Conv2d(128, 128, 4, bias=True, padding=1, stride=2, 
               gain_weight=gain_w, gain_bias=gain_b),
        MixtureReLU(),
        
        # Block 3
        Conv2d(128, 256, 3, bias=True, padding=1, 
               gain_weight=gain_w, gain_bias=gain_b),
        MixtureReLU(),
        Conv2d(256, 256, 4, bias=True, padding=1, stride=2, 
               gain_weight=gain_w, gain_bias=gain_b),
        MixtureReLU(),
        
        # Classifier
        Linear(256 * final_size * final_size, 512, 
               gain_weight=gain_w, gain_bias=gain_b),
        MixtureReLU(),
        Linear(512, num_classes, gain_weight=gain_w, gain_bias=gain_b),
    ]
    
    # Add classification head
    if head == "remax":
        layers.append(Remax())
    elif head == "softmax":
        layers.append(ClosedFormSoftmax())
    else:
        raise ValueError(f"Unknown head type: {head}")
    
    return Sequential(*layers)


# ============================================================================
# Data Loading
# ============================================================================

def load_cifar10(data_dir: str, batch_size: int, validation_split: float):
    """Load CIFAR-10 dataset with train/val/test splits."""
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    full_train = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    
    # Split into train and validation
    n_train = len(full_train)
    n_val = int(n_train * validation_split)
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train, [n_train - n_val, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True
    )
    
    print(f"  Training:   {n_train - n_val} samples")
    print(f"  Validation: {n_val} samples")
    print(f"  Test:       {len(test_dataset)} samples")
    
    return train_loader, val_loader, test_loader, 10, 3, 32


def load_mnist(data_dir: str, batch_size: int, validation_split: float):
    """Load MNIST dataset with train/val/test splits."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    
    full_train = torchvision.datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    
    n_train = len(full_train)
    n_val = int(n_train * validation_split)
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train, [n_train - n_val, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    
    test_dataset = torchvision.datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True
    )
    
    print(f"  Training:   {n_train - n_val} samples")
    print(f"  Validation: {n_val} samples")
    print(f"  Test:       {len(test_dataset)} samples")
    
    return train_loader, val_loader, test_loader, 10, 1, 28


def load_dataset(dataset: str, data_dir: str, batch_size: int, validation_split: float):
    """Load dataset by name."""
    print(f"\nLoading {dataset.upper()} dataset...")
    
    if dataset == "cifar10":
        return load_cifar10(data_dir, batch_size, validation_split)
    elif dataset == "mnist":
        return load_mnist(data_dir, batch_size, validation_split)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


# ============================================================================
# Metrics Computation
# ============================================================================

def compute_nll(probs: np.ndarray, targets: np.ndarray, eps: float = 1e-15) -> float:
    """Compute Negative Log-Likelihood."""
    probs = np.clip(probs, eps, 1 - eps)
    n_samples = len(targets)
    return -np.sum(np.log(probs[np.arange(n_samples), targets])) / n_samples


def compute_accuracy(probs: np.ndarray, targets: np.ndarray) -> float:
    """Compute classification accuracy (0-100)."""
    predictions = np.argmax(probs, axis=1)
    return np.mean(predictions == targets) * 100


def compute_error_rate(probs: np.ndarray, targets: np.ndarray) -> float:
    """Compute classification error rate (0-100)."""
    return 100.0 - compute_accuracy(probs, targets)


# ============================================================================
# Training and Evaluation
# ============================================================================

def one_hot_encode(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """Convert labels to one-hot encoding (flattened)."""
    batch_size = len(labels)
    one_hot = np.zeros((batch_size, num_classes), dtype=np.float32)
    one_hot[np.arange(batch_size), labels] = 1.0
    return one_hot.flatten()


def train_epoch(model: Sequential, out_updater: OutputUpdater, 
                train_loader, var_y: np.ndarray, num_classes: int, epoch: int):
    """Train for one epoch."""
    model.train()
    all_probs, all_targets = [], []
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        batch_images = images.numpy()
        batch_labels = labels.numpy()
        batch_size = batch_images.shape[0]
        batch_images = batch_images.reshape(batch_size, -1)
        
        m_pred, v_pred = model(batch_images)
        y = one_hot_encode(batch_labels, num_classes)
        
        out_updater.update(
            output_states=model.output_z_buffer,
            mu_obs=y,
            var_obs=var_y[:batch_size * num_classes],
            delta_states=model.input_delta_z_buffer,
        )
        
        model.backward()
        model.step()
        
        probs = np.reshape(m_pred, (batch_size, num_classes))
        all_probs.append(probs)
        all_targets.append(batch_labels)
        
        if (batch_idx + 1) % 50 == 0:
            print(f"  Epoch {epoch+1} - Batch {batch_idx+1}/{len(train_loader)}", end="\r")
    
    all_probs = np.vstack(all_probs)
    all_targets = np.concatenate(all_targets)
    
    return {
        "nll": compute_nll(all_probs, all_targets),
        "error": compute_error_rate(all_probs, all_targets)
    }


def evaluate(model: Sequential, data_loader, num_classes: int):
    """Evaluate the model on a dataset."""
    model.eval()
    all_probs, all_targets = [], []
    
    for images, labels in data_loader:
        batch_images = images.numpy()
        batch_labels = labels.numpy()
        batch_size = batch_images.shape[0]
        batch_images = batch_images.reshape(batch_size, -1)
        
        m_pred, _ = model(batch_images)
        probs = np.reshape(m_pred, (batch_size, num_classes))
        all_probs.append(probs)
        all_targets.append(batch_labels)
    
    all_probs = np.vstack(all_probs)
    all_targets = np.concatenate(all_targets)
    
    return {
        "nll": compute_nll(all_probs, all_targets),
        "error": compute_error_rate(all_probs, all_targets),
        "accuracy": compute_accuracy(all_probs, all_targets)
    }


# ============================================================================
# Main Training Function
# ============================================================================

def train(args):
    """Main training function."""
    # Create directories
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print(f"Training BNN on {args.dataset.upper()} with {args.head.upper()} head")
    print("=" * 70)
    
    # Load data
    train_loader, val_loader, test_loader, num_classes, input_channels, input_size = \
        load_dataset(args.dataset, args.data_dir, args.batch_size, args.val_split)
    
    # Build model
    print("\nBuilding model...")
    model = build_3cnn_model(
        num_classes=num_classes,
        head=args.head,
        gain_w=args.gain_w,
        gain_b=args.gain_b,
        input_channels=input_channels,
        input_size=input_size
    )
    
    # Move to device
    if pytagi.cuda.is_available() and args.device == "cuda":
        model.to_device("cuda")
        print(f"Using device: cuda")
    else:
        model.set_threads(8)
        print(f"Using device: cpu")
    
    out_updater = OutputUpdater(model.device)
    
    # Initialize scheduler
    scheduler = SigmaVScheduler(
        sigma_v_min=args.sigma_v_min,
        sigma_v_max=args.sigma_v_max,
        total_epochs=args.epochs,
        schedule_type=args.sigma_v_schedule
    )
    
    sigma_v = scheduler.get_sigma_v(0)
    var_y = np.full((args.batch_size * num_classes,), sigma_v**2, dtype=np.float32)
    
    print(f"\nConfiguration:")
    print(f"  Dataset: {args.dataset}, Classes: {num_classes}")
    print(f"  Architecture: 3-Block CNN with {args.head.upper()}")
    print(f"  Sigma_v: {args.sigma_v_max:.4f} -> {args.sigma_v_min:.4f} ({args.sigma_v_schedule})")
    
    # Training history
    history = {
        "train_nll": [], "train_error": [],
        "val_nll": [], "val_error": [],
        "test_nll": [], "test_error": [],
        "sigma_v": [],
    }
    
    best_val_nll = float("inf")
    best_epoch = -1
    model_name = f"{args.dataset}_{args.head}"
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print("-" * 70)
    
    start_time = time.time()
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        sigma_v = scheduler.get_sigma_v(epoch)
        var_y = np.full((args.batch_size * num_classes,), sigma_v**2, dtype=np.float32)
        
        train_metrics = train_epoch(model, out_updater, train_loader, var_y, num_classes, epoch)
        val_metrics = evaluate(model, val_loader, num_classes)
        test_metrics = evaluate(model, test_loader, num_classes)
        
        # Record history
        history["train_nll"].append(train_metrics["nll"])
        history["train_error"].append(train_metrics["error"])
        history["val_nll"].append(val_metrics["nll"])
        history["val_error"].append(val_metrics["error"])
        history["test_nll"].append(test_metrics["nll"])
        history["test_error"].append(test_metrics["error"])
        history["sigma_v"].append(sigma_v)
        
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch+1:3d}/{args.epochs} ({epoch_time:.1f}s) | σ_v: {sigma_v:.4f} | "
              f"Train: {train_metrics['nll']:.4f}/{train_metrics['error']:.2f}% | "
              f"Val: {val_metrics['nll']:.4f}/{val_metrics['error']:.2f}% | "
              f"Test: {test_metrics['nll']:.4f}/{test_metrics['error']:.2f}%")
        
        # Save best model
        if val_metrics["nll"] < best_val_nll:
            best_val_nll = val_metrics["nll"]
            best_epoch = epoch + 1
            
            model_path = os.path.join(args.save_dir, f"{model_name}_best.bin")
            if model.device == "cuda":
                model.params_to_host()
            model.save(model_path)
            if model.device == "cuda":
                model.params_to_device()
            print(f"  -> New best model saved! (Val NLL: {best_val_nll:.4f})")
    
    total_time = time.time() - start_time
    
    # Final summary
    print("-" * 70)
    print(f"\nTraining completed in {total_time/60:.1f} minutes")
    print(f"Best model from epoch {best_epoch} with Val NLL: {best_val_nll:.4f}")
    
    # Load and evaluate best model
    print("\nLoading best model for final evaluation...")
    model.load(os.path.join(args.save_dir, f"{model_name}_best.bin"))
    if model.device == "cuda":
        model.params_to_device()
    
    final_test = evaluate(model, test_loader, num_classes)
    print(f"\nFinal Test Results:")
    print(f"  NLL:      {final_test['nll']:.4f}")
    print(f"  Error:    {final_test['error']:.2f}%")
    print(f"  Accuracy: {final_test['accuracy']:.2f}%")
    
    # Save history
    history_path = os.path.join(args.results_dir, f"{model_name}_history.npz")
    np.savez(history_path, **history)
    print(f"\nTraining history saved to {history_path}")
    
    # Save results summary
    results_path = os.path.join(args.results_dir, f"{model_name}_results.txt")
    with open(results_path, "w") as f:
        f.write(f"BNN Training Results: {args.dataset.upper()} with {args.head.upper()}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Training epochs: {args.epochs}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Best epoch: {best_epoch}\n")
        f.write(f"Best validation NLL: {best_val_nll:.4f}\n\n")
        f.write("Final Test Results:\n")
        f.write(f"  NLL:      {final_test['nll']:.4f}\n")
        f.write(f"  Error:    {final_test['error']:.2f}%\n")
        f.write(f"  Accuracy: {final_test['accuracy']:.2f}%\n")
    print(f"Results saved to {results_path}")
    
    return history, final_test


# ============================================================================
# Entry Point
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train BNN with Remax/Softmax head on MNIST/CIFAR-10"
    )
    # Dataset and model
    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=["mnist", "cifar10"],
                        help="Dataset to train on (default: cifar10)")
    parser.add_argument("--head", type=str, default="remax",
                        choices=["remax", "softmax"],
                        help="Classification head type (default: remax)")
    
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of epochs (default: 50)")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size (default: 128)")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Validation split (default: 0.1)")
    
    # Model hyperparameters
    parser.add_argument("--gain-w", type=float, default=1.0,
                        help="Weight initialization gain (default: 1.0)")
    parser.add_argument("--gain-b", type=float, default=1.0,
                        help="Bias initialization gain (default: 1.0)")
    
    # Sigma_v scheduling
    parser.add_argument("--sigma-v-min", type=float, default=0.001,
                        help="Minimum observation noise std (default: 0.001)")
    parser.add_argument("--sigma-v-max", type=float, default=0.1,
                        help="Maximum observation noise std (default: 0.1)")
    parser.add_argument("--sigma-v-schedule", type=str, default="constant",
                        choices=["constant", "linear", "cosine", "exponential"],
                        help="Observation noise schedule (default: constant)")
    
    # Device and paths
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cpu", "cuda"],
                        help="Device to run on (default: cuda)")
    parser.add_argument("--data-dir", type=str, default=DATA_DIR,
                        help=f"Data directory (default: {DATA_DIR})")
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR,
                        help=f"Checkpoint directory (default: {SAVE_DIR})")
    parser.add_argument("--results-dir", type=str, default=RESULTS_DIR,
                        help=f"Results directory (default: {RESULTS_DIR})")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
