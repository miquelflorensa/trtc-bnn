"""
Training script for 3-block CNN with Remax head on CIFAR-10 using pytagi.

This script trains the TAGI_CNN_REMAX architecture with:
- 3 convolutional blocks with MixtureReLU activations
- Remax classification head
- Tracks train, validation, and test metrics
- Saves the model with best validation NLL
"""

import os
import sys
import time
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
    OutputUpdater,
)

import torch
import torchvision
import torchvision.transforms as transforms


# ============================================================================
# Configuration
# ============================================================================

# Weight initialization gains
GAIN_W = 1.0
GAIN_B = 1.0

# Training hyperparameters
EPOCHS = 50
BATCH_SIZE = 128
VALIDATION_SPLIT = 0.1

# Paths
DATA_DIR = "./data"
SAVE_DIR = "./checkpoints"
RESULTS_DIR = "./results"


# ============================================================================
# Model Definition
# ============================================================================

def build_3cnn_remax():
    """
    Build the 3-block CNN with Remax classification head.
    
    Architecture:
        Block 1: 32x32 -> 16x16 (64 channels)
        Block 2: 16x16 -> 8x8 (128 channels)
        Block 3: 8x8 -> 4x4 (256 channels)
        Classifier: 256*4*4 -> 512 -> 10 with Remax
    """
    model = Sequential(
        # --- Block 1: 32x32 ---
        # Standard Conv (Keep size)
        Conv2d(3, 64, 3, bias=True, padding=1, in_width=32, in_height=32, 
               gain_weight=GAIN_W, gain_bias=GAIN_B),
        MixtureReLU(),
        
        # Downsample (32 -> 16) | Kernel 4 fixes the division error
        Conv2d(64, 64, 4, bias=True, padding=1, stride=2, 
               gain_weight=GAIN_W, gain_bias=GAIN_B),
        MixtureReLU(),

        # --- Block 2: 16x16 ---
        Conv2d(64, 128, 3, bias=True, padding=1, 
               gain_weight=GAIN_W, gain_bias=GAIN_B),
        MixtureReLU(),
        
        # Downsample (16 -> 8) | Kernel 4 fixes the division error
        Conv2d(128, 128, 4, bias=True, padding=1, stride=2, 
               gain_weight=GAIN_W, gain_bias=GAIN_B),
        MixtureReLU(),

        # --- Block 3: 8x8 ---
        Conv2d(128, 256, 3, bias=True, padding=1, 
               gain_weight=GAIN_W, gain_bias=GAIN_B),
        MixtureReLU(),
        
        # Downsample (8 -> 4) | Kernel 4 fixes the division error
        Conv2d(256, 256, 4, bias=True, padding=1, stride=2, 
               gain_weight=GAIN_W, gain_bias=GAIN_B),
        MixtureReLU(),

        # --- Classifier ---
        Linear(256 * 4 * 4, 512, gain_weight=GAIN_W, gain_bias=GAIN_B),
        MixtureReLU(),
        Linear(512, 10, gain_weight=GAIN_W, gain_bias=GAIN_B),
        Remax()
    )
    
    return model


# ============================================================================
# Data Loading
# ============================================================================

def load_cifar10(data_dir: str, batch_size: int, validation_split: float):
    """
    Load CIFAR-10 dataset and split into train, validation, and test sets.
    
    Args:
        data_dir: Directory to store/load data
        batch_size: Batch size for data loaders
        validation_split: Fraction of training data for validation
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Normalization values for CIFAR-10
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    
    # Transforms with data augmentation for training
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
    
    # Load full training set
    full_train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    
    # Split into train and validation
    num_train = len(full_train_dataset)
    num_val = int(num_train * validation_split)
    num_train = num_train - num_val
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train_dataset, [num_train, num_val],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Load test set
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True
    )
    
    print(f"Dataset splits:")
    print(f"  Training:   {num_train} samples")
    print(f"  Validation: {num_val} samples")
    print(f"  Test:       {len(test_dataset)} samples")
    
    return train_loader, val_loader, test_loader


# ============================================================================
# Metrics Computation
# ============================================================================

def compute_nll(probs: np.ndarray, targets: np.ndarray, eps: float = 1e-15) -> float:
    """
    Compute Negative Log-Likelihood.
    
    Args:
        probs: Predicted probabilities of shape (N, C)
        targets: True labels of shape (N,)
        eps: Small constant for numerical stability
        
    Returns:
        Mean NLL value
    """
    probs = np.clip(probs, eps, 1 - eps)
    n_samples = len(targets)
    nll = -np.sum(np.log(probs[np.arange(n_samples), targets])) / n_samples
    return nll


def compute_accuracy(probs: np.ndarray, targets: np.ndarray) -> float:
    """
    Compute classification accuracy.
    
    Args:
        probs: Predicted probabilities of shape (N, C)
        targets: True labels of shape (N,)
        
    Returns:
        Accuracy value (0-100)
    """
    predictions = np.argmax(probs, axis=1)
    accuracy = np.mean(predictions == targets) * 100
    return accuracy


def compute_error_rate(probs: np.ndarray, targets: np.ndarray) -> float:
    """
    Compute classification error rate.
    
    Args:
        probs: Predicted probabilities of shape (N, C)
        targets: True labels of shape (N,)
        
    Returns:
        Error rate value (0-100)
    """
    return 100.0 - compute_accuracy(probs, targets)


# ============================================================================
# Training and Evaluation Functions
# ============================================================================

def one_hot_encode(labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """
    Convert labels to one-hot encoding.
    
    Args:
        labels: Class labels of shape (N,)
        num_classes: Number of classes
        
    Returns:
        One-hot encoded labels of shape (N * num_classes,) flattened
    """
    batch_size = len(labels)
    one_hot = np.zeros((batch_size, num_classes), dtype=np.float32)
    one_hot[np.arange(batch_size), labels] = 1.0

    # # Create smoothed labels
    # batch_size = len(labels)
    # one_hot = np.full((batch_size, num_classes), 0.1 / (num_classes - 1), dtype=np.float32)
    # one_hot[np.arange(batch_size), labels] = 0.9

    return one_hot.flatten()


def train_epoch(model: Sequential, out_updater: OutputUpdater, 
                train_loader, var_y: np.ndarray, epoch: int):
    """
    Train for one epoch.
    
    Args:
        model: The TAGI model
        out_updater: Output updater for backpropagation
        train_loader: Training data loader
        var_y: Observation variance
        epoch: Current epoch number
        
    Returns:
        Dictionary with training metrics
    """
    model.train()
    all_probs = []
    all_targets = []
    
    num_batches = len(train_loader)
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        # Convert to numpy
        batch_images = images.numpy()
        batch_labels = labels.numpy()
        
        # Flatten images: (B, C, H, W) -> (B, C*H*W)
        batch_size = batch_images.shape[0]
        batch_images = batch_images.reshape(batch_size, -1)
        
        # Forward pass
        m_pred, v_pred = model(batch_images)
        
        # Convert labels to one-hot encoding for Remax
        y = one_hot_encode(batch_labels, num_classes=10)
        
        # Update output layers
        out_updater.update(
            output_states=model.output_z_buffer,
            mu_obs=y,
            var_obs=var_y[:batch_size * 10],
            delta_states=model.input_delta_z_buffer,
        )
        
        # Backward pass and update parameters
        model.backward()
        model.step()
        
        # Reshape predictions and collect
        probs = np.reshape(m_pred, (batch_size, 10))
        all_probs.append(probs)
        all_targets.append(batch_labels)
        
        # Progress update
        if (batch_idx + 1) % 50 == 0:
            print(f"  Epoch {epoch+1} - Batch {batch_idx+1}/{num_batches}", end="\r")
    
    # Concatenate all predictions
    all_probs = np.vstack(all_probs)
    all_targets = np.concatenate(all_targets)
    
    # Compute metrics
    nll = compute_nll(all_probs, all_targets)
    error = compute_error_rate(all_probs, all_targets)
    
    return {"nll": nll, "error": error}


def evaluate(model: Sequential, data_loader, desc: str = "Evaluating"):
    """
    Evaluate the model on a dataset.
    
    Args:
        model: The TAGI model
        data_loader: Data loader for evaluation
        desc: Description for progress display
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    all_probs = []
    all_targets = []
    
    for batch_idx, (images, labels) in enumerate(data_loader):
        # Convert to numpy
        batch_images = images.numpy()
        batch_labels = labels.numpy()
        
        # Flatten images: (B, C, H, W) -> (B, C*H*W)
        batch_size = batch_images.shape[0]
        batch_images = batch_images.reshape(batch_size, -1)
        
        # Forward pass only (inference mode)
        m_pred, v_pred = model(batch_images)
        
        # Reshape predictions
        probs = np.reshape(m_pred, (batch_size, 10))
        all_probs.append(probs)
        all_targets.append(batch_labels)
    
    # Concatenate all predictions
    all_probs = np.vstack(all_probs)
    all_targets = np.concatenate(all_targets)
    
    # Compute metrics
    nll = compute_nll(all_probs, all_targets)
    error = compute_error_rate(all_probs, all_targets)
    accuracy = compute_accuracy(all_probs, all_targets)
    
    return {"nll": nll, "error": error, "accuracy": accuracy}


# ============================================================================
# Main Training Loop
# ============================================================================

def train(epochs: int = EPOCHS, batch_size: int = BATCH_SIZE, 
          validation_split: float = VALIDATION_SPLIT,
          data_dir: str = DATA_DIR, save_dir: str = SAVE_DIR,
          results_dir: str = RESULTS_DIR, sigma_v: float = 0.1,
          device: str = "cuda"):
    """
    Main training function.
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size
        validation_split: Fraction of training data for validation
        data_dir: Directory for data
        save_dir: Directory to save checkpoints
        results_dir: Directory to save results
        sigma_v: Observation noise standard deviation
        device: Device to run on ('cpu' or 'cuda')
    """
    # Create directories
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Training 3-Block CNN with Remax on CIFAR-10")
    print("=" * 70)
    
    # Load data
    print("\nLoading CIFAR-10 dataset...")
    train_loader, val_loader, test_loader = load_cifar10(
        data_dir, batch_size, validation_split
    )
    
    # Build model
    print("\nBuilding model...")
    model = build_3cnn_remax()
    
    # Move to device
    if pytagi.cuda.is_available() and device == "cuda":
        model.to_device("cuda")
        print(f"Using device: cuda")
    else:
        model.set_threads(8)
        print(f"Using device: cpu")
    
    # Create output updater
    out_updater = OutputUpdater(model.device)
    
    # Observation variance for Remax
    var_y = np.full((batch_size * 10,), sigma_v**2, dtype=np.float32)
    
    print(f"Model architecture: 3-Block CNN with Remax")
    print(f"  Input: 3x32x32 (CIFAR-10)")
    print(f"  Block 1: 64 channels, 32->16")
    print(f"  Block 2: 128 channels, 16->8")
    print(f"  Block 3: 256 channels, 8->4")
    print(f"  Classifier: 4096->512->10")
    
    # Training history
    history = {
        "train_nll": [],
        "train_error": [],
        "val_nll": [],
        "val_error": [],
        "test_nll": [],
        "test_error": [],
    }
    
    best_val_nll = float("inf")
    best_epoch = -1
    
    print(f"\nStarting training for {epochs} epochs...")
    print("-" * 70)
    
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Training
        train_metrics = train_epoch(model, out_updater, train_loader, var_y, epoch)
        
        # Validation
        val_metrics = evaluate(model, val_loader, "Validation")
        
        # Test
        test_metrics = evaluate(model, test_loader, "Test")
        
        # Record history
        history["train_nll"].append(train_metrics["nll"])
        history["train_error"].append(train_metrics["error"])
        history["val_nll"].append(val_metrics["nll"])
        history["val_error"].append(val_metrics["error"])
        history["test_nll"].append(test_metrics["nll"])
        history["test_error"].append(test_metrics["error"])
        
        epoch_time = time.time() - epoch_start
        
        # Print epoch summary
        print(f"Epoch {epoch+1:3d}/{epochs} ({epoch_time:.1f}s) | "
              f"Train NLL: {train_metrics['nll']:.4f}, Err: {train_metrics['error']:.2f}% | "
              f"Val NLL: {val_metrics['nll']:.4f}, Err: {val_metrics['error']:.2f}% | "
              f"Test NLL: {test_metrics['nll']:.4f}, Err: {test_metrics['error']:.2f}%")
        
        # Save best model based on validation NLL
        if val_metrics["nll"] < best_val_nll:
            best_val_nll = val_metrics["nll"]
            best_epoch = epoch + 1
            
            # Save model - need to transfer params to host first for CUDA
            model_path = os.path.join(save_dir, "3cnn_remax_best.bin")
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
    
    # Final test evaluation with best model
    print("\nLoading best model for final evaluation...")
    model.load(os.path.join(save_dir, "3cnn_remax_best.bin"))
    if model.device == "cuda":
        model.params_to_device()
    
    final_test = evaluate(model, test_loader, "Final Test")
    print(f"\nFinal Test Results (Best Model):")
    print(f"  NLL:      {final_test['nll']:.4f}")
    print(f"  Error:    {final_test['error']:.2f}%")
    print(f"  Accuracy: {final_test['accuracy']:.2f}%")
    
    # Save training history
    history_path = os.path.join(results_dir, "3cnn_remax_history.npz")
    np.savez(history_path, **history)
    print(f"\nTraining history saved to {history_path}")
    
    # Save final results
    results_path = os.path.join(results_dir, "3cnn_remax_results.txt")
    with open(results_path, "w") as f:
        f.write("3-Block CNN with Remax on CIFAR-10\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Training epochs: {epochs}\n")
        f.write(f"Batch size: {batch_size}\n")
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

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train 3-Block CNN with Remax on CIFAR-10"
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help=f"Number of epochs (default: {EPOCHS})")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Batch size (default: {BATCH_SIZE})")
    parser.add_argument("--val-split", type=float, default=VALIDATION_SPLIT,
                        help=f"Validation split (default: {VALIDATION_SPLIT})")
    parser.add_argument("--sigma-v", type=float, default=0.001,
                        help="Observation noise std (default: 0.1)")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cpu", "cuda"],
                        help="Device to run on (default: cuda)")
    parser.add_argument("--data-dir", type=str, default=DATA_DIR,
                        help=f"Data directory (default: {DATA_DIR})")
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR,
                        help=f"Checkpoint directory (default: {SAVE_DIR})")
    parser.add_argument("--results-dir", type=str, default=RESULTS_DIR,
                        help=f"Results directory (default: {RESULTS_DIR})")
    
    args = parser.parse_args()
    
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=args.val_split,
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        results_dir=args.results_dir,
        sigma_v=args.sigma_v,
        device=args.device
    )
