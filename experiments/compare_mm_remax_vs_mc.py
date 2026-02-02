"""
Experiment: Compare Analytical Methods vs Monte Carlo Ground Truth

This script trains a 3CNN on CIFAR-10 while tracking the accuracy of 
analytical moment approximations (MM-Remax) compared to Monte Carlo sampling.

Metrics tracked per epoch:
- MAE of μ_A (mean probability) for all classes and dominant class
- MAE of σ_A (std probability) for all classes and dominant class  
- Correlation coefficient between analytical and MC for μ_A and σ_A

The experiment uses two synchronized models:
1. model_remax: Full model with Remax (for training and getting probabilities)
2. model_logits: Same model without Remax (for getting logit distributions)

Both models are updated with the same parameter updates to stay synchronized.
"""

import os
import sys
import time
import json
import argparse
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

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
from torch.utils.data import DataLoader, Subset

# Import our analytical methods
from src.methods.mm_remax import mm_remax
from src.methods.mc_remax import mc_remax


# ============================================================================
# Configuration
# ============================================================================

GAIN_W = 0.1
GAIN_B = 0.1

EPOCHS = 50
BATCH_SIZE = 128
VALIDATION_SPLIT = 0.1
MC_SAMPLES = 10000  # Monte Carlo samples for ground truth

DATA_DIR = "./data"
SAVE_DIR = "./checkpoints"
RESULTS_DIR = "./results/method_comparison"


# ============================================================================
# Model Definitions
# ============================================================================

def build_3cnn_remax():
    """Build 3-block CNN with Remax head."""
    model = Sequential(
        Conv2d(3, 64, 3, bias=True, padding=1, in_width=32, in_height=32, 
               gain_weight=GAIN_W, gain_bias=GAIN_B),
        MixtureReLU(),
        Conv2d(64, 64, 4, bias=True, padding=1, stride=2, 
               gain_weight=GAIN_W, gain_bias=GAIN_B),
        MixtureReLU(),
        Conv2d(64, 128, 3, bias=True, padding=1, 
               gain_weight=GAIN_W, gain_bias=GAIN_B),
        MixtureReLU(),
        Conv2d(128, 128, 4, bias=True, padding=1, stride=2, 
               gain_weight=GAIN_W, gain_bias=GAIN_B),
        MixtureReLU(),
        Conv2d(128, 256, 3, bias=True, padding=1, 
               gain_weight=GAIN_W, gain_bias=GAIN_B),
        MixtureReLU(),
        Conv2d(256, 256, 4, bias=True, padding=1, stride=2, 
               gain_weight=GAIN_W, gain_bias=GAIN_B),
        MixtureReLU(),
        Linear(256 * 4 * 4, 512, gain_weight=GAIN_W, gain_bias=GAIN_B),
        MixtureReLU(),
        Linear(512, 10, gain_weight=GAIN_W, gain_bias=GAIN_B),
        Remax()
    )
    return model


def build_3cnn_logits():
    """Build 3-block CNN WITHOUT Remax (outputs logits)."""
    model = Sequential(
        Conv2d(3, 64, 3, bias=True, padding=1, in_width=32, in_height=32, 
               gain_weight=GAIN_W, gain_bias=GAIN_B),
        MixtureReLU(),
        Conv2d(64, 64, 4, bias=True, padding=1, stride=2, 
               gain_weight=GAIN_W, gain_bias=GAIN_B),
        MixtureReLU(),
        Conv2d(64, 128, 3, bias=True, padding=1, 
               gain_weight=GAIN_W, gain_bias=GAIN_B),
        MixtureReLU(),
        Conv2d(128, 128, 4, bias=True, padding=1, stride=2, 
               gain_weight=GAIN_W, gain_bias=GAIN_B),
        MixtureReLU(),
        Conv2d(128, 256, 3, bias=True, padding=1, 
               gain_weight=GAIN_W, gain_bias=GAIN_B),
        MixtureReLU(),
        Conv2d(256, 256, 4, bias=True, padding=1, stride=2, 
               gain_weight=GAIN_W, gain_bias=GAIN_B),
        MixtureReLU(),
        Linear(256 * 4 * 4, 512, gain_weight=GAIN_W, gain_bias=GAIN_B),
        MixtureReLU(),
        Linear(512, 10, gain_weight=GAIN_W, gain_bias=GAIN_B),
        # No Remax
    )
    return model


# ============================================================================
# Data Loading
# ============================================================================

def load_cifar10(data_dir: str, batch_size: int, validation_split: float):
    """Load CIFAR-10 with train/val/test splits."""
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )
    
    # Split training into train/val
    n_train = len(train_dataset)
    n_val = int(n_train * validation_split)
    indices = np.random.permutation(n_train)
    train_indices = indices[n_val:]
    val_indices = indices[:n_val]
    
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(train_dataset, val_indices)
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                              num_workers=2, drop_last=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                            num_workers=2, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=2, drop_last=True)
    
    return train_loader, val_loader, test_loader


# ============================================================================
# Metrics Computation
# ============================================================================

def compute_comparison_metrics(mu_z: np.ndarray, sigma_z: np.ndarray,
                                mu_remax: np.ndarray, sigma_remax: np.ndarray,
                                mc_samples: int = 10000):
    """
    Compare analytical MM-Remax against MC-Remax ground truth.
    
    Computes MAE for:
    - μ_A: Mean of Remax output
    - σ_A: Std of Remax output  
    - ρ_ZA: Correlation coefficient between input Z and output A
    
    Args:
        mu_z: Mean logits (B, K)
        sigma_z: Std of logits (B, K)
        mu_remax: Mean probabilities from Remax model (B, K) - for reference
        sigma_remax: Std probabilities from Remax model (B, K) - for reference
        mc_samples: Number of MC samples for ground truth
        
    Returns:
        Dictionary with comparison metrics (MAE for μ_A, σ_A, ρ_ZA)
    """
    batch_size, num_classes = mu_z.shape
    epsilon = 1e-10
    
    # Compute MM-Remax (analytical approximation)
    mm_results = mm_remax(mu_z, sigma_z**2)
    mu_mm = mm_results['mu_a']
    sigma_mm = np.sqrt(mm_results['sigma_a_sq'])
    cov_z_a_mm = mm_results['cov_z_a']
    
    # Compute MC-Remax (ground truth)
    mc_results = mc_remax(mu_z, sigma_z**2, n_samples=mc_samples)
    mu_mc = mc_results['mu_a']
    sigma_mc = np.sqrt(mc_results['sigma_a_sq'])
    cov_z_a_mc = mc_results['cov_z_a']
    
    # Compute correlation coefficient ρ_ZA = Cov(Z,A) / (σ_Z * σ_A)
    # Only compute where both stds are non-negligible to avoid numerical issues
    denom_mm = sigma_z * sigma_mm
    denom_mc = sigma_z * sigma_mc
    min_denom = 1e-6  # Minimum denominator to avoid explosion
    
    # Mask out entries with too-small denominators
    valid_mask = (denom_mm > min_denom) & (denom_mc > min_denom)
    
    rho_mm = np.where(valid_mask, cov_z_a_mm / (denom_mm + epsilon), 0.0)
    rho_mc = np.where(valid_mask, cov_z_a_mc / (denom_mc + epsilon), 0.0)
    
    # Clip to valid correlation range [-1, 1] for numerical stability
    rho_mm = np.clip(rho_mm, -1.0, 1.0)
    rho_mc = np.clip(rho_mc, -1.0, 1.0)
    
    # Get dominant class (highest mean probability from MM)
    dominant_class = np.argmax(mu_mm, axis=1)
    batch_indices = np.arange(batch_size)
    
    # ===== MAE for ALL classes =====
    mae_mu_all = np.mean(np.abs(mu_mm - mu_mc))
    mae_sigma_all = np.mean(np.abs(sigma_mm - sigma_mc))
    # Only compute rho MAE on valid entries
    if np.sum(valid_mask) > 0:
        mae_rho_all = np.mean(np.abs(rho_mm[valid_mask] - rho_mc[valid_mask]))
    else:
        mae_rho_all = 0.0
    
    # ===== MAE for DOMINANT class only =====
    mu_mm_dom = mu_mm[batch_indices, dominant_class]
    mu_mc_dom = mu_mc[batch_indices, dominant_class]
    sigma_mm_dom = sigma_mm[batch_indices, dominant_class]
    sigma_mc_dom = sigma_mc[batch_indices, dominant_class]
    rho_mm_dom = rho_mm[batch_indices, dominant_class]
    rho_mc_dom = rho_mc[batch_indices, dominant_class]
    valid_mask_dom = valid_mask[batch_indices, dominant_class]
    
    mae_mu_dom = np.mean(np.abs(mu_mm_dom - mu_mc_dom))
    mae_sigma_dom = np.mean(np.abs(sigma_mm_dom - sigma_mc_dom))
    # Only compute rho MAE on valid entries for dominant class
    if np.sum(valid_mask_dom) > 0:
        mae_rho_dom = np.mean(np.abs(rho_mm_dom[valid_mask_dom] - rho_mc_dom[valid_mask_dom]))
    else:
        mae_rho_dom = 0.0
    
    return {
        # MAE metrics - All classes
        'mae_mu_all': mae_mu_all,
        'mae_sigma_all': mae_sigma_all,
        'mae_rho_all': mae_rho_all,
        # MAE metrics - Dominant class
        'mae_mu_dominant': mae_mu_dom,
        'mae_sigma_dominant': mae_sigma_dom,
        'mae_rho_dominant': mae_rho_dom,
    }


def one_hot_encode(labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """Convert labels to one-hot encoding."""
    batch_size = len(labels)
    one_hot = np.zeros((batch_size, num_classes), dtype=np.float32)
    one_hot[np.arange(batch_size), labels] = 1.0
    return one_hot.flatten()


# ============================================================================
# Training Functions
# ============================================================================

def synchronize_models(model_remax: Sequential, model_logits: Sequential):
    """
    Synchronize weights from model_remax to model_logits.
    Both models share the same architecture (except Remax layer which has no weights).
    Uses state_dict/load_state_dict for weight transfer.
    """
    # Transfer parameters to host for state_dict access
    if model_remax.device == "cuda":
        model_remax.params_to_host()
    if model_logits.device == "cuda":
        model_logits.params_to_host()
    
    # Get state dict from remax model and load into logits model
    state_dict = model_remax.state_dict()
    model_logits.load_state_dict(state_dict)
    
    # Transfer back to device if using CUDA
    if model_remax.device == "cuda":
        model_remax.params_to_device()
    if model_logits.device == "cuda":
        model_logits.params_to_device()


def evaluate_epoch(model_remax: Sequential, model_logits: Sequential,
                   data_loader, mc_samples: int = 1000):
    """
    Evaluate comparison metrics on a dataset.
    
    Returns accuracy and comparison metrics.
    """
    model_remax.eval()
    model_logits.eval()
    
    all_metrics = []
    all_probs = []
    all_labels = []
    
    for images, labels in data_loader:
        batch_images = images.numpy()
        batch_labels = labels.numpy()
        batch_size = batch_images.shape[0]
        batch_flat = batch_images.reshape(batch_size, -1)
        
        # Forward pass on both models
        m_prob, v_prob = model_remax(batch_flat)
        m_logit, v_logit = model_logits(batch_flat)
        
        # Reshape
        mu_prob = np.reshape(m_prob, (batch_size, 10))
        var_prob = np.reshape(v_prob, (batch_size, 10))
        mu_logit = np.reshape(m_logit, (batch_size, 10))
        var_logit = np.reshape(v_logit, (batch_size, 10))
        
        # Compute comparison metrics
        metrics = compute_comparison_metrics(
            mu_logit, np.sqrt(var_logit),
            mu_prob, np.sqrt(var_prob),
            mc_samples=mc_samples
        )
        all_metrics.append(metrics)
        
        # Track for accuracy
        all_probs.append(mu_prob)
        all_labels.append(batch_labels)
    
    # Compute mean and std of metrics across batches
    avg_metrics = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics]
        avg_metrics[key] = np.mean(values)
        avg_metrics[f'{key}_std'] = np.std(values)
    
    # Compute accuracy
    probs = np.vstack(all_probs)
    labels = np.concatenate(all_labels)
    preds = np.argmax(probs, axis=1)
    accuracy = np.mean(preds == labels) * 100
    avg_metrics['accuracy'] = accuracy
    
    return avg_metrics


def train_epoch(model_remax: Sequential, model_logits: Sequential,
                out_updater: OutputUpdater, train_loader, var_y: np.ndarray):
    """
    Train for one epoch, updating both models synchronously.
    """
    model_remax.train()
    
    all_probs = []
    all_labels = []
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        batch_images = images.numpy()
        batch_labels = labels.numpy()
        batch_size = batch_images.shape[0]
        batch_flat = batch_images.reshape(batch_size, -1)
        
        # Forward pass on Remax model (for training)
        m_pred, v_pred = model_remax(batch_flat)
        
        # Convert labels to one-hot
        y = one_hot_encode(batch_labels, num_classes=10)
        
        # Update output layer
        out_updater.update(
            output_states=model_remax.output_z_buffer,
            mu_obs=y,
            var_obs=var_y[:batch_size * 10],
            delta_states=model_remax.input_delta_z_buffer,
        )
        
        # Backward pass and parameter update
        model_remax.backward()
        model_remax.step()
        
        # Track metrics
        probs = np.reshape(m_pred, (batch_size, 10))
        all_probs.append(probs)
        all_labels.append(batch_labels)
    
    # Compute training accuracy
    probs = np.vstack(all_probs)
    labels = np.concatenate(all_labels)
    preds = np.argmax(probs, axis=1)
    accuracy = np.mean(preds == labels) * 100
    
    return {'accuracy': accuracy}


# ============================================================================
# Main Experiment
# ============================================================================

def run_experiment(epochs: int = EPOCHS, batch_size: int = BATCH_SIZE,
                   validation_split: float = VALIDATION_SPLIT,
                   mc_samples: int = MC_SAMPLES,
                   data_dir: str = DATA_DIR, save_dir: str = SAVE_DIR,
                   results_dir: str = RESULTS_DIR,
                   sigma_v: float = 0.1, device: str = "cuda"):
    """
    Run the full experiment comparing MM-Remax vs MC-Remax over training.
    """
    # Create directories
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("Experiment: MM-Remax vs MC-Remax Comparison Over Training")
    print("="*70)
    
    # Load data
    print("\nLoading CIFAR-10...")
    train_loader, val_loader, test_loader = load_cifar10(
        data_dir, batch_size, validation_split
    )
    
    # Build models
    print("\nBuilding models...")
    model_remax = build_3cnn_remax()
    model_logits = build_3cnn_logits()
    
    use_cuda = pytagi.cuda.is_available() and device == "cuda"
    if use_cuda:
        model_remax.to_device("cuda")
        model_logits.to_device("cuda")
        print("Using device: cuda")
    else:
        model_remax.set_threads(8)
        model_logits.set_threads(8)
        print("Using device: cpu")
    
    # Synchronize initial weights
    print("Synchronizing model weights...")
    synchronize_models(model_remax, model_logits)
    
    # Create output updater
    out_updater = OutputUpdater(model_remax.device)
    
    # Observation variance
    var_y = np.full((batch_size * 10,), sigma_v**2, dtype=np.float32)
    
    print(f"\nExperiment Configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  MC samples: {mc_samples}")
    print(f"  Sigma_v: {sigma_v}")
    
    # History tracking (mean and std for each metric)
    history = {
        'epoch': [],
        'train_accuracy': [],
        'val_accuracy': [],
        'test_accuracy': [],
        # MAE metrics - All classes (mean and std)
        'mae_mu_all': [],
        'mae_mu_all_std': [],
        'mae_sigma_all': [],
        'mae_sigma_all_std': [],
        'mae_rho_all': [],
        'mae_rho_all_std': [],
        # MAE metrics - Dominant class (mean and std)
        'mae_mu_dominant': [],
        'mae_mu_dominant_std': [],
        'mae_sigma_dominant': [],
        'mae_sigma_dominant_std': [],
        'mae_rho_dominant': [],
        'mae_rho_dominant_std': [],
    }
    
    print(f"\nStarting training for {epochs} epochs...")
    print("-"*70)
    
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Train
        train_metrics = train_epoch(
            model_remax, model_logits, out_updater, train_loader, var_y
        )
        
        # Synchronize weights after training
        synchronize_models(model_remax, model_logits)
        
        # Evaluate on validation set with comparison metrics
        val_metrics = evaluate_epoch(
            model_remax, model_logits, val_loader, mc_samples=mc_samples
        )
        
        # Evaluate on test set
        test_metrics = evaluate_epoch(
            model_remax, model_logits, test_loader, mc_samples=mc_samples
        )
        
        epoch_time = time.time() - epoch_start
        
        # Record history
        history['epoch'].append(epoch + 1)
        history['train_accuracy'].append(train_metrics['accuracy'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['test_accuracy'].append(test_metrics['accuracy'])
        
        # Record comparison metrics (mean and std from validation set)
        for key in ['mae_mu_all', 'mae_sigma_all', 'mae_rho_all',
                    'mae_mu_dominant', 'mae_sigma_dominant', 'mae_rho_dominant']:
            history[key].append(val_metrics[key])
            history[f'{key}_std'].append(val_metrics[f'{key}_std'])
        
        # Print progress
        print(f"Epoch {epoch+1:3d}/{epochs} ({epoch_time:.1f}s) | "
              f"Train Acc: {train_metrics['accuracy']:.2f}% | "
              f"Val Acc: {val_metrics['accuracy']:.2f}% | "
              f"MAE μ: {val_metrics['mae_mu_all']:.4f}±{val_metrics['mae_mu_all_std']:.4f} | "
              f"MAE σ: {val_metrics['mae_sigma_all']:.4f}±{val_metrics['mae_sigma_all_std']:.4f} | "
              f"MAE ρ: {val_metrics['mae_rho_all']:.4f}±{val_metrics['mae_rho_all_std']:.4f}")
    
    total_time = time.time() - start_time
    print("-"*70)
    print(f"Training completed in {total_time/60:.1f} minutes")
    
    # Save results
    results_path = os.path.join(results_dir, "mm_remax_comparison.npz")
    np.savez(results_path, **{k: np.array(v) for k, v in history.items()})
    print(f"\nResults saved to: {results_path}")
    
    # Save as JSON for easy inspection
    json_path = os.path.join(results_dir, "mm_remax_comparison.json")
    with open(json_path, 'w') as f:
        json.dump({k: [float(x) for x in v] for k, v in history.items()}, f, indent=2)
    print(f"Results saved to: {json_path}")
    
    # Save final model
    model_path = os.path.join(save_dir, "3cnn_remax_comparison.bin")
    if use_cuda:
        model_remax.params_to_host()
    model_remax.save(model_path)
    print(f"Model saved to: {model_path}")
    
    # Print final summary
    print("\n" + "="*70)
    print("Final Results (Last Epoch)")
    print("="*70)
    print(f"Test Accuracy: {history['test_accuracy'][-1]:.2f}%")
    print(f"\nAll Classes:")
    print(f"  MAE μ_A: {history['mae_mu_all'][-1]:.6f}")
    print(f"  MAE σ_A: {history['mae_sigma_all'][-1]:.6f}")
    print(f"  MAE ρ_ZA: {history['mae_rho_all'][-1]:.6f}")
    print(f"\nDominant Class Only:")
    print(f"  MAE μ_A: {history['mae_mu_dominant'][-1]:.6f}")
    print(f"  MAE σ_A: {history['mae_sigma_dominant'][-1]:.6f}")
    print(f"  MAE ρ_ZA: {history['mae_rho_dominant'][-1]:.6f}")
    print("="*70)
    
    return history


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare MM-Remax vs MC-Remax over training epochs"
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help=f"Number of epochs (default: {EPOCHS})")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Batch size (default: {BATCH_SIZE})")
    parser.add_argument("--mc-samples", type=int, default=MC_SAMPLES,
                        help=f"MC samples for ground truth (default: {MC_SAMPLES})")
    parser.add_argument("--sigma-v", type=float, default=0.1,
                        help="Observation noise (default: 0.1)")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cpu", "cuda"],
                        help="Device (default: cuda)")
    parser.add_argument("--data-dir", type=str, default=DATA_DIR,
                        help=f"Data directory (default: {DATA_DIR})")
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR,
                        help=f"Checkpoint directory (default: {SAVE_DIR})")
    parser.add_argument("--results-dir", type=str, default=RESULTS_DIR,
                        help=f"Results directory (default: {RESULTS_DIR})")
    
    args = parser.parse_args()
    
    run_experiment(
        epochs=args.epochs,
        batch_size=args.batch_size,
        mc_samples=args.mc_samples,
        sigma_v=args.sigma_v,
        device=args.device,
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        results_dir=args.results_dir,
    )
