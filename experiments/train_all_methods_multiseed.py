"""
Unified Experiment: Compare All Analytical Methods vs Monte Carlo Ground Truth

This script trains 3CNN models on CIFAR-10 with three different output layers:
1. LL-Softmax (Linear Link Softmax)
2. MM-Softmax (Moment Matching Softmax)
3. MM-Remax (Moment Matching Remax)

For each method, we track the accuracy of analytical moment approximations 
compared to Monte Carlo sampling across multiple random seeds.

Metrics tracked per epoch:
- MAE of μ_A (mean probability) for all classes and dominant class
- MAE of σ_A (std probability) for all classes and dominant class  
- MAE of ρ_ZA (input-output correlation) for all classes and dominant class

The script supports:
- Multiple random seeds for statistical reliability
- Automatic averaging and std computation across seeds
- Synchronized model pairs (one with output layer, one without for MC comparison)
"""

import os
import sys
import time
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List

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
    ClosedFormSoftmax,
    OutputUpdater,
    Softmax,
)

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

# Import our analytical methods
from src.methods.ll_softmax import ll_softmax
from src.methods.mm_softmax import mm_softmax
from src.methods.mm_remax import mm_remax

# Try to import JAX accelerated versions, fall back to NumPy
try:
    from src.methods.mc_jax import mc_softmax_jax, mc_remax_jax, JAX_AVAILABLE
    if JAX_AVAILABLE:
        print("[INFO] JAX acceleration available for MC sampling")
        USE_JAX_DEFAULT = True
    else:
        print("[INFO] JAX not available, using NumPy MC sampling")
        USE_JAX_DEFAULT = False
except ImportError:
    print("[INFO] JAX module not found, using NumPy MC sampling")
    JAX_AVAILABLE = False
    USE_JAX_DEFAULT = False

# Import NumPy versions as fallback
from src.methods.mc_softmax import mc_softmax
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
EVAL_MC_FREQUENCY = 5  # Compute MC metrics every N epochs (1 = every epoch)

DATA_DIR = "./data"
SAVE_DIR = "./checkpoints"
RESULTS_DIR = "./results/method_comparison"


# ============================================================================
# Model Definitions
# ============================================================================

def build_3cnn_base():
    """Build base 3-block CNN without output layer."""
    layers = [
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
    ]
    return layers


def build_3cnn_with_output(method: str):
    """Build 3-block CNN with specified output layer."""
    base_layers = build_3cnn_base()
    
    if method == "ll-softmax":
        # LL-Softmax uses Softmax
        output_layer = Softmax()
    elif method == "mm-softmax":
        # MM-Softmax also uses ClosedFormSoftmax
        output_layer = ClosedFormSoftmax()
    elif method == "mm-remax":
        # MM-Remax uses Remax
        output_layer = Remax()
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return Sequential(*base_layers, output_layer)


def build_3cnn_logits():
    """Build 3-block CNN without output layer (for logit extraction)."""
    base_layers = build_3cnn_base()
    return Sequential(*base_layers)


# ============================================================================
# Data Loading
# ============================================================================

def load_cifar10(data_dir: str, batch_size: int, validation_split: float, seed: int):
    """Load CIFAR-10 with train/val/test splits using specified seed."""
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
    
    # Split training into train/val with fixed seed for reproducibility
    n_train = len(train_dataset)
    n_val = int(n_train * validation_split)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_train)
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

def compute_comparison_metrics(method: str, mu_z: np.ndarray, var_z: np.ndarray,
                                mc_samples: int = 10000, use_jax: bool = False,
                                seed: int = 0) -> Dict:
    """
    Compare analytical method against MC ground truth.
    
    Args:
        method: One of 'll-softmax', 'mm-softmax', 'mm-remax'
        mu_z: Mean logits (B, K)
        var_z: Variance of logits (B, K)
        mc_samples: Number of MC samples for ground truth
        use_jax: Whether to use JAX-accelerated MC sampling (much faster)
        seed: Random seed for MC sampling
        
    Returns:
        Dictionary with comparison metrics (MAE for μ_A, σ_A, ρ_ZA)
    """
    batch_size, num_classes = mu_z.shape
    epsilon = 1e-10
    
    # Compute analytical approximation
    if method == "ll-softmax":
        analytical_results = ll_softmax(mu_z, var_z)
    elif method == "mm-softmax":
        analytical_results = mm_softmax(mu_z, var_z)
    elif method == "mm-remax":
        analytical_results = mm_remax(mu_z, var_z)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    mu_analytical = analytical_results['mu_a']
    sigma_analytical = np.sqrt(analytical_results['sigma_a_sq'])
    cov_z_a_analytical = analytical_results['cov_z_a']
    
    # Compute MC ground truth (JAX or NumPy)
    if use_jax and JAX_AVAILABLE:
        # Use JAX-accelerated version (10-100x faster)
        if method in ["ll-softmax", "mm-softmax"]:
            mc_results = mc_softmax_jax(mu_z, var_z, n_samples=mc_samples, seed=seed)
        else:  # mm-remax
            mc_results = mc_remax_jax(mu_z, var_z, n_samples=mc_samples, seed=seed)
    else:
        # Use NumPy version
        if method in ["ll-softmax", "mm-softmax"]:
            mc_results = mc_softmax(mu_z, var_z, n_samples=mc_samples)
        else:  # mm-remax
            mc_results = mc_remax(mu_z, var_z, n_samples=mc_samples)
    
    mu_mc = mc_results['mu_a']
    sigma_mc = np.sqrt(mc_results['sigma_a_sq'])
    cov_z_a_mc = mc_results['cov_z_a']
    
    # Compute correlation coefficient ρ_ZA = Cov(Z,A) / (σ_Z * σ_A)
    sigma_z = np.sqrt(var_z)
    denom_analytical = sigma_z * sigma_analytical
    denom_mc = sigma_z * sigma_mc
    min_denom = 1e-6
    
    # Mask out entries with too-small denominators
    valid_mask = (denom_analytical > min_denom) & (denom_mc > min_denom)
    
    rho_analytical = np.where(valid_mask, cov_z_a_analytical / (denom_analytical + epsilon), 0.0)
    rho_mc = np.where(valid_mask, cov_z_a_mc / (denom_mc + epsilon), 0.0)
    
    # Clip to valid correlation range [-1, 1]
    rho_analytical = np.clip(rho_analytical, -1.0, 1.0)
    rho_mc = np.clip(rho_mc, -1.0, 1.0)
    
    # Get dominant class (highest mean probability from analytical)
    dominant_class = np.argmax(mu_analytical, axis=1)
    batch_indices = np.arange(batch_size)
    
    # ===== MAE for ALL classes =====
    mae_mu_all = np.mean(np.abs(mu_analytical - mu_mc))
    mae_sigma_all = np.mean(np.abs(sigma_analytical - sigma_mc))
    if np.sum(valid_mask) > 0:
        mae_rho_all = np.mean(np.abs(rho_analytical[valid_mask] - rho_mc[valid_mask]))
    else:
        mae_rho_all = 0.0
    
    # ===== MAE for DOMINANT class only =====
    mu_analytical_dom = mu_analytical[batch_indices, dominant_class]
    mu_mc_dom = mu_mc[batch_indices, dominant_class]
    sigma_analytical_dom = sigma_analytical[batch_indices, dominant_class]
    sigma_mc_dom = sigma_mc[batch_indices, dominant_class]
    rho_analytical_dom = rho_analytical[batch_indices, dominant_class]
    rho_mc_dom = rho_mc[batch_indices, dominant_class]
    valid_mask_dom = valid_mask[batch_indices, dominant_class]
    
    mae_mu_dom = np.mean(np.abs(mu_analytical_dom - mu_mc_dom))
    mae_sigma_dom = np.mean(np.abs(sigma_analytical_dom - sigma_mc_dom))
    if np.sum(valid_mask_dom) > 0:
        mae_rho_dom = np.mean(np.abs(rho_analytical_dom[valid_mask_dom] - rho_mc_dom[valid_mask_dom]))
    else:
        mae_rho_dom = 0.0
    
    return {
        'mae_mu_all': mae_mu_all,
        'mae_sigma_all': mae_sigma_all,
        'mae_rho_all': mae_rho_all,
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

def synchronize_models(model_with_output: Sequential, model_logits: Sequential):
    """
    Synchronize weights from model_with_output to model_logits.
    Uses state_dict/load_state_dict for weight transfer.
    """
    # Transfer parameters to host for state_dict access
    if model_with_output.device == "cuda":
        model_with_output.params_to_host()
    if model_logits.device == "cuda":
        model_logits.params_to_host()
    
    # Get state dict from output model and load into logits model
    state_dict = model_with_output.state_dict()
    model_logits.load_state_dict(state_dict)
    
    # Transfer back to device if using CUDA
    if model_with_output.device == "cuda":
        model_with_output.params_to_device()
    if model_logits.device == "cuda":
        model_logits.params_to_device()


def evaluate_epoch(method: str, model_with_output: Sequential, model_logits: Sequential,
                   data_loader, mc_samples: int = 1000, compute_mc: bool = True,
                   use_jax: bool = False, seed: int = 0) -> Dict:
    """
    Evaluate comparison metrics on a dataset.
    
    Args:
        compute_mc: If False, skip MC computation and return None for MC metrics
        use_jax: Whether to use JAX-accelerated MC sampling (much faster)
        seed: Random seed for MC sampling
    
    Returns accuracy and comparison metrics.
    """
    model_with_output.eval()
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
        m_prob, v_prob = model_with_output(batch_flat)
        m_logit, v_logit = model_logits(batch_flat)
        
        # Reshape
        mu_prob = np.reshape(m_prob, (batch_size, 10))
        var_prob = np.reshape(v_prob, (batch_size, 10))
        mu_logit = np.reshape(m_logit, (batch_size, 10))
        var_logit = np.reshape(v_logit, (batch_size, 10))
        
        # Compute comparison metrics only if requested
        if compute_mc:
            metrics = compute_comparison_metrics(
                method, mu_logit, var_logit, mc_samples=mc_samples,
                use_jax=use_jax, seed=seed
            )
            all_metrics.append(metrics)
        
        # Track for accuracy
        all_probs.append(mu_prob)
        all_labels.append(batch_labels)
    
    # Compute mean and std of metrics across batches
    avg_metrics = {}
    
    if compute_mc and len(all_metrics) > 0:
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            avg_metrics[key] = np.mean(values)
            avg_metrics[f'{key}_std'] = np.std(values)
    else:
        # Return None for MC metrics if not computed
        for key in ['mae_mu_all', 'mae_sigma_all', 'mae_rho_all',
                    'mae_mu_dominant', 'mae_sigma_dominant', 'mae_rho_dominant']:
            avg_metrics[key] = None
            avg_metrics[f'{key}_std'] = None
    
    # Compute accuracy (always computed)
    probs = np.vstack(all_probs)
    labels = np.concatenate(all_labels)
    preds = np.argmax(probs, axis=1)
    accuracy = np.mean(preds == labels) * 100
    avg_metrics['accuracy'] = accuracy
    
    return avg_metrics


def train_epoch(model_with_output: Sequential, model_logits: Sequential,
                out_updater: OutputUpdater, train_loader, var_y: np.ndarray) -> Dict:
    """
    Train for one epoch, updating both models synchronously.
    """
    model_with_output.train()
    
    all_probs = []
    all_labels = []
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        batch_images = images.numpy()
        batch_labels = labels.numpy()
        batch_size = batch_images.shape[0]
        batch_flat = batch_images.reshape(batch_size, -1)
        
        # Forward pass on output model (for training)
        m_pred, v_pred = model_with_output(batch_flat)
        
        # Convert labels to one-hot
        y = one_hot_encode(batch_labels, num_classes=10)
        
        # Update output layer
        out_updater.update(
            output_states=model_with_output.output_z_buffer,
            mu_obs=y,
            var_obs=var_y[:batch_size * 10],
            delta_states=model_with_output.input_delta_z_buffer,
        )
        
        # Backward pass and parameter update
        model_with_output.backward()
        model_with_output.step()
        
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
# Utility Functions
# ============================================================================

def list_existing_results(results_dir: str):
    """List all existing seed results."""
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Results directory does not exist: {results_dir}")
        return
    
    print(f"\n{'='*70}")
    print(f"Existing Results in {results_dir}")
    print(f"{'='*70}")
    
    methods = ["ll_softmax", "mm_softmax", "mm_remax"]
    for method in methods:
        seed_files = sorted(results_path.glob(f"{method}_seed*.npz"))
        if seed_files:
            seeds = [int(f.stem.split('seed')[1]) for f in seed_files]
            print(f"\n{method.replace('_', '-').upper()}:")
            print(f"  Completed seeds: {seeds}")
            print(f"  Total: {len(seeds)} seeds")
        else:
            print(f"\n{method.replace('_', '-').upper()}:")
            print(f"  No results found")
    
    # Check for aggregated results
    print(f"\n{'='*70}")
    print(f"Aggregated Results:")
    print(f"{'='*70}")
    for method in methods:
        agg_file = results_path / f"{method}_multiseed.npz"
        if agg_file.exists():
            data = np.load(agg_file, allow_pickle=True)
            num_seeds = data['num_seeds'] if 'num_seeds' in data.files else 'unknown'
            print(f"{method.replace('_', '-').upper()}: {num_seeds} seeds aggregated")
        else:
            print(f"{method.replace('_', '-').upper()}: No aggregated results")
    print(f"{'='*70}\n")


# ============================================================================
# Single Seed Experiment
# ============================================================================

def check_seed_exists(results_dir: str, method: str, seed: int) -> bool:
    """Check if results for this seed already exist."""
    method_clean = method.replace('-', '_')
    seed_file = Path(results_dir) / f"{method_clean}_seed{seed}.npz"
    return seed_file.exists()


def load_seed_results(results_dir: str, method: str, seed: int) -> Dict:
    """Load existing results for a seed."""
    method_clean = method.replace('-', '_')
    seed_file = Path(results_dir) / f"{method_clean}_seed{seed}.npz"
    
    if not seed_file.exists():
        return None
    
    data = np.load(seed_file, allow_pickle=True)
    history = {key: data[key].tolist() if isinstance(data[key], np.ndarray) else data[key] 
               for key in data.files}
    return history


def _convert_to_serializable(obj):
    """Convert numpy types to Python native types recursively."""
    if isinstance(obj, np.ndarray):
        return [_convert_to_serializable(item) for item in obj.tolist()]
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, list):
        return [_convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: _convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        return str(obj)


def save_seed_results(results_dir: str, method: str, seed: int, history: Dict):
    """Save results for a single seed."""
    method_clean = method.replace('-', '_')
    seed_file = Path(results_dir) / f"{method_clean}_seed{seed}.npz"
    
    # Save as npz
    np.savez(seed_file, **{k: np.array(v) if isinstance(v, list) else v 
                            for k, v in history.items()})
    print(f"Seed results saved to: {seed_file}")
    
    # Also save as JSON for easy inspection
    json_file = Path(results_dir) / f"{method_clean}_seed{seed}.json"
    json_serializable = _convert_to_serializable(history)
    
    with open(json_file, 'w') as f:
        json.dump(json_serializable, f, indent=2)


def run_single_seed_experiment(method: str, seed: int, epochs: int, batch_size: int,
                                validation_split: float, mc_samples: int,
                                data_dir: str, results_dir: str, sigma_v: float, 
                                device: str, eval_mc_frequency: int = 5,
                                use_jax: bool = False) -> Dict:
    """
    Run experiment for a single seed and method.
    Saves results independently for this seed.
    
    Args:
        use_jax: Whether to use JAX-accelerated MC sampling (10-100x faster)
    
    Returns:
        Dictionary with history for this seed
    """
    print(f"\n{'='*70}")
    print(f"Method: {method.upper()} | Seed: {seed}")
    print(f"{'='*70}")
    
    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Load data
    train_loader, val_loader, test_loader = load_cifar10(
        data_dir, batch_size, validation_split, seed
    )
    
    # Build models
    model_with_output = build_3cnn_with_output(method)
    model_logits = build_3cnn_logits()
    
    use_cuda = pytagi.cuda.is_available() and device == "cuda"
    if use_cuda:
        model_with_output.to_device("cuda")
        model_logits.to_device("cuda")
    else:
        model_with_output.set_threads(8)
        model_logits.set_threads(8)
    
    # Synchronize initial weights
    synchronize_models(model_with_output, model_logits)
    
    # Create output updater
    out_updater = OutputUpdater(model_with_output.device)
    
    # Observation variance
    var_y = np.full((batch_size * 10,), sigma_v**2, dtype=np.float32)
    
    # History tracking
    history = {
        'epoch': [],
        'train_accuracy': [],
        'val_accuracy': [],
        'test_accuracy': [],
        'mae_mu_all': [],
        'mae_mu_all_std': [],
        'mae_sigma_all': [],
        'mae_sigma_all_std': [],
        'mae_rho_all': [],
        'mae_rho_all_std': [],
        'mae_mu_dominant': [],
        'mae_mu_dominant_std': [],
        'mae_sigma_dominant': [],
        'mae_sigma_dominant_std': [],
        'mae_rho_dominant': [],
        'mae_rho_dominant_std': [],
    }
    
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Train
        train_metrics = train_epoch(
            model_with_output, model_logits, out_updater, train_loader, var_y
        )
        
        # Synchronize weights after training
        synchronize_models(model_with_output, model_logits)
        
        # Determine if we should compute MC metrics this epoch
        # Always compute on first, last, and every eval_mc_frequency epochs
        compute_mc = (epoch == 0 or epoch == epochs - 1 or 
                     (epoch + 1) % eval_mc_frequency == 0)
        
        # Evaluate on validation set
        val_metrics = evaluate_epoch(
            method, model_with_output, model_logits, val_loader, 
            mc_samples=mc_samples, compute_mc=compute_mc,
            use_jax=use_jax, seed=seed + epoch  # Different seed per epoch
        )
        
        # Evaluate on test set
        test_metrics = evaluate_epoch(
            method, model_with_output, model_logits, test_loader, 
            mc_samples=mc_samples, compute_mc=compute_mc,
            use_jax=use_jax, seed=seed + epoch + 1000  # Different seed from val
        )
        
        epoch_time = time.time() - epoch_start
        
        # Record history
        history['epoch'].append(epoch + 1)
        history['train_accuracy'].append(train_metrics['accuracy'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['test_accuracy'].append(test_metrics['accuracy'])
        
        for key in ['mae_mu_all', 'mae_sigma_all', 'mae_rho_all',
                    'mae_mu_dominant', 'mae_sigma_dominant', 'mae_rho_dominant']:
            history[key].append(val_metrics[key])
            history[f'{key}_std'].append(val_metrics[f'{key}_std'])
        
        # Print progress (every 5 epochs or last epoch)
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            if val_metrics['mae_mu_all'] is not None:
                print(f"Epoch {epoch+1:3d}/{epochs} ({epoch_time:.1f}s) | "
                      f"Val Acc: {val_metrics['accuracy']:.2f}% | "
                      f"MAE μ: {val_metrics['mae_mu_all']:.4f}")
            else:
                print(f"Epoch {epoch+1:3d}/{epochs} ({epoch_time:.1f}s) | "
                      f"Val Acc: {val_metrics['accuracy']:.2f}% | "
                      f"[MC metrics skipped]")
    
    total_time = time.time() - start_time
    print(f"Completed in {total_time/60:.1f} minutes")
    
    # Add seed and method to history for tracking
    history['seed'] = seed
    history['method'] = method
    
    # Save results for this seed immediately
    save_seed_results(results_dir, method, seed, history)
    
    return history


# ============================================================================
# Multi-Seed Experiment
# ============================================================================

def run_multi_seed_experiment(method: str, seeds: List[int], epochs: int, 
                               batch_size: int, validation_split: float,
                               mc_samples: int, data_dir: str, save_dir: str,
                               results_dir: str, sigma_v: float, device: str,
                               eval_mc_frequency: int = 5, use_jax: bool = False):
    """
    Run experiment across multiple seeds for a given method.
    Computes mean and std across seeds.
    Skips seeds that already have saved results.
    
    Args:
        eval_mc_frequency: Compute MC metrics every N epochs (1 = every epoch, 5 = every 5th epoch)
        use_jax: Whether to use JAX-accelerated MC sampling (10-100x faster)
    """
    print(f"\n{'='*70}")
    print(f"Multi-Seed Experiment: {method.upper()}")
    print(f"Seeds: {seeds}")
    print(f"{'='*70}")
    
    # Create directories
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    # Check which seeds already exist
    all_histories = []
    seeds_to_run = []
    seeds_loaded = []
    
    for seed in seeds:
        if check_seed_exists(results_dir, method, seed):
            print(f"\n[SKIP] Seed {seed} already completed. Loading existing results...")
            history = load_seed_results(results_dir, method, seed)
            if history is not None:
                all_histories.append(history)
                seeds_loaded.append(seed)
            else:
                print(f"[WARNING] Failed to load seed {seed}. Will re-run.")
                seeds_to_run.append(seed)
        else:
            seeds_to_run.append(seed)
    
    if seeds_loaded:
        print(f"\nLoaded {len(seeds_loaded)} existing seeds: {seeds_loaded}")
    
    if seeds_to_run:
        print(f"\nWill run {len(seeds_to_run)} new seeds: {seeds_to_run}")
    else:
        print(f"\nAll seeds already completed! Skipping training.")
    
    # Run remaining seeds
    for seed in seeds_to_run:
        history = run_single_seed_experiment(
            method=method,
            seed=seed,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            mc_samples=mc_samples,
            data_dir=data_dir,
            results_dir=results_dir,
            sigma_v=sigma_v,
            device=device,
            eval_mc_frequency=eval_mc_frequency,
            use_jax=use_jax
        )
        all_histories.append(history)
    
    # Aggregate results across seeds
    print(f"\nAggregating results across {len(all_histories)} seeds...")
    
    if len(all_histories) == 0:
        print("[ERROR] No seed results available to aggregate!")
        return None
    
    # Get all metric keys (excluding seed and method metadata)
    metric_keys = [k for k in all_histories[0].keys() if k not in ['seed', 'method']]
    
    # Compute mean and std across seeds for each metric and epoch
    aggregated = {}
    for key in metric_keys:
        # Stack values across seeds (seeds x epochs)
        values = np.array([h[key] for h in all_histories])
        
        # Compute mean and std across seeds
        aggregated[f'{key}_mean'] = np.mean(values, axis=0).tolist()
        aggregated[f'{key}_std'] = np.std(values, axis=0).tolist()
        
        # Also save individual seed data
        for seed_idx, seed in enumerate(seeds):
            aggregated[f'{key}_seed{seed}'] = all_histories[seed_idx][key]
    
    # Add metadata
    aggregated['seeds'] = seeds
    aggregated['num_seeds'] = len(seeds)
    aggregated['seeds_loaded'] = seeds_loaded
    aggregated['seeds_run'] = seeds_to_run
    aggregated['method'] = method
    
    # Save aggregated results
    method_clean = method.replace('-', '_')
    results_path = os.path.join(results_dir, f"{method_clean}_multiseed.npz")
    np.savez(results_path, **{k: np.array(v) if isinstance(v, list) else v 
                               for k, v in aggregated.items()})
    print(f"Results saved to: {results_path}")
    
    # Save as JSON for easy inspection
    json_path = os.path.join(results_dir, f"{method_clean}_multiseed.json")
    json_serializable = {}
    for k, v in aggregated.items():
        if isinstance(v, np.ndarray):
            json_serializable[k] = v.tolist()
        elif isinstance(v, (list, int, float, str)):
            json_serializable[k] = v
        else:
            json_serializable[k] = str(v)
    
    with open(json_path, 'w') as f:
        json.dump(json_serializable, f, indent=2)
    print(f"Results saved to: {json_path}")
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"Final Results for {method.upper()} (Mean ± Std across {len(seeds)} seeds)")
    print(f"{'='*70}")
    print(f"Seeds completed: {len(seeds)}")
    print(f"  - Loaded from disk: {len(seeds_loaded)}")
    print(f"  - Newly trained: {len(seeds_to_run)}")
    print(f"{'='*70}")
    final_epoch_idx = -1
    print(f"Test Accuracy: {aggregated['test_accuracy_mean'][final_epoch_idx]:.2f}% ± "
          f"{aggregated['test_accuracy_std'][final_epoch_idx]:.2f}%")
    print(f"\nAll Classes:")
    print(f"  MAE μ_A: {aggregated['mae_mu_all_mean'][final_epoch_idx]:.6f} ± "
          f"{aggregated['mae_mu_all_std'][final_epoch_idx]:.6f}")
    print(f"  MAE σ_A: {aggregated['mae_sigma_all_mean'][final_epoch_idx]:.6f} ± "
          f"{aggregated['mae_sigma_all_std'][final_epoch_idx]:.6f}")
    print(f"  MAE ρ_ZA: {aggregated['mae_rho_all_mean'][final_epoch_idx]:.6f} ± "
          f"{aggregated['mae_rho_all_std'][final_epoch_idx]:.6f}")
    print(f"\nDominant Class Only:")
    print(f"  MAE μ_A: {aggregated['mae_mu_dominant_mean'][final_epoch_idx]:.6f} ± "
          f"{aggregated['mae_mu_dominant_std'][final_epoch_idx]:.6f}")
    print(f"  MAE σ_A: {aggregated['mae_sigma_dominant_mean'][final_epoch_idx]:.6f} ± "
          f"{aggregated['mae_sigma_dominant_std'][final_epoch_idx]:.6f}")
    print(f"  MAE ρ_ZA: {aggregated['mae_rho_dominant_mean'][final_epoch_idx]:.6f} ± "
          f"{aggregated['mae_rho_dominant_std'][final_epoch_idx]:.6f}")
    print(f"{'='*70}")
    
    return aggregated


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare all analytical methods vs MC across multiple seeds"
    )
    parser.add_argument("--method", type=str, required=False,
                        choices=["ll-softmax", "mm-softmax", "mm-remax", "all"],
                        help="Method to train (or 'all' for all methods)")
    parser.add_argument("--seeds", type=int, nargs='+', default=[0, 1, 2, 3, 4],
                        help="Random seeds to use (default: 0 1 2 3 4)")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help=f"Number of epochs (default: {EPOCHS})")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Batch size (default: {BATCH_SIZE})")
    parser.add_argument("--mc-samples", type=int, default=MC_SAMPLES,
                        help=f"MC samples for ground truth (default: {MC_SAMPLES})")
    parser.add_argument("--eval-mc-frequency", type=int, default=EVAL_MC_FREQUENCY,
                        help=f"Compute MC metrics every N epochs (default: {EVAL_MC_FREQUENCY})")
    parser.add_argument("--use-jax", action="store_true", default=USE_JAX_DEFAULT,
                        help=f"Use JAX-accelerated MC sampling (10-100x faster, default: {USE_JAX_DEFAULT})")
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
    parser.add_argument("--list-results", action="store_true",
                        help="List existing results and exit")
    
    args = parser.parse_args()
    
    # If list-results flag is set, just list and exit
    if args.list_results:
        list_existing_results(args.results_dir)
        sys.exit(0)
    
    # Otherwise, method is required
    if args.method is None:
        parser.error("--method is required (unless using --list-results)")
    
    # Determine which methods to run
    if args.method == "all":
        methods = ["ll-softmax", "mm-softmax", "mm-remax"]
    else:
        methods = [args.method]
    
    # Print JAX status
    if args.use_jax:
        if JAX_AVAILABLE:
            print(f"\n[INFO] Using JAX-accelerated MC sampling (expect 10-100x speedup)")
        else:
            print(f"\n[WARNING] --use-jax specified but JAX not available")
            print(f"[WARNING] Install with: pip install jax jaxlib")
            print(f"[WARNING] Falling back to NumPy MC sampling")
    
    # Run experiments for each method
    for method in methods:
        run_multi_seed_experiment(
            method=method,
            seeds=args.seeds,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_split=VALIDATION_SPLIT,
            mc_samples=args.mc_samples,
            data_dir=args.data_dir,
            save_dir=args.save_dir,
            results_dir=args.results_dir,
            sigma_v=args.sigma_v,
            device=args.device,
            eval_mc_frequency=args.eval_mc_frequency,
            use_jax=args.use_jax
        )
    
    print(f"\n{'='*70}")
    print("All experiments completed!")
    print(f"{'='*70}")
