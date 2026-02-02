"""
Evaluation script for trained BNN models.

Generates:
1. ECE (Expected Calibration Error) plots
2. OOD entropy histograms (CIFAR-10 vs SVHN)
3. Qualitative examples (normal, high entropy, high variance predictions)
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from scipy.stats import gaussian_kde

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# pytagi imports
import pytagi
from pytagi.nn import (
    Sequential,
    Conv2d,
    Linear,
    MixtureReLU,
    Remax,
)

import torch
import torchvision
import torchvision.transforms as transforms


# ============================================================================
# Configuration
# ============================================================================

CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
NUM_CLASSES = 10

# Weight initialization gains (must match training)
GAIN_W = 0.1
GAIN_B = 0.1

# Plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 11


# ============================================================================
# Model Definition (must match training)
# ============================================================================

def build_3cnn_remax():
    """Build the 3-block CNN with Remax classification head."""
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
    """
    Build the 3-block CNN WITHOUT Remax (outputs logits).
    Used to visualize the logit distributions before Remax transformation.
    """
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
        # No Remax - outputs raw logits
    )
    return model


# ============================================================================
# Data Loading
# ============================================================================

def load_cifar10_test(data_dir: str, batch_size: int = 128):
    """Load CIFAR-10 test dataset."""
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=2, drop_last=True
    )
    
    return test_loader, test_dataset


def load_svhn_test(data_dir: str, batch_size: int = 128):
    """Load SVHN test dataset (OOD for CIFAR-10)."""
    # Use same normalization as CIFAR-10 for fair comparison
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    test_dataset = torchvision.datasets.SVHN(
        root=data_dir, split='test', download=True, transform=transform
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=2, drop_last=True
    )
    
    return test_loader, test_dataset


def load_cifar10_raw(data_dir: str):
    """Load CIFAR-10 without normalization for visualization."""
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )
    return dataset


# ============================================================================
# Inference
# ============================================================================

def run_inference(model: Sequential, data_loader, max_batches: int = None):
    """
    Run inference on a dataset.
    
    Returns:
        probs: Mean probabilities (N, C)
        var_probs: Variance of probabilities (N, C)
        labels: True labels (N,)
    """
    model.eval()
    all_probs = []
    all_var_probs = []
    all_labels = []
    
    for batch_idx, (images, labels) in enumerate(data_loader):
        if max_batches and batch_idx >= max_batches:
            break
            
        batch_images = images.numpy()
        batch_labels = labels.numpy()
        
        batch_size = batch_images.shape[0]
        batch_images = batch_images.reshape(batch_size, -1)
        
        # Forward pass
        m_pred, v_pred = model(batch_images)
        
        # Reshape
        probs = np.reshape(m_pred, (batch_size, NUM_CLASSES))
        var_probs = np.reshape(v_pred, (batch_size, NUM_CLASSES))
        
        all_probs.append(probs)
        all_var_probs.append(var_probs)
        all_labels.append(batch_labels)
    
    return (np.vstack(all_probs), 
            np.vstack(all_var_probs), 
            np.concatenate(all_labels))


def run_inference_single(model: Sequential, image: np.ndarray):
    """
    Run inference on a single image.
    
    Args:
        model: The model (with or without Remax)
        image: Image array (C, H, W) normalized
        
    Returns:
        mu: Mean outputs (C,)
        var: Variance of outputs (C,)
    """
    model.eval()
    
    # Reshape to (1, C*H*W)
    flat_image = image.reshape(1, -1)
    
    # Forward pass
    m_pred, v_pred = model(flat_image)
    
    # Reshape to (C,)
    mu = np.reshape(m_pred, (NUM_CLASSES,))
    var = np.reshape(v_pred, (NUM_CLASSES,))
    
    return mu, var


# ============================================================================
# ECE Computation and Plotting
# ============================================================================

def calculate_ece(confidences, predictions, labels, num_bins=15):
    """
    Calculate Expected Calibration Error (ECE).
    
    Args:
        confidences: Array of confidence scores (max probability for each prediction)
        predictions: Array of predicted class indices
        labels: Array of true labels
        num_bins: Number of bins to use for calibration
    
    Returns:
        ece: Expected Calibration Error
        bin_accuracies: Accuracy in each bin
        bin_confidences: Average confidence in each bin
        bin_counts: Number of samples in each bin
    """
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    accuracies = (predictions == labels)
    
    ece = 0.0
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.astype(float).mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].astype(float).mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            bin_accuracies.append(accuracy_in_bin)
            bin_confidences.append(avg_confidence_in_bin)
            bin_counts.append(in_bin.sum())
        else:
            bin_accuracies.append(0)
            bin_confidences.append(0)
            bin_counts.append(0)
    
    return ece, np.array(bin_accuracies), np.array(bin_confidences), np.array(bin_counts)


def plot_ece_curve(bin_confidences, bin_accuracies, bin_counts, ece, 
                   num_bins=15, save_path=None):
    """Plot Expected Calibration (Reliability) Diagram."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    bin_width = 1.0 / num_bins
    bin_centers = np.linspace(bin_width/2, 1 - bin_width/2, num_bins)
    
    # Plot Perfect Calibration Line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', alpha=0.5, linewidth=2)

    # Plot accuracy and gap bars
    for i, (center, acc, conf, count) in enumerate(zip(bin_centers, bin_accuracies, 
                                                        bin_confidences, bin_counts)):
        if count > 0:
            # Accuracy bar
            ax.bar(center, acc, width=bin_width * 0.9, color='#4A90E2', alpha=0.7, 
                   edgecolor='black', linewidth=0.5)
            
            # Gap bar
            gap = conf - acc
            if gap > 0:  # Overconfident
                ax.bar(center, gap, width=bin_width * 0.9, bottom=acc, color='#FF6B6B', 
                       alpha=0.5, edgecolor='black', linewidth=0.5)
            else:  # Underconfident
                ax.bar(center, -gap, width=bin_width * 0.9, bottom=acc + gap, color='#90EE90', 
                       alpha=0.5, edgecolor='black', linewidth=0.5)

    # Styling
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel('Confidence', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_title(f'Expected Calibration Plot (ECE)\nScore: {ece:.4f}', 
                 fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Legend
    legend_elements = [
        Line2D([0], [0], color='k', linestyle='--', label='Perfect Calibration'),
        plt.Rectangle((0, 0), 1, 1, fc='#4A90E2', alpha=0.7, label='Accuracy'),
        plt.Rectangle((0, 0), 1, 1, fc='#FF6B6B', alpha=0.5, label='Gap (Over-confident)'),
        plt.Rectangle((0, 0), 1, 1, fc='#90EE90', alpha=0.5, label='Gap (Under-confident)')
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"ECE plot saved to {save_path}")
    
    plt.show()


# ============================================================================
# OOD Detection - Entropy Histograms
# ============================================================================

def compute_entropy(probs, eps=1e-10):
    """Compute entropy of probability distributions."""
    return -np.sum(probs * np.log(probs + eps), axis=1)


def plot_kde(ax, data, color, label, fill_alpha=0.3):
    """Plot KDE with fill."""
    if len(data) > 1 and np.std(data) > 0:
        density = gaussian_kde(data)
        min_val, max_val = min(data), max(data)
        range_val = max_val - min_val
        xs = np.linspace(min_val - 0.1 * range_val, max_val + 0.1 * range_val, 200)
        ys = density(xs)
        ax.plot(xs, ys, color=color, label=label, linewidth=2)
        ax.fill_between(xs, ys, color=color, alpha=fill_alpha)
    else:
        ax.hist(data, bins=50, alpha=0.5, density=True, label=label, color=color)


def plot_ood_entropy_histograms(cifar_probs, cifar_var, svhn_probs, svhn_var, 
                                 save_path=None):
    """
    Plot entropy distributions for ID (CIFAR-10) vs OOD (SVHN) data.
    """
    # Compute entropies
    cifar_entropy = compute_entropy(cifar_probs)
    svhn_entropy = compute_entropy(svhn_probs)
    
    # Compute total variance
    cifar_total_var = np.sum(cifar_var, axis=1)
    svhn_total_var = np.sum(svhn_var, axis=1)
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: Entropy Distribution
    ax1 = axes[0]
    plot_kde(ax1, cifar_entropy, 'blue', 'CIFAR-10 (ID)')
    plot_kde(ax1, svhn_entropy, 'red', 'SVHN (OOD)')
    ax1.set_title('Probability Entropy Distribution\n(Predictive Uncertainty)', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Entropy (Higher = More Uncertain)', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Right: Total Variance Distribution
    ax2 = axes[1]
    plot_kde(ax2, cifar_total_var, 'blue', 'CIFAR-10 (ID)')
    plot_kde(ax2, svhn_total_var, 'red', 'SVHN (OOD)')
    ax2.set_title('Total Probability Variance Distribution\n(Model Uncertainty)', 
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('Total Variance (Higher = More Uncertain)', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"OOD entropy plot saved to {save_path}")
    
    plt.show()
    
    # Print statistics
    print("\n" + "="*60)
    print("Entropy Statistics:")
    print("="*60)
    print(f"CIFAR-10 (ID):  Mean={np.mean(cifar_entropy):.4f}, "
          f"Std={np.std(cifar_entropy):.4f}")
    print(f"SVHN (OOD):     Mean={np.mean(svhn_entropy):.4f}, "
          f"Std={np.std(svhn_entropy):.4f}")
    print(f"\nTotal Variance Statistics:")
    print(f"CIFAR-10 (ID):  Mean={np.mean(cifar_total_var):.4f}, "
          f"Std={np.std(cifar_total_var):.4f}")
    print(f"SVHN (OOD):     Mean={np.mean(svhn_total_var):.4f}, "
          f"Std={np.std(svhn_total_var):.4f}")
    print("="*60)


# ============================================================================
# Qualitative Examples
# ============================================================================

def plot_prediction_example(image, true_label, mu_probs, sigma_probs, 
                            image_idx=0, save_path=None):
    """
    Plot image with error bars for probabilities.
    
    Args:
        image: Image array (CHW format, values in [0, 1])
        true_label: True class label
        mu_probs: Mean probabilities (C,)
        sigma_probs: Standard deviation of probabilities (C,)
        image_idx: Image index for title
        save_path: Path to save figure
    """
    pred_label = np.argmax(mu_probs)
    
    # Create figure with 2 subplots
    fig, (ax_img, ax_probs) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Color scheme
    prob_color = '#4A90E2'
    
    classes = np.arange(NUM_CLASSES)
    
    # ===== Plot 1: Original Image =====
    # Convert CHW to HWC for display
    if image.shape[0] == 3:
        img_display = np.transpose(image, (1, 2, 0))
    else:
        img_display = image
    
    # Clip to valid range
    img_display = np.clip(img_display, 0, 1)
    
    ax_img.imshow(img_display)
    ax_img.axis('off')
    
    correct = pred_label == true_label
    ax_img.set_title(
        f'True: {CIFAR10_CLASSES[true_label]}\n'
        f'Predicted: {CIFAR10_CLASSES[pred_label]}',
        fontsize=12,
        fontweight='bold',
        color='green' if correct else 'red'
    )
    
    # ===== Plot 2: Probabilities with Error Bars =====
    for i in classes:
        # Draw error bars
        ax_probs.errorbar(i, mu_probs[i], yerr=sigma_probs[i], fmt='none', 
                          ecolor=prob_color, elinewidth=1.5, capsize=5)
        # Draw horizontal lines at mu +/- sigma
        ax_probs.plot([i - 0.2, i + 0.2], 
                      [mu_probs[i] + sigma_probs[i], mu_probs[i] + sigma_probs[i]], 
                      prob_color, linewidth=1.5)
        ax_probs.plot([i - 0.2, i + 0.2], 
                      [mu_probs[i] - sigma_probs[i], mu_probs[i] - sigma_probs[i]], 
                      prob_color, linewidth=1.5)
        
        # Draw mean marker - color based on prediction
        if i == true_label:
            marker_color = 'green'
        elif i == pred_label:
            marker_color = 'red'
        else:
            marker_color = prob_color
            
        ax_probs.scatter(i, mu_probs[i], color=marker_color, marker='s', 
                         s=50, zorder=3, edgecolor='k', linewidth=0.5)
    
    ax_probs.set_xticks(classes)
    ax_probs.set_xticklabels([CIFAR10_CLASSES[i] for i in classes], rotation=45, ha='right')
    ax_probs.set_xlim(-0.5, NUM_CLASSES - 0.5)
    ax_probs.set_ylim(-0.05, 1.05)
    ax_probs.grid(True, axis='y', linestyle=':', alpha=0.5)
    ax_probs.set_ylabel('Probability', fontsize=11)
    ax_probs.set_title('Output Probabilities (μ ± σ)', fontsize=12, fontweight='bold')
    
    # Add legend
    legend_handles = [
        Line2D([0], [0], marker='s', color='w', label='μ_prob', 
               markerfacecolor=prob_color, markersize=8),
        Line2D([0], [0], color=prob_color, lw=2, label='μ ± σ'),
        Line2D([0], [0], marker='s', color='w', label='True class', 
               markerfacecolor='green', markersize=8),
        Line2D([0], [0], marker='s', color='w', label='Predicted', 
               markerfacecolor='red', markersize=8),
    ]
    ax_probs.legend(handles=legend_handles, loc='upper right')
    
    # Calculate metrics
    entropy = -np.sum(mu_probs * np.log(mu_probs + 1e-10))
    total_var = np.sum(sigma_probs**2)
    
    plt.tight_layout()
    plt.suptitle(f'Image #{image_idx} | Entropy: {entropy:.3f} | Variance: {total_var:.4f}', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Example plot saved to {save_path}")
    
    plt.show()
    
    # Print statistics
    print(f"\n{'='*60}")
    print(f"Image #{image_idx} Analysis")
    print(f"{'='*60}")
    print(f"True Label: {CIFAR10_CLASSES[true_label]}")
    print(f"Predicted: {CIFAR10_CLASSES[pred_label]}")
    print(f"Prediction Probability: {mu_probs[pred_label]:.4f} ± {sigma_probs[pred_label]:.4f}")
    print(f"True Class Probability: {mu_probs[true_label]:.4f} ± {sigma_probs[true_label]:.4f}")
    print(f"Entropy: {entropy:.4f}")
    print(f"Total Variance: {total_var:.6f}")
    print(f"Correct: {'✓' if correct else '✗'}")
    print(f"{'='*60}\n")


def plot_prediction_with_logits(image, true_label, mu_logits, sigma_logits,
                                 mu_probs, sigma_probs, image_idx=0, save_path=None):
    """
    Plot image with error bars for both logits and probabilities.
    
    Three-panel figure:
        1. Left: Original image
        2. Middle: Logits with uncertainty (μ_z ± σ_z)
        3. Right: Probabilities with uncertainty (μ_a ± σ_a)
    
    Args:
        image: Image array (CHW format, values in [0, 1])
        true_label: True class label
        mu_logits: Mean logits (C,)
        sigma_logits: Standard deviation of logits (C,)
        mu_probs: Mean probabilities (C,)
        sigma_probs: Standard deviation of probabilities (C,)
        image_idx: Image index for title
        save_path: Path to save figure
    """
    pred_label = np.argmax(mu_probs)
    
    # Create figure with 3 subplots
    fig, (ax_img, ax_logits, ax_probs) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Color scheme
    logit_color = '#E24A4A'   # Red for logits
    prob_color = '#4A90E2'    # Blue for probabilities
    
    classes = np.arange(NUM_CLASSES)
    
    # ===== Plot 1: Original Image =====
    # Convert CHW to HWC for display
    if image.shape[0] == 3:
        img_display = np.transpose(image, (1, 2, 0))
    else:
        img_display = image
    
    # Clip to valid range
    img_display = np.clip(img_display, 0, 1)
    
    ax_img.imshow(img_display)
    ax_img.axis('off')
    
    correct = pred_label == true_label
    ax_img.set_title(
        f'True: {CIFAR10_CLASSES[true_label]}\n'
        f'Predicted: {CIFAR10_CLASSES[pred_label]}',
        fontsize=12,
        fontweight='bold',
        color='green' if correct else 'red'
    )
    
    # ===== Plot 2: Logits with Error Bars =====
    for i in classes:
        # Draw error bars
        ax_logits.errorbar(i, mu_logits[i], yerr=sigma_logits[i], fmt='none', 
                          ecolor=logit_color, elinewidth=1.5, capsize=5)
        # Draw horizontal lines at mu +/- sigma
        ax_logits.plot([i - 0.2, i + 0.2], 
                       [mu_logits[i] + sigma_logits[i], mu_logits[i] + sigma_logits[i]], 
                       logit_color, linewidth=1.5)
        ax_logits.plot([i - 0.2, i + 0.2], 
                       [mu_logits[i] - sigma_logits[i], mu_logits[i] - sigma_logits[i]], 
                       logit_color, linewidth=1.5)
        
        # Draw mean marker - color based on prediction
        if i == true_label:
            marker_color = 'green'
        elif i == pred_label:
            marker_color = 'red'
        else:
            marker_color = logit_color
            
        ax_logits.scatter(i, mu_logits[i], color=marker_color, marker='s', 
                          s=50, zorder=3, edgecolor='k', linewidth=0.5)
    
    ax_logits.set_xticks(classes)
    ax_logits.set_xticklabels([CIFAR10_CLASSES[i] for i in classes], rotation=45, ha='right')
    ax_logits.set_xlim(-0.5, NUM_CLASSES - 0.5)
    ax_logits.grid(True, axis='y', linestyle=':', alpha=0.5)
    ax_logits.set_ylabel('Logit Value', fontsize=11)
    ax_logits.set_title('Logits (μ_z ± σ_z)', fontsize=12, fontweight='bold')
    ax_logits.axhline(y=0, color='gray', linestyle='--', alpha=0.5)  # Zero line reference
    
    # Add legend for logits
    legend_handles_logits = [
        Line2D([0], [0], marker='s', color='w', label='μ_z', 
               markerfacecolor=logit_color, markersize=8),
        Line2D([0], [0], color=logit_color, lw=2, label='μ_z ± σ_z'),
        Line2D([0], [0], marker='s', color='w', label='True class', 
               markerfacecolor='green', markersize=8),
        Line2D([0], [0], marker='s', color='w', label='Predicted', 
               markerfacecolor='red', markersize=8),
    ]
    ax_logits.legend(handles=legend_handles_logits, loc='upper right', fontsize=9)
    
    # ===== Plot 3: Probabilities with Error Bars =====
    for i in classes:
        # Draw error bars
        ax_probs.errorbar(i, mu_probs[i], yerr=sigma_probs[i], fmt='none', 
                          ecolor=prob_color, elinewidth=1.5, capsize=5)
        # Draw horizontal lines at mu +/- sigma
        ax_probs.plot([i - 0.2, i + 0.2], 
                      [mu_probs[i] + sigma_probs[i], mu_probs[i] + sigma_probs[i]], 
                      prob_color, linewidth=1.5)
        ax_probs.plot([i - 0.2, i + 0.2], 
                      [mu_probs[i] - sigma_probs[i], mu_probs[i] - sigma_probs[i]], 
                      prob_color, linewidth=1.5)
        
        # Draw mean marker - color based on prediction
        if i == true_label:
            marker_color = 'green'
        elif i == pred_label:
            marker_color = 'red'
        else:
            marker_color = prob_color
            
        ax_probs.scatter(i, mu_probs[i], color=marker_color, marker='s', 
                         s=50, zorder=3, edgecolor='k', linewidth=0.5)
    
    ax_probs.set_xticks(classes)
    ax_probs.set_xticklabels([CIFAR10_CLASSES[i] for i in classes], rotation=45, ha='right')
    ax_probs.set_xlim(-0.5, NUM_CLASSES - 0.5)
    ax_probs.set_ylim(-0.05, 1.05)
    ax_probs.grid(True, axis='y', linestyle=':', alpha=0.5)
    ax_probs.set_ylabel('Probability', fontsize=11)
    ax_probs.set_title('Output Probabilities (μ_a ± σ_a)', fontsize=12, fontweight='bold')
    
    # Add legend for probabilities
    legend_handles_probs = [
        Line2D([0], [0], marker='s', color='w', label='μ_a', 
               markerfacecolor=prob_color, markersize=8),
        Line2D([0], [0], color=prob_color, lw=2, label='μ_a ± σ_a'),
        Line2D([0], [0], marker='s', color='w', label='True class', 
               markerfacecolor='green', markersize=8),
        Line2D([0], [0], marker='s', color='w', label='Predicted', 
               markerfacecolor='red', markersize=8),
    ]
    ax_probs.legend(handles=legend_handles_probs, loc='upper right', fontsize=9)
    
    # Calculate metrics
    entropy = -np.sum(mu_probs * np.log(mu_probs + 1e-10))
    prob_var = np.sum(sigma_probs**2)
    logit_var = np.sum(sigma_logits**2)
    
    plt.tight_layout()
    plt.suptitle(f'Image #{image_idx} | Entropy: {entropy:.3f} | '
                 f'Logit Var: {logit_var:.4f} | Prob Var: {prob_var:.4f}', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Example plot saved to {save_path}")
    
    plt.show()
    
    # Print statistics
    print(f"\n{'='*70}")
    print(f"Image #{image_idx} Analysis")
    print(f"{'='*70}")
    print(f"True Label: {CIFAR10_CLASSES[true_label]}")
    print(f"Predicted: {CIFAR10_CLASSES[pred_label]}")
    print(f"Correct: {'✓' if correct else '✗'}")
    print(f"\nLogits (μ_z ± σ_z):")
    print(f"  Pred class ({CIFAR10_CLASSES[pred_label]}): {mu_logits[pred_label]:.4f} ± {sigma_logits[pred_label]:.4f}")
    print(f"  True class ({CIFAR10_CLASSES[true_label]}): {mu_logits[true_label]:.4f} ± {sigma_logits[true_label]:.4f}")
    print(f"  Total logit variance: {logit_var:.6f}")
    print(f"\nProbabilities (μ_a ± σ_a):")
    print(f"  Pred class ({CIFAR10_CLASSES[pred_label]}): {mu_probs[pred_label]:.4f} ± {sigma_probs[pred_label]:.4f}")
    print(f"  True class ({CIFAR10_CLASSES[true_label]}): {mu_probs[true_label]:.4f} ± {sigma_probs[true_label]:.4f}")
    print(f"  Total prob variance: {prob_var:.6f}")
    print(f"  Entropy: {entropy:.4f}")
    print(f"{'='*70}\n")


def plot_qualitative_examples(model_remax, model_logits, raw_dataset, 
                               normalized_dataset, num_examples=3, save_dir=None,
                               batch_size=128):
    """
    Generate qualitative examples: normal, high entropy, high variance predictions.
    
    Uses two models:
        - model_remax: Full model with Remax head (for probabilities)
        - model_logits: Same model without Remax (for logits)
    
    Args:
        model_remax: Model with Remax head
        model_logits: Model without Remax (outputs logits)
        raw_dataset: CIFAR-10 test dataset without normalization (for display)
        normalized_dataset: CIFAR-10 test dataset with normalization (for inference)
        num_examples: Number of examples per category
        save_dir: Directory to save plots
        batch_size: Batch size for inference
    """
    print("\nRunning batch inference for qualitative analysis...")
    
    # Create data loader for batch inference
    data_loader = torch.utils.data.DataLoader(
        normalized_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=2, drop_last=False
    )
    
    # Run batch inference on both models
    model_remax.eval()
    model_logits.eval()
    
    all_probs = []
    all_var_probs = []
    all_logits = []
    all_var_logits = []
    all_labels = []
    
    for batch_idx, (images, labels) in enumerate(data_loader):
        batch_images = images.numpy()
        batch_labels = labels.numpy()
        
        batch_sz = batch_images.shape[0]
        batch_flat = batch_images.reshape(batch_sz, -1)
        
        # Get probabilities from Remax model
        m_prob, v_prob = model_remax(batch_flat)
        probs = np.reshape(m_prob, (batch_sz, NUM_CLASSES))
        var_probs = np.reshape(v_prob, (batch_sz, NUM_CLASSES))
        
        # Get logits from logits model
        m_logit, v_logit = model_logits(batch_flat)
        logits = np.reshape(m_logit, (batch_sz, NUM_CLASSES))
        var_logits = np.reshape(v_logit, (batch_sz, NUM_CLASSES))
        
        all_probs.append(probs)
        all_var_probs.append(var_probs)
        all_logits.append(logits)
        all_var_logits.append(var_logits)
        all_labels.append(batch_labels)
        
        if (batch_idx + 1) % 20 == 0:
            print(f"  Processed {(batch_idx + 1) * batch_size}/{len(normalized_dataset)} samples...")
    
    probs = np.vstack(all_probs)
    var_probs = np.vstack(all_var_probs)
    logits = np.vstack(all_logits)
    var_logits = np.vstack(all_var_logits)
    labels = np.concatenate(all_labels)
    
    predictions = np.argmax(probs, axis=1)
    std_probs = np.sqrt(var_probs)
    std_logits = np.sqrt(var_logits)
    
    # Compute metrics
    entropy = compute_entropy(probs)
    total_var = np.sum(var_probs, axis=1)
    confidence = np.max(probs, axis=1)
    
    # Find examples
    correct_mask = predictions == labels
    
    # 1. Normal predictions (high confidence, correct, low variance)
    normal_score = confidence * correct_mask.astype(float) - total_var * 10
    normal_indices = np.argsort(-normal_score)[:num_examples]
    
    # 2. High entropy predictions
    high_entropy_indices = np.argsort(-entropy)[:num_examples]
    
    # 3. High variance predictions
    high_var_indices = np.argsort(-total_var)[:num_examples]
    
    # 4. Incorrect but confident (interesting failures)
    incorrect_mask = ~correct_mask
    if incorrect_mask.sum() > 0:
        failure_score = confidence * incorrect_mask.astype(float)
        failure_indices = np.argsort(-failure_score)[:num_examples]
    else:
        failure_indices = []
    
    # Helper function to plot example
    def plot_example(idx, category, example_num):
        # Get raw image for display
        raw_img, _ = raw_dataset[idx]
        raw_img = raw_img.numpy()
        
        save_path = None
        if save_dir:
            save_path = os.path.join(save_dir, f"example_{category}_{example_num}.png")
        
        plot_prediction_with_logits(
            raw_img, labels[idx], 
            logits[idx], std_logits[idx],
            probs[idx], std_probs[idx],
            image_idx=idx, save_path=save_path
        )
    
    # Plot examples
    print("\n" + "="*70)
    print("NORMAL PREDICTIONS (High Confidence, Correct)")
    print("="*70)
    for i, idx in enumerate(normal_indices):
        plot_example(idx, "normal", i)
    
    print("\n" + "="*70)
    print("HIGH ENTROPY PREDICTIONS (Most Uncertain)")
    print("="*70)
    for i, idx in enumerate(high_entropy_indices):
        plot_example(idx, "high_entropy", i)
    
    print("\n" + "="*70)
    print("HIGH VARIANCE PREDICTIONS (Most Model Uncertainty)")
    print("="*70)
    for i, idx in enumerate(high_var_indices):
        plot_example(idx, "high_variance", i)
    
    if len(failure_indices) > 0:
        print("\n" + "="*70)
        print("CONFIDENT FAILURES (High Confidence, Incorrect)")
        print("="*70)
        for i, idx in enumerate(failure_indices):
            plot_example(idx, "failure", i)


# ============================================================================
# Rejection Curves
# ============================================================================

def calculate_rejection_metrics(uncertainties, predictions, labels):
    """Calculate accuracy vs rejection rate."""
    sorted_indices = np.argsort(-uncertainties)
    sorted_predictions = predictions[sorted_indices]
    sorted_labels = labels[sorted_indices]
    
    n_samples = len(uncertainties)
    rejection_rates = np.linspace(0, 0.95, 20)
    accuracies = []
    
    for rate in rejection_rates:
        n_rejected = int(n_samples * rate)
        
        if n_rejected == 0:
            kept_predictions = sorted_predictions
            kept_labels = sorted_labels
        else:
            kept_predictions = sorted_predictions[n_rejected:]
            kept_labels = sorted_labels[n_rejected:]
            
        if len(kept_labels) > 0:
            acc = np.mean(kept_predictions == kept_labels)
        else:
            acc = 1.0
            
        accuracies.append(acc)
        
    return rejection_rates, np.array(accuracies)


def plot_rejection_curves(probs, var_probs, predictions, labels, 
                          title_suffix='', save_path=None):
    """Plot rejection curves for different uncertainty metrics."""
    # Calculate Metrics
    confidences = np.max(probs, axis=1)
    uncertainty_conf = 1.0 - confidences
    
    epsilon = 1e-10
    entropy = -np.sum(probs * np.log(probs + epsilon), axis=1)
    
    total_variance = np.sum(var_probs, axis=1)
    
    # Calculate Curves
    rates_conf, acc_conf = calculate_rejection_metrics(uncertainty_conf, predictions, labels)
    rates_ent, acc_ent = calculate_rejection_metrics(entropy, predictions, labels)
    rates_var, acc_var = calculate_rejection_metrics(total_variance, predictions, labels)
    
    # Plot
    plt.figure(figsize=(10, 6))
    
    plt.plot(rates_conf * 100, acc_conf * 100, marker='o', label='1 - Confidence', linewidth=2)
    plt.plot(rates_ent * 100, acc_ent * 100, marker='s', label='Entropy', linewidth=2)
    plt.plot(rates_var * 100, acc_var * 100, marker='^', label='Total Variance', linewidth=2)
    
    plt.xlabel('Rejection Rate (%)', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title(f'Accuracy vs Rejection Rate{title_suffix}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    min_acc = np.min([acc_conf.min(), acc_ent.min(), acc_var.min()])
    plt.ylim(bottom=max(min_acc * 90, 0), top=102)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Rejection curves saved to {save_path}")
    
    plt.show()


# ============================================================================
# Main Evaluation Function
# ============================================================================

def evaluate_model(model_path: str, data_dir: str = "./data", 
                   output_dir: str = "./figures", batch_size: int = 128,
                   device: str = "cuda", num_examples: int = 3):
    """
    Run full evaluation pipeline.
    
    Args:
        model_path: Path to saved model
        data_dir: Directory with datasets
        output_dir: Directory to save figures
        batch_size: Batch size for inference
        device: Device to run on
        num_examples: Number of qualitative examples per category
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("Model Evaluation Pipeline")
    print("="*70)
    
    use_cuda = pytagi.cuda.is_available() and device == "cuda"
    
    # Build and load Remax model (full model with probabilities)
    print(f"\nLoading Remax model from {model_path}...")
    model_remax = build_3cnn_remax()
    
    if use_cuda:
        model_remax.to_device("cuda")
        print("Using device: cuda")
    else:
        model_remax.set_threads(8)
        print("Using device: cpu")
    
    model_remax.load(model_path)
    if use_cuda:
        model_remax.params_to_device()
    
    # Build and load Logits model (same architecture without Remax)
    print("Building logits model (without Remax)...")
    model_logits = build_3cnn_logits()
    
    if use_cuda:
        model_logits.to_device("cuda")
    else:
        model_logits.set_threads(8)
    
    # Transfer weights from remax model to logits model
    # Both models share the same weights except for the final Remax layer
    print("Transferring weights to logits model...")
    
    # Get parameters from remax model
    if use_cuda:
        model_remax.params_to_host()
    
    # Copy parameters (both models have same architecture except Remax)
    model_logits.load(model_path)
    
    if use_cuda:
        model_remax.params_to_device()
        model_logits.params_to_device()
    
    print("Both models ready for inference!")
    
    # Load datasets
    print("\nLoading datasets...")
    cifar_loader, cifar_dataset = load_cifar10_test(data_dir, batch_size)
    svhn_loader, _ = load_svhn_test(data_dir, batch_size)
    cifar_raw = load_cifar10_raw(data_dir)
    
    # Also load normalized dataset for single-sample inference
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    cifar_normalized = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )
    
    # Run inference on CIFAR-10 (using remax model)
    print("\nRunning inference on CIFAR-10...")
    cifar_probs, cifar_var, cifar_labels = run_inference(model_remax, cifar_loader)
    cifar_preds = np.argmax(cifar_probs, axis=1)
    
    # Compute accuracy
    accuracy = np.mean(cifar_preds == cifar_labels) * 100
    print(f"CIFAR-10 Test Accuracy: {accuracy:.2f}%")
    
    # Run inference on SVHN (OOD)
    print("Running inference on SVHN (OOD)...")
    svhn_probs, svhn_var, svhn_labels = run_inference(model_remax, svhn_loader)
    
    # ==========================================
    # 1. ECE Plot
    # ==========================================
    print("\n" + "-"*70)
    print("1. Generating ECE Plot...")
    print("-"*70)
    
    confidences = np.max(cifar_probs, axis=1)
    ece, bin_acc, bin_conf, bin_counts = calculate_ece(
        confidences, cifar_preds, cifar_labels, num_bins=15
    )
    print(f"ECE Score: {ece:.4f}")
    
    plot_ece_curve(
        bin_conf, bin_acc, bin_counts, ece, num_bins=15,
        save_path=os.path.join(output_dir, "ece_plot.png")
    )
    
    # ==========================================
    # 2. OOD Entropy Histograms
    # ==========================================
    print("\n" + "-"*70)
    print("2. Generating OOD Entropy Histograms...")
    print("-"*70)
    
    plot_ood_entropy_histograms(
        cifar_probs, cifar_var, svhn_probs, svhn_var,
        save_path=os.path.join(output_dir, "ood_entropy_histogram.png")
    )
    
    # ==========================================
    # 3. Rejection Curves
    # ==========================================
    print("\n" + "-"*70)
    print("3. Generating Rejection Curves...")
    print("-"*70)
    
    plot_rejection_curves(
        cifar_probs, cifar_var, cifar_preds, cifar_labels,
        title_suffix=' (CIFAR-10)',
        save_path=os.path.join(output_dir, "rejection_curves.png")
    )
    
    # ==========================================
    # 4. Qualitative Examples (with Logits + Probabilities)
    # ==========================================
    print("\n" + "-"*70)
    print("4. Generating Qualitative Examples (Logits + Probabilities)...")
    print("-"*70)
    
    plot_qualitative_examples(
        model_remax, model_logits, 
        cifar_raw, cifar_normalized,
        num_examples=num_examples,
        save_dir=output_dir,
        batch_size=batch_size
    )
    
    print("\n" + "="*70)
    print("Evaluation Complete!")
    print(f"All figures saved to: {output_dir}")
    print("="*70)


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained BNN model")
    parser.add_argument("--model-path", type=str, 
                        default="./checkpoints/3cnn_remax_best.bin",
                        help="Path to saved model")
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Directory with datasets")
    parser.add_argument("--output-dir", type=str, default="./figures",
                        help="Directory to save figures")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size for inference")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cpu", "cuda"],
                        help="Device to run on")
    parser.add_argument("--num-examples", type=int, default=3,
                        help="Number of qualitative examples per category")
    
    args = parser.parse_args()
    
    evaluate_model(
        model_path=args.model_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device=args.device,
        num_examples=args.num_examples
    )
