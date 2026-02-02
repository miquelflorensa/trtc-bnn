"""
Test script for the new qualitative example plots with logits and probabilities.

This script creates synthetic data to test the 3-panel plotting function
(Image | Logits | Probabilities) without needing a trained model.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Configuration
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
NUM_CLASSES = 10

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 11


def plot_prediction_with_logits(image, true_label, mu_logits, sigma_logits,
                                 mu_probs, sigma_probs, image_idx=0, save_path=None):
    """
    Plot image with error bars for both logits and probabilities.
    
    Three-panel figure:
        1. Left: Original image
        2. Middle: Logits with uncertainty (μ_z ± σ_z)
        3. Right: Probabilities with uncertainty (μ_a ± σ_a)
    """
    pred_label = np.argmax(mu_probs)
    
    # Create figure with 3 subplots
    fig, (ax_img, ax_logits, ax_probs) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Color scheme
    logit_color = '#E24A4A'   # Red for logits
    prob_color = '#4A90E2'    # Blue for probabilities
    
    classes = np.arange(NUM_CLASSES)
    
    # ===== Plot 1: Original Image =====
    if image.shape[0] == 3:
        img_display = np.transpose(image, (1, 2, 0))
    else:
        img_display = image
    
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
        ax_logits.errorbar(i, mu_logits[i], yerr=sigma_logits[i], fmt='none', 
                          ecolor=logit_color, elinewidth=1.5, capsize=5)
        ax_logits.plot([i - 0.2, i + 0.2], 
                       [mu_logits[i] + sigma_logits[i], mu_logits[i] + sigma_logits[i]], 
                       logit_color, linewidth=1.5)
        ax_logits.plot([i - 0.2, i + 0.2], 
                       [mu_logits[i] - sigma_logits[i], mu_logits[i] - sigma_logits[i]], 
                       logit_color, linewidth=1.5)
        
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
    ax_logits.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
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
        ax_probs.errorbar(i, mu_probs[i], yerr=sigma_probs[i], fmt='none', 
                          ecolor=prob_color, elinewidth=1.5, capsize=5)
        ax_probs.plot([i - 0.2, i + 0.2], 
                      [mu_probs[i] + sigma_probs[i], mu_probs[i] + sigma_probs[i]], 
                      prob_color, linewidth=1.5)
        ax_probs.plot([i - 0.2, i + 0.2], 
                      [mu_probs[i] - sigma_probs[i], mu_probs[i] - sigma_probs[i]], 
                      prob_color, linewidth=1.5)
        
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


def create_synthetic_example(example_type='normal'):
    """Create synthetic data for testing plots."""
    
    # Create a random colorful image
    np.random.seed(42 if example_type == 'normal' else 123)
    image = np.random.rand(3, 32, 32)
    
    if example_type == 'normal':
        # High confidence prediction
        true_label = 3  # cat
        
        # Logits: high for true class
        mu_logits = np.array([-1.5, -2.0, -0.8, 3.5, -1.2, -0.5, -1.8, -2.2, -1.0, -1.5])
        sigma_logits = np.array([0.3, 0.4, 0.35, 0.25, 0.3, 0.35, 0.4, 0.3, 0.35, 0.4])
        
        # Probabilities (after Remax)
        mu_probs = np.array([0.02, 0.01, 0.05, 0.85, 0.02, 0.03, 0.01, 0.005, 0.015, 0.01])
        sigma_probs = np.array([0.01, 0.005, 0.02, 0.03, 0.01, 0.015, 0.005, 0.003, 0.01, 0.005])
        
    elif example_type == 'high_entropy':
        # Uncertain prediction
        true_label = 5  # dog
        
        # Logits: similar values, high uncertainty
        mu_logits = np.array([0.5, 0.3, 0.8, 0.6, 0.4, 0.9, 0.2, 0.7, 0.5, 0.4])
        sigma_logits = np.array([0.8, 0.75, 0.9, 0.85, 0.7, 0.95, 0.8, 0.85, 0.75, 0.8])
        
        # Probabilities (after Remax) - more uniform
        mu_probs = np.array([0.09, 0.08, 0.12, 0.10, 0.08, 0.14, 0.07, 0.11, 0.10, 0.11])
        sigma_probs = np.array([0.04, 0.035, 0.05, 0.045, 0.035, 0.055, 0.03, 0.045, 0.04, 0.045])
        
    elif example_type == 'high_variance':
        # High variance in output
        true_label = 8  # ship
        
        # Logits with high variance
        mu_logits = np.array([-0.5, -1.0, -0.3, -0.8, -0.2, -0.6, -1.2, -0.4, 2.0, -0.7])
        sigma_logits = np.array([1.2, 1.5, 1.3, 1.4, 1.2, 1.3, 1.5, 1.2, 1.0, 1.4])
        
        # Probabilities with high variance
        mu_probs = np.array([0.05, 0.03, 0.07, 0.04, 0.08, 0.05, 0.02, 0.06, 0.50, 0.10])
        sigma_probs = np.array([0.06, 0.04, 0.07, 0.05, 0.08, 0.06, 0.03, 0.07, 0.12, 0.08])
        
    else:  # failure case
        # Confident but wrong
        true_label = 0  # airplane
        
        # Logits: confident on wrong class (cat instead of airplane)
        mu_logits = np.array([0.5, -1.5, -0.8, 3.2, -1.0, -0.5, -1.5, -2.0, -0.8, -1.2])
        sigma_logits = np.array([0.3, 0.4, 0.35, 0.2, 0.3, 0.35, 0.4, 0.35, 0.3, 0.35])
        
        # Probabilities: confident on cat (class 3)
        mu_probs = np.array([0.05, 0.02, 0.04, 0.80, 0.02, 0.03, 0.01, 0.01, 0.01, 0.01])
        sigma_probs = np.array([0.02, 0.01, 0.015, 0.025, 0.01, 0.012, 0.005, 0.005, 0.008, 0.008])
    
    # Normalize probabilities
    mu_probs = mu_probs / mu_probs.sum()
    
    return image, true_label, mu_logits, sigma_logits, mu_probs, sigma_probs


def main():
    """Test the 3-panel plotting function with synthetic examples."""
    from pathlib import Path
    
    output_dir = Path(__file__).parent.parent / "results" / "test_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("Testing 3-Panel Qualitative Example Plots")
    print("="*70)
    
    examples = [
        ('normal', 'High confidence correct prediction'),
        ('high_entropy', 'High entropy uncertain prediction'),
        ('high_variance', 'High variance prediction'),
        ('failure', 'Confident but incorrect prediction'),
    ]
    
    for example_type, description in examples:
        print(f"\n{'-'*70}")
        print(f"Example: {description}")
        print(f"{'-'*70}")
        
        image, true_label, mu_logits, sigma_logits, mu_probs, sigma_probs = \
            create_synthetic_example(example_type)
        
        save_path = output_dir / f"test_{example_type}.png"
        
        plot_prediction_with_logits(
            image, true_label, mu_logits, sigma_logits,
            mu_probs, sigma_probs, 
            image_idx=f"test_{example_type}",
            save_path=str(save_path)
        )
    
    print("\n" + "="*70)
    print(f"All test plots saved to: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
