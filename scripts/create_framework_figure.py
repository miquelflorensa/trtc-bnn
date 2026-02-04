#!/usr/bin/env python3
"""
MM-Remax Framework Figure for Paper.

Creates a 3-column figure showing:
1. Input image + Prior logits (3 Gaussians in one plot)
2. MM-Remax transformation -> Output probabilities
3. Bayesian update with true label -> Posterior logits
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from scipy.stats import norm
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.methods.mm_remax import mm_remax
from src.methods.mc_remax import mc_remax

# Configuration
plt.rcParams.update({
    'text.usetex': False,
    'font.family': 'serif',
    'font.size': 10,
})

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c']
CLASS_NAMES = ['Cat', 'Dog', 'Bird']
MU_Z_PRIOR = np.array([2.0, -0.5, -1.5])
SIGMA_Z_SQ_PRIOR = np.array([0.8, 1.0, 1.2])
TRUE_CLASS = 0
Y_TRUE = np.array([1.0, 0.0, 0.0])
SIGMA_Y = 0.1


def plot_gaussians(ax, mus, sigmas, colors, labels, title, x_range=None, alpha=0.3):
    if x_range is None:
        x_min = min(mus) - 3 * max(np.sqrt(sigmas))
        x_max = max(mus) + 3 * max(np.sqrt(sigmas))
        x_range = (x_min, x_max)
    x = np.linspace(x_range[0], x_range[1], 300)
    for mu, sigma_sq, color, label in zip(mus, sigmas, colors, labels):
        sigma = np.sqrt(sigma_sq)
        y = norm.pdf(x, mu, sigma)
        ax.plot(x, y, color=color, linewidth=2, label=label)
        ax.fill_between(x, y, alpha=alpha, color=color)
    ax.set_title(title, fontweight='bold', fontsize=11)
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.set_xlim(x_range)
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def bayesian_update(mu_z_prior, sigma_z_sq_prior, mu_y, sigma_y_sq, cov_z_y, y_obs):
    innovation = y_obs - mu_y
    kalman_gain = cov_z_y / (sigma_y_sq + 1e-10)
    mu_z_post = mu_z_prior + kalman_gain * innovation
    sigma_z_sq_post = sigma_z_sq_prior - kalman_gain * cov_z_y
    sigma_z_sq_post = np.maximum(sigma_z_sq_post, 1e-6)
    return mu_z_post, sigma_z_sq_post


def create_mm_remax_figure():
    fig = plt.figure(figsize=(16, 5))
    fig.suptitle('MM-Remax Framework: Uncertainty Propagation & Bayesian Update',
                 fontsize=14, fontweight='bold', y=0.98)

    # Layout: Image | Prior Logits | Output Probs | Posterior Logits
    ax_image = fig.add_axes([0.02, 0.15, 0.10, 0.65])  # Image on the left
    ax_prior = fig.add_axes([0.14, 0.15, 0.24, 0.65])  # Prior logits next to image
    ax_output = fig.add_axes([0.42, 0.15, 0.24, 0.65])  # Output probs
    ax_posterior = fig.add_axes([0.70, 0.15, 0.24, 0.65])  # Posterior logits

    # Input image
    np.random.seed(42)
    img = np.random.rand(32, 32, 3) * 0.3 + 0.3
    img[10:22, 10:22, :] = 0.6
    ax_image.imshow(img)
    ax_image.set_title('Input', fontweight='bold', fontsize=10)
    ax_image.axis('off')
    ax_image.text(0.5, -0.08, '(Cat)', transform=ax_image.transAxes, ha='center', fontsize=9, style='italic')

    labels = [f'{CLASS_NAMES[i]}: mu={MU_Z_PRIOR[i]:.1f}' for i in range(3)]
    plot_gaussians(ax_prior, MU_Z_PRIOR, SIGMA_Z_SQ_PRIOR, COLORS, labels,
                   'Prior Logit Distributions p(z)', x_range=(-4, 6))

    result = mm_remax(MU_Z_PRIOR, SIGMA_Z_SQ_PRIOR)
    mu_a = result['mu_a']
    sigma_a_sq = result['sigma_a_sq']
    cov_z_a = result['cov_z_a']

    print("MM-Remax Results:")
    print(f"  mu_a: {mu_a}")
    print(f"  sigma_a_sq: {sigma_a_sq}")

    labels_out = [f'{CLASS_NAMES[i]}: mu={mu_a[i]:.3f}' for i in range(3)]
    plot_gaussians(ax_output, mu_a, sigma_a_sq, COLORS, labels_out,
                   'Output Probability Distributions p(a)', x_range=(-0.2, 1.2))
    ax_output.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax_output.axvline(x=1, color='gray', linestyle='--', alpha=0.5)

    # Bayesian update
    sigma_y_sq_total = sigma_a_sq + SIGMA_Y**2
    mu_z_post, sigma_z_sq_post = bayesian_update(MU_Z_PRIOR, SIGMA_Z_SQ_PRIOR, mu_a, sigma_y_sq_total, cov_z_a, Y_TRUE)

    print(f"\nPosterior mu_z: {mu_z_post}")
    print(f"Posterior sigma_z_sq: {sigma_z_sq_post}")

    labels_post = [f'{CLASS_NAMES[i]}: mu={mu_z_post[i]:.2f}' for i in range(3)]
    plot_gaussians(ax_posterior, mu_z_post, sigma_z_sq_post, COLORS, labels_post,
                   'Posterior Logit Distributions p(z|y)', x_range=(-4, 6))

    # Arrows - smaller size to not cover labels
    fig.patches.append(FancyArrowPatch((0.385, 0.47), (0.415, 0.47), transform=fig.transFigure,
                                        arrowstyle='-|>', mutation_scale=12, linewidth=1.5, color='#2C3E50'))
    fig.text(0.40, 0.52, 'MM-Remax', ha='center', va='bottom', fontsize=9, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#AED6F1', edgecolor='#2980B9', linewidth=1.5))

    fig.patches.append(FancyArrowPatch((0.665, 0.47), (0.695, 0.47), transform=fig.transFigure,
                                        arrowstyle='-|>', mutation_scale=12, linewidth=1.5, color='#2C3E50'))
    fig.text(0.68, 0.52, 'Bayesian\nUpdate', ha='center', va='bottom', fontsize=9, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#F9E79F', edgecolor='#D4AC0D', linewidth=1.5))

    output_dir = Path(__file__).parent.parent / 'figures'
    output_dir.mkdir(exist_ok=True)
    fig.savefig(output_dir / 'mm_remax_framework.pdf', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(output_dir / 'mm_remax_framework.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nSaved to {output_dir}")
    plt.show()
    return fig


def create_mm_remax_figure_with_mc():
    """Create a 2-row figure: MM-Remax on top, MC-Remax on bottom."""
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle('Remax Framework: MM vs MC Comparison',
                 fontsize=14, fontweight='bold', y=0.98)

    # Row 1: MM-Remax
    ax_image1 = fig.add_axes([0.02, 0.55, 0.10, 0.35])
    ax_prior1 = fig.add_axes([0.14, 0.55, 0.24, 0.35])
    ax_output1 = fig.add_axes([0.42, 0.55, 0.24, 0.35])
    ax_posterior1 = fig.add_axes([0.70, 0.55, 0.24, 0.35])

    # Row 2: MC-Remax
    ax_image2 = fig.add_axes([0.02, 0.08, 0.10, 0.35])
    ax_prior2 = fig.add_axes([0.14, 0.08, 0.24, 0.35])
    ax_output2 = fig.add_axes([0.42, 0.08, 0.24, 0.35])
    ax_posterior2 = fig.add_axes([0.70, 0.08, 0.24, 0.35])

    # image is in figures/1074_cat.png
    #load image
    img_path = Path(__file__).parent.parent / 'figures' / '1074_cat.png'
    img = plt.imread(img_path)  
    

    for ax_img in [ax_image1, ax_image2]:
        ax_img.imshow(img)
        ax_img.axis('off')
        ax_img.text(0.5, -0.08, '(Cat)', transform=ax_img.transAxes, ha='center', fontsize=9, style='italic')

    ax_image1.set_title('Input', fontweight='bold', fontsize=10)
    ax_image2.set_title('Input', fontweight='bold', fontsize=10)

    # Prior logits (same for both)
    labels = [f'{CLASS_NAMES[i]}: mu={MU_Z_PRIOR[i]:.1f}' for i in range(3)]
    plot_gaussians(ax_prior1, MU_Z_PRIOR, SIGMA_Z_SQ_PRIOR, COLORS, labels,
                   'Prior Logit Distributions p(z)', x_range=(-4, 6))
    plot_gaussians(ax_prior2, MU_Z_PRIOR, SIGMA_Z_SQ_PRIOR, COLORS, labels,
                   'Prior Logit Distributions p(z)', x_range=(-4, 6))

    # Compute both methods first to determine common axis ranges
    result_mm = mm_remax(MU_Z_PRIOR, SIGMA_Z_SQ_PRIOR)
    mu_a_mm = result_mm['mu_a']
    sigma_a_sq_mm = result_mm['sigma_a_sq']
    cov_z_a_mm = result_mm['cov_z_a']

    result_mc = mc_remax(MU_Z_PRIOR, SIGMA_Z_SQ_PRIOR, n_samples=50000)
    mu_a_mc = result_mc['mu_a']
    sigma_a_sq_mc = result_mc['sigma_a_sq']
    cov_z_a_mc = result_mc['cov_z_a']

    # Compute posteriors for both
    sigma_y_sq_mm = sigma_a_sq_mm + SIGMA_Y**2
    mu_z_post_mm, sigma_z_sq_post_mm = bayesian_update(MU_Z_PRIOR, SIGMA_Z_SQ_PRIOR, mu_a_mm, sigma_y_sq_mm, cov_z_a_mm, Y_TRUE)

    sigma_y_sq_mc = sigma_a_sq_mc + SIGMA_Y**2
    mu_z_post_mc, sigma_z_sq_post_mc = bayesian_update(MU_Z_PRIOR, SIGMA_Z_SQ_PRIOR, mu_a_mc, sigma_y_sq_mc, cov_z_a_mc, Y_TRUE)

    # Fixed axis ranges for consistency
    logit_range = (-4, 6)
    prob_range = (-0.2, 1.2)

    # Compute common y-axis limits for probability plots
    x_prob = np.linspace(prob_range[0], prob_range[1], 300)
    max_y_prob = 0
    for mu, sigma_sq in zip(np.concatenate([mu_a_mm, mu_a_mc]), np.concatenate([sigma_a_sq_mm, sigma_a_sq_mc])):
        y = norm.pdf(x_prob, mu, np.sqrt(sigma_sq))
        max_y_prob = max(max_y_prob, np.max(y))

    # Compute common y-axis limits for posterior plots
    x_logit = np.linspace(logit_range[0], logit_range[1], 300)
    max_y_post = 0
    for mu, sigma_sq in zip(np.concatenate([mu_z_post_mm, mu_z_post_mc]), np.concatenate([sigma_z_sq_post_mm, sigma_z_sq_post_mc])):
        y = norm.pdf(x_logit, mu, np.sqrt(sigma_sq))
        max_y_post = max(max_y_post, np.max(y))

    # Row 1: MM-Remax
    labels_mm = [f'{CLASS_NAMES[i]}: mu={mu_a_mm[i]:.3f}' for i in range(3)]
    plot_gaussians(ax_output1, mu_a_mm, sigma_a_sq_mm, COLORS, labels_mm,
                   'MM-Remax: Output p(a)', x_range=prob_range)
    ax_output1.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax_output1.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
    ax_output1.set_ylim(0, max_y_prob * 1.1)

    labels_post_mm = [f'{CLASS_NAMES[i]}: mu={mu_z_post_mm[i]:.2f}' for i in range(3)]
    plot_gaussians(ax_posterior1, mu_z_post_mm, sigma_z_sq_post_mm, COLORS, labels_post_mm,
                   'MM: Posterior p(z|y)', x_range=logit_range)
    ax_posterior1.set_ylim(0, max_y_post * 1.1)

    # Row 2: MC-Remax
    labels_mc = [f'{CLASS_NAMES[i]}: mu={mu_a_mc[i]:.3f}' for i in range(3)]
    plot_gaussians(ax_output2, mu_a_mc, sigma_a_sq_mc, COLORS, labels_mc,
                   'MC-Remax: Output p(a)', x_range=prob_range)
    ax_output2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax_output2.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
    ax_output2.set_ylim(0, max_y_prob * 1.1)

    labels_post_mc = [f'{CLASS_NAMES[i]}: mu={mu_z_post_mc[i]:.2f}' for i in range(3)]
    plot_gaussians(ax_posterior2, mu_z_post_mc, sigma_z_sq_post_mc, COLORS, labels_post_mc,
                   'MC: Posterior p(z|y)', x_range=logit_range)
    ax_posterior2.set_ylim(0, max_y_post * 1.1)

    # Arrows Row 1 (MM)
    fig.patches.append(FancyArrowPatch((0.385, 0.72), (0.415, 0.72), transform=fig.transFigure,
                                        arrowstyle='-|>', mutation_scale=12, linewidth=1.5, color='#2C3E50'))
    fig.text(0.40, 0.76, 'MM-Remax', ha='center', va='bottom', fontsize=9, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#AED6F1', edgecolor='#2980B9', linewidth=1.5))

    fig.patches.append(FancyArrowPatch((0.665, 0.72), (0.695, 0.72), transform=fig.transFigure,
                                        arrowstyle='-|>', mutation_scale=12, linewidth=1.5, color='#2C3E50'))
    fig.text(0.68, 0.76, 'Bayesian\nUpdate', ha='center', va='bottom', fontsize=9, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#F9E79F', edgecolor='#D4AC0D', linewidth=1.5))

    # Arrows Row 2 (MC)
    fig.patches.append(FancyArrowPatch((0.385, 0.25), (0.415, 0.25), transform=fig.transFigure,
                                        arrowstyle='-|>', mutation_scale=12, linewidth=1.5, color='#2C3E50'))
    fig.text(0.40, 0.29, 'MC-Remax', ha='center', va='bottom', fontsize=9, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#D5F5E3', edgecolor='#27AE60', linewidth=1.5))

    fig.patches.append(FancyArrowPatch((0.665, 0.25), (0.695, 0.25), transform=fig.transFigure,
                                        arrowstyle='-|>', mutation_scale=12, linewidth=1.5, color='#2C3E50'))
    fig.text(0.68, 0.29, 'Bayesian\nUpdate', ha='center', va='bottom', fontsize=9, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#F9E79F', edgecolor='#D4AC0D', linewidth=1.5))

    # Row labels
    fig.text(0.005, 0.72, 'MM\n(Analytical)', ha='left', va='center', fontsize=10, fontweight='bold', rotation=90)
    fig.text(0.005, 0.25, 'MC\n(Sampling)', ha='left', va='center', fontsize=10, fontweight='bold', rotation=90)

    output_dir = Path(__file__).parent.parent / 'figures'
    output_dir.mkdir(exist_ok=True)
    fig.savefig(output_dir / 'mm_remax_framework_with_mc.pdf', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(output_dir / 'mm_remax_framework_with_mc.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nSaved comparison figure to {output_dir}")
    plt.show()
    return fig


if __name__ == "__main__":
    print("Creating MM-Remax only figure...")
    create_mm_remax_figure()
    print("\nCreating MM vs MC comparison figure...")
    create_mm_remax_figure_with_mc()
