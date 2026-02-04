"""
Generate individual plots from the BNN cycle diagram.
Each plot is saved separately without titles for flexible use in papers.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.image import imread
from scipy.stats import norm
import matplotlib.patches as mpatches
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.methods.mm_remax import mm_remax

# Configuration (same as main figure)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Colors
COLOR_CAT = '#2E86AB'      # Blue
COLOR_DOG = '#E94F37'      # Red  
COLOR_BIRD = '#F39C12'     # Orange
COLOR_ARROW = '#2C3E50'    # Arrow color
COLOR_REMAX = '#AED6F1'    # Light blue for Remax box
COLOR_UPDATE = '#F9E79F'   # Light yellow for update box
COLOR_BNN = '#B7E6B5'      # Light green for BNN
COLOR_PRIOR = '#FFE5B4'    # Light orange for prior
COLOR_POSTERIOR = '#CAFFBF' # Light green for posterior

COLORS = [COLOR_CAT, COLOR_DOG, COLOR_BIRD]
CLASS_NAMES = ['Cat', 'Dog', 'Bird']

# Prior values (same as in the single-column figure)
MU_Z_PRIOR = np.array([-0.2, 1.3, -1.0])
SIGMA_Z_SQ_PRIOR = np.array([1.0, 0.6, 3.0])

# Observation parameters
Y_TRUE = np.array([1.0, 0.0, 0.0])  # Cat is class 0
SIGMA_Y = 0.1  # Observation noise

# Plot configuration
GAUSS_X_RANGE = (-5, 5)
GAUSS_LINE_WIDTH = 1.8
GAUSS_ALPHA = 0.3

# Individual plot size (taller for horizontal error bars)
PLOT_SIZE = (4, 2.0)


def draw_probability_errorbar(ax, probs, stds):
    """Draw horizontal error bar plot (rotated 90 degrees from vertical)."""
    n_classes = len(probs)
    classes = np.arange(n_classes)
    
    # Plot horizontal error bars with vertical caps
    for i in classes:
        # Error bar horizontal line
        ax.errorbar(probs[i], i, xerr=stds[i], fmt='none', 
                   ecolor=COLORS[i], elinewidth=1.5, capsize=5)
        
        # Vertical lines at mean +/- std (caps)
        cap_height = 0.2
        ax.plot([probs[i] + stds[i], probs[i] + stds[i]], 
               [i - cap_height, i + cap_height], 
               COLORS[i], linewidth=1.5)
        ax.plot([probs[i] - stds[i], probs[i] - stds[i]], 
               [i - cap_height, i + cap_height], 
               COLORS[i], linewidth=1.5)
        
        # Square marker for mean
        ax.scatter(probs[i], i, color=COLORS[i], marker='s', s=30, 
                  zorder=3, edgecolor='k', linewidth=0.5)
    
    ax.set_yticks(classes)
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_ylim(-0.5, n_classes - 0.5)
    ax.set_xlim(-0.05, 1.25)  # Trim to end slightly after 1.0
    
    # Set 0 and 1 ticks on x-axis (no grid lines)
    ax.set_xticks([0.0, 1.0])
    ax.set_xticklabels(['0', '1'])
    
    # No grid lines
    ax.grid(False)
    # ax.set_xlabel('Probability')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Invert y-axis so Cat is at the top
    ax.invert_yaxis()


def draw_gaussians_inline(ax, mus, sigmas_sq, x_left, y_bottom, width, height):
    """Draw Gaussian distributions inline within the figure."""
    x = np.linspace(GAUSS_X_RANGE[0], GAUSS_X_RANGE[1], 200)
    
    # Create a mini axes for the gaussians
    ax_gauss = ax.inset_axes([x_left, y_bottom, width, height], transform=ax.transData)
    
    for i, (mu, sigma_sq) in enumerate(zip(mus, sigmas_sq)):
        sigma = np.sqrt(sigma_sq)
        y = norm.pdf(x, mu, sigma)
        ax_gauss.plot(x, y, color=COLORS[i], linewidth=GAUSS_LINE_WIDTH, label=CLASS_NAMES[i])
        ax_gauss.fill_between(x, y, alpha=GAUSS_ALPHA, color=COLORS[i])
    
    ax_gauss.set_xlim(GAUSS_X_RANGE)
    ax_gauss.set_ylim(bottom=0)
    ax_gauss.spines['top'].set_visible(False)
    ax_gauss.spines['right'].set_visible(False)
    ax_gauss.spines['left'].set_visible(False)
    ax_gauss.set_yticks([])
    
    # Set only 0 tick on x-axis
    ax_gauss.set_xticks([0])
    ax_gauss.set_xticklabels(['0'])
    ax_gauss.tick_params(axis='x', labelsize=6)
    ax_gauss.legend(loc='upper left', bbox_to_anchor=(0.8, 1.0), fontsize=6, framealpha=0.9)
    
    return ax_gauss


def create_prior_logits_plot(mu_z, sigma_z_sq, output_path):
    """Create Prior Logits plot."""
    fig, ax = plt.subplots(figsize=PLOT_SIZE)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.axis('off')
    
    # Draw gaussians
    draw_gaussians_inline(ax, mu_z, sigma_z_sq, 0.1, 0.15, 0.8, 0.7)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"Saved: {output_path}")


def create_prediction_plot(probs, stds, output_path):
    """Create Prediction probabilities plot."""
    fig, ax = plt.subplots(figsize=PLOT_SIZE)
    
    # Draw probability error bars
    draw_probability_errorbar(ax, probs, stds)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"Saved: {output_path}")


def create_ground_truth_plot(y_true, sigma_y, output_path):
    """Create Ground Truth observation plot."""
    fig, ax = plt.subplots(figsize=PLOT_SIZE)
    
    # Draw ground truth as error bars (only cat has value)
    draw_probability_errorbar(ax, y_true, [sigma_y, 0, 0])
    
    # Add observation error annotation
    # ax.text(0.5, -0.15, rf'obs error $\sigma_y={sigma_y}$', 
    #         fontsize=9, ha='center', va='top', fontweight='bold',
    #         transform=ax.transAxes)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"Saved: {output_path}")


def create_posterior_logits_plot(mu_z, sigma_z_sq, output_path):
    """Create Posterior Logits plot."""
    fig, ax = plt.subplots(figsize=PLOT_SIZE)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.axis('off')
    
    # Draw gaussians
    draw_gaussians_inline(ax, mu_z, sigma_z_sq, 0.1, 0.15, 0.8, 0.7)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"Saved: {output_path}")


def generate_all_individual_plots():
    """Generate all 4 individual plots."""
    
    # Compute MM-Remax results for prior
    result_prior = mm_remax(MU_Z_PRIOR, SIGMA_Z_SQ_PRIOR)
    mu_a_prior = result_prior['mu_a']
    sigma_a_sq_prior = result_prior['sigma_a_sq']
    sigma_a_prior = np.sqrt(sigma_a_sq_prior)
    cov_z_a = result_prior['cov_z_a']
    
    # Bayesian update (observe cat = class 0)
    sigma_y_sq = sigma_a_sq_prior + SIGMA_Y**2
    innovation = Y_TRUE - mu_a_prior
    kalman_gain = cov_z_a / (sigma_y_sq + 1e-10)
    mu_z_post = MU_Z_PRIOR + kalman_gain * innovation
    sigma_z_sq_post = SIGMA_Z_SQ_PRIOR - kalman_gain * cov_z_a
    sigma_z_sq_post = np.maximum(sigma_z_sq_post, 1e-6)
    
    # Compute MM-Remax results for posterior
    result_post = mm_remax(mu_z_post, sigma_z_sq_post)
    mu_a_post = result_post['mu_a']
    sigma_a_sq_post = result_post['sigma_a_sq']
    sigma_a_post = np.sqrt(sigma_a_sq_post)
    
    # Create output directory
    output_dir = Path(__file__).parent.parent / 'figures' / 'bnn_cycle_components'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating individual plots in {output_dir}...")
    print("=" * 60)
    
    # Generate each plot
    create_prior_logits_plot(MU_Z_PRIOR, SIGMA_Z_SQ_PRIOR, 
                             output_dir / '1_prior_logits.pdf')
    create_prior_logits_plot(MU_Z_PRIOR, SIGMA_Z_SQ_PRIOR, 
                             output_dir / '1_prior_logits.png')
    
    create_prediction_plot(mu_a_prior, sigma_a_prior, 
                          output_dir / '2_prediction.pdf')
    create_prediction_plot(mu_a_prior, sigma_a_prior, 
                          output_dir / '2_prediction.png')
    
    create_ground_truth_plot(Y_TRUE, SIGMA_Y, 
                            output_dir / '3_ground_truth.pdf')
    create_ground_truth_plot(Y_TRUE, SIGMA_Y, 
                            output_dir / '3_ground_truth.png')
    
    create_posterior_logits_plot(mu_z_post, sigma_z_sq_post, 
                                output_dir / '4_posterior_logits.pdf')
    create_posterior_logits_plot(mu_z_post, sigma_z_sq_post, 
                                output_dir / '4_posterior_logits.png')
    
    print("=" * 60)
    print("\nAll individual plots generated successfully!")
    print(f"\nData summary:")
    print(f"  Prior logits: {MU_Z_PRIOR}")
    print(f"  Prior probs:  {mu_a_prior}")
    print(f"  Posterior logits: {mu_z_post}")
    print(f"  Posterior probs:  {mu_a_post}")


if __name__ == '__main__':
    generate_all_individual_plots()
