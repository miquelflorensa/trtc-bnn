#!/usr/bin/env python3
"""
MM-Remax Framework Figure - Vertical Schematic Version for 2-column paper.

Creates a clean, vertical flowchart-style figure suitable for a single column.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.image import imread
from scipy.stats import norm
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.methods.mm_remax import mm_remax
from src.methods.mc_remax import mc_remax
from src.methods.mm_softmax import mm_softmax
from src.methods.mc_softmax import mc_softmax

# Configuration
plt.rcParams.update({
    'text.usetex': False,
    'font.family': 'serif',
    'font.size': 9,
    'axes.linewidth': 0.5,
})

# Colors
COLOR_CAT = '#2E86AB'      # Blue
COLOR_DOG = '#E94F37'      # Red  
COLOR_BIRD = '#F39C12'     # Orange
COLOR_ARROW = '#2C3E50'    # Arrow color
COLOR_REMAX = '#AED6F1'    # Light blue for Remax box
COLOR_UPDATE = '#F9E79F'   # Light yellow for update box

COLORS = [COLOR_CAT, COLOR_DOG, COLOR_BIRD]
CLASS_NAMES = ['Cat', 'Dog', 'Bird']

# Prior values
MU_Z_PRIOR = np.array([-0.2, 1.3, -1.0])
SIGMA_Z_SQ_PRIOR = np.array([1.0, 0.6, 3.0])


def draw_arrow(ax, start, end, label=None, label_pos='right', color=COLOR_ARROW, box_color=None, label_on_arrow=False):
    """Draw a styled arrow between two points.
    
    Args:
        label_on_arrow: If True, place the label directly on the arrow (centered).
                       If False, place it to the side with offset (default behavior).
    """
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle='-|>',
        mutation_scale=20,
        linewidth=2.5,
        color=color,
        connectionstyle='arc3,rad=0'
    )
    ax.add_patch(arrow)
    
    if label:
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        offset_y = 0
        
        if label_on_arrow:
            # Place label centered on the arrow
            offset_x = 0
            offset_y = 0.01
            ha = 'center'
        else:
            # Place label to the side with offset
            if label_pos == 'right':
                offset_x = 0.18
                ha = 'left'
            elif label_pos == 'center':
                offset_x = 0
                ha = 'center'
            else:
                offset_x = -0.18
                ha = 'right'
        
        if box_color:
            ax.text(mid_x + offset_x, mid_y + offset_y, label, fontsize=9, fontweight='bold',
                   ha=ha, va='center',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor=box_color, 
                            edgecolor=COLOR_ARROW, linewidth=2))
        else:
            ax.text(mid_x + offset_x, mid_y, label, fontsize=9, fontweight='bold',
                   ha=ha, va='center')


def draw_probability_bars(ax, probs, stds, y_center, bar_height=0.05, width=0.65):
    """Draw horizontal probability bars with uncertainty."""
    n_classes = len(probs)
    spacing = bar_height * 1.4
    y_positions = [y_center + spacing, y_center, y_center - spacing]
    
    x_left = 0.5 - width/2
    
    for i, (p, std, y_pos) in enumerate(zip(probs, stds, y_positions)):
        # Main bar
        bar_width = p * width
        rect = mpatches.FancyBboxPatch(
            (x_left, y_pos - bar_height/2), bar_width, bar_height,
            boxstyle="round,pad=0.01,rounding_size=0.01",
            facecolor=COLORS[i], edgecolor='none', alpha=0.85
        )
        ax.add_patch(rect)
        
        # Uncertainty whisker
        x_end = x_left + bar_width
        x_err_left = max(x_left, x_end - std * width)
        x_err_right = min(x_left + width, x_end + std * width)
        ax.plot([x_err_left, x_err_right], [y_pos, y_pos], 
               color=COLORS[i], linewidth=2, alpha=0.5)
        ax.plot([x_err_left, x_err_left], [y_pos - bar_height/2.5, y_pos + bar_height/2.5],
               color=COLORS[i], linewidth=2, alpha=0.5)
        ax.plot([x_err_right, x_err_right], [y_pos - bar_height/2.5, y_pos + bar_height/2.5],
               color=COLORS[i], linewidth=2, alpha=0.5)
        
        # Label
        ax.text(x_left - 0.03, y_pos, CLASS_NAMES[i], fontsize=8, ha='right', va='center',
               color=COLORS[i], fontweight='bold')
        ax.text(x_left + width + 0.03, y_pos, f'{p:.0%}', fontsize=8, ha='left', va='center')


def draw_gaussians_inline(ax, mus, sigmas_sq, y_center, x_left, width, height):
    """Draw Gaussian distributions inline within the figure."""
    x_range = (-4, 5)
    x = np.linspace(x_range[0], x_range[1], 200)
    
    # Create a mini axes for the gaussians
    ax_gauss = ax.inset_axes([x_left, y_center - height/2, width, height], transform=ax.transData)
    
    for i, (mu, sigma_sq) in enumerate(zip(mus, sigmas_sq)):
        sigma = np.sqrt(sigma_sq)
        y = norm.pdf(x, mu, sigma)
        ax_gauss.plot(x, y, color=COLORS[i], linewidth=1.8, label=CLASS_NAMES[i])
        ax_gauss.fill_between(x, y, alpha=0.3, color=COLORS[i])
    
    ax_gauss.set_xlim(x_range)
    ax_gauss.set_ylim(bottom=0)
    ax_gauss.spines['top'].set_visible(False)
    ax_gauss.spines['right'].set_visible(False)
    ax_gauss.spines['left'].set_visible(False)
    ax_gauss.set_yticks([])
    ax_gauss.tick_params(axis='x', labelsize=6)
    # Move legend to the right using bbox_to_anchor
    # (1.0, 1.0) means upper right corner, increase first value to move right
    ax_gauss.legend(loc='upper left', bbox_to_anchor=(0.8, 1.0), fontsize=6, framealpha=0.9)
    
    return ax_gauss


def create_vertical_figure():
    """Create a clean vertical schematic figure."""
    
    # Compute MM-Remax results
    result = mm_remax(MU_Z_PRIOR, SIGMA_Z_SQ_PRIOR)
    mu_a = result['mu_a']
    sigma_a_sq = result['sigma_a_sq']
    sigma_a = np.sqrt(sigma_a_sq)
    
    # Bayesian update (observe cat = class 0)
    Y_TRUE = np.array([1.0, 0.0, 0.0])
    SIGMA_Y = 0.2
    cov_z_a = result['cov_z_a']
    sigma_y_sq = sigma_a_sq + SIGMA_Y**2
    innovation = Y_TRUE - mu_a
    kalman_gain = cov_z_a / (sigma_y_sq + 1e-10)
    mu_z_post = MU_Z_PRIOR + kalman_gain * innovation
    sigma_z_sq_post = SIGMA_Z_SQ_PRIOR - kalman_gain * cov_z_a
    sigma_z_sq_post = np.maximum(sigma_z_sq_post, 1e-6)
    
    # Posterior probabilities (run remax on posterior)
    result_post = mm_remax(mu_z_post, sigma_z_sq_post)
    mu_a_post = result_post['mu_a']
    sigma_a_post = np.sqrt(result_post['sigma_a_sq'])
    
    # Create figure - single column width, taller for more spacing
    fig, ax = plt.subplots(figsize=(3.5, 7.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # ==========================================
    # CONFIGURATION - Adjust these positions
    # ==========================================
    gauss_height = 0.16
    
    # Vertical positions (0-1 scale, top to bottom)
    y_image_bottom = 0.8
    y_prior = 0.68
    y_output = 0.38
    y_posterior = 0.06
    
    # Arrow positions - YOU CAN ADJUST THESE
    # Each arrow defined as (start_y, end_y)
    arrow1_start = 0.87   # Below image
    arrow1_end = 0.76     # Above prior gaussians
    
    arrow2_start = 0.56   # Below prior gaussians
    arrow2_end = 0.45     # Above output bars

    arrow3_start = 0.26   # Below output bars
    arrow3_end = 0.15     # Above posterior gaussians
    # ==========================================
    
    # === 1. Input Image ===
    img_path = Path(__file__).parent.parent / 'figures' / '1074_cat.png'
    img_width_inches = 1.2  # Width in inches
    fig_width = 3.5  # Figure width in inches
    fig_height = 7.5  # Figure height in inches
    
    # Calculate image position to center it (in figure coordinates)
    img_width_fig = img_width_inches / fig_width
    img_height_fig = img_width_inches / fig_height  # Square image
    img_left = (1 - img_width_fig) / 2  # Center horizontally
    img_bottom_fig = y_image_bottom  # Position from bottom of figure
    
    if img_path.exists():
        img = imread(img_path)
        ax_img = fig.add_axes([img_left, img_bottom_fig, img_width_fig, img_height_fig])
        ax_img.imshow(img)
        ax_img.axis('off')
    else:
        ax.text(0.5, y_image_bottom + 0.08, '[Cat Image]', fontsize=10, ha='center', va='center', style='italic')
    
    # === 2. Prior Logits (Gaussian distributions) ===
    # Title on the left, rotated 90 degrees
    ax.text(0.02, y_prior, 'Prior Logits', fontsize=10, ha='center', va='center', 
            fontweight='bold', rotation=90)
    draw_gaussians_inline(ax, MU_Z_PRIOR, SIGMA_Z_SQ_PRIOR, y_prior, 0.12, 0.78, gauss_height)
    
    # === 3. Output Probabilities (horizontal bars) ===
    # Title on the left, rotated 90 degrees
    ax.text(0.02, y_output, 'Output\nProbabilities', fontsize=9, ha='center', va='center', 
            fontweight='bold', rotation=90)
    draw_probability_bars(ax, mu_a, sigma_a, y_output - 0.02, bar_height=0.035, width=0.55)
    
    # === 4. Posterior Logits (Gaussian distributions) ===
    # Title on the left, rotated 90 degrees
    ax.text(0.02, y_posterior, 'Posterior Logits', fontsize=10, ha='center', va='center', 
            fontweight='bold', rotation=90)
    draw_gaussians_inline(ax, mu_z_post, sigma_z_sq_post, y_posterior, 0.12, 0.78, gauss_height)
    
    # ==========================================
    # DRAW ALL ARROWS ON TOP (after all content)
    # All arrows are the same size now
    # Labels are placed directly on the arrows for better flow
    # ==========================================
    
    # Arrow 1: Image -> Prior (with BNN label)
    draw_arrow(ax, (0.5, arrow1_start), (0.5, arrow1_end),
               label='BNN', box_color="#B7E6B5", label_on_arrow=True)
    
    # Arrow 2: Prior -> Output (with MM-Remax label)
    draw_arrow(ax, (0.5, arrow2_start), (0.5, arrow2_end), 
               label='MM-Remax', box_color=COLOR_REMAX, label_on_arrow=True)
    
    # Arrow 3: Output -> Posterior (with Observe y = Cat label)
    draw_arrow(ax, (0.5, arrow3_start), (0.5, arrow3_end),
               label='Observe y = Cat', box_color=COLOR_UPDATE, label_on_arrow=True)
    
    # Save
    output_dir = Path(__file__).parent.parent / 'figures'
    output_dir.mkdir(exist_ok=True)
    
    fig.savefig(output_dir / 'mm_remax_schematic.pdf', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    fig.savefig(output_dir / 'mm_remax_schematic.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    print(f"Saved schematic figure to {output_dir}")
    print(f"\nPrior logits (mu_z): {MU_Z_PRIOR}")
    print(f"Prior probabilities (mu_a): {mu_a}")
    print(f"Posterior logits (mu_z_post): {mu_z_post}")
    print(f"Posterior probabilities (mu_a_post): {mu_a_post}")
    
    plt.show()
    return fig


def create_comparison_figure():
    """Create a side-by-side comparison: MM-Remax (left) vs MC-Remax (right)."""
    
    # Compute both methods
    result_mm = mm_remax(MU_Z_PRIOR, SIGMA_Z_SQ_PRIOR)
    mu_a_mm = result_mm['mu_a']
    sigma_a_sq_mm = result_mm['sigma_a_sq']
    sigma_a_mm = np.sqrt(sigma_a_sq_mm)
    cov_z_a_mm = result_mm['cov_z_a']
    
    result_mc = mc_remax(MU_Z_PRIOR, SIGMA_Z_SQ_PRIOR, n_samples=50000)
    mu_a_mc = result_mc['mu_a']
    sigma_a_sq_mc = result_mc['sigma_a_sq']
    sigma_a_mc = np.sqrt(sigma_a_sq_mc)
    cov_z_a_mc = result_mc['cov_z_a']
    
    # Bayesian updates for both methods
    Y_TRUE = np.array([1.0, 0.0, 0.0])
    SIGMA_Y = 0.2
    
    # MM-Remax update
    sigma_y_sq_mm = sigma_a_sq_mm + SIGMA_Y**2
    innovation_mm = Y_TRUE - mu_a_mm
    kalman_gain_mm = cov_z_a_mm / (sigma_y_sq_mm + 1e-10)
    mu_z_post_mm = MU_Z_PRIOR + kalman_gain_mm * innovation_mm
    sigma_z_sq_post_mm = SIGMA_Z_SQ_PRIOR - kalman_gain_mm * cov_z_a_mm
    sigma_z_sq_post_mm = np.maximum(sigma_z_sq_post_mm, 1e-6)
    result_post_mm = mm_remax(mu_z_post_mm, sigma_z_sq_post_mm)
    mu_a_post_mm = result_post_mm['mu_a']
    sigma_a_post_mm = np.sqrt(result_post_mm['sigma_a_sq'])
    
    # MC-Remax update
    sigma_y_sq_mc = sigma_a_sq_mc + SIGMA_Y**2
    innovation_mc = Y_TRUE - mu_a_mc
    kalman_gain_mc = cov_z_a_mc / (sigma_y_sq_mc + 1e-10)
    mu_z_post_mc = MU_Z_PRIOR + kalman_gain_mc * innovation_mc
    sigma_z_sq_post_mc = SIGMA_Z_SQ_PRIOR - kalman_gain_mc * cov_z_a_mc
    sigma_z_sq_post_mc = np.maximum(sigma_z_sq_post_mc, 1e-6)
    result_post_mc = mc_remax(mu_z_post_mc, sigma_z_sq_post_mc, n_samples=50000)
    mu_a_post_mc = result_post_mc['mu_a']
    sigma_a_post_mc = np.sqrt(result_post_mc['sigma_a_sq'])
    
    # Create figure - two columns side by side
    fig = plt.figure(figsize=(7, 7.5))
    
    # Left side: MM-Remax
    ax_left = fig.add_axes([0.05, 0.05, 0.45, 0.90])  # Added bottom margin
    ax_left.set_xlim(0, 1)
    ax_left.set_ylim(0, 1)
    ax_left.axis('off')
    
    # Right side: MC-Remax
    ax_right = fig.add_axes([0.52, 0.05, 0.45, 0.90])  # Added bottom margin
    ax_right.set_xlim(0, 1)
    ax_right.set_ylim(0, 1)
    ax_right.axis('off')
    
    # Configuration
    gauss_height = 0.16
    y_image_bottom = 0.76  # Moved up
    y_prior = 0.6
    y_output = 0.36
    y_posterior = 0.12
    
    arrow1_start = 0.75  # Adjusted
    arrow1_end = 0.69
    arrow2_start = 0.50
    arrow2_end = 0.44
    arrow3_start = 0.28
    arrow3_end = 0.22
    
    # Load image
    img_path = Path(__file__).parent.parent / 'figures' / '1074_cat.png'
    
    # Helper function to draw one column
    def draw_column(ax, mu_a, sigma_a, mu_z_post, sigma_z_sq_post, mu_a_post, sigma_a_post, 
                    method_label, method_color, ax_offset=0):
        # Image (smaller)
        if img_path.exists():
            img = imread(img_path)
            img_width = 0.22  # Reduced from 0.34
            img_height = img_width * 0.85  # Slightly compressed
            img_left = ax_offset + (0.45 - img_width) / 2  # Center in column width
            img_bottom_fig = 0.74  # Position in figure coordinates
            ax_img = fig.add_axes([img_left, img_bottom_fig, img_width, img_height])
            ax_img.imshow(img)
            ax_img.axis('off')
        
        # Prior Logits
        ax.text(0.02, y_prior, 'Prior Logits', fontsize=10, ha='center', va='center',
                fontweight='bold', rotation=90)
        draw_gaussians_inline(ax, MU_Z_PRIOR, SIGMA_Z_SQ_PRIOR, y_prior, 0.12, 0.78, gauss_height)
        
        # Output Probabilities
        ax.text(0.02, y_output, 'Output\nProbabilities', fontsize=9, ha='center', va='center',
                fontweight='bold', rotation=90)
        draw_probability_bars(ax, mu_a, sigma_a, y_output - 0.02, bar_height=0.035, width=0.55)
        
        # Posterior Logits
        ax.text(0.02, y_posterior, 'Posterior Logits', fontsize=10, ha='center', va='center',
                fontweight='bold', rotation=90)
        draw_gaussians_inline(ax, mu_z_post, sigma_z_sq_post, y_posterior, 0.12, 0.78, gauss_height)
        
        # Arrows
        draw_arrow(ax, (0.5, arrow1_start), (0.5, arrow1_end),
                   label='BNN', label_pos='right', box_color='#E8E8E8')
        draw_arrow(ax, (0.5, arrow2_start), (0.5, arrow2_end),
                   label=method_label.split('\n')[0], label_pos='right', box_color=method_color)
        draw_arrow(ax, (0.5, arrow3_start), (0.5, arrow3_end),
                   label='Observe y = Cat', label_pos='right', box_color=COLOR_UPDATE)
    
    # Draw left column (MM-Remax)
    draw_column(ax_left, mu_a_mm, sigma_a_mm, mu_z_post_mm, sigma_z_sq_post_mm, 
               mu_a_post_mm, sigma_a_post_mm, 'MM-Remax\n(Analytical)', COLOR_REMAX, ax_offset=0.05)
    
    # Draw right column (MC-Remax)
    draw_column(ax_right, mu_a_mc, sigma_a_mc, mu_z_post_mc, sigma_z_sq_post_mc,
               mu_a_post_mc, sigma_a_post_mc, 'MC-Remax\n(Sampling)', '#D5F5E3', ax_offset=0.52)
    
    # Add method labels at the bottom using figure coordinates
    fig.text(0.275, 0.015, 'MM-Remax (Analytical)', fontsize=10, ha='center', va='bottom',
            fontweight='bold', bbox=dict(boxstyle='round,pad=0.4', facecolor=COLOR_REMAX,
                                        edgecolor=COLOR_ARROW, linewidth=2))
    fig.text(0.745, 0.015, 'MC-Remax (Sampling)', fontsize=10, ha='center', va='bottom',
            fontweight='bold', bbox=dict(boxstyle='round,pad=0.4', facecolor='#D5F5E3',
                                        edgecolor=COLOR_ARROW, linewidth=2))
    
    # Save
    output_dir = Path(__file__).parent.parent / 'figures'
    output_dir.mkdir(exist_ok=True)
    
    fig.savefig(output_dir / 'mm_mc_comparison_schematic.pdf', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    fig.savefig(output_dir / 'mm_mc_comparison_schematic.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    print(f"\nSaved comparison figure to {output_dir}")
    print(f"\nMM-Remax results:")
    print(f"  Prior probs: {mu_a_mm}")
    print(f"  Posterior probs: {mu_a_post_mm}")
    print(f"\nMC-Remax results:")
    print(f"  Prior probs: {mu_a_mc}")
    print(f"  Posterior probs: {mu_a_post_mc}")
    
    plt.show()
    return fig


if __name__ == "__main__":
    print("Creating MM-Remax only figure...")
    create_vertical_figure()
    print("\n" + "="*60)
    print("Creating MM vs MC comparison figure...")
    create_comparison_figure()
