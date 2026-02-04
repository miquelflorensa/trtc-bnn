"""
Create a comprehensive BNN cycle diagram showing forward and backward passes.
This figure illustrates the complete Bayesian inference cycle in a BNN.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.gridspec import GridSpec
from matplotlib.image import imread
from scipy.stats import norm
import matplotlib.patches as mpatches
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.methods.mm_remax import mm_remax

# ============================================================================
# CONFIGURATION SECTION - Easy to modify all sizes and positions
# ============================================================================

# Style configuration for PRD two-column paper
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

pt = 1./72.27
jour_sizes = {"PRD": {"onecol": 246.*pt, "twocol": 510.*pt}}
my_width = jour_sizes["PRD"]["twocol"]
golden = (1 + 5 ** 0.5) / 2
figsize = (my_width, my_width / golden)

# Colors (matching the other figure)
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
SIGMA_Y = 0.2  # Observation noise

# --- FIGURE LAYOUT CONFIGURATION ---
# GridSpec layout (rows, columns)
LAYOUT_ROWS = 5
LAYOUT_COLS = 3
HSPACE = 0.6  # Horizontal space between subplots
WSPACE = 0.0  # Vertical space between subplots (reduced from 0.4 to bring columns closer)
LEFT_MARGIN = 0.08
RIGHT_MARGIN = 0.92
TOP_MARGIN = 0.95
BOTTOM_MARGIN = 0.08

# --- COMPONENT SIZES AND POSITIONS ---
# Input image size (in the top center box)
IMAGE_WIDTH_RATIO = 1  # Relative to box width
IMAGE_HEIGHT_RATIO = 1 # Relative to box height

# Gaussian plot configuration
GAUSS_X_RANGE = (-4, 5)
GAUSS_LINE_WIDTH = 1.8
GAUSS_ALPHA = 0.3

# Probability bar configuration
BAR_HEIGHT = 0.035
BAR_WIDTH = 0.55
BAR_ALPHA = 0.85

# --- ARROW CONFIGURATION (Figure coordinates 0-1) ---
# All arrows use figure coordinates for easy positioning
# Format: (x_start, y_start, x_end, y_end)

# Forward pass arrows (green)
ARROW_INPUT_TO_BNN = (0.5, 0.88, 0.5, 0.83)
ARROW_BNN_TO_PRIOR = (0.5, 0.78, 0.23, 0.68)
ARROW_PRIOR_TO_REMAX = (0.23, 0.55, 0.23, 0.49)
ARROW_REMAX_TO_PRED = (0.23, 0.41, 0.23, 0.35)
ARROW_PRED_TO_OBS = (0.23, 0.21, 0.42, 0.12)

# Backward pass arrows (red)
ARROW_OBS_TO_GT = (0.58, 0.12, 0.77, 0.21)
ARROW_GT_TO_POST = (0.77, 0.35, 0.77, 0.55)
ARROW_POST_TO_BNN = (0.77, 0.68, 0.5, 0.78)

# Side annotation arrows
FORWARD_ARROW_START = (0.02, 0.7)
FORWARD_ARROW_END = (0.02, 0.3)
BACKWARD_ARROW_START = (0.98, 0.3)
BACKWARD_ARROW_END = (0.98, 0.7)

# Arrow styling
ARROW_MUTATION_SCALE = 25
ARROW_LINE_WIDTH = 2.5
SIDE_ARROW_WIDTH = 3
COLOR_FORWARD = '#2E7D32'   # Green
COLOR_BACKWARD = '#C62828'  # Red

# ============================================================================
# END OF CONFIGURATION SECTION
# ============================================================================


def draw_probability_bars(ax, probs, stds, y_center, bar_height=BAR_HEIGHT, width=BAR_WIDTH):
    """Draw horizontal probability bars with uncertainty (matching style from other figure)."""
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
            facecolor=COLORS[i], edgecolor='none', alpha=BAR_ALPHA
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


def draw_gaussians_inline(ax, mus, sigmas_sq, x_left, y_bottom, width, height):
    """Draw Gaussian distributions inline within the figure (matching style from other figure)."""
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
    ax_gauss.tick_params(axis='x', labelsize=6)
    ax_gauss.legend(loc='upper left', bbox_to_anchor=(0.8, 1.0), fontsize=6, framealpha=0.9)
    
    return ax_gauss


def draw_gaussian_plot(ax, mus, sigmas_sq, title, color_bg):
    """Draw overlapping Gaussian distributions in a subplot."""
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.axis('off')
    
    # Add title
    ax.text(0.5, 0.95, title, fontsize=10, ha='center', va='top', fontweight='bold')
    
    # Draw gaussians using the inline function
    draw_gaussians_inline(ax, mus, sigmas_sq, 0.1, 0.1, 0.8, 0.7)


def draw_probability_plot(ax, probs, stds, title, color_bg, show_obs_error=False):
    """Draw probability bar plot with error bars."""
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.axis('off')
    
    # Add title
    ax.text(0.5, 0.95, title, fontsize=10, ha='center', va='top', fontweight='bold')
    
    # Draw probability bars at y=0.5
    draw_probability_bars(ax, probs, stds, 0.5, bar_height=BAR_HEIGHT, width=BAR_WIDTH)
    
    # Add observation error annotation if needed
    if show_obs_error:
        ax.text(0.5, 0.15, rf'obs error $\sigma_y={SIGMA_Y}$', 
                fontsize=9, ha='center', fontweight='bold')


def draw_box_with_text(ax, xy, width, height, text, color, fontsize=11):
    """Draw a fancy box with centered text."""
    box = FancyBboxPatch(xy, width, height,
                         boxstyle="round,pad=0.05",
                         facecolor=color, edgecolor='black',
                         linewidth=2, transform=ax.transData)
    ax.add_patch(box)
    
    # Add text in the center of the box
    text_x = xy[0] + width / 2
    text_y = xy[1] + height / 2
    ax.text(text_x, text_y, text, fontsize=fontsize, ha='center', va='center',
            fontweight='bold')


def draw_curved_arrow(fig, start, end, label=None, color=COLOR_ARROW, 
                      connectionstyle='arc3,rad=0', label_offset=(0, 0)):
    """Draw an arrow in figure coordinates."""
    arrow = FancyArrowPatch(start, end,
                           transform=fig.transFigure,
                           arrowstyle='-|>',
                           mutation_scale=ARROW_MUTATION_SCALE,
                           linewidth=ARROW_LINE_WIDTH,
                           color=color,
                           connectionstyle=connectionstyle)
    fig.patches.append(arrow)
    
    if label:
        mid_x = (start[0] + end[0]) / 2 + label_offset[0]
        mid_y = (start[1] + end[1]) / 2 + label_offset[1]
        fig.text(mid_x, mid_y, label, fontsize=10, ha='center', va='center',
                fontweight='bold', bbox=dict(boxstyle='round,pad=0.4',
                facecolor='white', edgecolor=color, linewidth=2))


# ============================================================================
# MAIN FIGURE GENERATION
# ============================================================================

def create_bnn_cycle_figure():
    """Create the BNN cycle diagram with real MM-Remax calculations."""
    
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
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    
    # Create grid layout
    gs = GridSpec(LAYOUT_ROWS, LAYOUT_COLS, figure=fig, 
                  hspace=HSPACE, wspace=WSPACE,
                  left=LEFT_MARGIN, right=RIGHT_MARGIN, 
                  top=TOP_MARGIN, bottom=BOTTOM_MARGIN)
    
    # ========================================================================
    # TOP CENTER: Input Image and BNN
    # ========================================================================
    ax_top = fig.add_subplot(gs[0, 1])
    ax_top.set_xlim([0, 10])
    ax_top.set_ylim([0, 10])
    ax_top.axis('off')
    
    # Input image box with actual cat image
    img_path = Path(__file__).parent.parent / 'figures' / '1074_cat.png'
    if img_path.exists():
        img = imread(img_path)
        # Calculate image position within the box
        box_left, box_bottom, box_width, box_height = 2, 5, 6, 3.5
        img_width = box_width * IMAGE_WIDTH_RATIO
        img_height = box_height * IMAGE_HEIGHT_RATIO
        img_left = box_left + (box_width - img_width) / 2
        img_bottom = box_bottom + (box_height - img_height) / 2
        
        # Draw box
        box = FancyBboxPatch((box_left, box_bottom), box_width, box_height,
                             boxstyle="round,pad=0.05",
                             facecolor=COLOR_BNN, edgecolor='black',
                             linewidth=2, transform=ax_top.transData)
        ax_top.add_patch(box)
        
        # Add image
        ax_top.imshow(img, extent=[img_left, img_left + img_width, 
                                   img_bottom, img_bottom + img_height])
    else:
        print(f"Image file not found: {img_path}. Using placeholder box instead.")
        draw_box_with_text(ax_top, (2, 5), 6, 3.5, 'Input Image\n(CAT)', COLOR_BNN, fontsize=12)
    
    # BNN box
    draw_box_with_text(ax_top, (3, 0.5), 4, 2.5, 'BNN', COLOR_BNN, fontsize=12)
    
    # ========================================================================
    # LEFT COLUMN: Forward Pass
    # ========================================================================
    
    # Plot 1: Prior Logits
    ax_prior = fig.add_subplot(gs[1, 0])
    draw_gaussian_plot(ax_prior, MU_Z_PRIOR, SIGMA_Z_SQ_PRIOR, 
                       r'\textbf{Prior Logits} $p(z)$', COLOR_PRIOR)
    
    # Plot 2: MM-Remax box
    ax_remax = fig.add_subplot(gs[2, 0])
    ax_remax.set_xlim([0, 10])
    ax_remax.set_ylim([0, 10])
    ax_remax.axis('off')
    draw_box_with_text(ax_remax, (1, 3), 8, 4, 'MM-Remax', COLOR_REMAX, fontsize=11)
    
    # Plot 3: Prediction probabilities
    ax_pred = fig.add_subplot(gs[3, 0])
    draw_probability_plot(ax_pred, mu_a_prior, sigma_a_prior,
                          r'\textbf{Prediction} $p(y|x)$', COLOR_REMAX,
                          show_obs_error=False)
    
    # ========================================================================
    # BOTTOM CENTER: Observation
    # ========================================================================
    ax_obs_text = fig.add_subplot(gs[4, 1])
    ax_obs_text.set_xlim([0, 10])
    ax_obs_text.set_ylim([0, 10])
    ax_obs_text.axis('off')
    ax_obs_text.text(5, 5, r'Observe $y = \mathrm{Cat}$', fontsize=12,
                    ha='center', va='center', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.6', facecolor=COLOR_UPDATE,
                             edgecolor='black', linewidth=2))
    
    # ========================================================================
    # RIGHT COLUMN: Backward Pass
    # ========================================================================
    
    # Plot 4: Observation (ground truth with noise)
    ax_obs = fig.add_subplot(gs[3, 2])
    draw_probability_plot(ax_obs, Y_TRUE, [SIGMA_Y, 0, 0],
                          r'\textbf{Ground Truth} $y$', COLOR_UPDATE,
                          show_obs_error=True)
    
    # Plot 5: Posterior Logits
    ax_post = fig.add_subplot(gs[1, 2])
    draw_gaussian_plot(ax_post, mu_z_post, sigma_z_sq_post,
                       r'\textbf{Posterior Logits} $p(z|y)$', COLOR_POSTERIOR)
    
    # ========================================================================
    # ARROWS CONNECTING EVERYTHING (using configured positions)
    # ========================================================================
    
    # Forward pass arrows (green)
    draw_curved_arrow(fig, ARROW_BNN_TO_PRIOR[0:2], ARROW_BNN_TO_PRIOR[2:4], 
                     color=COLOR_FORWARD)
    draw_curved_arrow(fig, ARROW_PRIOR_TO_REMAX[0:2], ARROW_PRIOR_TO_REMAX[2:4], 
                     color=COLOR_FORWARD)
    draw_curved_arrow(fig, ARROW_REMAX_TO_PRED[0:2], ARROW_REMAX_TO_PRED[2:4], 
                     color=COLOR_FORWARD)
    draw_curved_arrow(fig, ARROW_PRED_TO_OBS[0:2], ARROW_PRED_TO_OBS[2:4], 
                     color=COLOR_FORWARD)
    
    # Backward pass arrows (red)
    draw_curved_arrow(fig, ARROW_OBS_TO_GT[0:2], ARROW_OBS_TO_GT[2:4], 
                     color=COLOR_BACKWARD)
    draw_curved_arrow(fig, ARROW_GT_TO_POST[0:2], ARROW_GT_TO_POST[2:4], 
                     color=COLOR_BACKWARD)
    draw_curved_arrow(fig, ARROW_POST_TO_BNN[0:2], ARROW_POST_TO_BNN[2:4], 
                     label='Gaussian\nInference', color=COLOR_BACKWARD,
                     label_offset=(0, 0.02))
    
    # Arrow from input to BNN (in ax_top coordinates, converted to figure)
    arrow_input_bnn = FancyArrowPatch((5, 5), (5, 3),
                                     arrowstyle='-|>', mutation_scale=20,
                                     linewidth=2.5, color=COLOR_ARROW)
    ax_top.add_patch(arrow_input_bnn)
    
    # ========================================================================
    # SIDE ANNOTATIONS: Forward and Backward
    # ========================================================================
    
    # Left side: Forward arrow
    fig.text(0.02, 0.5, r'\textbf{Forward}', fontsize=12, rotation=90,
            ha='center', va='center', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#A5D6A7',
                     edgecolor=COLOR_FORWARD, linewidth=2))
    fig.patches.append(FancyArrowPatch(FORWARD_ARROW_START, FORWARD_ARROW_END,
                                      transform=fig.transFigure,
                                      arrowstyle='-|>', mutation_scale=ARROW_MUTATION_SCALE,
                                      linewidth=SIDE_ARROW_WIDTH, color=COLOR_FORWARD))
    
    # Right side: Backward arrow
    fig.text(0.98, 0.5, r'\textbf{Backwards / Backprop}', fontsize=11, rotation=90,
            ha='center', va='center', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFCDD2',
                     edgecolor=COLOR_BACKWARD, linewidth=2))
    fig.patches.append(FancyArrowPatch(BACKWARD_ARROW_START, BACKWARD_ARROW_END,
                                      transform=fig.transFigure,
                                      arrowstyle='-|>', mutation_scale=ARROW_MUTATION_SCALE,
                                      linewidth=SIDE_ARROW_WIDTH, color=COLOR_BACKWARD))
    
    # ========================================================================
    # SAVE FIGURE
    # ========================================================================
    output_dir = Path(__file__).parent.parent / 'figures'
    output_dir.mkdir(exist_ok=True)
    
    fig.savefig(output_dir / 'bnn_cycle_diagram.pdf', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    fig.savefig(output_dir / 'bnn_cycle_diagram.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    print(f"Saved BNN cycle diagram to {output_dir}")
    print("\nPrior logits (mu_z):", MU_Z_PRIOR)
    print("Prior probabilities (mu_a):", mu_a_prior)
    print("Posterior logits (mu_z_post):", mu_z_post)
    print("Posterior probabilities (mu_a_post):", mu_a_post)
    print("\nFigure shows:")
    print("  - Forward pass (green): Input -> BNN -> Prior -> MM-Remax -> Prediction")
    print("  - Observation: Ground truth with measurement noise")
    print("  - Backward pass (red): Observation -> Posterior -> Gaussian Inference -> BNN")
    
    plt.show()
    return fig


if __name__ == '__main__':
    create_bnn_cycle_figure()

