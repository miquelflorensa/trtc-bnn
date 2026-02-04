"""
Compare all methods (LL-Softmax, MM-Softmax, MM-Remax, Probit) against their MC sampling.
Creates a 5-row figure with one error chart per method.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.methods.ll_softmax import LLSoftmax
from src.methods.mm_softmax import MMSoftmax
from src.methods.mm_remax import MMRemax
from src.methods.probit import Probit

# LaTeX styling
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Define plot size based on PRD two-column
pt = 1./72.27
jour_sizes = {"PRD": {"onecol": 246.*pt, "twocol": 510.*pt}}
my_width = jour_sizes["PRD"]["twocol"]
golden = (1 + 5 ** 0.5) / 2
# Taller figure to accommodate 5 rows
figsize = (my_width, my_width * 1.2)

# Test configuration
MU_Z = np.array([-0.2, 1.3, -1.0, 0.5, -0.8, 5.1, -0.5, 0.3, -1.2, 0.7]) 
SIGMA_Z_SQ = np.array([1.0, 0.6, 3.0, 0.8, 1.2, 0.5, 1.5, 0.9, 2.0, 0.7])
N_SAMPLES = 100000

# Color palette - different color for each analytical method
INPUT_COLOR = "#1F3643"
INPUT_MARKER_COLOR = "#1F3643"
MC_COLOR = '#FF6B6B'
MC_MARKER_COLOR = '#FF6B6B'

# Method-specific colors
METHOD_COLORS = {
    'LL-Softmax': '#4A90E2',  # Blue
    'MM-Softmax': '#B8E986',  # Green
    'MM-Remax': '#9B59B6',    # Purple
    'Probit': '#F5A623'       # Orange
}


def monte_carlo_sampling(method_name, mu_z, sigma_z_sq, n_samples=100000):
    """
    Monte Carlo sampling for methods that don't have built-in MC.
    
    Args:
        method_name: Name of the method ('LL-Softmax', 'MM-Softmax', 'MM-Remax')
        mu_z: Mean of logits
        sigma_z_sq: Variance of logits
        n_samples: Number of samples
    
    Returns:
        dict with mu_a and sigma_a_sq
    """
    K = len(mu_z)
    rng = np.random.default_rng(42)
    
    # Sample Z ~ N(mu_z, sigma_z_sq)
    std_z = np.sqrt(sigma_z_sq)
    Z = rng.normal(mu_z, std_z, size=(n_samples, K))
    
    # Apply activation based on method
    if 'Softmax' in method_name:
        # Softmax: exp(z) / sum(exp(z))
        numerator = np.exp(Z)
    elif 'Remax' in method_name:
        # Remax: ReLU(z) / sum(ReLU(z))
        numerator = np.maximum(0, Z)
    else:
        raise ValueError(f"Unknown method: {method_name}")
    
    # Normalize
    denominator = np.sum(numerator, axis=1, keepdims=True)
    denominator = np.maximum(denominator, 1e-12)
    A = numerator / denominator
    
    # Compute moments
    mu_a = np.mean(A, axis=0)
    sigma_a_sq = np.var(A, axis=0)
    
    # Compute covariance
    cov_z_a = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            cov_z_a[i, j] = np.cov(Z[:, i], A[:, j])[0, 1]
    
    return {
        'mu_a': mu_a,
        'sigma_a_sq': sigma_a_sq,
        'cov_z_a': cov_z_a
    }


def plot_input_z(ax, mu_z, std_z):
    """
    Plot input Z with error bars.
    
    Args:
        ax: Matplotlib axis
        mu_z: Mean of input logits
        std_z: Std of input logits
    """
    K = len(mu_z)
    classes = np.arange(K)
    
    # Plot input Z error bars
    for i in classes:
        # Error bar
        ax.errorbar(i, mu_z[i], yerr=std_z[i], fmt='none', 
                   ecolor=INPUT_COLOR, elinewidth=1.5, capsize=5)
        # Horizontal caps at mean +/- std
        ax.plot([i - 0.2, i + 0.2], 
               [mu_z[i] + std_z[i], mu_z[i] + std_z[i]], 
               INPUT_COLOR, linewidth=1.5)
        ax.plot([i - 0.2, i + 0.2], 
               [mu_z[i] - std_z[i], mu_z[i] - std_z[i]], 
               INPUT_COLOR, linewidth=1.5)
        # Square marker
        ax.scatter(i, mu_z[i], color=INPUT_MARKER_COLOR, marker='s', 
                  s=30, zorder=3, edgecolor='k', linewidth=0.5)
    
    ax.set_xticks(classes)
    ax.set_xticklabels([])
    ax.set_xlim(-0.5, K - 0.5)
    ax.grid(True, axis='y', linestyle=':')
    ax.set_ylabel('Input Value', fontsize=9)
    
    # Input label on the left side
    ax.text(-0.05, 0.5, 'Input Z', transform=ax.transAxes, 
           fontsize=10, fontweight='bold', va='center', ha='right', rotation=90)
    
    # Legend for input
    legend_handles = [
        Line2D([0], [0], marker='s', color='w', label=r'$\mu_Z$', 
               markerfacecolor=INPUT_MARKER_COLOR, markersize=8),
        Line2D([0], [0], color=INPUT_COLOR, lw=2, 
               label=r'$\mu_Z \pm \sigma_Z$')
    ]
    ax.legend(handles=legend_handles, loc='upper right', fontsize=8)


def plot_method_comparison(ax, mu_a_analytical, std_a_analytical, mu_a_mc, std_a_mc, 
                           method_name, analytical_color, show_xlabel=False):
    """
    Plot analytical vs MC comparison for a single method.
    
    Args:
        ax: Matplotlib axis
        mu_a_analytical: Mean of output (analytical)
        std_a_analytical: Std of output (analytical)
        mu_a_mc: Mean of output (MC)
        std_a_mc: Std of output (MC)
        method_name: Name of the method for the title
        analytical_color: Color for the analytical error bars
        show_xlabel: Whether to show x-axis labels (only for bottom plot)
    """
    K = len(mu_a_analytical)
    classes = np.arange(K)
    
    # Plot for Output A (Analytical vs. Monte Carlo)
    for i in classes:
        # Monte Carlo error bars (plotted first to be underneath)
        ax.errorbar(i, mu_a_mc[i], yerr=std_a_mc[i], fmt='none', 
                   ecolor=MC_COLOR, elinewidth=1.5, capsize=5)
        ax.plot([i - 0.4, i + 0.4], 
               [mu_a_mc[i] + std_a_mc[i], mu_a_mc[i] + std_a_mc[i]], 
               MC_COLOR, linewidth=1.5)
        ax.plot([i - 0.4, i + 0.4], 
               [mu_a_mc[i] - std_a_mc[i], mu_a_mc[i] - std_a_mc[i]], 
               MC_COLOR, linewidth=1.5)
        ax.scatter(i, mu_a_mc[i], color=MC_MARKER_COLOR, marker='s', 
                  s=30, zorder=3, edgecolor='k', linewidth=0.5)

        # Analytical error bars (plotted second to be on top)
        ax.errorbar(i, mu_a_analytical[i], yerr=std_a_analytical[i], fmt='none', 
                   ecolor=analytical_color, elinewidth=1.5, capsize=5)
        ax.plot([i - 0.2, i + 0.2], 
               [mu_a_analytical[i] + std_a_analytical[i], mu_a_analytical[i] + std_a_analytical[i]], 
               analytical_color, linewidth=1.5)
        ax.plot([i - 0.2, i + 0.2], 
               [mu_a_analytical[i] - std_a_analytical[i], mu_a_analytical[i] - std_a_analytical[i]], 
               analytical_color, linewidth=1.5)
        ax.scatter(i, mu_a_analytical[i], color=analytical_color, 
                  marker='s', s=30, zorder=3, edgecolor='k', linewidth=0.5)

    ax.set_xticks(classes)
    if show_xlabel:
        ax.set_xticklabels([f'$A_{{{i}}}$' for i in classes])
    else:
        ax.set_xticklabels([])
    
    ax.set_xlim(-0.5, K - 0.5)
    ax.set_ylim(-0.1, 1.1)  # Set y-axis for A to [0, 1] interval
    ax.grid(True, axis='y', linestyle=':')
    # ax.set_ylabel('Probability', fontsize=9)
    
    # Method name on the left side
    ax.text(-0.05, 0.5, method_name, transform=ax.transAxes, 
           fontsize=10, fontweight='bold', va='center', ha='right', rotation=90)
    
    # Add legend with method-specific label
    method_label = method_name.replace('-', '-')  # Keep hyphen
    # Use \mathrm instead of \text for LaTeX compatibility
    legend_handles = [
        Line2D([0], [0], marker='s', color='w', 
               label=rf'$\mu_A^{{\mathrm{{{method_label}}}}}$', 
               markerfacecolor=analytical_color, markersize=8),
        Line2D([0], [0], color=analytical_color, lw=2, 
               label=rf'$\mu_A^{{\mathrm{{{method_label}}}}} \pm \sigma_A^{{\mathrm{{{method_label}}}}}$'),
        Line2D([0], [0], marker='s', color='w', label=r'$\mu_A^{\mathrm{MC}}$', 
               markerfacecolor=MC_MARKER_COLOR, markersize=8),
        Line2D([0], [0], color=MC_COLOR, lw=2.0, 
               label=r'$\mu_A^{\mathrm{MC}} \pm \sigma_A^{\mathrm{MC}}$')
    ]
    ax.legend(handles=legend_handles, loc='upper right', fontsize=7)
    
    # Add legend only to the top subplot
    return ax


def create_comparison_figure():
    """Create a 5-row figure comparing all methods."""
    
    print("Computing results for all methods...")
    print(f"Input: mu_z = {MU_Z}")
    print(f"       sigma_z_sq = {SIGMA_Z_SQ}")
    print()
    
    # Instantiate all methods
    ll_softmax = LLSoftmax()
    mm_softmax = MMSoftmax()
    mm_remax = MMRemax()
    probit = Probit()
    
    methods = [
        ("LL-Softmax", ll_softmax),
        ("MM-Softmax", mm_softmax),
        ("MM-Remax", mm_remax),
        ("Probit", probit),
    ]
    
    # Compute results for all methods
    results = []
    for name, method in methods:
        print(f"Computing {name}...")
        
        # Analytical - use forward() for existing methods, compute_moments() for Probit
        if hasattr(method, 'forward'):
            result_analytical = method.forward(MU_Z, SIGMA_Z_SQ)
        else:
            result_analytical = method.compute_moments(MU_Z, SIGMA_Z_SQ)
        
        mu_a_analytical = result_analytical['mu_a']
        std_a_analytical = np.sqrt(result_analytical['sigma_a_sq'])
        
        # Monte Carlo - check if method has compute_moments_mc
        if hasattr(method, 'compute_moments_mc'):
            result_mc = method.compute_moments_mc(MU_Z, SIGMA_Z_SQ, n_samples=N_SAMPLES)
        else:
            # Fall back to manual MC sampling
            result_mc = monte_carlo_sampling(name, MU_Z, SIGMA_Z_SQ, N_SAMPLES)
        
        mu_a_mc = result_mc['mu_a']
        std_a_mc = np.sqrt(result_mc['sigma_a_sq'])
        
        results.append({
            'name': name,
            'mu_a_analytical': mu_a_analytical,
            'std_a_analytical': std_a_analytical,
            'mu_a_mc': mu_a_mc,
            'std_a_mc': std_a_mc
        })
        
        print(f"  Analytical: mu_a = {mu_a_analytical}")
        print(f"  MC:         mu_a = {mu_a_mc}")
        print()
    
    # Create figure with 5 subplots (1 for input + 4 for methods)
    fig, axes = plt.subplots(5, 1, figsize=figsize, sharex=True)
    
    # Plot input Z in the first row
    std_z = np.sqrt(SIGMA_Z_SQ)
    plot_input_z(axes[0], MU_Z, std_z)
    
    # Plot each method in remaining rows
    for idx, (ax, result) in enumerate(zip(axes[1:], results)):
        is_last = (idx == len(results) - 1)
        method_color = METHOD_COLORS[result['name']]
        plot_method_comparison(
            ax,
            result['mu_a_analytical'],
            result['std_a_analytical'],
            result['mu_a_mc'],
            result['std_a_mc'],
            result['name'],
            method_color,
            show_xlabel=is_last
        )
    
    # Remove the old legend code since each subplot now has its own legend
    
    # Overall title
    fig.suptitle(r'\textbf{Analytical vs. Monte Carlo Comparison for All Methods}', 
                fontsize=12, y=0.995)
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path(__file__).parent.parent / 'figures'
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / 'all_methods_comparison.pdf'
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Saved figure to {output_path}")
    
    output_path_png = output_dir / 'all_methods_comparison.png'
    fig.savefig(output_path_png, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Saved figure to {output_path_png}")
    
    plt.show()


if __name__ == '__main__':
    create_comparison_figure()
