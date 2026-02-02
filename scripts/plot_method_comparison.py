"""
Plot comparison results: MM-Remax vs MC-Remax over training epochs.

Generates a 2x3 figure showing MAE of:
- μ_A (mean probability)
- σ_A (std probability)
- ρ_ZA (input-output correlation coefficient)

For both all classes and dominant class only.
With error bands showing ±1 std.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import json


# Plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'serif'


def load_results(results_path: str):
    """Load results from npz or json file."""
    if results_path.endswith('.npz'):
        data = np.load(results_path)
        return {key: data[key] for key in data.files}
    elif results_path.endswith('.json'):
        with open(results_path, 'r') as f:
            return json.load(f)
    else:
        raise ValueError(f"Unknown file format: {results_path}")


def plot_with_error_band(ax, epochs, mean, std, color, marker, label=None):
    """Plot a line with shaded error band (±1 std)."""
    mean = np.array(mean)
    std = np.array(std)
    
    # Plot the mean line
    ax.plot(epochs, mean, color=color, linewidth=2, 
            marker=marker, markersize=4, markevery=5, label=label)
    
    # Plot the error band (±1 std)
    ax.fill_between(epochs, mean - std, mean + std, 
                    color=color, alpha=0.2, edgecolor='none')


def plot_mae_comparison(history: dict, save_path: str = None):
    """
    Plot MAE of μ_A, σ_A, and ρ_ZA over training epochs with error bands.
    
    Creates a 2x3 grid:
        Top row: All classes (MAE μ_A, MAE σ_A, MAE ρ_ZA)
        Bottom row: Dominant class only (MAE μ_A, MAE σ_A, MAE ρ_ZA)
    """
    epochs = np.array(history['epoch'])
    
    # Check if std data is available
    has_std = 'mae_mu_all_std' in history
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Colors
    mu_color = '#4A90E2'    # Blue
    sigma_color = '#E24A4A'  # Red
    rho_color = '#50C878'    # Green
    
    # Row configurations
    rows = [
        ('all', 'All Classes'),
        ('dominant', 'Dominant Class'),
    ]
    
    for row_idx, (suffix, row_label) in enumerate(rows):
        # MAE μ_A
        ax = axes[row_idx, 0]
        mean_key = f'mae_mu_{suffix}'
        std_key = f'{mean_key}_std'
        if has_std and std_key in history:
            plot_with_error_band(ax, epochs, history[mean_key], history[std_key], mu_color, 'o')
        else:
            ax.plot(epochs, history[mean_key], color=mu_color, linewidth=2, 
                    marker='o', markersize=4, markevery=5)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('MAE', fontsize=12)
        ax.set_title(f'MAE of $\\mu_A$ ({row_label})', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(epochs[0], epochs[-1])
        
        # MAE σ_A
        ax = axes[row_idx, 1]
        mean_key = f'mae_sigma_{suffix}'
        std_key = f'{mean_key}_std'
        if has_std and std_key in history:
            plot_with_error_band(ax, epochs, history[mean_key], history[std_key], sigma_color, 's')
        else:
            ax.plot(epochs, history[mean_key], color=sigma_color, linewidth=2,
                    marker='s', markersize=4, markevery=5)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('MAE', fontsize=12)
        ax.set_title(f'MAE of $\\sigma_A$ ({row_label})', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(epochs[0], epochs[-1])
        
        # MAE ρ_ZA
        ax = axes[row_idx, 2]
        mean_key = f'mae_rho_{suffix}'
        std_key = f'{mean_key}_std'
        if has_std and std_key in history:
            plot_with_error_band(ax, epochs, history[mean_key], history[std_key], rho_color, '^')
        else:
            ax.plot(epochs, history[mean_key], color=rho_color, linewidth=2,
                    marker='^', markersize=4, markevery=5)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('MAE', fontsize=12)
        ax.set_title(f'MAE of $\\rho_{{ZA}}$ ({row_label})', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(epochs[0], epochs[-1])
    
    plt.suptitle('MM-Remax vs MC-Remax: Mean Absolute Error over Training',
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"MAE plot saved to: {save_path}")
    
    plt.show()


def main(results_path: str, output_dir: str = None):
    """Generate MAE comparison plot from results file."""
    
    # Load results
    print(f"Loading results from: {results_path}")
    history = load_results(results_path)
    
    # Determine output directory
    if output_dir is None:
        output_dir = Path(results_path).parent / "figures"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Saving figures to: {output_dir}")
    
    # Generate MAE comparison plot
    print("\nGenerating MAE comparison plot...")
    plot_mae_comparison(
        history, 
        save_path=str(Path(output_dir) / "mae_comparison.pdf")
    )
    
    print("\n" + "="*70)
    print("Plot generated successfully!")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot MM-Remax vs MC-Remax comparison results"
    )
    parser.add_argument("--results", type=str, 
                        default="./results/method_comparison/mm_remax_comparison.npz",
                        help="Path to results file (.npz or .json)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for figures")
    
    args = parser.parse_args()
    
    main(args.results, args.output_dir)
