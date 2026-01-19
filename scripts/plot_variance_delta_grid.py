#!/usr/bin/env python3
"""
Plot variance-delta grids comparing deterministic vs analytical probabilistic
transformations for Softmax and Remax.

Generates a multi-panel figure showing expected dominant-class output across
combinations of input variance (x-axis) and input mean separation (y-axis).

For each method (Softmax/Remax):
- Left panel: Deterministic expected value (doesn't depend on variance)
- Middle panel: Analytical probabilistic expected value (depends on both)
- Right panel (optional): Monte Carlo error comparison

Saves figures to the specified output directory.
"""
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.methods import (
    ll_softmax, deterministic_softmax,
    mm_softmax,
    mm_remax, deterministic_remax,
    mc_softmax, mc_remax
)

# Method configurations
METHOD_CONFIG = {
    'll_softmax': {
        'name': 'LL-Softmax',
        'det_func': deterministic_softmax,
        'ana_func': ll_softmax,
        'mc_func': mc_softmax,
    },
    'mm_softmax': {
        'name': 'MM-Softmax',
        'det_func': deterministic_softmax,
        'ana_func': mm_softmax,
        'mc_func': mc_softmax,
    },
    'mm_remax': {
        'name': 'MM-Remax',
        'det_func': deterministic_remax,
        'ana_func': mm_remax,
        'mc_func': mc_remax,
    },
}


def compute_grid(method: str, variances: np.ndarray, deltas: np.ndarray,
                 K: int = 10, mc_samples: int = 0, seed: int = 0):
    """
    Compute deterministic and analytical grids for a given method.
    
    Args:
        method: 'll_softmax', 'mm_softmax', or 'mm_remax'
        variances: Array of variance values
        deltas: Array of delta (mean separation) values
        K: Number of classes
        mc_samples: Number of MC samples (0 to disable)
        seed: Random seed for MC sampling
        
    Returns:
        det_vals: Deterministic values (shape: n_delta,)
        ana_vals: Analytical values (shape: n_delta, n_var)
        mc_vals: Monte Carlo values or None (shape: n_delta, n_var)
    """
    n_var = len(variances)
    n_delta = len(deltas)
    dominant_idx = 0
    
    # Select the appropriate functions
    if method not in METHOD_CONFIG:
        raise ValueError(f"Unknown method: {method}. Choose from {list(METHOD_CONFIG.keys())}")
    
    config = METHOD_CONFIG[method]
    det_func = config['det_func']
    ana_func = config['ana_func']
    mc_func = config['mc_func']
    
    # Deterministic: only depends on delta
    det_vals = np.zeros(n_delta)
    for j, d in enumerate(deltas):
        mu_z = np.zeros(K)
        mu_z[dominant_idx] = d
        probs = det_func(mu_z)
        det_vals[j] = probs[dominant_idx]
    
    # Analytical: depends on both delta and variance
    ana_vals = np.zeros((n_delta, n_var))
    for i in range(n_delta):
        mu_z = np.zeros(K)
        mu_z[dominant_idx] = deltas[i]
        for j in range(n_var):
            sigma_z_sq = np.full(K, variances[j])
            result = ana_func(mu_z, sigma_z_sq)
            ana_vals[i, j] = result['mu_a'][dominant_idx]
    
    # Monte Carlo (optional)
    mc_vals = None
    if mc_samples > 0:
        print(f'Computing Monte Carlo estimates for {method} on a {n_delta}x{n_var} '
              f'grid with {mc_samples} samples per point...')
        rng = np.random.default_rng(seed)
        mc_vals = np.zeros((n_delta, n_var))
        for i in range(n_delta):
            mu_z = np.zeros(K)
            mu_z[dominant_idx] = deltas[i]
            for j in range(n_var):
                sigma_z_sq = np.full(K, variances[j])
                result = mc_func(mu_z, sigma_z_sq, n_samples=mc_samples, rng=rng)
                mc_vals[i, j] = result['mu_a'][dominant_idx]
    
    return det_vals, ana_vals, mc_vals


def plot_grid(method: str, det_vals: np.ndarray, ana_vals: np.ndarray,
              mc_vals: np.ndarray, variances: np.ndarray, deltas: np.ndarray,
              output_path: str):
    """
    Plot the variance-delta grid for a single method.
    
    Args:
        method: 'll_softmax', 'mm_softmax', or 'mm_remax'
        det_vals: Deterministic values (shape: n_delta,)
        ana_vals: Analytical values (shape: n_delta, n_var)
        mc_vals: Monte Carlo values or None
        variances: Array of variance values
        deltas: Array of delta values
        output_path: Path to save the figure
    """
    # Try to use LaTeX rendering
    try:
        mpl.rc('text', usetex=True)
        mpl.rc('font', family='serif')
    except Exception:
        pass
    
    n_var = len(variances)
    n_panels = 3 if mc_vals is not None else 2
    
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4.5), 
                             constrained_layout=True)
    
    if n_panels == 2:
        ax0, ax1 = axes
        ax2 = None
    else:
        ax0, ax1, ax2 = axes
    
    # Create 2D array for deterministic (repeat across variance axis)
    det_img = np.repeat(det_vals[:, None], n_var, axis=1)
    
    # Shared color scale
    vmin = min(det_img.min(), ana_vals.min())
    vmax = max(det_img.max(), ana_vals.max())
    
    # Get method display name
    method_name = METHOD_CONFIG[method]['name']
    det_name = 'Softmax' if 'softmax' in method else 'Remax'
    
    # Left: Deterministic
    im0 = ax0.imshow(det_img, origin='lower', aspect='auto',
                     extent=(variances[0], variances[-1], deltas[0], deltas[-1]),
                     cmap='viridis', vmin=vmin, vmax=vmax)
    ax0.set_title(f'Deterministic {det_name} ($E[A_{{dom}}]$)')
    ax0.set_xlabel(r'Input variance: $\sigma_z^2$')
    ax0.set_ylabel(r'Input delta: $\mu_{\mathrm{dom}} - \overline{\mu}_{\mathrm{neg}}$')
    fig.colorbar(im0, ax=ax0, label=r'$E[A_{dom}]$')
    
    # Middle: Analytical
    im1 = ax1.imshow(ana_vals, origin='lower', aspect='auto',
                     extent=(variances[0], variances[-1], deltas[0], deltas[-1]),
                     cmap='viridis', vmin=vmin, vmax=vmax)
    ax1.set_title(f'Analytical {method_name} ($E[A_{{dom}}]$)')
    ax1.set_xlabel(r'Input variance: $\sigma_z^2$')
    ax1.set_ylabel('')
    fig.colorbar(im1, ax=ax1, label=r'$E[A_{dom}]$')
    
    # Right: Monte Carlo error (if computed)
    if mc_vals is not None:
        err = np.abs(mc_vals - ana_vals)
        im2 = ax2.imshow(err, origin='lower', aspect='auto',
                         extent=(variances[0], variances[-1], deltas[0], deltas[-1]),
                         cmap='inferno')
        ax2.set_title(r'Absolute error $|\mathrm{MC} - \mathrm{Analytic}|$')
        ax2.set_xlabel(r'Input variance: $\sigma_z^2$')
        ax2.set_ylabel('')
        fig.colorbar(im2, ax=ax2, label=r'$|E_{MC} - E_{Analytic}|$')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Saved {method} grid to: {output_path}')
    plt.close()


def plot_comparison_grid(softmax_ana: np.ndarray, remax_ana: np.ndarray,
                         variances: np.ndarray, deltas: np.ndarray,
                         output_path: str):
    """
    Plot side-by-side comparison of Softmax vs Remax analytical grids.
    
    Args:
        softmax_ana: Softmax analytical values
        remax_ana: Remax analytical values
        variances: Array of variance values
        deltas: Array of delta values
        output_path: Path to save the figure
    """
    try:
        mpl.rc('text', usetex=True)
        mpl.rc('font', family='serif')
    except Exception:
        pass
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    ax0, ax1 = axes
    
    # Shared color scale
    vmin = min(softmax_ana.min(), remax_ana.min())
    vmax = max(softmax_ana.max(), remax_ana.max())
    
    # Softmax
    im0 = ax0.imshow(softmax_ana, origin='lower', aspect='auto',
                     extent=(variances[0], variances[-1], deltas[0], deltas[-1]),
                     cmap='viridis', vmin=vmin, vmax=vmax)
    ax0.set_title(r'MM-Softmax: $E[A_{dom}]$')
    ax0.set_xlabel(r'Input variance: $\sigma_z^2$')
    ax0.set_ylabel(r'Input delta: $\mu_{\mathrm{dom}} - \overline{\mu}_{\mathrm{neg}}$')
    fig.colorbar(im0, ax=ax0, label=r'$E[A_{dom}]$')
    
    # Remax
    im1 = ax1.imshow(remax_ana, origin='lower', aspect='auto',
                     extent=(variances[0], variances[-1], deltas[0], deltas[-1]),
                     cmap='viridis', vmin=vmin, vmax=vmax)
    ax1.set_title(r'MM-Remax: $E[A_{dom}]$')
    ax1.set_xlabel(r'Input variance: $\sigma_z^2$')
    ax1.set_ylabel('')
    fig.colorbar(im1, ax=ax1, label=r'$E[A_{dom}]$')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Saved comparison grid to: {output_path}')
    plt.close()


def plot_all_methods_comparison(results: dict, variances: np.ndarray, 
                                 deltas: np.ndarray, output_path: str):
    """
    Plot comparison of all three analytical methods side by side.
    
    Args:
        results: Dict mapping method name to (det_vals, ana_vals, mc_vals)
        variances: Array of variance values
        deltas: Array of delta values
        output_path: Path to save the figure
    """
    try:
        mpl.rc('text', usetex=True)
        mpl.rc('font', family='serif')
    except Exception:
        pass
    
    methods = list(results.keys())
    n_methods = len(methods)
    
    fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 4.5), 
                             constrained_layout=True)
    if n_methods == 1:
        axes = [axes]
    
    # Find shared color scale across all methods
    all_vals = [results[m][1] for m in methods]  # ana_vals
    vmin = min(v.min() for v in all_vals)
    vmax = max(v.max() for v in all_vals)
    
    for ax, method in zip(axes, methods):
        ana_vals = results[method][1]
        method_name = METHOD_CONFIG[method]['name']
        
        im = ax.imshow(ana_vals, origin='lower', aspect='auto',
                       extent=(variances[0], variances[-1], deltas[0], deltas[-1]),
                       cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title(f'{method_name}: $E[A_{{dom}}]$')
        ax.set_xlabel(r'Input variance: $\sigma_z^2$')
        if ax == axes[0]:
            ax.set_ylabel(r'Input delta: $\mu_{\mathrm{dom}} - \overline{\mu}_{\mathrm{neg}}$')
        else:
            ax.set_ylabel('')
        fig.colorbar(im, ax=ax, label=r'$E[A_{dom}]$')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Saved all methods comparison to: {output_path}')
    plt.close()


def main():
    # Available methods
    available_methods = list(METHOD_CONFIG.keys())
    
    parser = argparse.ArgumentParser(
        description='Generate variance-delta grid plots for LL-Softmax, MM-Softmax, and MM-Remax'
    )
    parser.add_argument('--output-dir', default='./figures',
                        help='Directory to save figures')
    parser.add_argument('--method', 
                        choices=available_methods + ['all'],
                        default='all',
                        help='Which method(s) to plot')
    parser.add_argument('--K', type=int, default=10,
                        help='Number of classes')
    parser.add_argument('--var-min', type=float, default=0.0,
                        help='Minimum variance')
    parser.add_argument('--var-max', type=float, default=4.0,
                        help='Maximum variance')
    parser.add_argument('--delta-min', type=float, default=0.0,
                        help='Minimum delta (mean separation)')
    parser.add_argument('--delta-max', type=float, default=6.0,
                        help='Maximum delta')
    parser.add_argument('--n-var', type=int, default=201,
                        help='Number of variance grid points')
    parser.add_argument('--n-delta', type=int, default=201,
                        help='Number of delta grid points')
    parser.add_argument('--mc-samples', type=int, default=0,
                        help='Number of MC samples per point (0 = disabled)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for MC sampling')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine which methods to run
    if args.method == 'all':
        methods_to_run = available_methods
    else:
        methods_to_run = [args.method]
    
    # Create grids
    variances = np.linspace(args.var_min, args.var_max, args.n_var)
    deltas = np.linspace(args.delta_min, args.delta_max, args.n_delta)
    
    print("="*70)
    print("Generating Variance-Delta Grid Plots")
    print("="*70)
    print(f"Methods: {', '.join(methods_to_run)}")
    print(f"Classes (K): {args.K}")
    print(f"Variance range: [{args.var_min}, {args.var_max}] ({args.n_var} points)")
    print(f"Delta range: [{args.delta_min}, {args.delta_max}] ({args.n_delta} points)")
    print(f"Output directory: {args.output_dir}")
    print()
    
    results = {}
    
    # Compute and plot each method
    for method in methods_to_run:
        method_name = METHOD_CONFIG[method]['name']
        print(f"Computing {method_name} grid...")
        
        det_vals, ana_vals, mc_vals = compute_grid(
            method, variances, deltas, args.K, args.mc_samples, args.seed
        )
        results[method] = (det_vals, ana_vals, mc_vals)
        
        output_path = os.path.join(args.output_dir, f'{method}_variance_delta_grid.pdf')
        plot_grid(method, det_vals, ana_vals, mc_vals, variances, deltas, output_path)
    
    # Plot comparison if multiple methods were computed
    if len(results) > 1:
        print("\nGenerating comparison plot...")
        output_path = os.path.join(args.output_dir, 'all_methods_comparison.pdf')
        plot_all_methods_comparison(results, variances, deltas, output_path)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
