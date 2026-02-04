"""
Visualize results from exhaustive experiments.
Creates comprehensive plots showing MAE across different scenarios and methods.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
import seaborn as sns
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

# LaTeX styling
try:
    mpl.rc('text', usetex=True)
    mpl.rc('font', family='serif')
    print("[INFO] LaTeX rendering enabled for plots.")
except Exception as e:
    print(f"[WARN] LaTeX rendering failed: {e}")
    print("[INFO] Falling back to default matplotlib fonts.")

# Define plot size based on PRD two-column
pt = 1./72.27
jour_sizes = {"PRD": {"onecol": 246.*pt, "twocol": 510.*pt}}
my_width = jour_sizes["PRD"]["twocol"]

# Color palette - consistent with other scripts
METHOD_COLORS = {
    'LL-Softmax': '#4A90E2',  # Blue
    'MM-Softmax': '#B8E986',  # Green
    'MM-Remax': '#9B59B6',    # Purple
    'Probit': '#F5A623'       # Orange
}


def load_latest_results():
    """Load the most recent experiment results."""
    results_dir = Path(__file__).parent.parent / 'results' / 'exhaustive_experiments'
    
    # Find the latest CSV file
    csv_files = sorted(results_dir.glob('exhaustive_results_*.csv'))
    if not csv_files:
        raise FileNotFoundError("No experiment results found!")
    
    latest_csv = csv_files[-1]
    print(f"Loading results from: {latest_csv}")
    
    df = pd.read_csv(latest_csv)
    return df


def clean_row_name(name):
    """Clean scenario names for display."""
    return (name
            .replace("All Negative", "All Neg.")
            .replace("Competing", "Comp.")
            .replace("Dominant", "Dom.")
            .replace("Equal Positive", "Eq. Pos.")
            .replace("Random", "Rand."))


def create_combined_heatmap(df, output_dir):
    """
    Create a single wide heatmap showing all methods and metrics.
    Similar to the provided example with mu_a, sigma_a_sq, and cov_z_a.
    """
    print("\nCreating combined heatmap...")
    
    # Compute mean MAE for each scenario-method combination
    df_grouped = df.groupby(['scenario', 'method']).agg({
        'mae_mu_a': 'mean',
        'mae_sigma_a_sq': 'mean',
        'mae_cov_z_a': 'mean'
    }).reset_index()
    
    # Create separate DataFrames for each metric
    df_mu_a = df_grouped.pivot(index='scenario', columns='method', values='mae_mu_a')
    df_sigma_a = df_grouped.pivot(index='scenario', columns='method', values='mae_sigma_a_sq')
    df_cov_z_a = df_grouped.pivot(index='scenario', columns='method', values='mae_cov_z_a')
    
    # Clean up row names
    for df_metric in [df_mu_a, df_sigma_a, df_cov_z_a]:
        df_metric.index = [clean_row_name(idx) for idx in df_metric.index]
    
    # Sort index by variance level first, then by scenario type for better grouping
    def sort_key(scenario_name):
        # Extract variance level (Small Var, Medium Var, Large Var)
        if 'Small Var' in scenario_name:
            var_order = 0
        elif 'Medium Var' in scenario_name:
            var_order = 1
        elif 'Large Var' in scenario_name:
            var_order = 2
        else:
            var_order = 3
        # Extract scenario type
        scenario_type = scenario_name.split(' - ')[0] if ' - ' in scenario_name else scenario_name
        return (var_order, scenario_type)
    
    sorted_index = sorted(df_mu_a.index, key=sort_key)
    df_mu_a = df_mu_a.reindex(sorted_index)
    df_sigma_a = df_sigma_a.reindex(sorted_index)
    df_cov_z_a = df_cov_z_a.reindex(sorted_index)
    
    # Build combined DataFrame with columns ordered by metric
    methods = ['LL-Softmax', 'MM-Softmax', 'MM-Remax', 'Probit']
    
    df_combined = pd.DataFrame(index=df_mu_a.index)
    
    # Define metric labels
    label_mu = r'$\mu_A$'
    label_sigma = r'$\sigma_A^2$'
    label_cov = r'$\mathrm{Cov}(Z,A)$'
    
    # Add mu_a columns
    for m in methods:
        if m in df_mu_a.columns:
            col_name = f"{m} ({label_mu})"
            df_combined[col_name] = df_mu_a[m]
    
    # Add sigma_a columns
    for m in methods:
        if m in df_sigma_a.columns:
            col_name = f"{m} ({label_sigma})"
            df_combined[col_name] = df_sigma_a[m]
    
    # Add cov_z_a columns (skip Probit as it doesn't have this)
    for m in methods:
        if m != 'Probit' and m in df_cov_z_a.columns:
            col_name = f"{m} ({label_cov})"
            df_combined[col_name] = df_cov_z_a[m]
    
    # FIRST: Identify variance boundaries and label positions BEFORE modifying index
    # Find indices where variance level changes
    variance_boundaries = []
    prev_var = None
    for idx, scenario in enumerate(df_combined.index):
        if 'Small Var' in scenario:
            current_var = 'Small'
        elif 'Medium Var' in scenario:
            current_var = 'Medium'
        elif 'Large Var' in scenario:
            current_var = 'Large'
        else:
            current_var = None
        
        if prev_var is not None and current_var != prev_var:
            variance_boundaries.append(idx)
        prev_var = current_var
    
    # Calculate variance level label positions
    y_positions = []
    var_labels = []
    small_indices = [i for i, s in enumerate(df_combined.index) if 'Small Var' in s]
    medium_indices = [i for i, s in enumerate(df_combined.index) if 'Medium Var' in s]
    large_indices = [i for i, s in enumerate(df_combined.index) if 'Large Var' in s]
    
    if small_indices:
        # Add 0.5 to center in heatmap coordinates (row 0 is at 0.5, row 1 at 1.5, etc.)
        y_positions.append(np.mean(small_indices) + 0.5)
        var_labels.append('Small Variance')
    if medium_indices:
        y_positions.append(np.mean(medium_indices) + 0.5)
        var_labels.append('Medium Variance')
    if large_indices:
        y_positions.append(np.mean(large_indices) + 0.5)
        var_labels.append('Large Variance')
    
    # NOW: Remove variance level suffixes from y-axis labels since they're now grouped
    # Keep only the scenario type (e.g., "All Neg. - Small Var" -> "All Neg.")
    df_combined.index = [idx.split(' - ')[0] if ' - ' in idx else idx 
                         for idx in df_combined.index]
    
    # Determine figure size
    height = max(5.0, 0.3 * len(df_combined))
    fig, ax = plt.subplots(1, 1, figsize=(my_width, height))
    
    # Use log scale for better visualization with capped range
    positive_vals = df_combined[df_combined > 0].stack()
    if len(positive_vals) > 0:
        vmin = 1e-4  # Cap minimum at 10^-4
        vmax = 1e2   # Cap maximum at 10^2
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = None
    
    # Create heatmap without annotations
    sns.heatmap(df_combined, ax=ax,
                annot=False,  # Remove numbers from heatmap
                linewidths=0.5,
                cmap="plasma",
                norm=norm,
                cbar_kws={'label': 'MAE (Log Scale)' if norm is not None else 'MAE'})
    
    # Add visual separators between metric groups (vertical lines)
    ax.axvline(x=4, color='white', linewidth=3, linestyle='-')  # After mu_a (4 methods)
    ax.axvline(x=8, color='white', linewidth=3, linestyle='-')  # After sigma_a (4 methods)
    
    # Draw horizontal white lines at variance boundaries
    for boundary in variance_boundaries:
        ax.axhline(y=boundary, color='white', linewidth=2, linestyle='-')
    
    # Add metric group labels closer to the cells (reduced offset)
    ax.text(2, -0.1, r'MAE $\mu_A$', ha='center', va='bottom', fontsize=11, weight='bold', color='black')
    ax.text(6, -0.1, r'MAE $\sigma_A^2$', ha='center', va='bottom', fontsize=11, weight='bold', color='black')
    ax.text(9.5, -0.1, r'MAE $\mathrm{Cov}(Z,A)$', ha='center', va='bottom', fontsize=11, weight='bold', color='black')
    
    # Add variance labels on the right (using pre-calculated positions)
    for idx, (y_pos, label) in enumerate(zip(y_positions, var_labels)):
        # Center the text - no manual adjustments needed
        ax.text(len(df_combined.columns) + 0.2, y_pos, label, 
                ha='center', va='center', fontsize=11, weight='bold', 
                rotation=270, color='black')
    
    ax.set_ylabel('Configuration', fontsize=11, color='black')
    ax.set_xlabel('Method', fontsize=11, color='black')
    # Simplify x-tick labels to just show method names
    simplified_labels = []
    for col in df_combined.columns:
        method_name = col.split(' (')[0]
        simplified_labels.append(method_name)
    ax.set_xticklabels(simplified_labels, rotation=45, ha='right', fontsize=11, color='black')
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=11, color='black')
    
    # Save figure
    output_path = output_dir / 'mae_combined_heatmap.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    output_path_png = output_dir / 'mae_combined_heatmap.png'
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    
    plt.close()
    
    return df_combined


def plot_mae_by_scenario(df, output_dir):
    """
    Plot MAE for each metric grouped by scenario and variance level.
    """
    # Extract scenario type and variance level
    df['scenario_type'] = df['scenario'].apply(lambda x: ' - '.join(x.split(' - ')[:-1]))
    df['variance_level'] = df['scenario'].apply(lambda x: x.split(' - ')[-1])
    
    metrics = [
        ('mae_mu_a', r'$\mu_A$'),
        ('mae_sigma_a_sq', r'$\sigma_A^2$'),
        ('mae_cov_z_a', r'$\mathrm{Cov}(Z, A)$')
    ]
    
    variance_levels = ['Small Var', 'Medium Var', 'Large Var']
    scenario_types = df['scenario_type'].unique()
    
    for metric_name, metric_label in metrics:
        fig, axes = plt.subplots(1, 3, figsize=(my_width, my_width * 0.35), sharey=True)
        
        for idx, var_level in enumerate(variance_levels):
            ax = axes[idx]
            
            # Filter data for this variance level
            df_var = df[df['variance_level'] == var_level]
            
            # Prepare data for grouped bar plot
            methods = sorted(df['method'].unique())
            x = np.arange(len(scenario_types))
            width = 0.2
            
            for m_idx, method in enumerate(methods):
                df_method = df_var[df_var['method'] == method]
                
                # Skip if metric is NaN (e.g., cov_z_a for Probit)
                if metric_name == 'mae_cov_z_a' and method == 'Probit':
                    continue
                
                means = []
                for scenario_type in scenario_types:
                    vals = df_method[df_method['scenario_type'] == scenario_type][metric_name]
                    if len(vals) > 0 and not vals.isna().all():
                        means.append(vals.mean())
                    else:
                        means.append(0)
                
                offset = width * (m_idx - len(methods) / 2 + 0.5)
                ax.bar(x + offset, means, width, label=method if idx == 0 else "",
                       color=METHOD_COLORS[method], alpha=0.8)
            
            ax.set_xlabel(var_level, fontsize=10)
            ax.set_xticks(x)
            # Use clean_row_name for shorter labels and make them horizontal
            cleaned_labels = [clean_row_name(st) for st in scenario_types]
            ax.set_xticklabels(cleaned_labels, fontsize=8, rotation=0, ha='center')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3, which='both')
            
            if idx == 0:
                ax.set_ylabel(f'MAE {metric_label}', fontsize=10)
                ax.legend(fontsize=8, loc='upper left')
        
        plt.tight_layout()
        
        # Save figure
        metric_clean = metric_name.replace('mae_', '')
        output_path = output_dir / f'mae_{metric_clean}_by_scenario.pdf'
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Saved: {output_path}")
        
        output_path_png = output_dir / f'mae_{metric_clean}_by_scenario.png'
        plt.savefig(output_path_png, bbox_inches='tight', dpi=300)
        
        plt.close()


def plot_mae_heatmap(df, output_dir):
    """
    Create heatmap showing MAE for each method across all scenarios.
    """
    metrics = [
        ('mae_mu_a', r'MAE $\mu_A$'),
        ('mae_sigma_a_sq', r'MAE $\sigma_A^2$'),
        ('mae_cov_z_a', r'MAE $\mathrm{Cov}(Z, A)$')
    ]
    
    for metric_name, metric_label in metrics:
        # Skip cov_z_a for methods that don't compute it
        if metric_name == 'mae_cov_z_a':
            df_plot = df[df['method'] != 'Probit'].copy()
        else:
            df_plot = df.copy()
        
        # Compute mean MAE for each scenario-method combination
        pivot_data = df_plot.groupby(['scenario', 'method'])[metric_name].mean().unstack()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(my_width * 0.5, my_width * 0.7))
        
        # Plot heatmap
        im = ax.imshow(pivot_data.values, cmap='YlOrRd', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(pivot_data.columns)))
        ax.set_yticks(np.arange(len(pivot_data.index)))
        ax.set_xticklabels(pivot_data.columns, fontsize=9)
        ax.set_yticklabels(pivot_data.index, fontsize=8)
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(metric_label, fontsize=10)
        
        # Remove text annotations for cleaner plot
        
        ax.set_title(metric_label, fontsize=11)
        plt.tight_layout()
        
        # Save figure
        metric_clean = metric_name.replace('mae_', '')
        output_path = output_dir / f'heatmap_{metric_clean}.pdf'
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Saved: {output_path}")
        
        output_path_png = output_dir / f'heatmap_{metric_clean}.png'
        plt.savefig(output_path_png, bbox_inches='tight', dpi=300)
        
        plt.close()


def plot_method_comparison(df, output_dir):
    """
    Create box plots comparing methods across all scenarios.
    """
    metrics = [
        ('mae_mu_a', r'MAE $\mu_A$'),
        ('mae_sigma_a_sq', r'MAE $\sigma_A^2$'),
        ('mae_cov_z_a', r'MAE $\mathrm{Cov}(Z, A)$')
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(my_width, my_width * 0.35))
    
    for idx, (metric_name, metric_label) in enumerate(metrics):
        ax = axes[idx]
        
        methods = sorted(df['method'].unique())
        data_to_plot = []
        labels = []
        colors = []
        
        for method in methods:
            # Skip cov_z_a for Probit
            if metric_name == 'mae_cov_z_a' and method == 'Probit':
                continue
            
            method_data = df[df['method'] == method][metric_name].dropna()
            if len(method_data) > 0:
                data_to_plot.append(method_data)
                labels.append(method)
                colors.append(METHOD_COLORS[method])
        
        # Create box plot
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                       showfliers=False, widths=0.6)
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel(metric_label, fontsize=10)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='x', rotation=45)
        plt.setp(ax.get_xticklabels(), fontsize=9)
    
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / 'method_comparison_boxplots.pdf'
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Saved: {output_path}")
    
    output_path_png = output_dir / 'method_comparison_boxplots.png'
    plt.savefig(output_path_png, bbox_inches='tight', dpi=300)
    
    plt.close()


def print_best_method_summary(df):
    """
    Print summary of which method performs best in each scenario.
    """
    print("\n" + "="*80)
    print("BEST METHOD PER SCENARIO (lowest mean MAE)")
    print("="*80)
    
    metrics = ['mae_mu_a', 'mae_sigma_a_sq', 'mae_cov_z_a']
    
    for metric_name in metrics:
        print(f"\n{metric_name.upper()}")
        print("-" * 80)
        
        scenarios = df['scenario'].unique()
        
        for scenario in sorted(scenarios):
            df_scenario = df[df['scenario'] == scenario]
            
            # For cov_z_a, exclude Probit
            if metric_name == 'mae_cov_z_a':
                df_scenario = df_scenario[df_scenario['method'] != 'Probit']
            
            mean_mae = df_scenario.groupby('method')[metric_name].mean()
            
            if len(mean_mae) > 0 and not mean_mae.isna().all():
                best_method = mean_mae.idxmin()
                best_mae = mean_mae.min()
                print(f"{scenario:40s} -> {best_method:15s} (MAE: {best_mae:.6e})")


if __name__ == '__main__':
    # Load results
    df = load_latest_results()
    
    # Create output directory
    output_dir = Path(__file__).parent.parent / 'figures' / 'exhaustive_experiments'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating plots...")
    
    # Generate combined heatmap (main plot)
    df_combined = create_combined_heatmap(df, output_dir)
    
    # Generate additional plots
    plot_mae_by_scenario(df, output_dir)
    plot_mae_heatmap(df, output_dir)
    plot_method_comparison(df, output_dir)
    
    # Print summary
    print_best_method_summary(df)
    
    print("\n" + "="*80)
    print("All plots generated successfully!")
    print(f"Output directory: {output_dir}")
    print("="*80)
