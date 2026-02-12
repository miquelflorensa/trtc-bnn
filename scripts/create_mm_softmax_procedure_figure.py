"""
Generate a comprehensive figure demonstrating the moment matching procedure.
Layout: Single Column (3.25") split into 2 cols.
Style: Compact Error Bars (height ratios adjusted).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import gaussian_kde
from pathlib import Path
import sys

# --- Configuration for Paper-Ready Fonts ---
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=8)      
plt.rc('axes', labelsize=8)
plt.rc('xtick', labelsize=7)
plt.rc('ytick', labelsize=7)
plt.rc('legend', fontsize=6)
plt.rcParams['text.latex.preamble'] = r'\usepackage{times}'

# --- Figure Dimensions ---
FIG_WIDTH = 3.25  
FIG_HEIGHT = 4.2  # Reduced height further for compactness

# --- METHOD SELECTION ---
METHOD = 'softmax' # 'softmax' or 'remax'

# --- INPUT DATA ---
MU_Z = np.array([-0.4, -1.3, -0.2])
SIGMA_Z_SQ = np.array([0.2, 0.3, 0.2])
CLASS_NAMES = [r'$Z_0$', r'$Z_1$', r'$Z_2$']

# --- COLORS ---
COLOR_SUM = '#0B5345'   # Deep Dark Green
COLORS = ['#2E86AB', '#E94F37', '#F39C12'] 

# --- X-AXIS LIMITS CONFIGURATION ---
X_LIMITS_LOGITS = (-3, 1)   
X_LIMITS_EXP    = (0, 3)    
X_LIMITS_SUM    = None      
X_LIMITS_PROB   = (0, 1)    

# --- Monte Carlo Settings ---
N_MC_SAMPLES = 1000000

# Try import, fallback for standalone
try:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.methods.mm_softmax import mm_softmax
    from src.methods.mm_remax import mm_remax
except ImportError:
    print("Warning: Local src modules not found. Using dummy functions.")
    def mm_softmax(mu, sigma_sq): 
        return {'mu_e': np.exp(mu), 'sigma_e_sq': np.ones(3)*0.1, 'mu_e_tilde': 10.0, 
                'sigma_e_tilde_sq': 0.5, 'mu_a': np.array([0.3, 0.2, 0.5]), 'sigma_a_sq': np.ones(3)*0.01,
                'mu_ln_a': mu, 'sigma_ln_a_sq': sigma_sq}
    def mm_remax(mu, sigma_sq): return mm_softmax(mu, sigma_sq)

def compute_actual_distributions(mu_z, sigma_z_sq, n_samples=N_MC_SAMPLES, method='softmax'):
    K = len(mu_z)
    sigma_z = np.sqrt(sigma_z_sq)
    z_samples = np.random.randn(n_samples, K) * sigma_z[np.newaxis, :] + mu_z[np.newaxis, :]
    
    if method == 'softmax':
        e_samples = np.exp(z_samples)
    elif method == 'remax':
        e_samples = np.maximum(0, z_samples)
    
    e_sum_samples = np.sum(e_samples, axis=1)
    e_sum_samples_safe = np.maximum(e_sum_samples, 1e-30)
    a_samples = e_samples / e_sum_samples_safe[:, np.newaxis]
    
    return {'z_samples': z_samples, 'e_samples': e_samples, 
            'e_sum_samples': e_sum_samples, 'a_samples': a_samples}

def get_plot_limits(data_mu, data_sigma, manual_limits):
    if manual_limits is not None:
        low, high = manual_limits
        if low == high: return low - 1.0, high + 1.0
        return low, high
    low = np.min(data_mu - 3.0 * data_sigma)
    high = np.max(data_mu + 3.0 * data_sigma)
    return low, high

def create_figure():
    method_name = "MM-Softmax" if METHOD == 'softmax' else "MM-Remax"
    print(f"Generating {method_name} figure (Compact Error Bars)...")
    
    # 1. Analytic Calculations
    if METHOD == 'softmax': result = mm_softmax(MU_Z, SIGMA_Z_SQ)
    else: result = mm_remax(MU_Z, SIGMA_Z_SQ)
    
    mu_e = result['mu_e'] if METHOD == 'softmax' else result['mu_m']
    sigma_e = np.sqrt(result['sigma_e_sq'] if METHOD == 'softmax' else result['sigma_m_sq'])
    mu_e_tilde = result['mu_e_tilde'] if METHOD == 'softmax' else result.get('mu_m_tilde', np.sum(mu_e))
    sigma_e_tilde = np.sqrt(result['sigma_e_tilde_sq'] if METHOD == 'softmax' else result.get('sigma_m_tilde_sq', np.sum(sigma_e**2)))
    mu_a = result['mu_a']
    sigma_a = np.sqrt(result['sigma_a_sq'])
    sigma_z = np.sqrt(SIGMA_Z_SQ)

    # 2. Monte Carlo
    mc = compute_actual_distributions(MU_Z, SIGMA_Z_SQ, N_MC_SAMPLES, METHOD)
    
    # 3. Setup Plot
    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), layout='constrained')
    
    # HEIGHT RATIOS: 
    # Rows 0 & 2 (Error bars) get weight 0.7
    # Rows 1 & 3 (Distributions) get weight 1.2
    # This makes error bar plots about 60% the height of distribution plots
    gs = fig.add_gridspec(4, 2, width_ratios=[1, 1], height_ratios=[0.7, 1.2, 0.7, 1.2])

    # ==========================================
    # LEFT COLUMN 
    # ==========================================

    # --- BLOCK 1: INPUT LOGITS (Top Left) ---
    x_min_z, x_max_z = get_plot_limits(MU_Z, sigma_z, X_LIMITS_LOGITS)
    
    ax1 = fig.add_subplot(gs[0, 0])
    positions = np.arange(3)
    for i in positions:
        ax1.errorbar(MU_Z[i], i, xerr=sigma_z[i], fmt='none', ecolor=COLORS[i], elinewidth=1.0, capsize=2)
        ax1.plot([MU_Z[i]-sigma_z[i]]*2, [i-0.2, i+0.2], COLORS[i], lw=1.0)
        ax1.plot([MU_Z[i]+sigma_z[i]]*2, [i-0.2, i+0.2], COLORS[i], lw=1.0)
        ax1.scatter(MU_Z[i], i, color=COLORS[i], marker='s', s=10, zorder=3, edgecolor='k', linewidth=0.3)
    ax1.set_yticks(positions); ax1.set_yticklabels(CLASS_NAMES)
    ax1.set_xlim(x_min_z, x_max_z)
    ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
    ax1.invert_yaxis()
    ax1.set_title(r'\textbf{1. Input} $Z$', fontsize=9, pad=3)

    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    x = np.linspace(x_min_z, x_max_z, 500)
    for i, (mu, s2) in enumerate(zip(MU_Z, SIGMA_Z_SQ)):
        y = norm.pdf(x, mu, np.sqrt(s2))
        ax2.plot(x, y, color=COLORS[i], linewidth=1.0)
        ax2.fill_between(x, y, alpha=0.3, color=COLORS[i])
    ax2.set_yticks([]); ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False); ax2.spines['left'].set_visible(False)

    # --- BLOCK 3: SUM (Bottom Left) ---
    # Moved from Right Col (gs[0,1]) to Left Col (gs[2,0])
    x_min_sum, x_max_sum = get_plot_limits(np.array([mu_e_tilde]), np.array([sigma_e_tilde]), X_LIMITS_SUM)
    
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.errorbar(mu_e_tilde, 0, xerr=sigma_e_tilde, fmt='none', ecolor=COLOR_SUM, elinewidth=1.0, capsize=2)
    ax5.plot([mu_e_tilde-sigma_e_tilde]*2, [-0.1, 0.1], COLOR_SUM, lw=1.0)
    ax5.plot([mu_e_tilde+sigma_e_tilde]*2, [-0.1, 0.1], COLOR_SUM, lw=1.0)
    ax5.scatter(mu_e_tilde, 0, color=COLOR_SUM, marker='s', s=15, zorder=3, edgecolor='k', lw=0.3)
    ax5.set_yticks([0]); ax5.set_yticklabels([r'$\tilde{M}$'])
    ax5.set_ylim(-0.5, 0.5)
    ax5.set_xlim(x_min_sum, x_max_sum)
    ax5.spines['top'].set_visible(False); ax5.spines['right'].set_visible(False); ax5.invert_yaxis()
    ax5.set_title(r'\textbf{3. Sum} $\tilde{M}$', fontsize=9, pad=3)

    ax6 = fig.add_subplot(gs[3, 0], sharex=ax5)
    x = np.linspace(x_min_sum, x_max_sum, 500)
    y = norm.pdf(x, mu_e_tilde, sigma_e_tilde)
    ax6.plot(x, y, COLOR_SUM, lw=1.0)
    ax6.fill_between(x, y, alpha=0.3, color=COLOR_SUM)
    kde = gaussian_kde(mc['e_sum_samples'])
    ax6.plot(x, kde(x), COLOR_SUM, lw=1.0, ls='--', alpha=0.8)
    ax6.set_yticks([]); ax6.spines['top'].set_visible(False); ax6.spines['right'].set_visible(False); ax6.spines['left'].set_visible(False)


    # ==========================================
    # RIGHT COLUMN
    # ==========================================

    # --- BLOCK 2: TRANSFORMED (Top Right) ---
    # Moved from Left Col (gs[2,0]) to Right Col (gs[0,1])
    x_min_e, x_max_e = get_plot_limits(mu_e, sigma_e, X_LIMITS_EXP)
    
    ax3 = fig.add_subplot(gs[0, 1])
    for i in positions:
        ax3.errorbar(mu_e[i], i, xerr=sigma_e[i], fmt='none', ecolor=COLORS[i], elinewidth=1.0, capsize=2)
        ax3.plot([mu_e[i]-sigma_e[i]]*2, [i-0.2, i+0.2], COLORS[i], lw=1.0)
        ax3.plot([mu_e[i]+sigma_e[i]]*2, [i-0.2, i+0.2], COLORS[i], lw=1.0)
        ax3.scatter(mu_e[i], i, color=COLORS[i], marker='s', s=10, zorder=3, edgecolor='k', lw=0.3)
    ax3.set_yticks(positions); ax3.set_yticklabels([rf'$M_{i}$' for i in range(3)])
    ax3.set_xlim(x_min_e, x_max_e)
    ax3.spines['top'].set_visible(False); ax3.spines['right'].set_visible(False); ax3.invert_yaxis()
    ax3.set_title(r'\textbf{2. Transform} $M = \exp(Z)$', fontsize=9, pad=3)

    ax4 = fig.add_subplot(gs[1, 1], sharex=ax3)
    x = np.linspace(x_min_e, x_max_e, 500)
    for i in range(3):
        y = norm.pdf(x, mu_e[i], sigma_e[i])
        ax4.plot(x, y, color=COLORS[i], lw=1.0)
        ax4.fill_between(x, y, alpha=0.2, color=COLORS[i])
        kde = gaussian_kde(mc['e_samples'][:, i])
        ax4.plot(x, kde(x), color=COLORS[i], lw=1.0, ls='--', alpha=0.8)
    ax4.set_yticks([]); ax4.spines['top'].set_visible(False); ax4.spines['right'].set_visible(False); ax4.spines['left'].set_visible(False)

    # --- LEGEND MOVED HERE (Top Right) ---
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0],[0], color='black', lw=1.0, label='MM'), 
                       Line2D([0],[0], color='black', lw=1.0, ls='--', label='MC')]
    ax4.legend(handles=legend_elements, loc='upper right', framealpha=0.8, fontsize=6, handlelength=1.5)

    # --- BLOCK 4: OUTPUT PROBABILITIES (Bottom Right) ---
    # Stays in Right Col, but indices gs[2,1] and gs[3,1] are now correct relative to new layout
    x_min_a, x_max_a = get_plot_limits(mu_a, sigma_a, X_LIMITS_PROB)
    
    ax7 = fig.add_subplot(gs[2, 1])
    for i in positions:
        ax7.errorbar(mu_a[i], i, xerr=sigma_a[i], fmt='none', ecolor=COLORS[i], elinewidth=1.0, capsize=2)
        ax7.plot([mu_a[i]-sigma_a[i]]*2, [i-0.2, i+0.2], COLORS[i], lw=1.0)
        ax7.plot([mu_a[i]+sigma_a[i]]*2, [i-0.2, i+0.2], COLORS[i], lw=1.0)
        ax7.scatter(mu_a[i], i, color=COLORS[i], marker='s', s=10, zorder=3, edgecolor='k', lw=0.3)
    ax7.set_yticks(positions); ax7.set_yticklabels([rf'$A_{i}$' for i in range(3)])
    ax7.set_xlim(x_min_a, x_max_a)
    ax7.set_xticks([0, 0.5, 1])
    ax7.set_xticklabels(['0', '0.5', '1'])
    ax7.spines['top'].set_visible(False); ax7.spines['right'].set_visible(False); ax7.invert_yaxis()
    ax7.set_title(r'\textbf{4. Output} $A = M / \tilde{M}$', fontsize=9, pad=3)

    ax8 = fig.add_subplot(gs[3, 1], sharex=ax7)
    x = np.linspace(x_min_a, x_max_a, 500)
    for i in range(3):
        y = norm.pdf(x, mu_a[i], sigma_a[i])
        ax8.plot(x, y, color=COLORS[i], lw=1.0)
        ax8.fill_between(x, y, alpha=0.2, color=COLORS[i])
        kde = gaussian_kde(mc['a_samples'][:, i])
        ax8.plot(x, kde(x), color=COLORS[i], lw=1.0, ls='--', alpha=0.8)
    
    # Legend removed from here
    ax8.set_yticks([]); ax8.spines['top'].set_visible(False); ax8.spines['right'].set_visible(False); ax8.spines['left'].set_visible(False)

    # --- SAVE ---
    output_dir = Path(__file__).parent.parent / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    fpath = output_dir / f'mm_{METHOD}_procedure_compact.pdf'
    fig.savefig(fpath, dpi=300)
    print(f"Saved to {fpath}")
    plt.close(fig)

if __name__ == '__main__':
    create_figure()