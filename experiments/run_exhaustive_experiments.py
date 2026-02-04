"""
Exhaustive experiments comparing all methods across different scenarios.
Computes MAE (Mean Absolute Error) with respect to Monte Carlo sampling.

Scenarios:
1. Random/Initialization (Small, Medium, Large variance)
2. One Dominant Logit (Small, Medium, Large variance)
3. Multiple Competing Logits (Small, Medium, Large variance)
4. All Negative Logits (Small, Medium, Large variance)
5. Equal Positive Logits (Small, Medium, Large variance)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
from tqdm import tqdm
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.methods.ll_softmax import LLSoftmax
from src.methods.mm_softmax import MMSoftmax
from src.methods.mm_remax import MMRemax
from src.methods.probit import Probit
from src.methods.mc_softmax import mc_softmax
from src.methods.mc_remax import mc_remax


def get_mc_baseline(method_name, mu_z, sigma_z_sq, n_samples=100000):
    """
    Get Monte Carlo baseline using the proper implementation for each method.
    
    Args:
        method_name: Name of the method
        mu_z: Mean of logits (K,)
        sigma_z_sq: Variance of logits (K,)
        n_samples: Number of MC samples
    
    Returns:
        dict with mu_a, sigma_a_sq, cov_z_a (element-wise covariance vector)
    """
    if method_name == 'Probit':
        # Probit has its own MC method
        probit = Probit()
        result = probit.compute_moments_mc(mu_z, sigma_z_sq, n_samples)
        # Probit doesn't have closed-form covariance, so we don't compare it
        return {
            'mu_a': result['mu_a'],
            'sigma_a_sq': result['sigma_a_sq'],
            'cov_z_a': None
        }
        
    elif method_name in ['LL-Softmax', 'MM-Softmax']:
        # Both use softmax activation - use mc_softmax
        result = mc_softmax(mu_z, sigma_z_sq, n_samples)
        return {
            'mu_a': result['mu_a'],
            'sigma_a_sq': result['sigma_a_sq'],
            'cov_z_a': result['cov_z_a']
        }
        
    elif method_name == 'MM-Remax':
        # Use mc_remax for Remax
        result = mc_remax(mu_z, sigma_z_sq, n_samples)
        return {
            'mu_a': result['mu_a'],
            'sigma_a_sq': result['sigma_a_sq'],
            'cov_z_a': result['cov_z_a']
        }
    else:
        raise ValueError(f"Unknown method: {method_name}")


def compute_mae(analytical, mc_baseline):
    """
    Compute Mean Absolute Error between analytical and MC results.
    
    Args:
        analytical: dict with mu_a, sigma_a_sq, cov_z_a
        mc_baseline: dict with mu_a, sigma_a_sq, cov_z_a
    
    Returns:
        dict with mae_mu_a, mae_sigma_a_sq, mae_cov_z_a
    """
    mae_mu_a = np.mean(np.abs(analytical['mu_a'] - mc_baseline['mu_a']))
    mae_sigma_a_sq = np.mean(np.abs(analytical['sigma_a_sq'] - mc_baseline['sigma_a_sq']))
    
    # For covariance, compute MAE only if both are available
    if analytical['cov_z_a'] is not None and mc_baseline['cov_z_a'] is not None:
        mae_cov_z_a = np.mean(np.abs(analytical['cov_z_a'] - mc_baseline['cov_z_a']))
    else:
        mae_cov_z_a = None
    
    return {
        'mae_mu_a': mae_mu_a,
        'mae_sigma_a_sq': mae_sigma_a_sq,
        'mae_cov_z_a': mae_cov_z_a
    }


def run_experiment_for_analysis(scenario_name, mu_z, sigma_z_sq, n_samples_mc):
    """
    Run all methods on a single configuration and compute MAE.
    
    Args:
        scenario_name: Name of the scenario
        mu_z: Mean of logits (K,)
        sigma_z_sq: Variance of logits (K,)
        n_samples_mc: Number of MC samples
    
    Returns:
        List of result dictionaries (one per method)
    """
    methods = {
        'LL-Softmax': LLSoftmax(),
        'MM-Softmax': MMSoftmax(),
        'MM-Remax': MMRemax(),
        'Probit': Probit()
    }
    
    results_list = []
    
    for method_name, method_obj in methods.items():
        # Compute analytical results
        if hasattr(method_obj, 'forward'):
            analytical = method_obj.forward(mu_z, sigma_z_sq)
        elif hasattr(method_obj, 'compute_moments'):
            analytical = method_obj.compute_moments(mu_z, sigma_z_sq)
        else:
            raise ValueError(f"Method {method_name} has no forward or compute_moments method")
        
        # Probit doesn't compute covariance analytically
        if method_name == 'Probit':
            analytical['cov_z_a'] = None
        
        # Compute MC baseline using proper implementation
        mc_baseline = get_mc_baseline(method_name, mu_z, sigma_z_sq, n_samples_mc)
        
        # Compute MAE
        mae_results = compute_mae(analytical, mc_baseline)
        
        # Store results
        result_dict = {
            'scenario': scenario_name,
            'method': method_name,
            'mae_mu_a': mae_results['mae_mu_a'],
            'mae_sigma_a_sq': mae_results['mae_sigma_a_sq'],
            'mae_cov_z_a': mae_results['mae_cov_z_a'],
            'mean_mu_z': np.mean(mu_z),
            'std_mu_z': np.std(mu_z),
            'mean_sigma_z_sq': np.mean(sigma_z_sq),
            'std_sigma_z_sq': np.std(sigma_z_sq)
        }
        results_list.append(result_dict)
    
    return results_list


def save_results(all_results_df, first_configs):
    """
    Save experiment results to disk.
    
    Args:
        all_results_df: List of result dictionaries
        first_configs: Dict mapping scenario names to (mu_z, sigma_z_sq) tuples
    """
    # Create results directory
    results_dir = Path(__file__).parent.parent / 'results' / 'exhaustive_experiments'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results_df)
    
    # Save raw results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = results_dir / f'exhaustive_results_{timestamp}.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nSaved raw results to {csv_path}")
    
    # Compute summary statistics grouped by scenario and method
    summary = df.groupby(['scenario', 'method']).agg({
        'mae_mu_a': ['mean', 'std', 'min', 'max'],
        'mae_sigma_a_sq': ['mean', 'std', 'min', 'max'],
        'mae_cov_z_a': ['mean', 'std', 'min', 'max']
    }).round(6)
    
    summary_path = results_dir / f'summary_statistics_{timestamp}.csv'
    summary.to_csv(summary_path)
    print(f"Saved summary statistics to {summary_path}")
    
    # Save first config of each scenario for reproducibility
    configs_dict = {
        scenario: {
            'mu_z': mu_z.tolist(),
            'sigma_z_sq': sigma_z_sq.tolist()
        }
        for scenario, (mu_z, sigma_z_sq) in first_configs.items()
    }
    configs_path = results_dir / f'first_configs_{timestamp}.json'
    with open(configs_path, 'w') as f:
        json.dump(configs_dict, f, indent=2)
    print(f"Saved first configs to {configs_path}")
    
    # Print summary to console
    print("\n" + "="*80)
    print("SUMMARY STATISTICS (MAE with respect to MC sampling)")
    print("="*80)
    print(summary.to_string())
    print("="*80)
    
    return df, summary


if __name__ == '__main__':
    # --- Experiment Configuration ---
    K = 10
    N_SAMPLES_MC = 100000
    N_EXPERIMENTS_PER_CONFIG = 100
    
    all_results_df = []
    np.random.seed(42)
    
    # Store first config of each type for plotting
    first_configs = {}

    print("Running exhaustive experiments...")
    print(f"Configuration: K={K}, N_MC={N_SAMPLES_MC}, N_per_config={N_EXPERIMENTS_PER_CONFIG}")
    print(f"Total experiments: {N_EXPERIMENTS_PER_CONFIG * 15} (15 scenarios)")
    print()

    # --- SCENARIO 1: RANDOM/INITIALIZATION ---
    # Small Variance
    for i in tqdm(range(N_EXPERIMENTS_PER_CONFIG), desc="Random - Small Var"):
        mu_z = np.random.normal(loc=0.0, scale=0.5, size=K)
        sigma_z_sq = np.random.uniform(low=0.001, high=0.1, size=K)
        if i == 0: first_configs['Random - Small Var'] = (mu_z.copy(), sigma_z_sq.copy())
        results_list = run_experiment_for_analysis("Random - Small Var", mu_z, sigma_z_sq, N_SAMPLES_MC)
        all_results_df.extend(results_list)
        
    # Medium Variance (Original config)
    for i in tqdm(range(N_EXPERIMENTS_PER_CONFIG), desc="Random - Medium Var"):
        mu_z = np.random.normal(loc=0.0, scale=0.5, size=K)
        sigma_z_sq = np.random.uniform(low=0.5, high=3.0, size=K)
        if i == 0: first_configs['Random - Medium Var'] = (mu_z.copy(), sigma_z_sq.copy())
        results_list = run_experiment_for_analysis("Random - Medium Var", mu_z, sigma_z_sq, N_SAMPLES_MC)
        all_results_df.extend(results_list)
    
    # Large Variance
    for i in tqdm(range(N_EXPERIMENTS_PER_CONFIG), desc="Random - Large Var"):
        mu_z = np.random.normal(loc=0.0, scale=0.5, size=K)
        sigma_z_sq = np.random.uniform(low=5.0, high=10.0, size=K)
        if i == 0: first_configs['Random - Large Var'] = (mu_z.copy(), sigma_z_sq.copy())
        results_list = run_experiment_for_analysis("Random - Large Var", mu_z, sigma_z_sq, N_SAMPLES_MC)
        all_results_df.extend(results_list)


    # --- SCENARIO 2: ONE DOMINANT LOGIT ---
    # Small Variance
    for i in tqdm(range(N_EXPERIMENTS_PER_CONFIG), desc="Dominant - Small Var"):
        pos_logit_mu = np.random.uniform(low=2.0, high=4.0)
        neg_logit_mus = np.random.uniform(low=-4.0, high=-1.0, size=K - 1)
        mu_z = np.append(neg_logit_mus, pos_logit_mu)
        np.random.shuffle(mu_z)
        sigma_z_sq = np.random.uniform(low=0.001, high=0.1, size=K)
        if i == 0: first_configs['Dominant - Small Var'] = (mu_z.copy(), sigma_z_sq.copy())
        results_list = run_experiment_for_analysis("Dominant - Small Var", mu_z, sigma_z_sq, N_SAMPLES_MC)
        all_results_df.extend(results_list)
        
    # Medium Variance (Original config)
    for i in tqdm(range(N_EXPERIMENTS_PER_CONFIG), desc="Dominant - Medium Var"):
        pos_logit_mu = np.random.uniform(low=2.0, high=4.0)
        neg_logit_mus = np.random.uniform(low=-4.0, high=-1.0, size=K - 1)
        mu_z = np.append(neg_logit_mus, pos_logit_mu)
        np.random.shuffle(mu_z)
        sigma_z_sq = np.random.uniform(low=0.5, high=3.0, size=K)
        if i == 0: first_configs['Dominant - Medium Var'] = (mu_z.copy(), sigma_z_sq.copy())
        results_list = run_experiment_for_analysis("Dominant - Medium Var", mu_z, sigma_z_sq, N_SAMPLES_MC)
        all_results_df.extend(results_list)
    
    # Large Variance
    for i in tqdm(range(N_EXPERIMENTS_PER_CONFIG), desc="Dominant - Large Var"):
        pos_logit_mu = np.random.uniform(low=2.0, high=4.0)
        neg_logit_mus = np.random.uniform(low=-4.0, high=-1.0, size=K - 1)
        mu_z = np.append(neg_logit_mus, pos_logit_mu)
        np.random.shuffle(mu_z)
        sigma_z_sq = np.random.uniform(low=5.0, high=10.0, size=K)
        if i == 0: first_configs['Dominant - Large Var'] = (mu_z.copy(), sigma_z_sq.copy())
        results_list = run_experiment_for_analysis("Dominant - Large Var", mu_z, sigma_z_sq, N_SAMPLES_MC)
        all_results_df.extend(results_list)
        
        
    # --- SCENARIO 3: MULTIPLE COMPETING LOGITS ---
    # Small Variance
    for i in tqdm(range(N_EXPERIMENTS_PER_CONFIG), desc="Competing - Small Var"):
        num_positive = np.random.randint(2, 4)
        pos_logit_mus = np.random.uniform(low=2.0, high=4.0, size=num_positive)
        neg_logit_mus = np.random.uniform(low=-4.0, high=-1.0, size=K - num_positive)
        mu_z = np.concatenate([pos_logit_mus, neg_logit_mus])
        np.random.shuffle(mu_z)
        sigma_z_sq = np.random.uniform(low=0.001, high=0.1, size=K)
        if i == 0: first_configs['Competing - Small Var'] = (mu_z.copy(), sigma_z_sq.copy())
        results_list = run_experiment_for_analysis("Competing - Small Var", mu_z, sigma_z_sq, N_SAMPLES_MC)
        all_results_df.extend(results_list)
    
    # Medium Variance (Original config)
    for i in tqdm(range(N_EXPERIMENTS_PER_CONFIG), desc="Competing - Medium Var"):
        num_positive = np.random.randint(2, 4)
        pos_logit_mus = np.random.uniform(low=2.0, high=4.0, size=num_positive)
        neg_logit_mus = np.random.uniform(low=-4.0, high=-1.0, size=K - num_positive)
        mu_z = np.concatenate([pos_logit_mus, neg_logit_mus])
        np.random.shuffle(mu_z)
        sigma_z_sq = np.random.uniform(low=0.5, high=3.0, size=K)
        if i == 0: first_configs['Competing - Medium Var'] = (mu_z.copy(), sigma_z_sq.copy())
        results_list = run_experiment_for_analysis("Competing - Medium Var", mu_z, sigma_z_sq, N_SAMPLES_MC)
        all_results_df.extend(results_list)
        
    # Large Variance
    for i in tqdm(range(N_EXPERIMENTS_PER_CONFIG), desc="Competing - Large Var"):
        num_positive = np.random.randint(2, 4)
        pos_logit_mus = np.random.uniform(low=2.0, high=4.0, size=num_positive)
        neg_logit_mus = np.random.uniform(low=-4.0, high=-1.0, size=K - num_positive)
        mu_z = np.concatenate([pos_logit_mus, neg_logit_mus])
        np.random.shuffle(mu_z)
        sigma_z_sq = np.random.uniform(low=5.0, high=10.0, size=K)
        if i == 0: first_configs['Competing - Large Var'] = (mu_z.copy(), sigma_z_sq.copy())
        results_list = run_experiment_for_analysis("Competing - Large Var", mu_z, sigma_z_sq, N_SAMPLES_MC)
        all_results_df.extend(results_list)
        
        
    # --- SCENARIO 4: ALL NEGATIVE LOGITS ---
    # Small Variance
    for i in tqdm(range(N_EXPERIMENTS_PER_CONFIG), desc="All Negative - Small Var"):
        mu_z = np.random.uniform(low=-4.0, high=-1.0, size=K)
        sigma_z_sq = np.random.uniform(low=0.001, high=0.1, size=K)
        if i == 0: first_configs['All Negative - Small Var'] = (mu_z.copy(), sigma_z_sq.copy())
        results_list = run_experiment_for_analysis("All Negative - Small Var", mu_z, sigma_z_sq, N_SAMPLES_MC)
        all_results_df.extend(results_list)

    # Medium Variance
    for i in tqdm(range(N_EXPERIMENTS_PER_CONFIG), desc="All Negative - Medium Var"):
        mu_z = np.random.uniform(low=-4.0, high=-1.0, size=K)
        sigma_z_sq = np.random.uniform(low=0.5, high=3.0, size=K)
        if i == 0: first_configs['All Negative - Medium Var'] = (mu_z.copy(), sigma_z_sq.copy())
        results_list = run_experiment_for_analysis("All Negative - Medium Var", mu_z, sigma_z_sq, N_SAMPLES_MC)
        all_results_df.extend(results_list)
        
    # Large Variance
    for i in tqdm(range(N_EXPERIMENTS_PER_CONFIG), desc="All Negative - Large Var"):
        mu_z = np.random.uniform(low=-4.0, high=-1.0, size=K)
        sigma_z_sq = np.random.uniform(low=5.0, high=10.0, size=K)
        if i == 0: first_configs['All Negative - Large Var'] = (mu_z.copy(), sigma_z_sq.copy())
        results_list = run_experiment_for_analysis("All Negative - Large Var", mu_z, sigma_z_sq, N_SAMPLES_MC)
        all_results_df.extend(results_list)
        
        
    # --- SCENARIO 5: EQUAL POSITIVE LOGITS ---
    # Small Variance
    for i in tqdm(range(N_EXPERIMENTS_PER_CONFIG), desc="Equal Positive - Small Var"):
        mu_z_base = np.random.uniform(low=2.0, high=4.0)
        mu_z = np.full(K, mu_z_base) + np.random.uniform(low=-0.2, high=0.2, size=K)
        sigma_z_sq = np.random.uniform(low=0.001, high=0.1, size=K)
        if i == 0: first_configs['Equal Positive - Small Var'] = (mu_z.copy(), sigma_z_sq.copy())
        results_list = run_experiment_for_analysis("Equal Positive - Small Var", mu_z, sigma_z_sq, N_SAMPLES_MC)
        all_results_df.extend(results_list)

    # Medium Variance
    for i in tqdm(range(N_EXPERIMENTS_PER_CONFIG), desc="Equal Positive - Medium Var"):
        mu_z_base = np.random.uniform(low=2.0, high=4.0)
        mu_z = np.full(K, mu_z_base) + np.random.uniform(low=-0.2, high=0.2, size=K)
        sigma_z_sq = np.random.uniform(low=0.5, high=3.0, size=K)
        if i == 0: first_configs['Equal Positive - Medium Var'] = (mu_z.copy(), sigma_z_sq.copy())
        results_list = run_experiment_for_analysis("Equal Positive - Medium Var", mu_z, sigma_z_sq, N_SAMPLES_MC)
        all_results_df.extend(results_list)
        
    # Large Variance
    for i in tqdm(range(N_EXPERIMENTS_PER_CONFIG), desc="Equal Positive - Large Var"):
        mu_z_base = np.random.uniform(low=2.0, high=4.0)
        mu_z = np.full(K, mu_z_base) + np.random.uniform(low=-0.2, high=0.2, size=K)
        sigma_z_sq = np.random.uniform(low=5.0, high=10.0, size=K)
        if i == 0: first_configs['Equal Positive - Large Var'] = (mu_z.copy(), sigma_z_sq.copy())
        results_list = run_experiment_for_analysis("Equal Positive - Large Var", mu_z, sigma_z_sq, N_SAMPLES_MC)
        all_results_df.extend(results_list)

    # Save results
    print("\n" + "="*80)
    print("Experiments complete! Saving results...")
    print("="*80)
    df, summary = save_results(all_results_df, first_configs)
    
    print(f"\nTotal experiments run: {len(all_results_df)}")
    print(f"Results saved in: results/exhaustive_experiments/")
