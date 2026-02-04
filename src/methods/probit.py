"""
Probit (Norm CDF) method for Bayesian Neural Networks.
Based on 'Rethinking Approximate Gaussian Inference' paper.
"""

import numpy as np
from scipy.stats import norm
from scipy.special import owens_t


class Probit:
    """
    Probit activation function using analytical moment propagation.
    Uses the normal CDF (Φ) as the activation function.
    """
    
    def __init__(self):
        """Initialize Probit method."""
        self.name = "Probit"
    
    def compute_moments(self, mu_z, sigma_z_sq):
        """
        Compute output moments analytically using Probit approximation.
        
        Based on 'Rethinking Approximate Gaussian Inference' paper.
        Uses Owen's T function for second moment calculation.
        
        Args:
            mu_z: Mean of logits (K,)
            sigma_z_sq: Variance of logits (K,)
        
        Returns:
            dict with keys:
                - mu_a: Mean of output probabilities (K,)
                - sigma_a_sq: Variance of output probabilities (K,)
                - cov_z_a: Covariance between logits and outputs (K, K)
        """
        var_z = sigma_z_sq
        
        # 1. Unnormalized Moments (Q = Φ(z))
        denom_1 = np.sqrt(1 + var_z)
        denom_2 = np.sqrt(1 + 2 * var_z)
        
        # E[Q] - Equation 13 from paper
        E_Q = norm.cdf(mu_z / denom_1)
        
        # E[Q^2] - Equation 22 using Owen's T function
        T_val = owens_t(mu_z / denom_1, 1 / denom_2)
        E_Q2 = E_Q - 2 * T_val
        
        # 2. Normalized Moments (A = Q / sum(Q))
        sum_E_Q = np.sum(E_Q)
        
        # Mean of output probabilities
        mu_a = E_Q / sum_E_Q
        
        # Variance: E[A^2] - (E[A])^2
        # Approximation: E[A^2] ≈ E[Q^2] / (sum E[Q])^2
        # Note: The paper assumes sum is deterministic for variance calculation
        sigma_a_sq = (E_Q2 / (sum_E_Q**2)) - (mu_a**2)
        
        # Ensure non-negative variances
        sigma_a_sq = np.maximum(sigma_a_sq, 0)
        
        return {
            'mu_a': mu_a,
            'sigma_a_sq': sigma_a_sq,
        }
    
    def compute_moments_mc(self, mu_z, sigma_z_sq, n_samples=100000):
        """
        Compute output moments using Monte Carlo sampling.
        
        Args:
            mu_z: Mean of logits (K,)
            sigma_z_sq: Variance of logits (K,)
            n_samples: Number of Monte Carlo samples
        
        Returns:
            dict with keys:
                - mu_a: Mean of output probabilities (K,)
                - sigma_a_sq: Variance of output probabilities (K,)
                - cov_z_a: Covariance between logits and outputs (K, K)
        """
        K = len(mu_z)
        rng = np.random.default_rng(42)
        
        # Sample Z ~ N(mu_z, sigma_z_sq)
        std_z = np.sqrt(sigma_z_sq)
        Z = rng.normal(mu_z, std_z, size=(n_samples, K))
        
        # Apply Probit activation: Q = Φ(Z)
        Q = norm.cdf(Z)
        
        # Normalize to get probabilities: A = Q / sum(Q)
        denominator = np.sum(Q, axis=1, keepdims=True)
        denominator = np.maximum(denominator, 1e-12)  # Avoid division by zero
        A = Q / denominator
        
        # Compute moments
        mu_a = np.mean(A, axis=0)
        sigma_a_sq = np.var(A, axis=0)
        
        # Compute covariance between Z and A
        cov_z_a = np.zeros((K, K))
        for i in range(K):
            for j in range(K):
                cov_z_a[i, j] = np.cov(Z[:, i], A[:, j])[0, 1]
        
        return {
            'mu_a': mu_a,
            'sigma_a_sq': sigma_a_sq,
            'cov_z_a': cov_z_a
        }
    
    def __repr__(self):
        return f"Probit(name='{self.name}')"
