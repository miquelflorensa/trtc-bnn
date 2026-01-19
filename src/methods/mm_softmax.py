"""
Moment-Matching Softmax (MM-Softmax)

Calculates the moments of the probabilistic softmax output analytically
using log-normal moment matching.

This is the second proposed method for analytically tractable real-to-categorical
transformations in Bayesian Neural Networks.
"""

import numpy as np
from typing import Dict


def mm_softmax(
    mu_z: np.ndarray, 
    sigma_z_sq: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Calculates the moments and correlation coefficients of the probabilistic 
    softmax output analytically using moment matching.

    Supports both single samples and batched inputs.

    Args:
        mu_z: Mean of the input Gaussian vector Z.
              Shape: (K,) for single sample or (B, K) for batch of B samples.
        sigma_z_sq: Variance of the input Gaussian vector Z.
              Shape: (K,) for single sample or (B, K) for batch.

    Returns:
        dict: A dictionary containing:
            - mu_a: Expected value of softmax outputs. Shape: (K,) or (B, K)
            - sigma_a_sq: Variance of softmax outputs. Shape: (K,) or (B, K)
            - cov_z_a: Covariance between inputs Z and outputs A. Shape: (K,) or (B, K)
            - rho_z_a: Correlation coefficient between Z and A. Shape: (K,) or (B, K)
            - mu_e: Mean of exponentiated variables E = exp(Z). Shape: (K,) or (B, K)
            - sigma_e_sq: Variance of E. Shape: (K,) or (B, K)
            - mu_e_tilde: Mean of sum of E. Shape: () or (B,)
            - sigma_e_tilde_sq: Variance of sum of E. Shape: () or (B,)
            - mu_ln_e_tilde: Mean of log(sum(E)). Shape: () or (B,)
            - sigma_ln_e_tilde_sq: Variance of log(sum(E)). Shape: () or (B,)
            - mu_ln_a: Mean of log(A). Shape: (K,) or (B, K)
            - sigma_ln_a_sq: Variance of log(A). Shape: (K,) or (B, K)
    """
    # Handle both single sample and batch inputs
    single_sample = mu_z.ndim == 1
    if single_sample:
        mu_z = mu_z[np.newaxis, :]  # (1, K)
        sigma_z_sq = sigma_z_sq[np.newaxis, :]  # (1, K)
    
    # 1. Moments of the exponentiated variables E_i = exp(Z_i)
    mu_e = np.exp(mu_z + 0.5 * sigma_z_sq)  # (B, K)
    sigma_e_sq = mu_e**2 * (np.exp(sigma_z_sq) - 1)  # (B, K)
    cov_z_e = sigma_z_sq * mu_e  # (B, K)

    # 2. Moments of the sum E_tilde = sum(E_j)
    mu_e_tilde = np.sum(mu_e, axis=-1, keepdims=True)  # (B, 1)
    sigma_e_tilde_sq = np.sum(sigma_e_sq, axis=-1, keepdims=True)  # (B, 1)
    cov_e_e_tilde = sigma_e_sq  # Simplified due to independence assumption (B, K)

    # 3. Moments of ln(E_tilde) assuming E_tilde is Log-Normal
    sigma_ln_e_tilde_sq = np.log(1 + sigma_e_tilde_sq / (mu_e_tilde**2))  # (B, 1)
    mu_ln_e_tilde = np.log(mu_e_tilde) - 0.5 * sigma_ln_e_tilde_sq  # (B, 1)

    # 4. Covariance between Z_i and ln(E_tilde)
    cov_z_ln_e_tilde = np.log(1 + cov_e_e_tilde / (mu_e * mu_e_tilde))  # (B, K)

    # 5. Moments of the Log-Space Output ln(A_i)
    # The derivation shows ln(A_i) = Z_i - ln(E_tilde)
    mu_ln_a = mu_z - mu_ln_e_tilde  # (B, K)
    sigma_ln_a_sq = sigma_z_sq + sigma_ln_e_tilde_sq - 2 * cov_z_ln_e_tilde  # (B, K)

    # 6. Moments of the Final Output A_i (approximated as Log-Normal)
    mu_a = np.exp(mu_ln_a + 0.5 * sigma_ln_a_sq)  # (B, K)
    mu_a_sum = np.sum(mu_a, axis=-1, keepdims=True)  # (B, 1)
    mu_a = mu_a / mu_a_sum  # Normalize to ensure sum(A_i) = 1
    sigma_a_sq = mu_a**2 * (np.exp(sigma_ln_a_sq) - 1)  # (B, K)

    # 7. Covariance between Z_i and A_i
    cov_z_a = mu_a * (sigma_z_sq - cov_z_ln_e_tilde)  # (B, K)

    # 8. Correlation Coefficient between Z_i and A_i
    sigma_z = np.sqrt(sigma_z_sq)
    sigma_a = np.sqrt(np.maximum(sigma_a_sq, 1e-9))
    rho_z_a = np.zeros_like(cov_z_a)
    non_zero_mask = (sigma_z * sigma_a) > 1e-9
    rho_z_a[non_zero_mask] = cov_z_a[non_zero_mask] / (sigma_z[non_zero_mask] * sigma_a[non_zero_mask])

    # Squeeze back to original shape if single sample
    if single_sample:
        mu_a = mu_a.squeeze(0)
        sigma_a_sq = sigma_a_sq.squeeze(0)
        cov_z_a = cov_z_a.squeeze(0)
        rho_z_a = rho_z_a.squeeze(0)
        mu_e = mu_e.squeeze(0)
        sigma_e_sq = sigma_e_sq.squeeze(0)
        mu_e_tilde = mu_e_tilde.squeeze()
        sigma_e_tilde_sq = sigma_e_tilde_sq.squeeze()
        mu_ln_e_tilde = mu_ln_e_tilde.squeeze()
        sigma_ln_e_tilde_sq = sigma_ln_e_tilde_sq.squeeze()
        mu_ln_a = mu_ln_a.squeeze(0)
        sigma_ln_a_sq = sigma_ln_a_sq.squeeze(0)

    return {
        "mu_a": mu_a, 
        "sigma_a_sq": sigma_a_sq, 
        "cov_z_a": cov_z_a,
        "rho_z_a": rho_z_a, 
        "mu_e": mu_e, 
        "sigma_e_sq": sigma_e_sq,
        "mu_e_tilde": mu_e_tilde, 
        "sigma_e_tilde_sq": sigma_e_tilde_sq,
        "mu_ln_e_tilde": mu_ln_e_tilde, 
        "sigma_ln_e_tilde_sq": sigma_ln_e_tilde_sq,
        "mu_ln_a": mu_ln_a, 
        "sigma_ln_a_sq": sigma_ln_a_sq
    }


def deterministic_softmax(mu_z: np.ndarray) -> np.ndarray:
    """
    Compute the deterministic softmax of the input means.
    
    Args:
        mu_z: Mean of the input. Shape: (K,) or (B, K)
        
    Returns:
        Softmax probabilities. Shape: (K,) or (B, K)
    """
    single_sample = mu_z.ndim == 1
    if single_sample:
        mu_z = mu_z[np.newaxis, :]
    
    # Numerical stability: subtract max
    mu_z_stable = mu_z - np.max(mu_z, axis=-1, keepdims=True)
    e = np.exp(mu_z_stable)
    result = e / np.sum(e, axis=-1, keepdims=True)
    
    if single_sample:
        result = result.squeeze(0)
    
    return result


class MMSoftmax:
    """
    Moment-Matching Softmax class for probabilistic classification.
    
    This implements the analytically tractable softmax transformation
    that propagates uncertainty from input Gaussians to output probabilities
    using log-normal moment matching.
    """
    
    def __init__(self):
        """Initialize MM-Softmax."""
        pass
    
    def forward(self, mu_z: np.ndarray, sigma_z_sq: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Forward pass computing all moments.
        
        Args:
            mu_z: Mean of input Gaussian. Shape: (K,) or (B, K)
            sigma_z_sq: Variance of input Gaussian. Shape: (K,) or (B, K)
            
        Returns:
            Dictionary with all computed moments
        """
        return mm_softmax(mu_z, sigma_z_sq)
    
    def get_expected_output(self, mu_z: np.ndarray, sigma_z_sq: np.ndarray) -> np.ndarray:
        """
        Get the expected value of the softmax output.
        
        Args:
            mu_z: Mean of input Gaussian. Shape: (K,) or (B, K)
            sigma_z_sq: Variance of input Gaussian. Shape: (K,) or (B, K)
            
        Returns:
            Expected softmax probabilities. Shape: (K,) or (B, K)
        """
        return mm_softmax(mu_z, sigma_z_sq)["mu_a"]
    
    def get_output_variance(self, mu_z: np.ndarray, sigma_z_sq: np.ndarray) -> np.ndarray:
        """
        Get the variance of the softmax output.
        
        Args:
            mu_z: Mean of input Gaussian. Shape: (K,) or (B, K)
            sigma_z_sq: Variance of input Gaussian. Shape: (K,) or (B, K)
            
        Returns:
            Variance of softmax outputs. Shape: (K,) or (B, K)
        """
        return mm_softmax(mu_z, sigma_z_sq)["sigma_a_sq"]
    
    @staticmethod
    def deterministic(mu_z: np.ndarray) -> np.ndarray:
        """
        Compute deterministic softmax (no uncertainty propagation).
        
        Args:
            mu_z: Mean of input. Shape: (K,) or (B, K)
            
        Returns:
            Softmax probabilities. Shape: (K,) or (B, K)
        """
        return deterministic_softmax(mu_z)
