"""
Locally Linearized Softmax (LL-Softmax)

Calculates the moments of the softmax output using the Gaussian Multiplicative 
Approximation (GMA) with local linearization.

This is the first proposed method for analytically tractable real-to-categorical
transformations in Bayesian Neural Networks.
"""

import numpy as np
from typing import Dict, Union


def ll_softmax(
    mu_z: np.ndarray, 
    sigma_z_sq: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Calculates the moments of the decomposed locally linearized softmax
    using the Gaussian Multiplicative Approximation (GMA).

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
            - cov_z_a: Covariance between Z and A. Shape: (K,) or (B, K)
            - rho_z_a: Correlation coefficient between Z and A. Shape: (K,) or (B, K)
    """
    # Handle both single sample and batch inputs
    single_sample = mu_z.ndim == 1
    if single_sample:
        mu_z = mu_z[np.newaxis, :]  # (1, K)
        sigma_z_sq = sigma_z_sq[np.newaxis, :]  # (1, K)
    
    # Intermediate moments based on the GMA derivations
    mu_e = np.exp(mu_z + 0.5 * sigma_z_sq)  # (B, K)
    sigma_e_sq = mu_e**2 * (np.exp(sigma_z_sq) - 1)  # (B, K)
    
    mu_e_sum = np.sum(mu_e, axis=-1, keepdims=True)  # (B, 1)
    sigma_e_sum_sq = np.sum(sigma_e_sq, axis=-1, keepdims=True)  # (B, 1)
    
    mu_e_sum_inv = 1.0 / mu_e_sum  # (B, 1)
    sigma_e_sum_inv_sq = sigma_e_sum_sq / (mu_e_sum**4)  # (B, 1)
    
    cov_z_e = sigma_z_sq * mu_e  # (B, K)
    cov_z_e_sum_inv = -(cov_z_e / (mu_e_sum**2))  # (B, K)
    
    # Enforcing sum to one by assuming cov(E_i, E_sum_inv) = 0
    mu_a = mu_e * mu_e_sum_inv  # (B, K)
    
    # Variance and Covariance using full GMA formula
    sigma_a_sq = (
        mu_e**2 * sigma_e_sum_inv_sq + 
        mu_e_sum_inv**2 * sigma_e_sq +
        sigma_e_sq * sigma_e_sum_inv_sq
    )  # (B, K)
    
    cov_z_a = (
        cov_z_e * mu_e_sum_inv +
        cov_z_e_sum_inv * mu_e
    )  # (B, K)
    
    # Calculate correlation coefficient
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

    return {
        "mu_a": mu_a, 
        "sigma_a_sq": sigma_a_sq, 
        "cov_z_a": cov_z_a, 
        "rho_z_a": rho_z_a
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


class LLSoftmax:
    """
    Locally Linearized Softmax class for probabilistic classification.
    
    This implements the GMA-based softmax transformation that propagates 
    uncertainty from input Gaussians to output probabilities using local
    linearization.
    """
    
    def __init__(self):
        """Initialize LL-Softmax."""
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
        return ll_softmax(mu_z, sigma_z_sq)
    
    def get_expected_output(self, mu_z: np.ndarray, sigma_z_sq: np.ndarray) -> np.ndarray:
        """
        Get the expected value of the softmax output.
        
        Args:
            mu_z: Mean of input Gaussian. Shape: (K,) or (B, K)
            sigma_z_sq: Variance of input Gaussian. Shape: (K,) or (B, K)
            
        Returns:
            Expected softmax probabilities. Shape: (K,) or (B, K)
        """
        return ll_softmax(mu_z, sigma_z_sq)["mu_a"]
    
    def get_output_variance(self, mu_z: np.ndarray, sigma_z_sq: np.ndarray) -> np.ndarray:
        """
        Get the variance of the softmax output.
        
        Args:
            mu_z: Mean of input Gaussian. Shape: (K,) or (B, K)
            sigma_z_sq: Variance of input Gaussian. Shape: (K,) or (B, K)
            
        Returns:
            Variance of softmax outputs. Shape: (K,) or (B, K)
        """
        return ll_softmax(mu_z, sigma_z_sq)["sigma_a_sq"]
    
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
