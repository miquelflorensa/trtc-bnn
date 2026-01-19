"""
Moment-Matching Remax (MM-Remax)

Calculates the moments of the probabilistic Remax output analytically
using log-normal moment matching through the ReLU and normalization.

Remax = ReLU(Z) / sum(ReLU(Z))

This is the third proposed method for analytically tractable real-to-categorical
transformations in Bayesian Neural Networks. It is more numerically stable than
softmax as it avoids the exponential function.
"""

import numpy as np
from scipy.stats import norm
from typing import Dict


def _phi(x: np.ndarray) -> np.ndarray:
    """Probability Density Function (PDF) of the standard normal distribution."""
    return norm.pdf(x)


def _Phi(x: np.ndarray) -> np.ndarray:
    """Cumulative Distribution Function (CDF) of the standard normal distribution."""
    return norm.cdf(x)


def mm_remax(
    mu_z: np.ndarray, 
    sigma_z_sq: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Calculates the moments of the Remax output analytically using moment matching.
    
    Remax(Z) = ReLU(Z) / sum(ReLU(Z))
    
    where ReLU is applied element-wise and the normalization ensures outputs sum to 1.

    Supports both single samples and batched inputs.

    Args:
        mu_z: Mean of the input Gaussian vector Z.
              Shape: (K,) for single sample or (B, K) for batch of B samples.
        sigma_z_sq: Variance of the input Gaussian vector Z.
              Shape: (K,) for single sample or (B, K) for batch.

    Returns:
        dict: A dictionary containing:
            - mu_a: Expected value of Remax outputs. Shape: (K,) or (B, K)
            - sigma_a_sq: Variance of Remax outputs. Shape: (K,) or (B, K)
            - mu_m: Mean of ReLU outputs M = ReLU(Z). Shape: (K,) or (B, K)
            - sigma_m_sq: Variance of ReLU outputs. Shape: (K,) or (B, K)
            - cov_z_m: Covariance between Z and M. Shape: (K,) or (B, K)
            - cov_z_a: Covariance between Z and A. Shape: (K,) or (B, K)
            - mu_ln_a: Mean of log(A). Shape: (K,) or (B, K)
            - sigma_ln_a_sq: Variance of log(A). Shape: (K,) or (B, K)
    """
    epsilon = 1e-7
    
    # Handle both single sample and batch inputs
    single_sample = mu_z.ndim == 1
    if single_sample:
        mu_z = mu_z[np.newaxis, :]  # (1, K)
        sigma_z_sq = sigma_z_sq[np.newaxis, :]  # (1, K)
    
    sigma_z = np.sqrt(sigma_z_sq)  # (B, K)
    
    # Standardized input for CDF/PDF calculations
    # Handle case where sigma_z is very small
    alpha = np.divide(mu_z, sigma_z, out=np.zeros_like(mu_z), where=sigma_z > epsilon)  # (B, K)
    
    # 1. Moments of M = ReLU(Z) = max(0, Z)
    # Using the analytical formula for the mean of a rectified Gaussian
    mu_m = np.maximum(sigma_z * _phi(alpha) + mu_z * _Phi(alpha), epsilon)  # (B, K)
    
    # Variance of rectified Gaussian
    sigma_m_sq = np.maximum(
        (mu_z**2 + sigma_z_sq) * _Phi(alpha) + mu_z * sigma_z * _phi(alpha) - mu_m**2, 
        epsilon
    )  # (B, K)
    
    # Covariance between Z and M
    cov_z_m = sigma_z_sq * _Phi(alpha)  # (B, K)
    
    # 2. Log-space moments of M (treating M as approximately log-normal)
    sigma_ln_m_sq = np.log(1 + sigma_m_sq / (mu_m**2))  # (B, K)
    mu_ln_m = np.log(mu_m) - 0.5 * sigma_ln_m_sq  # (B, K)
    
    # 3. Moments of M_tilde = sum(M)
    mu_m_tilde = np.sum(mu_m, axis=-1, keepdims=True)  # (B, 1)
    sigma_m_tilde_sq = np.sum(sigma_m_sq, axis=-1, keepdims=True)  # (B, 1)
    cov_m_m_tilde = sigma_m_sq  # (B, K)
    
    # 4. Log-space moments of M_tilde
    sigma_ln_m_tilde_sq = np.log(1 + sigma_m_tilde_sq / (mu_m_tilde**2))  # (B, 1)
    mu_ln_m_tilde = np.log(mu_m_tilde) - 0.5 * sigma_ln_m_tilde_sq  # (B, 1)
    
    # 5. Covariance in log-space
    cov_ln_m_ln_m_tilde = np.log(1 + cov_m_m_tilde / (mu_m * mu_m_tilde))  # (B, K)
    
    # 6. Moments of ln(A) = ln(M) - ln(M_tilde)
    mu_ln_a = mu_ln_m - mu_ln_m_tilde  # (B, K)
    sigma_ln_a_sq = sigma_ln_m_sq + sigma_ln_m_tilde_sq - 2 * cov_ln_m_ln_m_tilde  # (B, K)
    
    # 7. Final moments of A (treating as log-normal)
    mu_a = np.maximum(np.exp(mu_ln_a + 0.5 * sigma_ln_a_sq), epsilon)  # (B, K)
    sigma_a_sq = mu_a**2 * (np.exp(sigma_ln_a_sq) - 1)  # (B, K)
    
    # 8. Covariance between Z and A (for gradient computation)
    cov_z_a = mu_a * cov_z_m * (1/mu_m - 1/mu_m_tilde)  # (B, K)
    
    # Squeeze back to original shape if single sample
    if single_sample:
        mu_a = mu_a.squeeze(0)
        sigma_a_sq = sigma_a_sq.squeeze(0)
        mu_m = mu_m.squeeze(0)
        sigma_m_sq = sigma_m_sq.squeeze(0)
        cov_z_m = cov_z_m.squeeze(0)
        cov_z_a = cov_z_a.squeeze(0)
        mu_ln_a = mu_ln_a.squeeze(0)
        sigma_ln_a_sq = sigma_ln_a_sq.squeeze(0)

    return {
        "mu_a": mu_a, 
        "sigma_a_sq": sigma_a_sq,
        "mu_m": mu_m, 
        "sigma_m_sq": sigma_m_sq,
        "cov_z_m": cov_z_m,
        "cov_z_a": cov_z_a,
        "mu_ln_a": mu_ln_a,
        "sigma_ln_a_sq": sigma_ln_a_sq,
    }


def deterministic_remax(mu_z: np.ndarray) -> np.ndarray:
    """
    Compute the deterministic Remax of the input means.
    
    Remax(z) = ReLU(z) / sum(ReLU(z))
    
    Args:
        mu_z: Mean of the input. Shape: (K,) or (B, K)
        
    Returns:
        Remax probabilities. Shape: (K,) or (B, K)
    """
    single_sample = mu_z.ndim == 1
    if single_sample:
        mu_z = mu_z[np.newaxis, :]
    
    m = np.maximum(mu_z, 0)  # ReLU: (B, K)
    m_sum = np.sum(m, axis=-1, keepdims=True)  # (B, 1)
    
    # Avoid division by zero - if all ReLU outputs are 0, use uniform
    m_sum = np.maximum(m_sum, 1e-10)
    result = m / m_sum
    
    if single_sample:
        result = result.squeeze(0)
    
    return result


class MMRemax:
    """
    Moment-Matching Remax class for probabilistic classification.
    
    This implements the analytically tractable Remax transformation
    that propagates uncertainty from input Gaussians to output probabilities
    using log-normal moment matching through ReLU.
    
    Remax is more numerically stable than softmax for BNNs as it uses ReLU
    instead of exponentials.
    """
    
    def __init__(self):
        """Initialize MM-Remax."""
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
        return mm_remax(mu_z, sigma_z_sq)
    
    def get_expected_output(self, mu_z: np.ndarray, sigma_z_sq: np.ndarray) -> np.ndarray:
        """
        Get the expected value of the Remax output.
        
        Args:
            mu_z: Mean of input Gaussian. Shape: (K,) or (B, K)
            sigma_z_sq: Variance of input Gaussian. Shape: (K,) or (B, K)
            
        Returns:
            Expected Remax probabilities. Shape: (K,) or (B, K)
        """
        return mm_remax(mu_z, sigma_z_sq)["mu_a"]
    
    def get_output_variance(self, mu_z: np.ndarray, sigma_z_sq: np.ndarray) -> np.ndarray:
        """
        Get the variance of the Remax output.
        
        Args:
            mu_z: Mean of input Gaussian. Shape: (K,) or (B, K)
            sigma_z_sq: Variance of input Gaussian. Shape: (K,) or (B, K)
            
        Returns:
            Variance of Remax outputs. Shape: (K,) or (B, K)
        """
        return mm_remax(mu_z, sigma_z_sq)["sigma_a_sq"]
    
    @staticmethod
    def deterministic(mu_z: np.ndarray) -> np.ndarray:
        """
        Compute deterministic Remax (no uncertainty propagation).
        
        Args:
            mu_z: Mean of input. Shape: (K,) or (B, K)
            
        Returns:
            Remax probabilities. Shape: (K,) or (B, K)
        """
        return deterministic_remax(mu_z)
