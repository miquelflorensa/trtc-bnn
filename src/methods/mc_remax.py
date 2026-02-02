"""
Monte Carlo Remax (MC-Remax)

Estimates the moments of the probabilistic Remax output using Monte Carlo sampling.
This serves as the ground truth baseline for comparing analytical approximations.

Remax = ReLU(Z) / sum(ReLU(Z))
"""

import numpy as np
from typing import Dict, Optional


def mc_remax(
    mu_z: np.ndarray, 
    sigma_z_sq: np.ndarray,
    n_samples: int = 10000,
    rng: Optional[np.random.Generator] = None
) -> Dict[str, np.ndarray]:
    """
    Estimates the moments of the Remax output using Monte Carlo sampling.
    
    Samples Z ~ Normal(mu_z, sigma_z_sq) and computes Remax A for each sample,
    returning the empirical moments.
    
    Remax(Z) = ReLU(Z) / sum(ReLU(Z))

    Supports both single samples and batched inputs.

    Args:
        mu_z: Mean of the input Gaussian vector Z.
              Shape: (K,) for single sample or (B, K) for batch of B samples.
        sigma_z_sq: Variance of the input Gaussian vector Z.
              Shape: (K,) for single sample or (B, K) for batch.
        n_samples: Number of Monte Carlo samples to draw.
        rng: NumPy random generator for reproducibility. If None, creates a new one.

    Returns:
        dict: A dictionary containing:
            - mu_a: Expected value of Remax outputs (empirical mean). Shape: (K,) or (B, K)
            - sigma_a_sq: Variance of Remax outputs (empirical variance). Shape: (K,) or (B, K)
            - mu_m: Expected value of ReLU outputs. Shape: (K,) or (B, K)
            - sigma_m_sq: Variance of ReLU outputs. Shape: (K,) or (B, K)
            - cov_z_a: Covariance between Z and A. Shape: (K,) or (B, K)
            - samples: Raw Remax samples. Shape: (n_samples, K) or (n_samples, B, K)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Handle both single sample and batch inputs
    single_sample = mu_z.ndim == 1
    if single_sample:
        mu_z = mu_z[np.newaxis, :]  # (1, K)
        sigma_z_sq = sigma_z_sq[np.newaxis, :]  # (1, K)
    
    B, K = mu_z.shape
    sigma_z = np.sqrt(sigma_z_sq)  # (B, K)
    
    # Draw samples: shape (n_samples, B, K)
    Z = rng.normal(loc=mu_z, scale=sigma_z, size=(n_samples, B, K))
    
    # Compute ReLU: M = max(0, Z)
    M = np.maximum(Z, 0)  # (n_samples, B, K)
    
    # Compute Remax: A = M / sum(M)
    M_sum = np.sum(M, axis=-1, keepdims=True)  # (n_samples, B, 1)
    # Avoid division by zero - if all ReLU outputs are 0, use uniform
    M_sum = np.maximum(M_sum, 1e-10)
    A = M / M_sum  # (n_samples, B, K)
    
    # Compute empirical moments
    mu_a = np.mean(A, axis=0)  # (B, K)
    sigma_a_sq = np.var(A, axis=0)  # (B, K)
    mu_m = np.mean(M, axis=0)  # (B, K)
    sigma_m_sq = np.var(M, axis=0)  # (B, K)
    
    # Compute covariance between Z and A: Cov(Z, A) = E[Z*A] - E[Z]*E[A]
    cov_z_a = np.mean(Z * A, axis=0) - mu_z * mu_a  # (B, K)
    
    # Squeeze back to original shape if single sample
    if single_sample:
        mu_a = mu_a.squeeze(0)
        sigma_a_sq = sigma_a_sq.squeeze(0)
        mu_m = mu_m.squeeze(0)
        sigma_m_sq = sigma_m_sq.squeeze(0)
        cov_z_a = cov_z_a.squeeze(0)
        A = A.squeeze(1)  # (n_samples, K)
    
    return {
        "mu_a": mu_a,
        "sigma_a_sq": sigma_a_sq,
        "mu_m": mu_m,
        "sigma_m_sq": sigma_m_sq,
        "cov_z_a": cov_z_a,
        "samples": A
    }


def mc_remax_expected(
    mu_z: np.ndarray, 
    sigma_z_sq: np.ndarray,
    n_samples: int = 10000,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Estimate E[A] for Remax by Monte Carlo sampling.
    
    Convenience function that only returns the expected value.
    
    Args:
        mu_z: Mean of input Gaussian. Shape: (K,) or (B, K)
        sigma_z_sq: Variance of input Gaussian. Shape: (K,) or (B, K)
        n_samples: Number of Monte Carlo samples.
        rng: Random generator for reproducibility.
        
    Returns:
        Expected Remax probabilities. Shape: (K,) or (B, K)
    """
    return mc_remax(mu_z, sigma_z_sq, n_samples, rng)["mu_a"]


class MCRemax:
    """
    Monte Carlo Remax class for probabilistic classification.
    
    This estimates the Remax output distribution using Monte Carlo sampling,
    serving as the ground truth baseline for comparing analytical methods.
    """
    
    def __init__(self, n_samples: int = 10000, seed: Optional[int] = None):
        """
        Initialize MC-Remax.
        
        Args:
            n_samples: Default number of Monte Carlo samples.
            seed: Random seed for reproducibility.
        """
        self.n_samples = n_samples
        self.rng = np.random.default_rng(seed)
    
    def forward(
        self, 
        mu_z: np.ndarray, 
        sigma_z_sq: np.ndarray,
        n_samples: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Forward pass computing moments via Monte Carlo sampling.
        
        Args:
            mu_z: Mean of input Gaussian. Shape: (K,) or (B, K)
            sigma_z_sq: Variance of input Gaussian. Shape: (K,) or (B, K)
            n_samples: Number of samples (uses default if None).
            
        Returns:
            Dictionary with computed moments and samples.
        """
        n = n_samples if n_samples is not None else self.n_samples
        return mc_remax(mu_z, sigma_z_sq, n, self.rng)
    
    def get_expected_output(
        self, 
        mu_z: np.ndarray, 
        sigma_z_sq: np.ndarray,
        n_samples: Optional[int] = None
    ) -> np.ndarray:
        """
        Get the expected value of the Remax output.
        
        Args:
            mu_z: Mean of input Gaussian. Shape: (K,) or (B, K)
            sigma_z_sq: Variance of input Gaussian. Shape: (K,) or (B, K)
            n_samples: Number of samples (uses default if None).
            
        Returns:
            Expected Remax probabilities. Shape: (K,) or (B, K)
        """
        n = n_samples if n_samples is not None else self.n_samples
        return mc_remax(mu_z, sigma_z_sq, n, self.rng)["mu_a"]
    
    def get_output_variance(
        self, 
        mu_z: np.ndarray, 
        sigma_z_sq: np.ndarray,
        n_samples: Optional[int] = None
    ) -> np.ndarray:
        """
        Get the variance of the Remax output.
        
        Args:
            mu_z: Mean of input Gaussian. Shape: (K,) or (B, K)
            sigma_z_sq: Variance of input Gaussian. Shape: (K,) or (B, K)
            n_samples: Number of samples (uses default if None).
            
        Returns:
            Variance of Remax outputs. Shape: (K,) or (B, K)
        """
        n = n_samples if n_samples is not None else self.n_samples
        return mc_remax(mu_z, sigma_z_sq, n, self.rng)["sigma_a_sq"]
