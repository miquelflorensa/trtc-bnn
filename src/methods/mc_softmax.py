"""
Monte Carlo Softmax (MC-Softmax)

Estimates the moments of the probabilistic softmax output using Monte Carlo sampling.
This serves as the ground truth baseline for comparing analytical approximations.
"""

import numpy as np
from typing import Dict, Optional


def mc_softmax(
    mu_z: np.ndarray, 
    sigma_z_sq: np.ndarray,
    n_samples: int = 10000,
    rng: Optional[np.random.Generator] = None
) -> Dict[str, np.ndarray]:
    """
    Estimates the moments of the softmax output using Monte Carlo sampling.
    
    Samples Z ~ Normal(mu_z, sigma_z_sq) and computes softmax A for each sample,
    returning the empirical moments.

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
            - mu_a: Expected value of softmax outputs (empirical mean). Shape: (K,) or (B, K)
            - sigma_a_sq: Variance of softmax outputs (empirical variance). Shape: (K,) or (B, K)
            - cov_z_a: Covariance between Z and A. Shape: (K,) or (B, K)
            - samples: Raw softmax samples. Shape: (n_samples, K) or (n_samples, B, K)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Handle both single sample and batch inputs
    single_sample = mu_z.ndim == 1
    if single_sample:
        mu_z = mu_z[np.newaxis, :]  # (1, K)
        sigma_z_sq = sigma_z_sq[np.newaxis, :]  # (1, K)
    
    B, K = mu_z.shape
    sigma_z = np.sqrt(np.maximum(sigma_z_sq, 0))  # (B, K), ensure non-negative
    
    # Draw samples: shape (n_samples, B, K)
    Z = rng.normal(loc=mu_z, scale=sigma_z, size=(n_samples, B, K))
    
    # Handle any NaN/inf values that might have been generated
    Z = np.nan_to_num(Z, nan=0.0, posinf=500.0, neginf=-500.0)
    
    # Compute softmax per sample (numerically stable)
    Z_max = np.max(Z, axis=-1, keepdims=True)  # (n_samples, B, 1)
    e = np.exp(Z - Z_max)  # (n_samples, B, K)
    e_sum = np.sum(e, axis=-1, keepdims=True)
    e_sum = np.maximum(e_sum, 1e-30)  # Prevent division by zero
    A = e / e_sum  # (n_samples, B, K)
    
    # Compute empirical moments
    mu_a = np.mean(A, axis=0)  # (B, K)
    sigma_a_sq = np.var(A, axis=0)  # (B, K)
    
    # Compute covariance between Z and A: Cov(Z, A) = E[Z*A] - E[Z]*E[A]
    cov_z_a = np.mean(Z * A, axis=0) - mu_z * mu_a  # (B, K)
    
    # Squeeze back to original shape if single sample
    if single_sample:
        mu_a = mu_a.squeeze(0)
        sigma_a_sq = sigma_a_sq.squeeze(0)
        cov_z_a = cov_z_a.squeeze(0)
        A = A.squeeze(1)  # (n_samples, K)
    
    return {
        "mu_a": mu_a,
        "sigma_a_sq": sigma_a_sq,
        "cov_z_a": cov_z_a,
        "samples": A
    }


def mc_softmax_expected(
    mu_z: np.ndarray, 
    sigma_z_sq: np.ndarray,
    n_samples: int = 10000,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Estimate E[A] for softmax by Monte Carlo sampling.
    
    Convenience function that only returns the expected value.
    
    Args:
        mu_z: Mean of input Gaussian. Shape: (K,) or (B, K)
        sigma_z_sq: Variance of input Gaussian. Shape: (K,) or (B, K)
        n_samples: Number of Monte Carlo samples.
        rng: Random generator for reproducibility.
        
    Returns:
        Expected softmax probabilities. Shape: (K,) or (B, K)
    """
    return mc_softmax(mu_z, sigma_z_sq, n_samples, rng)["mu_a"]


class MCSoftmax:
    """
    Monte Carlo Softmax class for probabilistic classification.
    
    This estimates the softmax output distribution using Monte Carlo sampling,
    serving as the ground truth baseline for comparing analytical methods.
    """
    
    def __init__(self, n_samples: int = 10000, seed: Optional[int] = None):
        """
        Initialize MC-Softmax.
        
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
        return mc_softmax(mu_z, sigma_z_sq, n, self.rng)
    
    def get_expected_output(
        self, 
        mu_z: np.ndarray, 
        sigma_z_sq: np.ndarray,
        n_samples: Optional[int] = None
    ) -> np.ndarray:
        """
        Get the expected value of the softmax output.
        
        Args:
            mu_z: Mean of input Gaussian. Shape: (K,) or (B, K)
            sigma_z_sq: Variance of input Gaussian. Shape: (K,) or (B, K)
            n_samples: Number of samples (uses default if None).
            
        Returns:
            Expected softmax probabilities. Shape: (K,) or (B, K)
        """
        n = n_samples if n_samples is not None else self.n_samples
        return mc_softmax(mu_z, sigma_z_sq, n, self.rng)["mu_a"]
    
    def get_output_variance(
        self, 
        mu_z: np.ndarray, 
        sigma_z_sq: np.ndarray,
        n_samples: Optional[int] = None
    ) -> np.ndarray:
        """
        Get the variance of the softmax output.
        
        Args:
            mu_z: Mean of input Gaussian. Shape: (K,) or (B, K)
            sigma_z_sq: Variance of input Gaussian. Shape: (K,) or (B, K)
            n_samples: Number of samples (uses default if None).
            
        Returns:
            Variance of softmax outputs. Shape: (K,) or (B, K)
        """
        n = n_samples if n_samples is not None else self.n_samples
        return mc_softmax(mu_z, sigma_z_sq, n, self.rng)["sigma_a_sq"]
