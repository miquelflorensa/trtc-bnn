"""
Negative Log-Likelihood (NLL) computation for BNN evaluation.
"""

import numpy as np


def compute_nll(predictions: np.ndarray, targets: np.ndarray, eps: float = 1e-15) -> float:
    """
    Compute Negative Log-Likelihood.
    
    Args:
        predictions: Predicted probabilities of shape (N, C)
        targets: True labels of shape (N,) or one-hot (N, C)
        eps: Small constant for numerical stability
        
    Returns:
        NLL value
    """
    raise NotImplementedError


def compute_nll_per_sample(predictions: np.ndarray, targets: np.ndarray, 
                           eps: float = 1e-15) -> np.ndarray:
    """
    Compute NLL for each sample.
    
    Args:
        predictions: Predicted probabilities of shape (N, C)
        targets: True labels of shape (N,) or one-hot (N, C)
        eps: Small constant for numerical stability
        
    Returns:
        NLL for each sample of shape (N,)
    """
    raise NotImplementedError


def compute_nll_with_uncertainty(mu: np.ndarray, var: np.ndarray, 
                                  targets: np.ndarray) -> float:
    """
    Compute NLL considering predictive uncertainty.
    
    Args:
        mu: Mean predictions
        var: Variance of predictions
        targets: True labels
        
    Returns:
        NLL value accounting for uncertainty
    """
    raise NotImplementedError
