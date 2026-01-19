"""
Accuracy metrics for BNN evaluation.
"""

import numpy as np


def compute_accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Compute classification accuracy.
    
    Args:
        predictions: Predicted probabilities of shape (N, C) or class indices (N,)
        targets: True labels of shape (N,)
        
    Returns:
        Accuracy value
    """
    raise NotImplementedError


def compute_top_k_accuracy(predictions: np.ndarray, targets: np.ndarray, 
                           k: int = 5) -> float:
    """
    Compute top-k classification accuracy.
    
    Args:
        predictions: Predicted probabilities of shape (N, C)
        targets: True labels of shape (N,)
        k: Number of top predictions to consider
        
    Returns:
        Top-k accuracy value
    """
    raise NotImplementedError


def compute_error_rate(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Compute classification error rate.
    
    Args:
        predictions: Predicted probabilities or class indices
        targets: True labels
        
    Returns:
        Error rate
    """
    raise NotImplementedError
