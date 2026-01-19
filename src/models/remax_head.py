"""
Remax classification head for Bayesian Neural Networks.

Implementation of the Remax activation function for real-to-categorical
transformations in BNNs.
"""

import numpy as np


class RemaxHead:
    """
    Remax classification head for BNNs.
    
    Provides analytically tractable real-to-categorical transformation.
    """
    
    def __init__(self, num_classes: int):
        """
        Initialize Remax head.
        
        Args:
            num_classes: Number of output classes
        """
        self.num_classes = num_classes
        
    def forward(self, mu: np.ndarray, var: np.ndarray):
        """
        Forward pass through Remax head.
        
        Args:
            mu: Mean of the input distribution
            var: Variance of the input distribution
            
        Returns:
            Categorical probabilities
        """
        raise NotImplementedError
        
    def compute_probability(self, mu: np.ndarray, var: np.ndarray):
        """
        Compute class probabilities from Gaussian inputs.
        
        Args:
            mu: Mean vector
            var: Variance vector
            
        Returns:
            Class probabilities
        """
        raise NotImplementedError
