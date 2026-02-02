"""
MNIST data loading and preprocessing.
"""

import numpy as np
from typing import Tuple


class MNISTLoader:
    """Data loader for MNIST dataset."""
    
    def __init__(self, data_dir: str = "./data"):
        """
        Initialize MNIST loader.
        
        Args:
            data_dir: Directory to store/load data
        """
        self.data_dir = data_dir
        
    def load(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load MNIST dataset.
        
        Returns:
            Tuple of (x_train, y_train, x_test, y_test)
        """
        raise NotImplementedError
        
    def preprocess(self, x: np.ndarray) -> np.ndarray:
        """
        Preprocess MNIST images.
        
        Args:
            x: Raw image data
            
        Returns:
            Preprocessed images
        """
        raise NotImplementedError
        
    def get_dataloader(self, batch_size: int = 32, shuffle: bool = True):
        """
        Get data loader for training.
        
        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle data
            
        Returns:
            Data loader
        """
        raise NotImplementedError
