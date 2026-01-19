"""
Unit tests for metrics computation.
"""

import numpy as np
import pytest
import sys
sys.path.append('..')

from src.metrics.nll import compute_nll, compute_nll_per_sample
from src.metrics.ece import compute_ece, compute_mce
from src.metrics.accuracy import compute_accuracy


class TestNLL:
    """Tests for NLL computation."""
    
    def test_nll_perfect_predictions(self):
        """NLL should be 0 for perfect predictions."""
        # TODO: Implement test
        pass
    
    def test_nll_random_predictions(self):
        """NLL should be positive for random predictions."""
        # TODO: Implement test
        pass
    
    def test_nll_shape(self):
        """NLL per sample should have correct shape."""
        # TODO: Implement test
        pass


class TestECE:
    """Tests for ECE computation."""
    
    def test_ece_perfect_calibration(self):
        """ECE should be 0 for perfectly calibrated predictions."""
        # TODO: Implement test
        pass
    
    def test_ece_range(self):
        """ECE should be between 0 and 1."""
        # TODO: Implement test
        pass


class TestAccuracy:
    """Tests for accuracy computation."""
    
    def test_accuracy_perfect(self):
        """Accuracy should be 1 for perfect predictions."""
        # TODO: Implement test
        pass
    
    def test_accuracy_random(self):
        """Accuracy should be around 1/num_classes for random predictions."""
        # TODO: Implement test
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
