"""
Unit tests for proposed methods.
"""

import numpy as np
import pytest
import sys
sys.path.append('..')

from src.methods.method_1 import Method1
from src.methods.method_2 import Method2
from src.methods.method_3 import Method3
from src.methods.mc_samples import MCSampler


class TestMethod1:
    """Tests for Method 1."""
    
    def test_probability_sum_to_one(self):
        """Probabilities should sum to 1."""
        # TODO: Implement test
        pass
    
    def test_probability_non_negative(self):
        """All probabilities should be non-negative."""
        # TODO: Implement test
        pass


class TestMethod2:
    """Tests for Method 2."""
    
    def test_probability_sum_to_one(self):
        """Probabilities should sum to 1."""
        # TODO: Implement test
        pass
    
    def test_probability_non_negative(self):
        """All probabilities should be non-negative."""
        # TODO: Implement test
        pass


class TestMethod3:
    """Tests for Method 3."""
    
    def test_probability_sum_to_one(self):
        """Probabilities should sum to 1."""
        # TODO: Implement test
        pass
    
    def test_probability_non_negative(self):
        """All probabilities should be non-negative."""
        # TODO: Implement test
        pass


class TestMCSampler:
    """Tests for MC sampler."""
    
    def test_convergence(self):
        """MC estimates should converge with more samples."""
        # TODO: Implement test
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
