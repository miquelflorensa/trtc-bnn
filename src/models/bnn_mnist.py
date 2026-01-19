"""
BNN architectures for MNIST dataset using cuTAGI.
"""


class BNN_MNIST:
    """Bayesian Neural Network for MNIST classification."""
    
    def __init__(self, architecture: str = "default"):
        """
        Initialize BNN for MNIST.
        
        Args:
            architecture: Model architecture type
        """
        self.architecture = architecture
        
    def build_model(self):
        """Build the BNN model."""
        raise NotImplementedError
        
    def forward(self, x):
        """Forward pass through the network."""
        raise NotImplementedError
