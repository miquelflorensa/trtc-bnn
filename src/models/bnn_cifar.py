"""
BNN architectures for CIFAR dataset using cuTAGI.
"""


class BNN_CIFAR:
    """Bayesian Neural Network for CIFAR classification."""
    
    def __init__(self, architecture: str = "default"):
        """
        Initialize BNN for CIFAR.
        
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
