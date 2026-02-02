"""
Test script for the SigmaVScheduler class.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

# Import the scheduler from the training script
import importlib.util
spec = importlib.util.spec_from_file_location("train_module", 
                                               Path(__file__).parent / "train_3cnn_cifar10.py")
train_module = importlib.util.module_from_spec(spec)

# Define the SigmaVScheduler locally for testing
class SigmaVScheduler:
    """
    Scheduler for observation noise (sigma_v) during training.
    
    Supports different scheduling strategies:
    - 'constant': No change (sigma_v stays at sigma_v_min)
    - 'linear': Linear interpolation from sigma_v_min to sigma_v_max
    - 'cosine': Cosine annealing from sigma_v_min to sigma_v_max
    - 'exponential': Exponential decay/growth from sigma_v_min to sigma_v_max
    """
    
    def __init__(self, sigma_v_min: float, sigma_v_max: float, 
                 total_epochs: int, schedule_type: str = 'constant'):
        """
        Initialize the scheduler.
        
        Args:
            sigma_v_min: Minimum (ending) sigma_v value
            sigma_v_max: Maximum (starting) sigma_v value
            total_epochs: Total number of training epochs
            schedule_type: Type of schedule ('constant', 'linear', 'cosine', 'exponential')
        """
        self.sigma_v_min = sigma_v_min
        self.sigma_v_max = sigma_v_max
        self.total_epochs = total_epochs
        self.schedule_type = schedule_type
        
    def get_sigma_v(self, epoch: int) -> float:
        """
        Get sigma_v value for the current epoch.
        Decays from sigma_v_max (start) to sigma_v_min (end).
        
        Args:
            epoch: Current epoch (0-indexed)
            
        Returns:
            sigma_v value for this epoch
        """
        if self.schedule_type == 'constant':
            return self.sigma_v_max
            
        # Progress through training (0.0 to 1.0)
        progress = epoch / max(self.total_epochs - 1, 1)
        
        if self.schedule_type == 'linear':
            # Linear decay from max to min
            return self.sigma_v_max - (self.sigma_v_max - self.sigma_v_min) * progress
            
        elif self.schedule_type == 'cosine':
            # Cosine annealing from max to min
            return self.sigma_v_min + (self.sigma_v_max - self.sigma_v_min) * \
                   (1 + np.cos(progress * np.pi)) / 2
                   
        elif self.schedule_type == 'exponential':
            # Exponential decay from max to min
            if self.sigma_v_min > 0 and self.sigma_v_max > 0:
                ratio = self.sigma_v_min / self.sigma_v_max
                return self.sigma_v_max * (ratio ** progress)
            else:
                # Fallback to linear if either is 0
                return self.sigma_v_max - (self.sigma_v_max - self.sigma_v_min) * progress
                
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")


def test_schedulers():
    """Test and visualize different scheduler types."""
    
    # Test parameters
    epochs = 50
    sigma_v_min = 0.001  # Ending value (low noise for fine-tuning)
    sigma_v_max = 0.1    # Starting value (high noise for regularization)
    
    # Create schedulers
    schedulers = {
        'constant': SigmaVScheduler(sigma_v_max, sigma_v_max, epochs, 'constant'),
        'linear': SigmaVScheduler(sigma_v_min, sigma_v_max, epochs, 'linear'),
        'cosine': SigmaVScheduler(sigma_v_min, sigma_v_max, epochs, 'cosine'),
        'exponential': SigmaVScheduler(sigma_v_min, sigma_v_max, epochs, 'exponential'),
    }
    
    # Generate values
    epochs_list = np.arange(epochs)
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, (name, scheduler) in enumerate(schedulers.items()):
        values = [scheduler.get_sigma_v(epoch) for epoch in epochs_list]
        
        axes[idx].plot(epochs_list, values, linewidth=2, marker='o', markersize=3)
        axes[idx].set_title(f'{name.capitalize()} Schedule', fontsize=14, fontweight='bold')
        axes[idx].set_xlabel('Epoch', fontsize=12)
        axes[idx].set_ylabel('σ_v (Observation Noise)', fontsize=12)
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_xlim(-1, epochs)
        
        # Add min/max horizontal lines
        axes[idx].axhline(y=sigma_v_max, color='red', linestyle='--', 
                         alpha=0.5, label=f'Start (Max): {sigma_v_max}')
        if name != 'constant':
            axes[idx].axhline(y=sigma_v_min, color='green', linestyle='--', 
                             alpha=0.5, label=f'End (Min): {sigma_v_min}')
        axes[idx].legend()
        
        # Print first and last values
        print(f"\n{name.capitalize()} Schedule:")
        print(f"  Epoch 0:  σ_v = {values[0]:.6f} (START - high noise)")
        print(f"  Epoch 25: σ_v = {values[25]:.6f}")
        print(f"  Epoch 49: σ_v = {values[-1]:.6f} (END - low noise)")
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(__file__).parent.parent / "results" / "sigma_v_schedules.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n\nPlot saved to: {output_path}")
    
    # Also save as PNG
    png_path = output_path.with_suffix('.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {png_path}")
    
    plt.show()


if __name__ == "__main__":
    print("Testing SigmaVScheduler...")
    print("=" * 70)
    test_schedulers()
    print("\n" + "=" * 70)
    print("Test complete!")
