"""
Configuration utilities.
"""

import yaml
from typing import Dict, Any
from pathlib import Path


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_config(config: Dict[str, Any], config_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save config
    """
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def get_default_config() -> Dict[str, Any]:
    """
    Get default training configuration.
    
    Returns:
        Default configuration dictionary
    """
    return {
        "training": {
            "epochs": 100,
            "batch_size": 32,
            "learning_rate": 0.001,
        },
        "model": {
            "architecture": "default",
            "classification_head": "remax",
        },
        "data": {
            "dataset": "mnist",
            "validation_split": 0.1,
        }
    }
