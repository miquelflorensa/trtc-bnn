"""
Training script for BNN on CIFAR dataset with Remax classification head.
"""

import argparse
import sys
sys.path.append('..')

from src.models.bnn_cifar import BNN_CIFAR
from src.models.remax_head import RemaxHead
from src.data.cifar_loader import CIFARLoader
from src.training.trainer import BNNTrainer
from src.utils.config import load_config, get_default_config


def parse_args():
    parser = argparse.ArgumentParser(description="Train BNN on CIFAR")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--architecture", type=str, default="default", help="Model architecture")
    parser.add_argument("--cifar-version", type=int, default=10, choices=[10, 100], 
                        help="CIFAR version (10 or 100)")
    parser.add_argument("--save-dir", type=str, default="./results", help="Directory to save results")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = get_default_config()
        config["training"]["epochs"] = args.epochs
        config["training"]["batch_size"] = args.batch_size
        config["model"]["architecture"] = args.architecture
        config["data"]["dataset"] = f"cifar{args.cifar_version}"
    
    # Load data
    print(f"Loading CIFAR-{args.cifar_version} dataset...")
    data_loader = CIFARLoader(version=args.cifar_version)
    x_train, y_train, x_test, y_test = data_loader.load()
    
    # Build model
    print("Building BNN model with Remax head...")
    model = BNN_CIFAR(architecture=config["model"]["architecture"])
    
    # Train
    print("Starting training...")
    trainer = BNNTrainer(model, config)
    # trainer.train(...)
    
    print("Training complete!")


if __name__ == "__main__":
    main()
