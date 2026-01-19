"""
Master script to run all experiments for the paper.
"""

import subprocess
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run all experiments")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--experiments", nargs="+", 
                        default=["mnist", "cifar10", "cifar100", "comparison"],
                        help="Which experiments to run")
    return parser.parse_args()


def run_command(cmd, dry_run=False):
    print(f"Running: {cmd}")
    if not dry_run:
        subprocess.run(cmd, shell=True, check=True)


def main():
    args = parse_args()
    
    experiments = {
        "mnist": "python scripts/train_mnist.py --config configs/mnist_default.yaml",
        "cifar10": "python scripts/train_cifar.py --config configs/cifar10_default.yaml --cifar-version 10",
        "cifar100": "python scripts/train_cifar.py --config configs/cifar100_default.yaml --cifar-version 100",
        "comparison": "python scripts/compare_methods.py --mc-samples 10000",
    }
    
    for exp_name in args.experiments:
        if exp_name in experiments:
            print(f"\n{'='*50}")
            print(f"Running experiment: {exp_name}")
            print(f"{'='*50}")
            run_command(experiments[exp_name], args.dry_run)
        else:
            print(f"Unknown experiment: {exp_name}")
    
    print("\n" + "="*50)
    print("All experiments completed!")


if __name__ == "__main__":
    main()
