"""
Master script to run all experiments for the paper.

Usage:
    python scripts/run_all_experiments.py                    # Run all
    python scripts/run_all_experiments.py --experiments mnist cifar10
    python scripts/run_all_experiments.py --dry-run          # Print commands only
"""

import subprocess
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run all experiments for the paper")
    parser.add_argument("--dry-run", action="store_true", 
                        help="Print commands without executing")
    parser.add_argument("--experiments", nargs="+", 
                        default=["mnist_remax", "mnist_softmax", "cifar10_remax", 
                                 "cifar10_softmax", "comparison_remax", "comparison_softmax"],
                        help="Which experiments to run")
    return parser.parse_args()


def run_command(cmd, dry_run=False):
    print(f"  $ {cmd}")
    if not dry_run:
        subprocess.run(cmd, shell=True, check=True)


def main():
    args = parse_args()
    
    # Define all experiments using the consolidated train.py
    experiments = {
        # Training experiments
        "mnist_remax": "python scripts/train.py --dataset mnist --head remax --epochs 50",
        "mnist_softmax": "python scripts/train.py --dataset mnist --head softmax --epochs 50",
        "cifar10_remax": "python scripts/train.py --dataset cifar10 --head remax --epochs 50",
        "cifar10_softmax": "python scripts/train.py --dataset cifar10 --head softmax --epochs 50",
        
        # Method comparison experiments
        "comparison_remax": "python scripts/compare_mm_remax_vs_mc.py --epochs 50",
        "comparison_softmax": "python scripts/compare_mm_softmax_vs_mc.py --epochs 50",
        
        # Generate figures
        "figures": "python scripts/generate_figures.py --results-dir results --output-dir figures",
    }
    
    for exp_name in args.experiments:
        if exp_name in experiments:
            print(f"\n{'='*60}")
            print(f"Running experiment: {exp_name}")
            print(f"{'='*60}")
            run_command(experiments[exp_name], args.dry_run)
        else:
            print(f"Unknown experiment: {exp_name}")
            print(f"Available: {list(experiments.keys())}")
    
    print("\n" + "="*60)
    print("All experiments completed!")


if __name__ == "__main__":
    main()
