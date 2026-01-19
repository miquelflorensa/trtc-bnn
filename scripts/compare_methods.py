"""
Script to compare the three proposed methods with MC sampling baseline.
"""

import argparse
import numpy as np
import sys
sys.path.append('..')

from src.methods.method_1 import Method1
from src.methods.method_2 import Method2
from src.methods.method_3 import Method3
from src.methods.mc_samples import MCSampler
from src.methods.comparison import MethodComparison
from src.utils.visualization import plot_method_comparison


def parse_args():
    parser = argparse.ArgumentParser(description="Compare proposed methods with MC sampling")
    parser.add_argument("--mc-samples", type=int, default=1000, help="Number of MC samples")
    parser.add_argument("--num-tests", type=int, default=100, help="Number of test cases")
    parser.add_argument("--num-classes", type=int, default=10, help="Number of classes")
    parser.add_argument("--output-dir", type=str, default="./results", help="Output directory")
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("Initializing methods...")
    comparison = MethodComparison(mc_samples=args.mc_samples)
    
    print(f"Running comparison with {args.num_tests} test cases...")
    
    # Generate test cases
    test_cases = []
    for i in range(args.num_tests):
        mu = np.random.randn(args.num_classes)
        var = np.abs(np.random.randn(args.num_classes)) + 0.1
        test_cases.append({"mu": mu, "var": var})
    
    # Run comparison
    results = comparison.run_benchmark(test_cases)
    
    print("\n=== Comparison Results ===")
    print(f"Method 1 - Mean Error: {results.get('method_1_error', 'N/A')}")
    print(f"Method 2 - Mean Error: {results.get('method_2_error', 'N/A')}")
    print(f"Method 3 - Mean Error: {results.get('method_3_error', 'N/A')}")
    
    # Plot comparison
    plot_method_comparison(results, save_path=f"{args.output_dir}/method_comparison.pdf")
    print(f"\nPlot saved to {args.output_dir}/method_comparison.pdf")


if __name__ == "__main__":
    main()
