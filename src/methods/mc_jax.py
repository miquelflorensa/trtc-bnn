"""
JAX-accelerated Monte Carlo Methods

Ultra-fast GPU/TPU-accelerated Monte Carlo sampling using JAX.
Can be 10-100x faster than NumPy implementations depending on hardware.

Installation:
    pip install jax jaxlib  # CPU
    pip install -U "jax[cuda12]"  # CUDA 12
    pip install -U "jax[cuda11]"  # CUDA 11

Usage:
    Import these instead of the numpy versions in your training code:
    
    from src.methods.mc_jax import mc_softmax_jax, mc_remax_jax
    
    # Use exactly like numpy versions
    results = mc_softmax_jax(mu_z, sigma_z_sq, n_samples=10000)
"""

import numpy as np
from typing import Dict, Optional

# Try to import JAX, fall back to NumPy if not available
try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap
    JAX_AVAILABLE = True
    print("[INFO] JAX acceleration available")
except ImportError:
    JAX_AVAILABLE = False
    print("[WARNING] JAX not available. Install with: pip install jax jaxlib")
    print("[WARNING] Falling back to NumPy (slower)")


if JAX_AVAILABLE:
    # ========================================================================
    # JAX-Accelerated Softmax MC
    # ========================================================================
    
    @jit
    def _softmax_jax(z: jnp.ndarray) -> jnp.ndarray:
        """Numerically stable softmax."""
        z_max = jnp.max(z, axis=-1, keepdims=True)
        e = jnp.exp(z - z_max)
        return e / jnp.sum(e, axis=-1, keepdims=True)
    
    def _mc_softmax_single_jax(mu_z: jnp.ndarray, sigma_z: jnp.ndarray, 
                                key: jnp.ndarray, n_samples: int) -> Dict[str, jnp.ndarray]:
        """
        MC softmax for a single batch element.
        
        Args:
            mu_z: (K,) mean vector
            sigma_z: (K,) std vector
            key: JAX random key
            n_samples: number of MC samples (must be static for JIT)
        """
        K = mu_z.shape[0]
        # Sample Z ~ N(mu_z, sigma_z^2): shape (n_samples, K)
        Z = mu_z + sigma_z * jax.random.normal(key, (n_samples, K))
        
        # Compute softmax: (n_samples, K)
        A = vmap(_softmax_jax)(Z)
        
        # Compute moments
        mu_a = jnp.mean(A, axis=0)  # (K,)
        sigma_a_sq = jnp.var(A, axis=0)  # (K,)
        cov_z_a = jnp.mean(Z * A, axis=0) - mu_z * mu_a  # (K,)
        
        return {
            'mu_a': mu_a,
            'sigma_a_sq': sigma_a_sq,
            'cov_z_a': cov_z_a
        }
    
    # Vectorize over batch dimension and JIT with static n_samples
    def _make_mc_softmax_batch_jax(n_samples: int):
        """Create a JIT-compiled batch function for a specific n_samples."""
        return jit(vmap(
            lambda mu_z, sigma_z, key: _mc_softmax_single_jax(mu_z, sigma_z, key, n_samples),
            in_axes=(0, 0, 0)
        ))
    
    # Cache compiled functions for different n_samples
    _mc_softmax_batch_cache = {}
    
    def mc_softmax_jax(mu_z: np.ndarray, sigma_z_sq: np.ndarray,
                       n_samples: int = 10000, seed: int = 0) -> Dict[str, np.ndarray]:
        """
        JAX-accelerated Monte Carlo softmax.
        
        **10-100x faster than NumPy version** depending on hardware.
        
        Args:
            mu_z: Mean of input Gaussian. Shape: (K,) or (B, K)
            sigma_z_sq: Variance of input Gaussian. Shape: (K,) or (B, K)
            n_samples: Number of MC samples
            seed: Random seed for reproducibility
            
        Returns:
            dict with 'mu_a', 'sigma_a_sq', 'cov_z_a' (NumPy arrays)
        """
        # Handle single sample
        single_sample = mu_z.ndim == 1
        if single_sample:
            mu_z = mu_z[np.newaxis, :]
            sigma_z_sq = sigma_z_sq[np.newaxis, :]
        
        B, K = mu_z.shape
        
        # Convert to JAX arrays
        mu_z_jax = jnp.array(mu_z)
        sigma_z_jax = jnp.sqrt(jnp.maximum(jnp.array(sigma_z_sq), 0))
        
        # Create random keys for each batch element
        key = jax.random.PRNGKey(seed)
        keys = jax.random.split(key, B)
        
        # Get or create compiled function for this n_samples
        if n_samples not in _mc_softmax_batch_cache:
            _mc_softmax_batch_cache[n_samples] = _make_mc_softmax_batch_jax(n_samples)
        
        # Run vectorized MC computation (JIT-compiled, GPU-accelerated)
        results = _mc_softmax_batch_cache[n_samples](mu_z_jax, sigma_z_jax, keys)
        
        # Convert back to NumPy
        mu_a = np.array(results['mu_a'])
        sigma_a_sq = np.array(results['sigma_a_sq'])
        cov_z_a = np.array(results['cov_z_a'])
        
        # Squeeze if single sample
        if single_sample:
            mu_a = mu_a.squeeze(0)
            sigma_a_sq = sigma_a_sq.squeeze(0)
            cov_z_a = cov_z_a.squeeze(0)
        
        return {
            'mu_a': mu_a,
            'sigma_a_sq': sigma_a_sq,
            'cov_z_a': cov_z_a
        }
    
    # ========================================================================
    # JAX-Accelerated Remax MC
    # ========================================================================
    
    @jit
    def _remax_jax(z: jnp.ndarray) -> jnp.ndarray:
        """Compute Remax: ReLU(Z) / sum(ReLU(Z))."""
        # ReLU: max(0, Z)
        m = jnp.maximum(z, 0)
        # Remax: M / sum(M), avoid division by zero
        m_sum = jnp.sum(m, axis=-1, keepdims=True)
        m_sum = jnp.maximum(m_sum, 1e-10)
        return m / m_sum
    
    def _mc_remax_single_jax(mu_z: jnp.ndarray, sigma_z: jnp.ndarray,
                             key: jnp.ndarray, n_samples: int) -> Dict[str, jnp.ndarray]:
        """MC remax for a single batch element."""
        K = mu_z.shape[0]
        # Sample Z ~ N(mu_z, sigma_z^2): shape (n_samples, K)
        Z = mu_z + sigma_z * jax.random.normal(key, (n_samples, K))
        
        # Compute remax: (n_samples, K)
        A = vmap(_remax_jax)(Z)
        
        # Compute moments
        mu_a = jnp.mean(A, axis=0)
        sigma_a_sq = jnp.var(A, axis=0)
        cov_z_a = jnp.mean(Z * A, axis=0) - mu_z * mu_a
        
        return {
            'mu_a': mu_a,
            'sigma_a_sq': sigma_a_sq,
            'cov_z_a': cov_z_a
        }
    
    # Vectorize over batch dimension and JIT with static n_samples
    def _make_mc_remax_batch_jax(n_samples: int):
        """Create a JIT-compiled batch function for a specific n_samples."""
        return jit(vmap(
            lambda mu_z, sigma_z, key: _mc_remax_single_jax(mu_z, sigma_z, key, n_samples),
            in_axes=(0, 0, 0)
        ))
    
    # Cache compiled functions for different n_samples
    _mc_remax_batch_cache = {}
    
    def mc_remax_jax(mu_z: np.ndarray, sigma_z_sq: np.ndarray,
                     n_samples: int = 10000, seed: int = 0) -> Dict[str, np.ndarray]:
        """
        JAX-accelerated Monte Carlo remax.
        
        **10-100x faster than NumPy version** depending on hardware.
        
        Args:
            mu_z: Mean of input Gaussian. Shape: (K,) or (B, K)
            sigma_z_sq: Variance of input Gaussian. Shape: (K,) or (B, K)
            n_samples: Number of MC samples
            seed: Random seed for reproducibility
            
        Returns:
            dict with 'mu_a', 'sigma_a_sq', 'cov_z_a' (NumPy arrays)
        """
        # Handle single sample
        single_sample = mu_z.ndim == 1
        if single_sample:
            mu_z = mu_z[np.newaxis, :]
            sigma_z_sq = sigma_z_sq[np.newaxis, :]
        
        B, K = mu_z.shape
        
        # Convert to JAX arrays
        mu_z_jax = jnp.array(mu_z)
        sigma_z_jax = jnp.sqrt(jnp.maximum(jnp.array(sigma_z_sq), 0))
        
        # Create random keys
        key = jax.random.PRNGKey(seed)
        keys = jax.random.split(key, B)
        
        # Get or create compiled function for this n_samples
        if n_samples not in _mc_remax_batch_cache:
            _mc_remax_batch_cache[n_samples] = _make_mc_remax_batch_jax(n_samples)
        
        # Run vectorized MC computation
        results = _mc_remax_batch_cache[n_samples](mu_z_jax, sigma_z_jax, keys)
        
        # Convert back to NumPy
        mu_a = np.array(results['mu_a'])
        sigma_a_sq = np.array(results['sigma_a_sq'])
        cov_z_a = np.array(results['cov_z_a'])
        
        # Squeeze if single sample
        if single_sample:
            mu_a = mu_a.squeeze(0)
            sigma_a_sq = sigma_a_sq.squeeze(0)
            cov_z_a = cov_z_a.squeeze(0)
        
        return {
            'mu_a': mu_a,
            'sigma_a_sq': sigma_a_sq,
            'cov_z_a': cov_z_a
        }

else:
    # Fallback to NumPy versions if JAX not available
    from src.methods.mc_softmax import mc_softmax
    from src.methods.mc_remax import mc_remax
    
    def mc_softmax_jax(mu_z: np.ndarray, sigma_z_sq: np.ndarray,
                       n_samples: int = 10000, seed: int = 0) -> Dict[str, np.ndarray]:
        """Fallback to NumPy implementation."""
        rng = np.random.default_rng(seed)
        results = mc_softmax(mu_z, sigma_z_sq, n_samples=n_samples, rng=rng)
        return {k: results[k] for k in ['mu_a', 'sigma_a_sq', 'cov_z_a']}
    
    def mc_remax_jax(mu_z: np.ndarray, sigma_z_sq: np.ndarray,
                     n_samples: int = 10000, seed: int = 0) -> Dict[str, np.ndarray]:
        """Fallback to NumPy implementation."""
        rng = np.random.default_rng(seed)
        results = mc_remax(mu_z, sigma_z_sq, n_samples=n_samples, rng=rng)
        return {k: results[k] for k in ['mu_a', 'sigma_a_sq', 'cov_z_a']}


# ============================================================================
# Benchmarking Utility
# ============================================================================

def benchmark_jax_vs_numpy():
    """
    Benchmark JAX vs NumPy implementations.
    
    Run this to see the speedup on your hardware:
        python -c "from src.methods.mc_jax import benchmark_jax_vs_numpy; benchmark_jax_vs_numpy()"
    """
    import time
    
    print("\n" + "="*70)
    print("JAX vs NumPy Monte Carlo Benchmark")
    print("="*70)
    
    # Test parameters
    B, K = 128, 10  # Typical batch size and num classes
    n_samples = 10000
    
    mu_z = np.random.randn(B, K).astype(np.float32)
    sigma_z_sq = np.abs(np.random.randn(B, K).astype(np.float32))
    
    if JAX_AVAILABLE:
        # Warmup JAX (JIT compilation)
        print("\nWarming up JAX (compiling JIT functions)...")
        _ = mc_softmax_jax(mu_z[:1], sigma_z_sq[:1], n_samples=100, seed=0)
        _ = mc_remax_jax(mu_z[:1], sigma_z_sq[:1], n_samples=100, seed=0)
        
        # Benchmark JAX
        print("\nBenchmarking JAX...")
        start = time.time()
        for _ in range(5):
            _ = mc_softmax_jax(mu_z, sigma_z_sq, n_samples=n_samples, seed=0)
        jax_softmax_time = (time.time() - start) / 5
        
        start = time.time()
        for _ in range(5):
            _ = mc_remax_jax(mu_z, sigma_z_sq, n_samples=n_samples, seed=0)
        jax_remax_time = (time.time() - start) / 5
    
    # Benchmark NumPy
    print("Benchmarking NumPy...")
    from src.methods.mc_softmax import mc_softmax
    from src.methods.mc_remax import mc_remax
    
    start = time.time()
    for _ in range(5):
        _ = mc_softmax(mu_z, sigma_z_sq, n_samples=n_samples)
    numpy_softmax_time = (time.time() - start) / 5
    
    start = time.time()
    for _ in range(5):
        _ = mc_remax(mu_z, sigma_z_sq, n_samples=n_samples)
    numpy_remax_time = (time.time() - start) / 5
    
    # Print results
    print("\n" + "="*70)
    print("Results (averaged over 5 runs)")
    print("="*70)
    print(f"Configuration: Batch size={B}, Classes={K}, MC samples={n_samples}")
    print()
    
    if JAX_AVAILABLE:
        print(f"MC Softmax:")
        print(f"  NumPy:  {numpy_softmax_time:.3f}s")
        print(f"  JAX:    {jax_softmax_time:.3f}s")
        print(f"  Speedup: {numpy_softmax_time/jax_softmax_time:.1f}x faster")
        print()
        print(f"MC Remax:")
        print(f"  NumPy:  {numpy_remax_time:.3f}s")
        print(f"  JAX:    {jax_remax_time:.3f}s")
        print(f"  Speedup: {numpy_remax_time/jax_remax_time:.1f}x faster")
    else:
        print(f"MC Softmax (NumPy): {numpy_softmax_time:.3f}s")
        print(f"MC Remax (NumPy):   {numpy_remax_time:.3f}s")
        print()
        print("[INFO] Install JAX to see speedup comparison")
    
    print("="*70)
    
    # Estimate time savings
    if JAX_AVAILABLE:
        batches_per_epoch = 4000 // B  # ~32 batches for val set
        epochs_with_mc = 10  # e.g., every 5 epochs over 50 total
        
        numpy_total = (numpy_softmax_time + numpy_remax_time) * batches_per_epoch * epochs_with_mc
        jax_total = (jax_softmax_time + jax_remax_time) * batches_per_epoch * epochs_with_mc
        
        print(f"\nTime per seed with MC every 5 epochs (50 total epochs):")
        print(f"  NumPy: ~{numpy_total/60:.1f} minutes")
        print(f"  JAX:   ~{jax_total/60:.1f} minutes")
        print(f"  Time saved: ~{(numpy_total-jax_total)/60:.1f} minutes per seed")
        print(f"  For 20 seeds: ~{(numpy_total-jax_total)*20/3600:.1f} hours saved!")
        print("="*70 + "\n")


if __name__ == "__main__":
    benchmark_jax_vs_numpy()
