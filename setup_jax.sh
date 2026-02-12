#!/bin/bash

# JAX Setup Script for TRTC-BNN
# This script helps you install and test JAX acceleration

set -e  # Exit on error

echo "========================================================================"
echo "JAX Acceleration Setup for TRTC-BNN"
echo "========================================================================"
echo ""

# Check CUDA version
echo "[1/4] Checking CUDA availability..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
    echo "✓ CUDA detected: version $CUDA_VERSION"
    
    # Determine which JAX version to install
    if [[ "$CUDA_VERSION" =~ ^12\. ]]; then
        JAX_VERSION="cuda12"
        echo "  → Will install JAX for CUDA 12"
    elif [[ "$CUDA_VERSION" =~ ^11\. ]]; then
        JAX_VERSION="cuda11"
        echo "  → Will install JAX for CUDA 11"
    else
        echo "  ⚠ Unsupported CUDA version: $CUDA_VERSION"
        echo "  → Will install CPU-only JAX"
        JAX_VERSION="cpu"
    fi
else
    echo "⚠ CUDA not detected"
    echo "  → Will install CPU-only JAX (slower)"
    JAX_VERSION="cpu"
fi
echo ""

# Install JAX
echo "[2/4] Installing JAX..."
if [ "$JAX_VERSION" = "cpu" ]; then
    pip install jax jaxlib
else
    pip install -U "jax[${JAX_VERSION}]"
fi
echo "✓ JAX installation complete"
echo ""

# Verify installation
echo "[3/4] Verifying JAX installation..."
python -c "
import jax
import jax.numpy as jnp

print('✓ JAX imported successfully')
print(f'  JAX version: {jax.__version__}')

devices = jax.devices()
print(f'  Available devices: {devices}')

if 'cuda' in str(devices[0]).lower() or 'gpu' in str(devices[0]).lower():
    print('  ✓ GPU acceleration available!')
else:
    print('  ⚠ Running on CPU (slower)')
"
echo ""

# Run tests
echo "[4/4] Running verification tests..."
python experiments/test_jax_mc.py

echo ""
echo "========================================================================"
echo "Setup Complete!"
echo "========================================================================"
echo ""
echo "Next steps:"
echo "  1. Run a quick test:"
echo "     python experiments/train_all_methods_multiseed.py \\"
echo "         --method mm-remax --seeds 0 1 --epochs 5 --use-jax"
echo ""
echo "  2. Run full 20-seed experiment:"
echo "     python experiments/train_all_methods_multiseed.py \\"
echo "         --method all --seeds {0..19} --epochs 50 \\"
echo "         --mc-samples 10000 --eval-mc-frequency 5 --use-jax"
echo ""
echo "See experiments/JAX_ACCELERATION.md for more details."
echo "========================================================================"
