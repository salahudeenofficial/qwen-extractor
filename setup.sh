#!/bin/bash
# ================================================================
# Qwen-Image-Edit-2509 Setup Script
# Base Image: vastai/base-image:cuda-12.4.1-cudnn-devel-ubuntu22.04-py310-ipv2
# ================================================================

set -e  # Exit on error

echo "========================================"
echo "🚀 Qwen-Image-Edit-2509 Setup Script"
echo "========================================"
echo ""

# Print system info
echo "📋 System Information:"
echo "  Python: $(python3 --version)"
echo "  CUDA: $(nvcc --version 2>/dev/null | grep release || echo 'nvcc not found')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"
echo ""

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA 12.4 support
echo ""
echo "🔧 Installing PyTorch with CUDA 12.4 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install diffusers from git (required for QwenImageEditPlusPipeline)
echo ""
echo "🔧 Installing diffusers from GitHub (latest version required)..."
pip install git+https://github.com/huggingface/diffusers

# Install transformers from git (required for huggingface-hub 1.x compatibility)
echo ""
echo "🔧 Installing transformers from GitHub (for huggingface-hub 1.x support)..."
pip install git+https://github.com/huggingface/transformers

# Install other requirements
echo ""
echo "🔧 Installing other dependencies..."
pip install -r requirements.txt

# Verify installation
echo ""
echo "✅ Verifying installation..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')

import diffusers
print(f'Diffusers version: {diffusers.__version__}')

import transformers
print(f'Transformers version: {transformers.__version__}')

# Check if QwenImageEditPlusPipeline is available
from diffusers import QwenImageEditPlusPipeline
print('✅ QwenImageEditPlusPipeline is available!')
"

echo ""
echo "========================================"
echo "✅ Setup completed successfully!"
echo "========================================"
echo ""
echo "To run a test, execute:"
echo "  python test_qwen_edit.py"
echo ""
echo "For custom input:"
echo "  python test_qwen_edit.py --input your_image.png --prompt 'Your edit prompt'"
echo ""
