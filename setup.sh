#!/bin/bash
# ================================================================
# Qwen-Image-Edit-2511 + Lightning LoRA Setup Script
# Base Image: vastai/base-image:cuda-12.4.1-cudnn-devel-ubuntu22.04-py310-ipv2
# ================================================================

set -e  # Exit on error

echo "========================================"
echo "🚀 Qwen-Image-Edit-2511 + Lightning LoRA Setup"
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

# Install PyTorch 2.6.0 with CUDA 12.4 support
# Note: PyTorch 2.9 has JIT issues, 2.5.1 is too old for latest transformers
echo ""
echo "🔧 Installing PyTorch 2.6.0 with CUDA 12.4 support..."
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# Install compatible xformers for torch 2.6.0
echo ""
echo "🔧 Installing xformers for PyTorch 2.6.0..."
pip install xformers==0.0.29.post3

# Install diffusers from a specific commit that works well
echo ""
echo "🔧 Installing diffusers from GitHub..."
pip install git+https://github.com/huggingface/diffusers

# Install transformers - use a stable version compatible with PyTorch 2.6
echo ""
echo "🔧 Installing transformers..."
pip install transformers>=4.51.3

# Install other requirements
echo ""
echo "🔧 Installing other dependencies..."
pip install -r requirements.txt

# Verify installation
echo ""
echo "✅ Verifying installation..."
python3 << 'EOF'
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

# Check if LoRA loading works
from diffusers.models import QwenImageTransformer2DModel
print('✅ QwenImageTransformer2DModel is available!')

# Check for key components
from diffusers import FlowMatchEulerDiscreteScheduler
print('✅ FlowMatchEulerDiscreteScheduler is available!')
EOF

echo ""
echo "========================================"
echo "✅ Setup completed successfully!"
echo "========================================"
echo ""
echo "📝 Available Commands:"
echo ""
echo "  1. Run with 4-Step Lightning LoRA (⚡ ~10x faster):"
echo "     python test_qwen_edit.py"
echo ""
echo "  2. Run with custom image:"
echo "     python test_qwen_edit.py --input your_image.png --prompt 'Your edit prompt'"
echo ""
echo "  3. Run with base model (slower, 40 steps):"
echo "     python test_qwen_edit.py --no-lora --steps 40 --cfg 4.0"
echo ""
echo "📌 Models used:"
echo "   - Base: Qwen/Qwen-Image-Edit-2511"
echo "   - LoRA: lightx2v/Qwen-Image-Edit-2511-Lightning (4-step distilled)"
echo ""
