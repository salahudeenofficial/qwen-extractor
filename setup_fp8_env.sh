#!/bin/bash
# ================================================================
# LightX2V FP8 Environment Setup (Compatible Versions)
# 
# This script sets up the correct PyTorch version for LightX2V FP8
# to work with sgl_kernel and flash_attn
# ================================================================

set -e

echo "========================================"
echo "🚀 LightX2V FP8 Environment Setup"
echo "========================================"
echo ""

# Print system info
echo "📋 Current Environment:"
python3 --version
echo ""
echo "Current PyTorch version:"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null || echo "PyTorch not installed"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "NVIDIA SMI not available"
echo ""

# Step 1: Uninstall current incompatible packages
echo "📦 Step 1: Removing incompatible packages..."
pip uninstall -y torch torchvision torchaudio flash-attn sgl-kernel 2>/dev/null || true

# Step 2: Install PyTorch 2.5.1 with CUDA 12.4 (compatible with sgl_kernel)
echo ""
echo "📦 Step 2: Installing PyTorch 2.5.1 (compatible version)..."
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# Step 3: Install flash-attn from source
echo ""
echo "📦 Step 3: Installing Flash Attention (building from source)..."
pip install flash-attn --no-build-isolation

# Step 4: Install sgl_kernel
echo ""
echo "📦 Step 4: Installing sgl_kernel..."
pip install sgl-kernel

# Step 5: Reinstall diffusers (compatible version)
echo ""
echo "📦 Step 5: Installing diffusers..."
pip install git+https://github.com/huggingface/diffusers

# Step 6: Install other dependencies
echo ""
echo "📦 Step 6: Installing other dependencies..."
pip install transformers>=4.51.3 accelerate safetensors einops Pillow requests tqdm

# Step 7: Reinstall LightX2V
echo ""
echo "📦 Step 7: Reinstalling LightX2V..."
cd /workspace/try_og_pipeline/LightX2V
pip install -v -e .
cd /workspace/try_og_pipeline

# Verify installation
echo ""
echo "========================================"
echo "✅ Verifying Installation"
echo "========================================"

python3 << 'EOF'
import torch
print(f"✅ PyTorch: {torch.__version__}")
print(f"   CUDA available: {torch.cuda.is_available()}")
print(f"   CUDA version: {torch.version.cuda}")

try:
    import flash_attn
    print(f"✅ Flash Attention: {flash_attn.__version__}")
except Exception as e:
    print(f"❌ Flash Attention: {e}")

try:
    import sgl_kernel
    print(f"✅ sgl_kernel loaded successfully")
except Exception as e:
    print(f"❌ sgl_kernel: {e}")

try:
    from lightx2v import LightX2VPipeline
    print(f"✅ LightX2V imported successfully")
except Exception as e:
    print(f"❌ LightX2V: {e}")

try:
    from diffusers import QwenImageEditPlusPipeline
    print(f"✅ Diffusers Qwen pipeline available")
except Exception as e:
    print(f"❌ Diffusers: {e}")
EOF

echo ""
echo "========================================"
echo "🎉 Setup Complete!"
echo "========================================"
echo ""
echo "Now run the FP8 test:"
echo "  python test_lightx2v_vton.py --mode fp8 --person person.jpg --cloth cloth.png"
echo ""
