# Qwen-Image-Edit-2511 with Lightning LoRA ⚡

Run **Qwen-Image-Edit-2511** with the **4-Step Lightning LoRA** for ~10x faster inference on Vast.ai GPU instances.

## 📦 FP8 Quantization Options

This repository includes several FP8 quantization approaches for reduced VRAM usage:

### Option 1: FP8 Layerwise Casting (Recommended)
```bash
python test_qwen_edit_fp8_layerwise.py --mode layerwise --input your_image.png
```
- Stores weights in FP8, computes in BF16
- ~50% VRAM reduction (15-20GB instead of 25-35GB)
- Works on most GPUs

### Option 2: TorchAO FP8 (RTX 4000+/A100/H100)
```bash
python test_qwen_edit_fp8_layerwise.py --mode torchao --input your_image.png
```
- Hardware-accelerated FP8 quantization
- Requires GPU with compute capability ≥ 8.9
- Best speed/memory/quality trade-off

### Option 3: 1038lab FP8 Model (Pre-quantized)
```python
from diffusers import QwenImageEditPipeline
pipe = QwenImageEditPipeline.from_pretrained(
    "1038lab/Qwen-Image-Edit-2511-FP8", torch_dtype=torch.bfloat16
).to("cuda")
```

### ComfyUI FP8 Weights Compatibility
⚠️ **Note**: ComfyUI FP8 weights (e.g., `qwen_image_edit_2511_fp8_e4m3fn_scaled.safetensors`) have different naming conventions and aren't directly compatible with diffusers. Use the converter script:
```bash
python convert_comfy_fp8.py --input comfy_weights.safetensors --analyze-only
```

## 🚀 Key Features

| Feature | Description |
|---------|-------------|
| **Base Model** | [Qwen/Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) - Latest image editing model |
| **Lightning LoRA** | [lightx2v/Qwen-Image-Edit-2511-Lightning](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning) |
| **Speed** | 4 steps vs 40 steps = ~10x faster! |
| **Quality** | Maintains editing quality with step distillation |

## 🖥️ Recommended Instance Configuration

| Parameter | Value |
|-----------|-------|
| **Base Image** | `vastai/base-image:cuda-12.4.1-cudnn-devel-ubuntu22.04-py310-ipv2` |
| **GPU** | NVIDIA L40S (45GB VRAM) or similar |
| **Disk Space** | 100GB+ (model is ~60GB + LoRA) |
| **System RAM** | 64GB+ recommended |
| **Python** | 3.10 |
| **CUDA** | 12.4.1+ |

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/salahudeenofficial/try_og_pipeline.git
cd try_og_pipeline
```

### 2. Run Setup

```bash
chmod +x setup.sh
./setup.sh
```

This will:
- Install PyTorch with CUDA 12.4 support
- Install diffusers and transformers from GitHub (latest)
- Install all other dependencies
- Verify the installation

### 3. Run a Test (with 4-Step Lightning LoRA ⚡)

**Basic test with sample image:**
```bash
python test_qwen_edit.py
```

**With your own image:**
```bash
python test_qwen_edit.py \
    --input your_image.png \
    --prompt "Transform into a watercolor painting"
```

**Multi-image editing (2-3 images):**
```bash
python test_qwen_edit.py \
    --input person1.png person2.png \
    --prompt "Both people are standing together in a beautiful garden"
```

### 4. Run with Base Model (no LoRA, slower but potentially higher quality)

```bash
python test_qwen_edit.py --no-lora --steps 40 --cfg 4.0
```

## 📋 Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--input`, `-i` | Path to input image(s) | Downloads sample |
| `--prompt`, `-p` | Edit prompt | "Transform into oil painting..." |
| `--output`, `-o` | Output path | `output_image.png` |
| `--steps` | Inference steps | 4 (LoRA) / 40 (base) |
| `--cfg` | True CFG scale | 1.0 (LoRA) / 4.0 (base) |
| `--seed` | Random seed | 42 |
| `--no-lora` | Use base model | False |
| `--lora-dir` | LoRA weights directory | `./lora_weights` |
| `--skip-cuda-check` | Skip CUDA check | False |

## 🎨 Example Prompts

**Style Transfer:**
- "Transform this into a beautiful oil painting"
- "Make this look like a pencil sketch"
- "Convert to anime style with vibrant colors"

**Lighting Effects:**
- "Add dramatic sunset lighting"
- "Create a cinematic night scene with neon lights"
- "Add soft studio lighting"

**Scene Editing:**
- "Place the person in a coffee shop"
- "Change the background to a beach at sunset"
- "Add snow to the scene"

**Person Editing:**
- "Make the person wear a red dress"
- "Change the hairstyle to curly blonde"
- "Add sunglasses and a hat"

**Multi-Image (2-3 images):**
- "The two people are shaking hands in an office"
- "Place the product next to the person"
- "Combine these images into one scene"

## 📊 Expected Performance

### With 4-Step Lightning LoRA ⚡
| Metric | Expected Value |
|--------|----------------|
| **Model Load Time** | 2-5 minutes (first run) |
| **Inference Time** | ~3-8 seconds |
| **VRAM Usage** | ~25-35GB |
| **Speed Improvement** | ~10x faster |

### With Base Model (40 steps)
| Metric | Expected Value |
|--------|----------------|
| **Model Load Time** | 2-5 minutes (first run) |
| **Inference Time** | ~30-60 seconds |
| **VRAM Usage** | ~25-35GB |

## 🔧 Troubleshooting

### Out of Memory (OOM)
If you get OOM errors, try:
```bash
# Use fewer steps
python test_qwen_edit.py --steps 2

# Or use smaller input images (resize before running)
```

### LoRA Download Failed
Manually download from: https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning
```bash
pip install "huggingface_hub[cli]"
huggingface-cli download lightx2v/Qwen-Image-Edit-2511-Lightning \
    --local-dir ./lora_weights
```

### Pipeline Not Found
Ensure diffusers is installed from git:
```bash
pip install git+https://github.com/huggingface/diffusers
```

### CUDA Errors
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
nvidia-smi
```

## 📁 Project Structure

```
try_og_pipeline/
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── setup.sh              # Setup script (run first)
├── test_qwen_edit.py     # Main test script
├── lora_weights/         # LoRA weights (downloaded automatically)
│   └── Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors
└── output_image.png      # Generated output
```

## 📚 References

- [Qwen-Image-Edit-2511 on Hugging Face](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)
- [Lightning LoRA on Hugging Face](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning)
- [Qwen-Image-Lightning GitHub](https://github.com/ModelTC/Qwen-Image-Lightning)
- [Diffusers Documentation](https://huggingface.co/docs/diffusers)
- [Vast.ai GPU Instances](https://vast.ai/)

## 📄 License

This code is provided for testing purposes. 
- Qwen-Image-Edit-2511 is licensed under Apache 2.0
- Lightning LoRA follows the base model license

Please refer to the respective model cards for full license details.
