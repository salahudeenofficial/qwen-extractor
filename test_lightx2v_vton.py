#!/usr/bin/env python3
"""
LightX2V Virtual Try-On Test Script

This script uses the LightX2V framework for high-performance inference
with Qwen-Image-Edit-2511 model for virtual try-on tasks.

Features:
- FP8 quantization support (~50% memory reduction)
- 4-step Lightning LoRA (~10x speedup)
- CPU offloading for low-VRAM GPUs
- CFG parallelism for faster inference

Performance (on H100):
- BF16 + 4-step LoRA: ~4-6 seconds
- FP8 + 4-step distillation: ~3-4 seconds
- Combined optimizations: up to 42x speedup vs base model
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image


# Virtual Try-On Prompts
VTON_PROMPT_CN = """将图片 1 中的绿色遮罩区域仅用于判断服装属于上半身或下半身，不要将服装限制在遮罩范围内。

将图片 2 中的服装自然地穿戴到图片 1 中的人物身上，保持图片 2 中服装的完整形状、袖长和轮廓。无论图片 2 是单独的服装图还是人物穿着该服装的图，都应准确地转移服装，同时保留其原始面料质感、材质细节和颜色准确性。

确保图片 1 中人物的面部、头发和皮肤完全保持不变。光照与阴影应自然匹配图片 1 的环境，但服装的材质外观必须忠实于图片 2。

保持边缘平滑融合、阴影逼真，整体效果自然且不改变人物的身份特征。"""

VTON_PROMPT_EN = """Use the green mask area in image 1 only to determine if the garment belongs to upper or lower body, do not restrict the garment to the mask area.

Naturally dress the person in image 1 with the garment from image 2, maintaining the complete shape, sleeve length, and silhouette of the garment from image 2. Whether image 2 shows the garment alone or on a model, accurately transfer the garment while preserving its original fabric texture, material details, and color accuracy.

Ensure the face, hair, and skin of the person in image 1 remain completely unchanged. Lighting and shadows should naturally match the environment of image 1, but the material appearance of the garment must stay faithful to image 2.

Keep edges smoothly blended, shadows realistic, and the overall effect natural without altering the person's identity features."""


def check_cuda():
    """Check CUDA availability and print GPU info."""
    print("=" * 60)
    print("🔍 CUDA Environment Check")
    print("=" * 60)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_memory = props.total_memory / (1024 ** 3)
            print(f"\n📌 GPU {i}: {props.name}")
            print(f"   Total Memory: {total_memory:.2f} GB")
            print(f"   Compute Capability: {props.major}.{props.minor}")
            
            # Check FP8 hardware support
            fp8_hw = props.major >= 9 or (props.major == 8 and props.minor >= 9)
            print(f"   FP8 Hardware: {'✅ Native' if fp8_hw else '⚠️ Emulated'}")
    else:
        print("❌ CUDA is not available!")
        sys.exit(1)
    
    print("=" * 60)
    return True


def find_model_paths():
    """Find model paths in common locations."""
    possible_paths = [
        "models/Qwen-Image-Edit-2511",
        "./Qwen-Image-Edit-2511",
        "/models/Qwen-Image-Edit-2511",
        os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen-Image-Edit-2511"),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None


def find_lora_path():
    """Find LoRA weights path."""
    possible_paths = [
        "models/Qwen-Image-Edit-2511-Lightning/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-fp32.safetensors",
        "models/Qwen-Image-Edit-2511-Lightning/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
        "lora_weights/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None


def find_fp8_path():
    """Find FP8 quantized weights path."""
    possible_paths = [
        "models/Qwen-Image-Edit-2511-Lightning/qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning.safetensors",
        "models/qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning.safetensors",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None


def run_lightx2v_vton(
    person_image_path: str,
    cloth_image_path: str,
    output_path: str,
    mode: str = "lora",  # "lora", "fp8", or "base"
    enable_offload: bool = False,
    prompt: str = None,
    seed: int = 42,
    steps: int = 4,
):
    """
    Run Virtual Try-On using LightX2V framework.
    
    Args:
        person_image_path: Path to person image (with green mask)
        cloth_image_path: Path to cloth/garment image
        output_path: Path to save output
        mode: "lora" for BF16+LoRA, "fp8" for FP8+distillation, "base" for 40-step base
        enable_offload: Enable CPU offloading for low VRAM
        prompt: VTON prompt (uses Chinese prompt by default)
        seed: Random seed
        steps: Inference steps (4 for distilled, 40 for base)
    """
    from lightx2v import LightX2VPipeline
    
    print("\n" + "=" * 60)
    print("🚀 Initializing LightX2V Pipeline")
    print("=" * 60)
    print(f"Mode: {mode}")
    print(f"Offloading: {'✅ Enabled' if enable_offload else '❌ Disabled'}")
    print("-" * 60)
    
    # Find model path
    model_path = find_model_paths()
    if model_path is None:
        print("❌ Could not find Qwen-Image-Edit-2511 model!")
        print("   Run: huggingface-cli download Qwen/Qwen-Image-Edit-2511 --local-dir models/Qwen-Image-Edit-2511")
        sys.exit(1)
    
    print(f"Model path: {model_path}")
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        
        mem_before = torch.cuda.memory_allocated() / (1024 ** 3)
        print(f"📊 GPU Memory before loading: {mem_before:.2f} GB")
    
    start_time = time.time()
    
    # Initialize LightX2V pipeline
    pipe = LightX2VPipeline(
        model_path=model_path,
        model_cls="qwen-image-edit-2511",
        task="i2i",
    )
    
    # Enable offloading if requested (for low VRAM)
    if enable_offload:
        print("\n🔧 Enabling CPU offloading...")
        pipe.enable_offload(
            cpu_offload=True,
            offload_granularity="block",
            text_encoder_offload=True,
            vae_offload=False,
        )
        print("✅ Offloading enabled")
    
    # Configure based on mode
    if mode == "fp8":
        fp8_path = find_fp8_path()
        if fp8_path is None:
            print("⚠️ FP8 weights not found, falling back to LoRA mode")
            mode = "lora"
        else:
            print(f"\n🔧 Enabling FP8 quantization...")
            print(f"   FP8 weights: {fp8_path}")
            pipe.enable_quantize(
                dit_quantized=True,
                dit_quantized_ckpt=fp8_path,
                quant_scheme="fp8-sgl"
            )
            steps = 4  # FP8 model has distillation baked in
            print("✅ FP8 quantization enabled")
    
    if mode == "lora":
        lora_path = find_lora_path()
        if lora_path is None:
            print("⚠️ LoRA weights not found!")
            print("   Run: huggingface-cli download lightx2v/Qwen-Image-Edit-2511-Lightning --local-dir models/Qwen-Image-Edit-2511-Lightning")
            sys.exit(1)
        
        print(f"\n🔧 Loading 4-step Lightning LoRA...")
        print(f"   LoRA path: {lora_path}")
        pipe.enable_lora([
            {"path": lora_path, "strength": 1.0},
        ])
        steps = 4
        print("✅ LoRA loaded")
    
    if mode == "base":
        steps = 40
        print("\n⚠️ Using base model (40 steps, slower)")
    
    # Determine attention mode (don't import flash_attn directly - it may be broken)
    # LightX2V will handle the fallback internally
    attn_mode = "sdpa"  # Safe default
    
    # Check if flash_attn is available and working
    try:
        import importlib.util
        if importlib.util.find_spec("flash_attn") is not None:
            # Check if it actually loads without error
            import flash_attn
            attn_mode = "flash_attn2"  # or "flash_attn3" if available
            print(f"\n🔧 Using Flash Attention")
    except Exception as e:
        print(f"\n🔧 Flash Attention unavailable ({type(e).__name__}), using SDPA")
        attn_mode = "sdpa"
    
    # Create generator
    print(f"\n🔧 Creating generator (steps={steps})...")
    pipe.create_generator(
        attn_mode=attn_mode,
        auto_resize=True,
        infer_steps=steps,
        guidance_scale=1.0,
    )
    
    init_time = time.time() - start_time
    print(f"✅ Pipeline initialized in {init_time:.2f} seconds")
    
    # Memory usage
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"📊 GPU Memory: {allocated:.2f} GB / {total:.2f} GB")
    
    print("=" * 60)
    
    # Prepare image paths for multi-image input
    # LightX2V uses comma-separated paths for multi-image
    image_paths = f"{person_image_path},{cloth_image_path}"
    
    # Use prompt
    if prompt is None:
        prompt = VTON_PROMPT_CN
    
    print("\n" + "=" * 60)
    print("👗 Running Virtual Try-On Inference")
    print("=" * 60)
    print(f"Person image: {person_image_path}")
    print(f"Cloth image: {cloth_image_path}")
    print(f"Steps: {steps}")
    print(f"Mode: {mode}")
    print(f"Seed: {seed}")
    print("-" * 60)
    
    # Reset peak memory tracking
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    print("\n🔄 Starting inference...")
    infer_start = time.time()
    
    # Generate
    pipe.generate(
        seed=seed,
        image_path=image_paths,
        prompt=prompt,
        negative_prompt="",
        save_result_path=output_path,
    )
    
    inference_time = time.time() - infer_start
    
    print(f"\n✅ Inference completed!")
    print(f"⏱️  Time: {inference_time:.2f} seconds")
    print(f"⚡ Speed: {inference_time / steps:.2f} seconds/step")
    print(f"💾 Output saved to: {os.path.abspath(output_path)}")
    
    # Memory stats
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"📊 GPU Memory - Current: {allocated:.2f} GB, Peak: {peak:.2f} GB")
    
    print("=" * 60)
    
    return output_path, inference_time


def create_comparison_image(
    person_path: str,
    cloth_path: str,
    result_path: str,
    output_path: str
):
    """Create side-by-side comparison."""
    person_img = Image.open(person_path).convert("RGB")
    cloth_img = Image.open(cloth_path).convert("RGB")
    result_img = Image.open(result_path).convert("RGB")
    
    target_height = 768
    
    def resize_to_height(img, height):
        ratio = height / img.height
        new_width = int(img.width * ratio)
        return img.resize((new_width, height), Image.LANCZOS)
    
    person_resized = resize_to_height(person_img, target_height)
    cloth_resized = resize_to_height(cloth_img, target_height)
    result_resized = resize_to_height(result_img, target_height)
    
    total_width = person_resized.width + cloth_resized.width + result_resized.width + 40
    comparison = Image.new('RGB', (total_width, target_height + 60), color=(40, 40, 40))
    
    x_offset = 10
    comparison.paste(person_resized, (x_offset, 50))
    x_offset += person_resized.width + 10
    comparison.paste(cloth_resized, (x_offset, 50))
    x_offset += cloth_resized.width + 10
    comparison.paste(result_resized, (x_offset, 50))
    
    try:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(comparison)
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        x_offset = 10
        draw.text((x_offset + person_resized.width//2 - 30, 15), "Person", fill=(255, 255, 255), font=font)
        x_offset += person_resized.width + 10
        draw.text((x_offset + cloth_resized.width//2 - 25, 15), "Cloth", fill=(255, 255, 255), font=font)
        x_offset += cloth_resized.width + 10
        draw.text((x_offset + result_resized.width//2 - 30, 15), "Result", fill=(255, 255, 255), font=font)
    except Exception as e:
        print(f"Note: Could not add labels: {e}")
    
    comparison.save(output_path)
    print(f"📸 Comparison image saved to: {output_path}")
    return comparison


def main():
    parser = argparse.ArgumentParser(
        description="LightX2V Virtual Try-On with Qwen-Image-Edit-2511",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with BF16 + 4-step Lightning LoRA
  python test_lightx2v_vton.py --mode lora

  # Run with FP8 + distillation (fastest, lowest memory)
  python test_lightx2v_vton.py --mode fp8

  # Run with CPU offloading (for low VRAM GPUs)
  python test_lightx2v_vton.py --mode fp8 --offload

  # Custom images
  python test_lightx2v_vton.py --person my_photo.jpg --cloth my_garment.png --mode fp8
        """
    )
    parser.add_argument("--person", "-p", type=str, default="person.jpg",
                        help="Path to person image (with green mask)")
    parser.add_argument("--cloth", "-c", type=str, default="cloth.png",
                        help="Path to cloth/garment image")
    parser.add_argument("--output", "-o", type=str, default="outputs/vton_lightx2v_result.png",
                        help="Output path for result")
    parser.add_argument("--mode", "-m", type=str, default="lora",
                        choices=["lora", "fp8", "base"],
                        help="Inference mode: lora (BF16+LoRA), fp8 (FP8+distill), base (40 steps)")
    parser.add_argument("--offload", action="store_true",
                        help="Enable CPU offloading for low VRAM")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--prompt-lang", type=str, default="cn", choices=["cn", "en"],
                        help="Prompt language")
    parser.add_argument("--no-comparison", action="store_true",
                        help="Skip creating comparison image")
    
    args = parser.parse_args()
    
    print("\n" + "⚡" * 30)
    print("   LIGHTX2V VIRTUAL TRY-ON TEST")
    print("   Qwen-Image-Edit-2511 ⚡")
    print("⚡" * 30 + "\n")
    
    # Check CUDA
    check_cuda()
    
    # Validate inputs
    if not os.path.exists(args.person):
        print(f"❌ Person image not found: {args.person}")
        sys.exit(1)
    
    if not os.path.exists(args.cloth):
        print(f"❌ Cloth image not found: {args.cloth}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    
    # Select prompt
    prompt = VTON_PROMPT_CN if args.prompt_lang == "cn" else VTON_PROMPT_EN
    
    # Run inference
    result_path, inference_time = run_lightx2v_vton(
        person_image_path=args.person,
        cloth_image_path=args.cloth,
        output_path=args.output,
        mode=args.mode,
        enable_offload=args.offload,
        prompt=prompt,
        seed=args.seed,
    )
    
    # Create comparison
    if not args.no_comparison:
        comparison_path = args.output.replace(".png", "_comparison.png")
        create_comparison_image(args.person, args.cloth, result_path, comparison_path)
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    print(f"Framework: LightX2V")
    print(f"Model: Qwen-Image-Edit-2511")
    print(f"Mode: {args.mode}")
    print(f"Offload: {'Yes' if args.offload else 'No'}")
    print(f"Inference Time: {inference_time:.2f}s")
    print(f"Output: {os.path.abspath(result_path)}")
    print("=" * 60)
    
    print("\n" + "✅" * 30)
    print("   VIRTUAL TRY-ON COMPLETED!")
    print("✅" * 30 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
