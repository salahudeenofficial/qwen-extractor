#!/usr/bin/env python3
"""
1038lab FP8 Qwen-Image-Edit-2511 Test Script for Virtual Try-On

This script tests the 1038lab/Qwen-Image-Edit-2511-FP8 model with 
the 4-step Lightning LoRA for virtual try-on accuracy comparison.

Model: 1038lab/Qwen-Image-Edit-2511-FP8
LoRA: lightx2v/Qwen-Image-Edit-2511-Lightning (4-step)

Memory: ~15-22GB VRAM (vs ~35GB for BF16)
Speed: ~4-8 seconds for 4 steps
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image


# Virtual Try-On Prompt (Chinese - optimized for Qwen)
VTON_PROMPT_CN = """将图片 1 中的绿色遮罩区域仅用于判断服装属于上半身或下半身，不要将服装限制在遮罩范围内。

将图片 2 中的服装自然地穿戴到图片 1 中的人物身上，保持图片 2 中服装的完整形状、袖长和轮廓。无论图片 2 是单独的服装图还是人物穿着该服装的图，都应准确地转移服装，同时保留其原始面料质感、材质细节和颜色准确性。

确保图片 1 中人物的面部、头发和皮肤完全保持不变。光照与阴影应自然匹配图片 1 的环境，但服装的材质外观必须忠实于图片 2。

保持边缘平滑融合、阴影逼真，整体效果自然且不改变人物的身份特征。"""

# Alternative English prompt
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
    else:
        print("❌ CUDA is not available!")
        sys.exit(1)
    
    print("=" * 60)
    return True


def download_lora_weights(lora_dir: str = "./lora_weights"):
    """Download the 4-step Lightning LoRA weights."""
    from huggingface_hub import hf_hub_download
    
    os.makedirs(lora_dir, exist_ok=True)
    
    lora_filename = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
    lora_path = os.path.join(lora_dir, lora_filename)
    
    if os.path.exists(lora_path):
        print(f"✅ LoRA weights already exist: {lora_path}")
        return lora_path
    
    print("\n" + "=" * 60)
    print("📥 Downloading 4-Step Lightning LoRA Weights")
    print("=" * 60)
    print("Repository: lightx2v/Qwen-Image-Edit-2511-Lightning")
    print(f"File: {lora_filename}")
    print("-" * 60)
    
    try:
        downloaded_path = hf_hub_download(
            repo_id="lightx2v/Qwen-Image-Edit-2511-Lightning",
            filename=lora_filename,
            local_dir=lora_dir,
            local_dir_use_symlinks=False,
        )
        print(f"✅ LoRA weights downloaded to: {downloaded_path}")
        return downloaded_path
    except Exception as e:
        print(f"❌ Failed to download LoRA weights: {e}")
        return None


def load_1038lab_fp8_pipeline(
    lora_path: str = None,
    device: str = "cuda"
):
    """
    Load the 1038lab FP8 quantized Qwen-Image-Edit-2511 model.
    
    This model is pre-quantized to FP8 and packaged in diffusers format,
    making it directly compatible with the diffusers library.
    """
    from diffusers import QwenImageEditPlusPipeline, FlowMatchEulerDiscreteScheduler
    
    model_id = "1038lab/Qwen-Image-Edit-2511-FP8"
    
    print("\n" + "=" * 60)
    print("🚀 Loading 1038lab FP8 Qwen-Image-Edit-2511")
    print("=" * 60)
    print(f"Model: {model_id}")
    print(f"LoRA: {lora_path if lora_path else 'None'}")
    print("This is a pre-quantized FP8 model for reduced VRAM usage")
    print("-" * 60)
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        
        mem_before = torch.cuda.memory_allocated() / (1024 ** 3)
        print(f"📊 GPU Memory before loading: {mem_before:.2f} GB")
    
    start_time = time.time()
    
    # Scheduler config for 4-step Lightning LoRA
    # This scheduler configuration is optimized for the distilled model
    scheduler_config = {
        "base_image_seq_len": 256,
        "base_shift": math.log(3),  # shift=3 used in distillation
        "invert_sigmas": False,
        "max_image_seq_len": 8192,
        "max_shift": math.log(3),   # shift=3 used in distillation
        "num_train_timesteps": 1000,
        "shift": 1.0,
        "shift_terminal": None,     # Required for distilled model
        "stochastic_sampling": False,
        "time_shift_type": "exponential",
        "use_beta_sigmas": False,
        "use_dynamic_shifting": True,
        "use_exponential_sigmas": False,
        "use_karras_sigmas": False,
    }
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
    
    # Load the FP8 model
    print("Loading FP8 pipeline (this may take a few minutes on first run)...")
    
    try:
        pipeline = QwenImageEditPlusPipeline.from_pretrained(
            model_id,
            scheduler=scheduler,
            torch_dtype=torch.bfloat16,  # Compute dtype
            device_map="balanced",
        )
        print("✅ FP8 model loaded successfully!")
        
    except Exception as e:
        print(f"⚠️ Failed to load FP8 model: {e}")
        print("Falling back to standard BF16 model...")
        
        pipeline = QwenImageEditPlusPipeline.from_pretrained(
            "Qwen/Qwen-Image-Edit-2511",
            scheduler=scheduler,
            torch_dtype=torch.bfloat16,
            device_map="balanced",
        )
    
    # Load LoRA if provided
    if lora_path and os.path.exists(lora_path):
        print(f"\n📦 Loading 4-step Lightning LoRA...")
        print(f"   Path: {lora_path}")
        try:
            pipeline.load_lora_weights(lora_path)
            print("✅ LoRA loaded successfully!")
        except Exception as e:
            print(f"⚠️ LoRA loading failed: {e}")
            print("   Continuing without LoRA (may require more steps)")
    
    load_time = time.time() - start_time
    print(f"\n✅ Pipeline ready in {load_time:.2f} seconds")
    
    # Memory usage
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"📊 GPU Memory: {allocated:.2f} GB / {total:.2f} GB ({allocated/total*100:.1f}%)")
    
    pipeline.set_progress_bar_config(disable=False)
    print("=" * 60)
    
    return pipeline


def run_vton_inference(
    pipeline, 
    person_image: Image.Image,
    cloth_image: Image.Image,
    prompt: str,
    output_path: str = "vton_output.png",
    num_inference_steps: int = 4,
    true_cfg_scale: float = 1.0,
    seed: int = 42
):
    """Run Virtual Try-On inference."""
    print("\n" + "=" * 60)
    print("👗 Running Virtual Try-On Inference")
    print("=" * 60)
    print(f"Person image size: {person_image.size}")
    print(f"Cloth image size: {cloth_image.size}")
    print(f"Inference steps: {num_inference_steps}")
    print(f"True CFG scale: {true_cfg_scale}")
    print(f"Seed: {seed}")
    print("-" * 60)
    
    # Display prompt (truncated for readability)
    prompt_preview = prompt[:150] + "..." if len(prompt) > 150 else prompt
    print(f"Prompt: {prompt_preview}")
    print("-" * 60)
    
    # Prepare inputs
    inputs = {
        "image": [person_image, cloth_image],  # Multi-image input
        "prompt": prompt,
        "generator": torch.Generator(device="cuda").manual_seed(seed),
        "true_cfg_scale": true_cfg_scale,
        "negative_prompt": " ",  # Empty but not None
        "num_inference_steps": num_inference_steps,
    }
    
    # Reset peak memory tracking
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    print("\n🔄 Starting inference...")
    start_time = time.time()
    
    with torch.inference_mode():
        output = pipeline(**inputs)
    
    inference_time = time.time() - start_time
    
    print(f"\n✅ Inference completed!")
    print(f"⏱️  Time: {inference_time:.2f} seconds")
    print(f"⚡ Speed: {inference_time / num_inference_steps:.2f} seconds/step")
    
    # Save output
    output_image = output.images[0]
    output_image.save(output_path)
    print(f"💾 Output saved to: {os.path.abspath(output_path)}")
    
    # Memory stats
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"📊 GPU Memory - Current: {allocated:.2f} GB, Peak: {peak:.2f} GB")
    
    print("=" * 60)
    return output_image, inference_time


def create_comparison_image(
    person_img: Image.Image,
    cloth_img: Image.Image,
    result_img: Image.Image,
    output_path: str = "vton_comparison.png"
):
    """Create a side-by-side comparison image."""
    # Resize all images to same height for comparison
    target_height = 768
    
    def resize_to_height(img, height):
        ratio = height / img.height
        new_width = int(img.width * ratio)
        return img.resize((new_width, height), Image.LANCZOS)
    
    person_resized = resize_to_height(person_img, target_height)
    cloth_resized = resize_to_height(cloth_img, target_height)
    result_resized = resize_to_height(result_img, target_height)
    
    # Create comparison canvas
    total_width = person_resized.width + cloth_resized.width + result_resized.width + 40
    comparison = Image.new('RGB', (total_width, target_height + 60), color=(40, 40, 40))
    
    # Paste images
    x_offset = 10
    comparison.paste(person_resized, (x_offset, 50))
    x_offset += person_resized.width + 10
    comparison.paste(cloth_resized, (x_offset, 50))
    x_offset += cloth_resized.width + 10
    comparison.paste(result_resized, (x_offset, 50))
    
    # Add labels using PIL
    try:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(comparison)
        
        # Try to use a nice font, fallback to default
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
        description="Test 1038lab FP8 Qwen-Image-Edit for Virtual Try-On",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic VTON test with default images
  python test_1038lab_fp8_vton.py

  # With custom images
  python test_1038lab_fp8_vton.py --person my_photo.jpg --cloth my_garment.png

  # Compare with different seeds
  python test_1038lab_fp8_vton.py --seed 123 --output result_seed123.png
        """
    )
    parser.add_argument("--person", "-p", type=str, default="person.jpg",
                        help="Path to person image (with green mask on clothing area)")
    parser.add_argument("--cloth", "-c", type=str, default="cloth.png",
                        help="Path to cloth/garment image")
    parser.add_argument("--output", "-o", type=str, default="vton_1038lab_fp8_result.png",
                        help="Output path for result image")
    parser.add_argument("--steps", type=int, default=4,
                        help="Number of inference steps (4 for Lightning LoRA)")
    parser.add_argument("--cfg", type=float, default=1.0,
                        help="True CFG scale (1.0 for Lightning LoRA)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--lora-dir", type=str, default="./lora_weights",
                        help="Directory for LoRA weights")
    parser.add_argument("--no-lora", action="store_true",
                        help="Skip LoRA loading (use base steps ~28-40)")
    parser.add_argument("--prompt-lang", type=str, default="cn", choices=["cn", "en"],
                        help="Prompt language: cn (Chinese, better) or en (English)")
    parser.add_argument("--no-comparison", action="store_true",
                        help="Skip creating comparison image")
    
    args = parser.parse_args()
    
    print("\n" + "👗" * 30)
    print("   1038LAB FP8 QWEN VIRTUAL TRY-ON TEST")
    print("   With 4-Step Lightning LoRA ⚡")
    print("👗" * 30 + "\n")
    
    # Check CUDA
    check_cuda()
    
    # Validate input files
    if not os.path.exists(args.person):
        print(f"❌ Person image not found: {args.person}")
        print("   Please provide a person image with green mask on the clothing area.")
        sys.exit(1)
    
    if not os.path.exists(args.cloth):
        print(f"❌ Cloth image not found: {args.cloth}")
        print("   Please provide a garment/cloth image.")
        sys.exit(1)
    
    # Load images
    print("\n📷 Loading input images...")
    person_img = Image.open(args.person).convert("RGB")
    cloth_img = Image.open(args.cloth).convert("RGB")
    print(f"   Person: {person_img.size} from {args.person}")
    print(f"   Cloth: {cloth_img.size} from {args.cloth}")
    
    # Download LoRA if needed
    lora_path = None
    if not args.no_lora:
        lora_path = download_lora_weights(args.lora_dir)
        if lora_path is None:
            print("⚠️ LoRA download failed. Using more inference steps...")
            args.steps = 28  # Fall back to more steps
            args.cfg = 3.5
    
    # Adjust steps if no LoRA
    if args.no_lora:
        print("\n⚠️ Running without LoRA - adjusting parameters:")
        args.steps = 28
        args.cfg = 3.5
        print(f"   Steps: {args.steps}, CFG: {args.cfg}")
    
    # Load pipeline
    pipeline = load_1038lab_fp8_pipeline(
        lora_path=lora_path,
    )
    
    # Select prompt
    prompt = VTON_PROMPT_CN if args.prompt_lang == "cn" else VTON_PROMPT_EN
    
    # Run inference
    result_img, inference_time = run_vton_inference(
        pipeline=pipeline,
        person_image=person_img,
        cloth_image=cloth_img,
        prompt=prompt,
        output_path=args.output,
        num_inference_steps=args.steps,
        true_cfg_scale=args.cfg,
        seed=args.seed,
    )
    
    # Create comparison image
    if not args.no_comparison:
        comparison_path = args.output.replace(".png", "_comparison.png")
        create_comparison_image(person_img, cloth_img, result_img, comparison_path)
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    print(f"Model: 1038lab/Qwen-Image-Edit-2511-FP8")
    print(f"LoRA: {'4-Step Lightning ⚡' if lora_path else 'None'}")
    print(f"Steps: {args.steps}")
    print(f"CFG Scale: {args.cfg}")
    print(f"Inference Time: {inference_time:.2f}s")
    print(f"Output: {os.path.abspath(args.output)}")
    print("=" * 60)
    
    print("\n" + "✅" * 30)
    print("   VIRTUAL TRY-ON TEST COMPLETED!")
    print("✅" * 30 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
