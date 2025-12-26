#!/usr/bin/env python3
"""
Qwen-Image-Edit-2511 FP8 Test Script with 4-Step Lightning LoRA

Uses the FP8 quantized model for ~50% memory reduction:
- VRAM: ~20-22GB instead of ~43GB
- Speed: ~4-6 seconds for 4 steps

Models:
- FP8 Base + LoRA: lightx2v/Qwen-Image-Edit-2511-Lightning
  - qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning.safetensors
"""

import argparse
import math
import os
import sys
import time
from io import BytesIO
from pathlib import Path

import requests
import torch
from PIL import Image


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
    else:
        print("❌ CUDA is not available!")
        sys.exit(1)
    
    print("=" * 60)
    return True


def download_fp8_model(model_dir: str = "./models"):
    """Download the FP8 quantized model with Lightning LoRA baked in."""
    from huggingface_hub import hf_hub_download
    
    os.makedirs(model_dir, exist_ok=True)
    
    # FP8 model with 4-step Lightning LoRA already fused
    model_filename = "qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning.safetensors"
    model_path = os.path.join(model_dir, model_filename)
    
    if os.path.exists(model_path):
        print(f"✅ FP8 model already exists: {model_path}")
        return model_path
    
    print("\n" + "=" * 60)
    print("📥 Downloading FP8 Quantized Model with Lightning LoRA")
    print("=" * 60)
    print("Repository: lightx2v/Qwen-Image-Edit-2511-Lightning")
    print(f"File: {model_filename}")
    print("This model uses ~50% less VRAM than BF16!")
    print("-" * 60)
    
    try:
        downloaded_path = hf_hub_download(
            repo_id="lightx2v/Qwen-Image-Edit-2511-Lightning",
            filename=model_filename,
            local_dir=model_dir,
            local_dir_use_symlinks=False,
        )
        print(f"✅ FP8 model downloaded to: {downloaded_path}")
        return downloaded_path
    except Exception as e:
        print(f"❌ Failed to download FP8 model: {e}")
        return None


def load_pipeline_fp8(device: str = "cuda"):
    """Load the FP8 quantized pipeline by replacing transformer weights."""
    from diffusers import QwenImageEditPlusPipeline, FlowMatchEulerDiscreteScheduler
    from diffusers.models import QwenImageTransformer2DModel
    
    base_model_id = "Qwen/Qwen-Image-Edit-2511"
    fp8_model_id = "drbaph/Qwen-Image-Edit-2511-FP8"
    
    print("\n" + "=" * 60)
    print("🚀 Loading Qwen-Image-Edit-2511 FP8 Pipeline")
    print("=" * 60)
    print(f"Base Model: {base_model_id}")
    print(f"FP8 Transformer: {fp8_model_id}")
    print(f"Expected VRAM: ~20-22GB (50% less than BF16)")
    print("-" * 60)
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()
    
    start_time = time.time()
    
    # Scheduler config for 4-step distilled model
    scheduler_config = {
        "base_image_seq_len": 256,
        "base_shift": math.log(3),
        "invert_sigmas": False,
        "max_image_seq_len": 8192,
        "max_shift": math.log(3),
        "num_train_timesteps": 1000,
        "shift": 1.0,
        "shift_terminal": None,
        "stochastic_sampling": False,
        "time_shift_type": "exponential",
        "use_beta_sigmas": False,
        "use_dynamic_shifting": True,
        "use_exponential_sigmas": False,
        "use_karras_sigmas": False,
    }
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
    
    # Load FP8 transformer separately
    print("Loading FP8 quantized transformer...")
    transformer = QwenImageTransformer2DModel.from_pretrained(
        fp8_model_id,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )
    
    # Load the pipeline with FP8 transformer
    print("Loading full pipeline with FP8 transformer...")
    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        base_model_id,
        transformer=transformer,
        scheduler=scheduler,
        torch_dtype=torch.bfloat16,
        device_map="balanced",
    )
    
    print(f"✅ FP8 Pipeline loaded in {time.time() - start_time:.2f} seconds")
    
    # Print memory usage
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"📊 GPU Memory: {allocated:.2f} GB / {total:.2f} GB")
    
    pipeline.set_progress_bar_config(disable=False)
    
    print("=" * 60)
    return pipeline


def load_pipeline_standard(model_id: str = "Qwen/Qwen-Image-Edit-2511",
                           lora_path: str = None,
                           device: str = "cuda"):
    """Load the standard BF16 pipeline with optional LoRA."""
    from diffusers import QwenImageEditPlusPipeline, FlowMatchEulerDiscreteScheduler
    
    print("\n" + "=" * 60)
    print("🚀 Loading Qwen-Image-Edit-2511 Pipeline (BF16)")
    print("=" * 60)
    print(f"Model: {model_id}")
    print(f"LoRA: {lora_path if lora_path else 'None'}")
    print("-" * 60)
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()
    
    start_time = time.time()
    
    # Scheduler config for 4-step distilled model
    scheduler_config = {
        "base_image_seq_len": 256,
        "base_shift": math.log(3),
        "invert_sigmas": False,
        "max_image_seq_len": 8192,
        "max_shift": math.log(3),
        "num_train_timesteps": 1000,
        "shift": 1.0,
        "shift_terminal": None,
        "stochastic_sampling": False,
        "time_shift_type": "exponential",
        "use_beta_sigmas": False,
        "use_dynamic_shifting": True,
        "use_exponential_sigmas": False,
        "use_karras_sigmas": False,
    }
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
    
    # Load pipeline
    print("Loading pipeline...")
    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        model_id,
        scheduler=scheduler,
        torch_dtype=torch.bfloat16,
        device_map="balanced",
    )
    
    # Load LoRA if provided
    if lora_path and os.path.exists(lora_path):
        print(f"Loading LoRA from: {lora_path}")
        pipeline.load_lora_weights(lora_path)
        print("✅ LoRA loaded!")
    
    print(f"✅ Pipeline loaded in {time.time() - start_time:.2f} seconds")
    
    # Print memory usage
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"📊 GPU Memory: {allocated:.2f} GB / {total:.2f} GB")
    
    pipeline.set_progress_bar_config(disable=False)
    
    print("=" * 60)
    return pipeline


def run_inference(pipeline, 
                  images: list,
                  prompt: str,
                  output_path: str = "output_image.png",
                  num_inference_steps: int = 4,
                  true_cfg_scale: float = 1.0,
                  seed: int = 42):
    """Run inference with the pipeline."""
    print("\n" + "=" * 60)
    print("🎨 Running Virtual Try-On Inference")
    print("=" * 60)
    print(f"Number of input images: {len(images)}")
    print(f"Inference steps: {num_inference_steps}")
    print(f"True CFG scale: {true_cfg_scale}")
    print(f"Seed: {seed}")
    print("-" * 60)
    print(f"Prompt:\n{prompt[:200]}..." if len(prompt) > 200 else f"Prompt:\n{prompt}")
    print("-" * 60)
    
    inputs = {
        "image": images,
        "prompt": prompt,
        "generator": torch.Generator(device="cuda").manual_seed(seed),
        "true_cfg_scale": true_cfg_scale,
        "negative_prompt": " ",
        "num_inference_steps": num_inference_steps,
    }
    
    start_time = time.time()
    
    with torch.inference_mode():
        output = pipeline(**inputs)
    
    inference_time = time.time() - start_time
    print(f"\n✅ Inference completed in {inference_time:.2f} seconds")
    print(f"⚡ Speed: {inference_time / num_inference_steps:.2f} seconds per step")
    
    # Save output
    output_image = output.images[0]
    output_image.save(output_path)
    print(f"💾 Output saved to: {os.path.abspath(output_path)}")
    
    # Print final memory usage
    if torch.cuda.is_available():
        max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"📊 Peak GPU Memory: {max_allocated:.2f} GB")
    
    print("=" * 60)
    return output_image


def main():
    parser = argparse.ArgumentParser(description="Qwen-Image-Edit-2511 FP8 Test")
    parser.add_argument("--person", "-p", type=str, required=True,
                        help="Path to person image (with green mask)")
    parser.add_argument("--cloth", "-c", type=str, required=True,
                        help="Path to cloth/garment image")
    parser.add_argument("--output", "-o", type=str, default="vton_output.png",
                        help="Output path")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Custom prompt (uses VTON prompt by default)")
    parser.add_argument("--steps", type=int, default=4,
                        help="Inference steps")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--use-fp8", action="store_true",
                        help="Use FP8 quantized model (experimental)")
    parser.add_argument("--lora-dir", type=str, default="./lora_weights",
                        help="Directory for LoRA weights")
    
    args = parser.parse_args()
    
    # Default VTON prompt
    if args.prompt is None:
        args.prompt = """将图片 1 中的绿色遮罩区域仅用于判断服装属于上半身或下半身，不要将服装限制在遮罩范围内。

将图片 2 中的服装自然地穿戴到图片 1 中的人物身上，保持图片 2 中服装的完整形状、袖长和轮廓。无论图片 2 是单独的服装图还是人物穿着该服装的图，都应准确地转移服装，同时保留其原始面料质感、材质细节和颜色准确性。

确保图片 1 中人物的面部、头发和皮肤完全保持不变。光照与阴影应自然匹配图片 1 的环境，但服装的材质外观必须忠实于图片 2。

保持边缘平滑融合、阴影逼真，整体效果自然且不改变人物的身份特征。"""
    
    print("\n" + "👗" * 30)
    print("   QWEN VIRTUAL TRY-ON TEST")
    print("👗" * 30 + "\n")
    
    # Check CUDA
    check_cuda()
    
    # Check input files
    if not os.path.exists(args.person):
        print(f"❌ Person image not found: {args.person}")
        sys.exit(1)
    if not os.path.exists(args.cloth):
        print(f"❌ Cloth image not found: {args.cloth}")
        sys.exit(1)
    
    # Load images
    print("\n📷 Loading input images...")
    person_img = Image.open(args.person).convert("RGB")
    cloth_img = Image.open(args.cloth).convert("RGB")
    print(f"   Person image: {person_img.size}")
    print(f"   Cloth image: {cloth_img.size}")
    
    # Load pipeline
    if args.use_fp8:
        # FP8 mode - uses 1038lab/Qwen-Image-Edit-2511-FP8
        print("\n🔧 Using FP8 quantized model for reduced VRAM usage...")
        pipeline = load_pipeline_fp8()
    else:
        # Standard BF16 with LoRA
        from huggingface_hub import hf_hub_download
        
        lora_filename = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
        lora_path = os.path.join(args.lora_dir, lora_filename)
        
        if not os.path.exists(lora_path):
            print("📥 Downloading LoRA weights...")
            os.makedirs(args.lora_dir, exist_ok=True)
            hf_hub_download(
                repo_id="lightx2v/Qwen-Image-Edit-2511-Lightning",
                filename=lora_filename,
                local_dir=args.lora_dir,
                local_dir_use_symlinks=False,
            )
        
        pipeline = load_pipeline_standard(lora_path=lora_path)
    
    # Run inference with both images
    output_image = run_inference(
        pipeline=pipeline,
        images=[person_img, cloth_img],
        prompt=args.prompt,
        output_path=args.output,
        num_inference_steps=args.steps,
        seed=args.seed,
    )
    
    print("\n" + "✅" * 30)
    print("   VIRTUAL TRY-ON COMPLETED!")
    print("✅" * 30 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
