#!/usr/bin/env python3
"""
Qwen-Image-Edit-2511 with FP8 Layerwise Casting

This script uses FP8 layerwise casting for ~50% VRAM reduction
while maintaining BF16 compute precision.

Memory comparison:
- BF16: ~25-35GB VRAM
- FP8 Layerwise: ~15-20GB VRAM
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
            print(f"   Compute Capability: {props.major}.{props.minor}")
            
            # Check FP8 support
            fp8_supported = props.major >= 9 or (props.major == 8 and props.minor >= 9)
            print(f"   FP8 Hardware Support: {'✅ Yes' if fp8_supported else '⚠️ Limited (software emulation)'}")
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


def load_pipeline_with_fp8_layerwise(
    model_id: str = "Qwen/Qwen-Image-Edit-2511",
    lora_path: str = None,
    use_fp8_storage: bool = True,
    device: str = "cuda"
):
    """
    Load the pipeline with optional FP8 layerwise casting.
    
    FP8 layerwise casting stores weights in FP8 but computes in BF16.
    This provides ~50% VRAM reduction with minimal quality loss.
    """
    from diffusers import QwenImageEditPlusPipeline, FlowMatchEulerDiscreteScheduler
    
    print("\n" + "=" * 60)
    print("🚀 Loading Qwen-Image-Edit-2511 Pipeline")
    print("=" * 60)
    print(f"Model: {model_id}")
    print(f"FP8 Layerwise Casting: {'✅ Enabled' if use_fp8_storage else '❌ Disabled'}")
    print(f"LoRA: {lora_path if lora_path else 'None'}")
    print("-" * 60)
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()
    
    start_time = time.time()
    
    # Memory before loading
    if torch.cuda.is_available():
        mem_before = torch.cuda.memory_allocated() / (1024 ** 3)
        print(f"📊 GPU Memory before loading: {mem_before:.2f} GB")
    
    # Scheduler config for 4-step Lightning LoRA
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
    
    # Memory after loading
    if torch.cuda.is_available():
        mem_after_load = torch.cuda.memory_allocated() / (1024 ** 3)
        print(f"📊 GPU Memory after loading: {mem_after_load:.2f} GB")
    
    # Enable FP8 layerwise casting if requested
    if use_fp8_storage:
        print("\n🔧 Enabling FP8 layerwise casting...")
        try:
            # Try the newer API first
            pipeline.enable_layerwise_casting(
                storage_dtype=torch.float8_e4m3fn,
                compute_dtype=torch.bfloat16,
            )
            print("✅ FP8 layerwise casting enabled!")
            
            # Check memory after FP8 conversion
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                import gc
                gc.collect()
                mem_after_fp8 = torch.cuda.memory_allocated() / (1024 ** 3)
                savings = mem_after_load - mem_after_fp8
                print(f"📊 GPU Memory after FP8: {mem_after_fp8:.2f} GB")
                print(f"💾 Memory saved: {savings:.2f} GB ({savings/mem_after_load*100:.1f}%)")
                
        except AttributeError:
            print("⚠️ FP8 layerwise casting not available in this diffusers version")
            print("   Continuing with BF16 weights")
        except Exception as e:
            print(f"⚠️ FP8 layerwise casting failed: {e}")
            print("   Continuing with BF16 weights")
    
    # Load LoRA if provided
    if lora_path and os.path.exists(lora_path):
        print(f"\n📦 Loading LoRA from: {lora_path}")
        pipeline.load_lora_weights(lora_path)
        print("✅ LoRA loaded!")
    
    load_time = time.time() - start_time
    print(f"\n✅ Pipeline loaded in {load_time:.2f} seconds")
    
    # Final memory usage
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"📊 Final GPU Memory: {allocated:.2f} GB / {total:.2f} GB")
    
    pipeline.set_progress_bar_config(disable=False)
    
    print("=" * 60)
    return pipeline


def load_pipeline_with_torchao_fp8(
    model_id: str = "Qwen/Qwen-Image-Edit-2511",
    lora_path: str = None,
    device: str = "cuda"
):
    """
    Load the pipeline with torchao FP8 quantization.
    
    Requires:
    - GPU with compute capability >= 8.9 (RTX 4090, A100, H100)
    - torchao installed: pip install torchao
    """
    from diffusers import QwenImageEditPlusPipeline, FlowMatchEulerDiscreteScheduler
    
    print("\n" + "=" * 60)
    print("🚀 Loading Pipeline with TorchAO FP8 Quantization")
    print("=" * 60)
    
    # Check GPU compute capability
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        cc = f"{props.major}.{props.minor}"
        if props.major < 8 or (props.major == 8 and props.minor < 9):
            print(f"⚠️ GPU compute capability {cc} < 8.9")
            print("   FP8 may run in software emulation mode (slower)")
    
    try:
        from diffusers import PipelineQuantizationConfig, TorchAoConfig
    except ImportError:
        print("❌ TorchAO quantization not available")
        print("   Install with: pip install torchao")
        return None
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()
    
    start_time = time.time()
    
    # Scheduler config
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
    
    # Configure FP8 quantization for transformer
    try:
        pipeline_quant_config = PipelineQuantizationConfig(
            quant_mapping={
                "transformer": TorchAoConfig("float8wo_e4m3")  # FP8 weight-only
            }
        )
        
        print("Loading pipeline with FP8 quantization...")
        pipeline = QwenImageEditPlusPipeline.from_pretrained(
            model_id,
            scheduler=scheduler,
            quantization_config=pipeline_quant_config,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        
    except Exception as e:
        print(f"❌ FP8 quantization failed: {e}")
        print("   Falling back to standard BF16 loading")
        
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
    
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"📊 GPU Memory: {allocated:.2f} GB / {total:.2f} GB")
    
    pipeline.set_progress_bar_config(disable=False)
    print("=" * 60)
    
    return pipeline


def run_inference(
    pipeline, 
    images: list,
    prompt: str,
    output_path: str = "output_image.png",
    num_inference_steps: int = 4,
    true_cfg_scale: float = 1.0,
    seed: int = 42
):
    """Run inference with the pipeline."""
    print("\n" + "=" * 60)
    print("🎨 Running Image Edit Inference")
    print("=" * 60)
    print(f"Number of input images: {len(images)}")
    print(f"Inference steps: {num_inference_steps}")
    print(f"True CFG scale: {true_cfg_scale}")
    print(f"Seed: {seed}")
    print("-" * 60)
    
    inputs = {
        "image": images,
        "prompt": prompt,
        "generator": torch.Generator(device="cuda").manual_seed(seed),
        "true_cfg_scale": true_cfg_scale,
        "negative_prompt": " ",
        "num_inference_steps": num_inference_steps,
    }
    
    # Reset peak memory tracking
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
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
    
    # Print memory usage
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"📊 GPU Memory - Current: {allocated:.2f} GB, Peak: {peak:.2f} GB")
    
    print("=" * 60)
    return output_image


def main():
    parser = argparse.ArgumentParser(
        description="Qwen-Image-Edit-2511 with FP8 Memory Optimization"
    )
    parser.add_argument("--input", "-i", type=str, nargs="+", default=None,
                        help="Path to input image(s)")
    parser.add_argument("--prompt", "-p", type=str, 
                        default="Transform this into a beautiful oil painting with dramatic lighting",
                        help="Edit prompt")
    parser.add_argument("--output", "-o", type=str, default="output_fp8.png",
                        help="Output path")
    parser.add_argument("--steps", type=int, default=4,
                        help="Inference steps")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--lora-dir", type=str, default="./lora_weights",
                        help="Directory for LoRA weights")
    parser.add_argument("--mode", type=str, default="layerwise",
                        choices=["layerwise", "torchao", "bf16"],
                        help="FP8 mode: layerwise (recommended), torchao, or bf16 (baseline)")
    
    args = parser.parse_args()
    
    print("\n" + "🎨" * 30)
    print("   QWEN-IMAGE-EDIT-2511 FP8 TEST")
    print("🎨" * 30 + "\n")
    
    # Check CUDA
    check_cuda()
    
    # Download LoRA weights
    lora_path = download_lora_weights(args.lora_dir)
    
    # Prepare input images
    if args.input is None:
        # Check for default test images
        test_images = ["sample_input.png", "person.jpg"]
        input_paths = []
        for img in test_images:
            if os.path.exists(img):
                input_paths.append(img)
                break
        
        if not input_paths:
            print("❌ No input image provided. Use --input flag.")
            sys.exit(1)
    else:
        input_paths = args.input
        for path in input_paths:
            if not os.path.exists(path):
                print(f"❌ Input image not found: {path}")
                sys.exit(1)
    
    # Load images
    images = [Image.open(p).convert("RGB") for p in input_paths]
    print(f"\n📷 Loaded {len(images)} image(s)")
    for i, (img, path) in enumerate(zip(images, input_paths)):
        print(f"   Image {i+1}: {img.size} from {path}")
    
    # Load pipeline based on mode
    if args.mode == "layerwise":
        pipeline = load_pipeline_with_fp8_layerwise(
            lora_path=lora_path,
            use_fp8_storage=True,
        )
    elif args.mode == "torchao":
        pipeline = load_pipeline_with_torchao_fp8(
            lora_path=lora_path,
        )
    else:  # bf16 baseline
        pipeline = load_pipeline_with_fp8_layerwise(
            lora_path=lora_path,
            use_fp8_storage=False,
        )
    
    if pipeline is None:
        print("❌ Failed to load pipeline")
        sys.exit(1)
    
    # Run inference
    output_image = run_inference(
        pipeline=pipeline,
        images=images,
        prompt=args.prompt,
        output_path=args.output,
        num_inference_steps=args.steps,
        seed=args.seed,
    )
    
    print("\n" + "✅" * 30)
    print("   TEST COMPLETED!")
    print("✅" * 30 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
