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

Performance (on L40/L40S with optimizations):
- BF16 + 4-step LoRA: ~6-10 seconds
- FP8 + 4-step distillation: ~5-7 seconds
- FP8 + TeaCache: ~4-6 seconds

L40/L40S Notes:
- Uses Flash Attention 2 (FA3 is Hopper-only)
- Native FP8 support (SM 8.9)
- 48GB VRAM - no offloading needed
"""

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image

# Set CUDA arch for L40/L40S (Ada Lovelace SM 8.9)
os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '8.9')

# Enable TF32 for faster matmul on Ada Lovelace GPUs
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


# Virtual Try-On Prompts
VTON_PROMPT_CN = """执行图像编辑操作：删除并移除人物及配饰。
将人物、人体区域以及所有配饰完全擦除、清空、不可见。

严禁出现任何人体或人体部位。
包括但不限于：
人、模特、身体、皮肤、腿、脚、脚踝、手、手臂、手指、头部、脸部、头发、颈部、肩膀、躯干、人体轮廓、人体阴影。

严禁出现任何配饰或附属物。
包括但不限于：
鞋子、袜子、帽子、眼镜、墨镜、围巾、手套、腰带、首饰、项链、耳环、戒指、手表、包、背包、手提包、钱包、钥匙、耳机、发饰。

仅保留完整整套核心服装（上衣、下装、外套）。
不包含任何配饰或穿戴附属物。

服装必须以无人体支撑的形式展示（悬浮或平铺）。

颜色与输入图像必须 100% 完全一致。
不允许任何颜色变化、色相偏移、饱和度变化或亮度变化。
保留原始面料纹理、材质和图案。

背景必须为纯白色。

如检测到人体或配饰残留，必须继续删除，仅保留核心服装。"""

VTON_PROMPT_EN = VTON_PROMPT_CN


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
        # Correct filename with version suffix
        "models/Qwen-Image-Edit-2511-Lightning/qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning_4steps_v1.0.safetensors",
        # Old filename (fallback)
        "models/Qwen-Image-Edit-2511-Lightning/qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning.safetensors",
        "models/qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning.safetensors",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None


def calculate_720p_resolution(input_image_path: str):
    """
    Calculate 720p output resolution maintaining input aspect ratio.
    720p = 720 pixels on the shorter side.
    """
    img = Image.open(input_image_path)
    width, height = img.size
    aspect_ratio = width / height
    
    # For 720p: shorter side = 720
    if width >= height:
        # Landscape or square: height = 720
        target_height = 720
        target_width = int(720 * aspect_ratio)
    else:
        # Portrait: width = 720
        target_width = 720
        target_height = int(720 / aspect_ratio)
    
    # Ensure dimensions are divisible by 16 (required for diffusion models)
    target_width = (target_width // 16) * 16
    target_height = (target_height // 16) * 16
    
    return target_width, target_height, aspect_ratio


def run_lightx2v_vton(
    input_image_path: str,
    output_path: str,
    mode: str = "lora",  # "lora", "fp8", or "base"
    enable_offload: bool = False,
    prompt: str = None,
    seed: int = 42,
    steps: int = 4,
    target_width: int = None,
    target_height: int = None,
    enable_teacache: bool = False,
    teacache_thresh: float = 0.26,
    warmup: bool = False,
):
    """
    Run Virtual Try-On using LightX2V framework.
    
    Args:
        input_image_path: Path to input image
        output_path: Path to save output
        mode: "lora" for BF16+LoRA, "fp8" for FP8+distillation, "base" for 40-step base
        enable_offload: Enable CPU offloading for low VRAM
        prompt: VTON prompt (uses Chinese prompt by default)
        seed: Random seed
        steps: Inference steps (4 for distilled, 40 for base)
        target_width: Target output width (auto-calculated if None)
        target_height: Target output height (auto-calculated if None)
        enable_teacache: Enable TeaCache for ~1.5-2x faster inference
        teacache_thresh: TeaCache threshold (0.26 default, lower = faster but less quality)
        warmup: Run warmup inference for consistent timing
    """
    from lightx2v import LightX2VPipeline
    
    # Calculate 720p resolution if not specified
    if target_width is None or target_height is None:
        target_width, target_height, aspect_ratio = calculate_720p_resolution(input_image_path)
        print(f"\n📐 Auto-calculated 720p resolution: {target_width}x{target_height} (AR: {aspect_ratio:.2f})")
    
    print("\n" + "=" * 60)
    print("🚀 Initializing LightX2V Pipeline")
    print("=" * 60)
    print(f"Mode: {mode}")
    print(f"Offloading: {'✅ Enabled' if enable_offload else '❌ Disabled'}")
    print(f"Output Resolution: {target_width}x{target_height}")
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
    
    # Determine attention mode
    # L40/L40S: Use flash_attn2 (FA3 is Hopper-only)
    # LightX2V supported modes: "flash_attn2", "flash_attn3", "torch_sdpa", "sage_attn2"
    attn_mode = "torch_sdpa"  # Safe default - uses PyTorch SDPA
    
    # Detect GPU and choose optimal attention backend
    gpu_name = torch.cuda.get_device_name(0).lower() if torch.cuda.is_available() else ""
    
    # Check if flash_attn is available
    try:
        import importlib.util
        if importlib.util.find_spec("flash_attn") is not None:
            import flash_attn
            # L40/L40S or other Ada GPUs: Use FA2 (FA3 is Hopper-only)
            if "l40" in gpu_name or "4090" in gpu_name or "4080" in gpu_name:
                attn_mode = "flash_attn2"
                print(f"\n🔧 L40/Ada GPU detected → Using Flash Attention 2")
            elif "h100" in gpu_name or "h200" in gpu_name:
                # Only Hopper GPUs can use FA3
                try:
                    from flash_attn_interface import flash_attn_func
                    attn_mode = "flash_attn3"
                    print(f"\n🔧 Hopper GPU detected → Using Flash Attention 3")
                except ImportError:
                    attn_mode = "flash_attn2"
                    print(f"\n🔧 Hopper GPU but FA3 not installed → Using Flash Attention 2")
            else:
                attn_mode = "flash_attn2"
                print(f"\n🔧 Using Flash Attention 2")
    except Exception as e:
        print(f"\n🔧 Flash Attention unavailable ({type(e).__name__}), using PyTorch SDPA")
        attn_mode = "torch_sdpa"
    
    # Create generator with resolution settings
    print(f"\n🔧 Creating generator (steps={steps})...")
    
    # IMPORTANT: Detect input image aspect ratio BEFORE creating generator
    # to prevent zooming/cropping
    input_img_check = Image.open(input_image_path)
    orig_w, orig_h = input_img_check.size
    orig_ratio = orig_w / orig_h
    input_img_check.close()
    
    # Adjust target dimensions to match input aspect ratio
    if orig_ratio < 1:  # Portrait (taller than wide)
        # Keep height at target, adjust width
        actual_height = target_height
        actual_width = int(target_height * orig_ratio)
    else:  # Landscape or square
        # Keep width at target, adjust height
        actual_width = target_width
        actual_height = int(target_width / orig_ratio)
    
    # Ensure dimensions are multiples of 16
    actual_width = max(16, (actual_width // 16) * 16)
    actual_height = max(16, (actual_height // 16) * 16)
    
    print(f"📐 Input: {orig_w}x{orig_h} → Output: {actual_width}x{actual_height} (preserving aspect ratio)")
    
    # Determine aspect_ratio for LightX2V resolution control
    # LightX2V default mappings:
    #   "16:9": [1664, 928], "9:16": [928, 1664], "1:1": [1328, 1328]
    #   "4:3": [1472, 1140], "3:4": [768, 1024]
    if orig_ratio < 0.8:  # Portrait (taller than wide)
        target_aspect_ratio = "3:4"  # Maps to 768x1024
        print(f"   Setting aspect_ratio: 3:4 (portrait) → 768x1024")
    elif orig_ratio > 1.2:  # Landscape (wider than tall)  
        target_aspect_ratio = "4:3"  # Maps to 1472x1140
        print(f"   Setting aspect_ratio: 4:3 (landscape) → 1472x1140")
    else:  # Square-ish
        target_aspect_ratio = "1:1"  # Maps to 1328x1328
        print(f"   Setting aspect_ratio: 1:1 (square) → 1328x1328")
    
    # Pass aspect_ratio directly to create_generator (this is the proper LightX2V API)
    pipe.create_generator(
        attn_mode=attn_mode,
        infer_steps=steps,
        guidance_scale=1.0,
        width=actual_width,
        height=actual_height,
        aspect_ratio=target_aspect_ratio,
    )
    
    # Monkey-patch run_pipeline to inject aspect_ratio into input_info
    # This is needed because:
    # 1. set_config() filters out aspect_ratio (it's in ALL_INPUT_INFO_KEYS)
    # 2. set_input_info() for i2i task doesn't populate aspect_ratio
    # 3. get_custom_shape() checks input_info.aspect_ratio first
    original_run_pipeline = pipe.runner.run_pipeline
    def patched_run_pipeline(input_info):
        input_info.aspect_ratio = target_aspect_ratio
        # Also need to set _auto_resize in config (get_custom_shape checks it)
        pipe.runner.config["_auto_resize"] = False
        print(f"   Injected aspect_ratio into input_info: {target_aspect_ratio}")
        return original_run_pipeline(input_info)
    pipe.runner.run_pipeline = patched_run_pipeline
    
    # Enable TeaCache AFTER creating generator (model must be loaded first)
    # We monkey-patch the transformer_infer with our TeaCache implementation
    teacache_enabled = False
    if enable_teacache:
        print(f"\n⚡ Enabling TeaCache (threshold={teacache_thresh})...")
        # Import our TeaCache implementation
        import sys
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        try:
            from teacache_transformer_infer import QwenImageTeaCacheTransformerInfer
            
            # Get the current transformer_infer
            orig_transformer_infer = pipe.runner.model.transformer_infer
            
            # Create new TeaCache infer instance with same config
            config = orig_transformer_infer.config.copy()
            config["teacache_thresh"] = teacache_thresh
            config["coefficients"] = [1.0]  # Linear scaling
            config["infer_steps"] = steps
            
            # Create TeaCache wrapper
            teacache_infer = QwenImageTeaCacheTransformerInfer(config)
            teacache_infer.scheduler = orig_transformer_infer.scheduler
            teacache_infer.infer_func = orig_transformer_infer.infer_func
            
            # Replace the transformer_infer
            pipe.runner.model.transformer_infer = teacache_infer
            
            teacache_enabled = True
            print("✅ TeaCache enabled")
            print(f"   Threshold: {teacache_thresh}")
            print(f"   Expected: Skip 1-2 transformer passes → 25-50% speedup")
        except Exception as e:
            print(f"⚠️ TeaCache setup failed: {e}")
            import traceback
            traceback.print_exc()
            print("   Continuing without TeaCache...")
    
    init_time = time.time() - start_time
    print(f"✅ Pipeline initialized in {init_time:.2f} seconds")
    
    # Memory usage
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"📊 GPU Memory: {allocated:.2f} GB / {total:.2f} GB")
    
    print("=" * 60)
    
    # Pre-resize images to calculated resolution (actual_width x actual_height)
    import tempfile
    temp_dir = tempfile.mkdtemp()
    
    # Resize input image to aspect-ratio-correct dimensions
    input_img = Image.open(input_image_path)
    input_resized = input_img.resize((actual_width, actual_height), Image.LANCZOS)
    input_temp = os.path.join(temp_dir, "input_resized.png")
    input_resized.save(input_temp)
    print(f"📐 Input resized to: {actual_width}x{actual_height}")
    
    # Use resized images
    image_paths = input_temp
    
    # Use prompt
    if prompt is None:
        prompt = VTON_PROMPT_CN
    
    print("\n" + "=" * 60)
    print("👗 Running Output Extraction Inference")
    print("=" * 60)
    print(f"Input image: {input_image_path} -> {target_width}x{target_height}")
    print(f"Steps: {steps}")
    print(f"Mode: {mode}")
    print(f"Seed: {seed}")
    print("-" * 60)
    
    # Reset peak memory tracking
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    print("\n🔄 Starting inference...")
    infer_start = time.time()
    
    # Generate - resolution is controlled by input image size
    pipe.generate(
        seed=seed,
        image_path=image_paths,
        prompt=prompt,
        negative_prompt="",
        save_result_path=output_path,
    )
    
    # Cleanup temp files
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    
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
    input_path: str,
    result_path: str,
    output_path: str
):
    """Create side-by-side comparison."""
    input_img = Image.open(input_path).convert("RGB")
    result_img = Image.open(result_path).convert("RGB")
    
    target_height = 768
    
    def resize_to_height(img, height):
        ratio = height / img.height
        new_width = int(img.width * ratio)
        return img.resize((new_width, height), Image.LANCZOS)
    
    input_resized = resize_to_height(input_img, target_height)
    result_resized = resize_to_height(result_img, target_height)
    
    total_width = input_resized.width + result_resized.width + 30
    comparison = Image.new('RGB', (total_width, target_height + 60), color=(40, 40, 40))
    
    x_offset = 10
    comparison.paste(input_resized, (x_offset, 50))
    x_offset += input_resized.width + 10
    comparison.paste(result_resized, (x_offset, 50))
    
    try:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(comparison)
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        x_offset = 10
        draw.text((x_offset + input_resized.width//2 - 30, 15), "Input", fill=(255, 255, 255), font=font)
        x_offset += input_resized.width + 10
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
    parser.add_argument("--input", "-i", type=str, default="input.jpg",
                        help="Path to input image")
    parser.add_argument("--output", "-o", type=str, default="outputs/outfit_extractor_result.png",
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
    parser.add_argument("--width", type=int, default=None,
                        help="Target output width (default: auto-calculate for 720p)")
    parser.add_argument("--height", type=int, default=None,
                        help="Target output height (default: auto-calculate for 720p)")
    parser.add_argument("--resolution", type=str, default="720p",
                        choices=["480p", "720p", "1080p", "auto"],
                        help="Target resolution preset (default: 720p)")
    parser.add_argument("--teacache", action="store_true",
                        help="Enable TeaCache for faster inference (quality trade-off)")
    parser.add_argument("--teacache-thresh", type=float, default=0.05,
                        help="TeaCache threshold (lower = better quality, default: 0.05)")
    parser.add_argument("--warmup", action="store_true",
                        help="Run warmup inference before timing")
    
    args = parser.parse_args()
    
    print("\n" + "⚡" * 30)
    print("   LIGHTX2V VIRTUAL TRY-ON TEST")
    print("   Qwen-Image-Edit-2511 ⚡")
    print("⚡" * 30 + "\n")
    
    # Check CUDA
    check_cuda()
    
    # Validate inputs
    if not os.path.exists(args.input):
        print(f"❌ Input image not found: {args.input}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    
    # Select prompt
    prompt = VTON_PROMPT_CN if args.prompt_lang == "cn" else VTON_PROMPT_EN
    
    # Calculate resolution
    target_width = args.width
    target_height = args.height
    
    if target_width is None or target_height is None:
        # Auto-calculate based on resolution preset
        img = Image.open(args.input)
        w, h = img.size
        aspect_ratio = w / h
        
        # Resolution presets (shorter side)
        res_map = {"480p": 480, "720p": 720, "1080p": 1080, "auto": None}
        target_short = res_map.get(args.resolution)
        
        if target_short and args.resolution != "auto":
            if w >= h:
                target_height = target_short
                target_width = int(target_short * aspect_ratio)
            else:
                target_width = target_short
                target_height = int(target_short / aspect_ratio)
            
            # Ensure divisible by 16
            target_width = (target_width // 16) * 16
            target_height = (target_height // 16) * 16
        
        print(f"📐 Resolution: {args.resolution} -> {target_width}x{target_height}")
    
    # Run inference
    result_path, inference_time = run_lightx2v_vton(
        input_image_path=args.input,
        output_path=args.output,
        mode=args.mode,
        enable_offload=args.offload,
        prompt=prompt,
        seed=args.seed,
        target_width=target_width,
        target_height=target_height,
        enable_teacache=args.teacache,
        teacache_thresh=args.teacache_thresh,
        warmup=args.warmup,
    )
    
    # Create comparison
    if not args.no_comparison:
        comparison_path = args.output.replace(".png", "_comparison.png")
        create_comparison_image(args.input, result_path, comparison_path)
    
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
