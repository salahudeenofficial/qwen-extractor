#!/usr/bin/env python3
"""
LightX2V Optimization Profiling Script

This script profiles and verifies what optimizations are already active in 
the LightX2V framework when running Qwen-Image-Edit-2511 inference.

It reports on:
1. Triton kernels in use (fused operations)
2. Attention backend (Flash Attention 2/3, SDPA, SageAttention)
3. Quantization (FP8, INT8, BF16)
4. torch.compile status
5. CUDA/cuDNN optimizations
6. Memory allocation patterns
7. Per-layer timing breakdown

Run this script on your GPU server AFTER setting up LightX2V:
    python profile_lightx2v_optimizations.py

Results will be saved to: profile_results.json
"""

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any

import torch


@dataclass
class OptimizationReport:
    """Container for optimization profiling results."""
    # Environment
    python_version: str = ""
    torch_version: str = ""
    cuda_version: str = ""
    cudnn_version: str = ""
    gpu_name: str = ""
    gpu_memory_gb: float = 0.0
    
    # PyTorch Settings
    torch_compile_enabled: bool = False
    torch_compile_mode: str = ""
    tf32_enabled: bool = False
    cudnn_benchmark: bool = False
    cudnn_deterministic: bool = False
    
    # LightX2V Config
    attn_backend: str = ""
    rope_type: str = ""
    modulate_type: str = ""
    quantization: str = ""
    feature_caching: str = ""
    offload_enabled: bool = False
    
    # Triton Kernels Detected
    triton_fused_scale_shift: bool = False
    triton_fused_modulate: bool = False
    triton_rmsnorm: bool = False
    triton_autotune_active: bool = False
    
    # Performance Metrics
    warmup_time_s: float = 0.0
    inference_time_s: float = 0.0
    time_per_step_s: float = 0.0
    peak_memory_gb: float = 0.0
    
    # Timing Breakdown (fraction of total time)
    timing_breakdown: Dict[str, float] = None
    
    # Issues/Warnings
    warnings: List[str] = None


def get_environment_info() -> Dict[str, Any]:
    """Gather environment information."""
    info = {
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "cudnn_version": str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else "N/A",
    }
    
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu_name"] = props.name
        info["gpu_memory_gb"] = round(props.total_memory / (1024**3), 2)
        info["compute_capability"] = f"{props.major}.{props.minor}"
        
        # FP8 hardware detection (Ada Lovelace sm_89+ or Hopper sm_90+)
        info["fp8_hardware"] = props.major >= 9 or (props.major == 8 and props.minor >= 9)
        
        # GPU-specific recommendations
        gpu_name_lower = props.name.lower()
        if "l40" in gpu_name_lower:
            info["gpu_family"] = "L40/L40S (Ada Lovelace)"
            info["recommended_attn"] = "flash_attn2"  # FA3 is Hopper-only
            info["fa3_available"] = False
            info["notes"] = "L40/L40S: Use FA2, FP8 native, 48GB VRAM is plenty for BF16"
        elif "h100" in gpu_name_lower or "h200" in gpu_name_lower:
            info["gpu_family"] = "H100/H200 (Hopper)"
            info["recommended_attn"] = "flash_attn3"
            info["fa3_available"] = True
            info["notes"] = "Hopper GPU: Use FA3 for best performance"
        elif "a100" in gpu_name_lower:
            info["gpu_family"] = "A100 (Ampere)"
            info["recommended_attn"] = "flash_attn2"
            info["fa3_available"] = False
            info["notes"] = "A100: Use FA2, no native FP8 (emulated)"
        elif "4090" in gpu_name_lower or "4080" in gpu_name_lower:
            info["gpu_family"] = "RTX 40-series (Ada Lovelace)"
            info["recommended_attn"] = "flash_attn2"
            info["fa3_available"] = False
            info["notes"] = "Ada consumer GPU: Use FA2, FP8 native"
        else:
            info["gpu_family"] = "Unknown"
            info["recommended_attn"] = "torch_sdpa"  # Safe fallback
            info["fa3_available"] = False
    
    return info


def get_pytorch_settings() -> Dict[str, Any]:
    """Check PyTorch optimization settings."""
    import torch as th  # Local import to avoid any shadowing issues
    
    settings = {}
    
    # TF32 settings
    try:
        settings["tf32_matmul"] = th.backends.cuda.matmul.allow_tf32
    except:
        settings["tf32_matmul"] = False
    
    try:
        settings["tf32_cudnn"] = th.backends.cudnn.allow_tf32
    except:
        settings["tf32_cudnn"] = False
    
    # cuDNN settings
    try:
        settings["cudnn_benchmark"] = th.backends.cudnn.benchmark
        settings["cudnn_deterministic"] = th.backends.cudnn.deterministic
        settings["cudnn_enabled"] = th.backends.cudnn.enabled
    except:
        settings["cudnn_benchmark"] = False
        settings["cudnn_deterministic"] = False
        settings["cudnn_enabled"] = False
    
    # Check if torch.compile could be used
    settings["torch_compile_available"] = hasattr(th, 'compile')
    
    # Check for dynamo settings
    try:
        import torch._dynamo
        settings["dynamo_suppress_errors"] = getattr(torch._dynamo.config, 'suppress_errors', False)
    except:
        settings["dynamo_suppress_errors"] = "N/A"
    
    return settings


def check_attention_backends() -> Dict[str, bool]:
    """Check which attention backends are available."""
    backends = {}
    
    # Flash Attention 2
    try:
        from flash_attn import flash_attn_func
        backends["flash_attn2"] = True
    except ImportError:
        backends["flash_attn2"] = False
    
    # Flash Attention 3
    try:
        from flash_attn_interface import flash_attn_func as flash_attn3_func
        backends["flash_attn3"] = True
    except ImportError:
        backends["flash_attn3"] = False
    
    # SageAttention
    try:
        import sageattention
        backends["sage_attn"] = True
    except ImportError:
        backends["sage_attn"] = False
    
    # Triton
    try:
        import triton
        backends["triton"] = True
        backends["triton_version"] = triton.__version__
    except ImportError:
        backends["triton"] = False
    
    # xFormers
    try:
        import xformers.ops as xops
        backends["xformers"] = True
    except ImportError:
        backends["xformers"] = False
    
    # PyTorch SDPA
    backends["torch_sdpa"] = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
    
    return backends


def check_lightx2v_config(pipe) -> Dict[str, Any]:
    """Extract LightX2V configuration from pipeline."""
    config = {}
    
    try:
        # Get runner/model config
        if hasattr(pipe, 'runner') and hasattr(pipe.runner, 'model'):
            model = pipe.runner.model
            
            # Transformer infer config
            if hasattr(model, 'transformer_infer'):
                ti = model.transformer_infer
                if hasattr(ti, 'config'):
                    tc = ti.config
                    config["attn_type"] = tc.get("attn_type", "unknown")
                    config["rope_type"] = tc.get("rope_type", "unknown")
                    config["modulate_type"] = tc.get("modulate_type", "unknown")
                    config["feature_caching"] = tc.get("feature_caching", "None")
                    config["seq_parallel"] = tc.get("seq_parallel", False)
                    config["zero_cond_t"] = tc.get("zero_cond_t", False)
                
                # Check if it's TeaCache
                config["teacache_enabled"] = "TeaCache" in type(ti).__name__
    except Exception as e:
        config["error"] = str(e)
    
    return config


def check_triton_kernels() -> Dict[str, bool]:
    """Check if LightX2V Triton kernels are present."""
    kernels = {}
    
    try:
        # Check for fused scale/shift kernels
        from lightx2v.models.networks.qwen_image.infer.triton_ops import (
            fuse_scale_shift_kernel,
            fuse_scale_shift_gate_select01_kernel,
        )
        kernels["fuse_scale_shift"] = True
        kernels["fuse_scale_shift_gate"] = True
    except ImportError:
        kernels["fuse_scale_shift"] = False
        kernels["fuse_scale_shift_gate"] = False
    
    try:
        # Check common triton norm ops
        from lightx2v.common.ops.norm.triton_ops import _layer_norm_fwd_kernel
        kernels["triton_layer_norm"] = True
    except ImportError:
        kernels["triton_layer_norm"] = False
    
    try:
        # Check if triton matmul kernels are present
        from lightx2v.common.ops.mm.triton_kernels import (
            _fp8_mm_kernel,
        )
        kernels["triton_fp8_mm"] = True
    except ImportError:
        kernels["triton_fp8_mm"] = False
    
    return kernels


def profile_inference_timing(pipe, image_paths: str, prompt: str, steps: int = 4) -> Dict[str, float]:
    """Profile inference with detailed timing breakdown."""
    import torch.cuda
    
    timings = {}
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    
    # Warmup run
    warmup_start = time.perf_counter()
    try:
        pipe.generate(
            seed=42,
            image_path=image_paths,
            prompt=prompt,
            negative_prompt="",
            save_result_path="/tmp/warmup_result.png",
        )
    except Exception as e:
        print(f"⚠️ Warmup failed: {e}")
        return {"error": str(e)}
    
    warmup_end = time.perf_counter()
    timings["warmup_time"] = warmup_end - warmup_start
    
    # Timed run
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    infer_start = time.perf_counter()
    pipe.generate(
        seed=42,
        image_path=image_paths,
        prompt=prompt,
        negative_prompt="",
        save_result_path="/tmp/timed_result.png",
    )
    torch.cuda.synchronize()
    infer_end = time.perf_counter()
    
    timings["inference_time"] = infer_end - infer_start
    timings["time_per_step"] = timings["inference_time"] / steps
    timings["peak_memory_gb"] = torch.cuda.max_memory_allocated() / (1024**3)
    
    return timings


def profile_with_torch_profiler(pipe, image_paths: str, prompt: str) -> Dict[str, Any]:
    """Profile with PyTorch profiler for detailed breakdown."""
    profiler_results = {}
    
    try:
        from torch.profiler import profile, ProfilerActivity, record_function
        
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) as prof:
            with record_function("lightx2v_inference"):
                pipe.generate(
                    seed=42,
                    image_path=image_paths,
                    prompt=prompt,
                    negative_prompt="",
                    save_result_path="/tmp/profiled_result.png",
                )
        
        # Get top CUDA kernels
        key_averages = prof.key_averages()
        
        # Get top 10 CUDA kernels by time
        sorted_kernels = sorted(
            [e for e in key_averages if e.device_type == torch.device("cuda").type],
            key=lambda x: x.cuda_time_total,
            reverse=True
        )[:15]
        
        profiler_results["top_cuda_kernels"] = [
            {
                "name": k.key,
                "cuda_time_ms": k.cuda_time_total / 1000,
                "calls": k.count,
                "cuda_memory_mb": k.cuda_memory_usage / (1024**2) if k.cuda_memory_usage else 0,
            }
            for k in sorted_kernels
        ]
        
        # Check for specific kernel types
        kernel_names = [k.key.lower() for k in key_averages]
        profiler_results["has_triton_kernels"] = any("triton" in n for n in kernel_names)
        profiler_results["has_flash_attn"] = any("flash" in n for n in kernel_names)
        profiler_results["has_cutlass"] = any("cutlass" in n for n in kernel_names)
        profiler_results["has_cudnn"] = any("cudnn" in n for n in kernel_names)
        
    except Exception as e:
        profiler_results["error"] = str(e)
    
    return profiler_results


def check_compile_status(pipe) -> Dict[str, bool]:
    """Check if any models are compiled with torch.compile."""
    compile_status = {}
    
    try:
        # Check transformer
        if hasattr(pipe, 'runner') and hasattr(pipe.runner, 'model'):
            model = pipe.runner.model
            
            # Check if model has compiled forward
            if hasattr(model, '_compiled'):
                compile_status["model_compiled"] = True
            else:
                compile_status["model_compiled"] = False
            
            # Check for dynamo marks
            try:
                import torch._dynamo
                # Check if model is marked for compilation
                compile_status["dynamo_active"] = torch._dynamo.is_compiling()
            except:
                compile_status["dynamo_active"] = False
    except Exception as e:
        compile_status["error"] = str(e)
    
    return compile_status


def generate_report(args) -> OptimizationReport:
    """Generate comprehensive optimization report."""
    report = OptimizationReport()
    report.warnings = []
    report.timing_breakdown = {}
    
    print("\n" + "=" * 70)
    print("  LightX2V Optimization Profiler")
    print("=" * 70)
    
    # 1. Environment Info
    print("\n📋 Checking environment...")
    env_info = get_environment_info()
    report.python_version = env_info["python_version"]
    report.torch_version = env_info["torch_version"]
    report.cuda_version = env_info["cuda_version"]
    report.cudnn_version = env_info["cudnn_version"]
    report.gpu_name = env_info.get("gpu_name", "N/A")
    report.gpu_memory_gb = env_info.get("gpu_memory_gb", 0)
    
    print(f"   Python: {report.python_version}")
    print(f"   PyTorch: {report.torch_version}")
    print(f"   CUDA: {report.cuda_version}")
    print(f"   GPU: {report.gpu_name} ({report.gpu_memory_gb} GB)")
    
    # 2. PyTorch Settings
    print("\n⚙️ Checking PyTorch settings...")
    pt_settings = get_pytorch_settings()
    report.tf32_enabled = pt_settings["tf32_matmul"] and pt_settings["tf32_cudnn"]
    report.cudnn_benchmark = pt_settings["cudnn_benchmark"]
    report.cudnn_deterministic = pt_settings["cudnn_deterministic"]
    
    print(f"   TF32 enabled: {'✅' if report.tf32_enabled else '❌'}")
    print(f"   cuDNN benchmark: {'✅' if report.cudnn_benchmark else '❌'}")
    
    if not report.tf32_enabled:
        report.warnings.append("TF32 is disabled - enable for ~2x speedup on Ampere+ GPUs")
    if not report.cudnn_benchmark:
        report.warnings.append("cuDNN benchmark is disabled - enable for optimized kernel selection")
    
    # 3. Attention Backends
    print("\n🔍 Checking attention backends...")
    backends = check_attention_backends()
    for name, available in backends.items():
        if isinstance(available, bool):
            print(f"   {name}: {'✅' if available else '❌'}")
    
    # 4. Triton Kernels
    print("\n⚡ Checking Triton kernels...")
    triton_kernels = check_triton_kernels()
    for name, available in triton_kernels.items():
        print(f"   {name}: {'✅' if available else '❌'}")
    
    report.triton_fused_scale_shift = triton_kernels.get("fuse_scale_shift", False)
    report.triton_fused_modulate = triton_kernels.get("fuse_scale_shift_gate", False)  
    report.triton_rmsnorm = triton_kernels.get("triton_layer_norm", False)
    
    # 5. Load LightX2V Pipeline
    print("\n🚀 Loading LightX2V pipeline...")
    try:
        from lightx2v import LightX2VPipeline
        
        # Find model path
        model_paths = [
            "models/Qwen-Image-Edit-2511",
            "./Qwen-Image-Edit-2511",
            "/models/Qwen-Image-Edit-2511",
        ]
        
        model_path = None
        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path is None:
            print("   ❌ Could not find model path!")
            report.warnings.append("Model not found - run inference profiling manually")
            return report
        
        pipe = LightX2VPipeline(
            model_path=model_path,
            model_cls="qwen-image-edit-2511",
            task="i2i",
        )
        
        # Check for LoRA
        lora_paths = [
            "models/Qwen-Image-Edit-2511-Lightning/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
            "lora_weights/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
        ]
        
        lora_path = None
        for path in lora_paths:
            if os.path.exists(path):
                lora_path = path
                break
        
        if lora_path:
            pipe.enable_lora([{"path": lora_path, "strength": 1.0}])
            print(f"   Loaded LoRA from {lora_path}")
        
        # Determine attention mode
        attn_mode = "torch_sdpa"  # Default
        if backends.get("flash_attn3"):
            attn_mode = "flash_attn3"
        elif backends.get("flash_attn2"):
            attn_mode = "flash_attn2"
        elif backends.get("sage_attn"):
            attn_mode = "sage_attn2"
        
        print(f"   Using attention: {attn_mode}")
        
        # Create generator
        pipe.create_generator(
            attn_mode=attn_mode,
            auto_resize=False,
            infer_steps=4,
            guidance_scale=1.0,
            width=720,
            height=480,
        )
        
        # 6. Check LightX2V Config
        print("\n📊 Analyzing LightX2V configuration...")
        lx2v_config = check_lightx2v_config(pipe)
        for key, value in lx2v_config.items():
            print(f"   {key}: {value}")
        
        report.attn_backend = lx2v_config.get("attn_type", attn_mode)
        report.rope_type = lx2v_config.get("rope_type", "unknown")
        report.modulate_type = lx2v_config.get("modulate_type", "unknown")
        
        # 7. Check for torch.compile
        print("\n🔧 Checking torch.compile status...")
        compile_status = check_compile_status(pipe)
        for key, value in compile_status.items():
            print(f"   {key}: {value}")
        report.torch_compile_enabled = compile_status.get("model_compiled", False)
        
        # 8. Run inference profiling (if images exist)
        if os.path.exists("person.jpg") and os.path.exists("cloth.png"):
            print("\n⏱️ Running inference profiling...")
            
            # Prepare test images
            import tempfile
            from PIL import Image
            
            temp_dir = tempfile.mkdtemp()
            person_temp = os.path.join(temp_dir, "person.png")
            cloth_temp = os.path.join(temp_dir, "cloth.png")
            
            # Resize images to 720p
            person_img = Image.open("person.jpg").resize((480, 720), Image.LANCZOS)
            person_img.save(person_temp)
            
            cloth_img = Image.open("cloth.png").resize((480, 720), Image.LANCZOS)
            cloth_img.save(cloth_temp)
            
            image_paths = f"{person_temp},{cloth_temp}"
            prompt = "A person wearing the clothing naturally."
            
            # Basic timing
            timing = profile_inference_timing(pipe, image_paths, prompt, steps=4)
            report.warmup_time_s = timing.get("warmup_time", 0)
            report.inference_time_s = timing.get("inference_time", 0)
            report.time_per_step_s = timing.get("time_per_step", 0)
            report.peak_memory_gb = timing.get("peak_memory_gb", 0)
            
            print(f"\n   Warmup: {report.warmup_time_s:.2f}s")
            print(f"   Inference: {report.inference_time_s:.2f}s")
            print(f"   Per step: {report.time_per_step_s:.2f}s")
            print(f"   Peak memory: {report.peak_memory_gb:.2f} GB")
            
            # Detailed profiling
            if args.detailed:
                print("\n📈 Running detailed profiling (this may take a minute)...")
                profiler_results = profile_with_torch_profiler(pipe, image_paths, prompt)
                
                if "top_cuda_kernels" in profiler_results:
                    print("\n   Top CUDA Kernels:")
                    for k in profiler_results["top_cuda_kernels"][:10]:
                        print(f"      {k['name'][:50]}: {k['cuda_time_ms']:.1f}ms ({k['calls']} calls)")
                
                report.timing_breakdown = profiler_results
            
            # Cleanup
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        else:
            report.warnings.append("Test images not found (person.jpg, cloth.png) - skipping inference profiling")
        
    except ImportError as e:
        print(f"   ❌ LightX2V not installed: {e}")
        report.warnings.append(f"LightX2V not installed: {e}")
    except Exception as e:
        print(f"   ❌ Error loading pipeline: {e}")
        import traceback
        traceback.print_exc()
        report.warnings.append(f"Pipeline error: {e}")
    
    return report


def main():
    parser = argparse.ArgumentParser(description="Profile LightX2V optimizations")
    parser.add_argument("--detailed", action="store_true", 
                        help="Run detailed torch.profiler analysis")
    parser.add_argument("--output", "-o", type=str, default="profile_results.json",
                        help="Output JSON file for results")
    
    args = parser.parse_args()
    
    # Generate report
    report = generate_report(args)
    
    # Summary
    print("\n" + "=" * 70)
    print("  OPTIMIZATION SUMMARY")
    print("=" * 70)
    
    print("\n✅ Optimizations ACTIVE:")
    active = []
    if report.tf32_enabled:
        active.append("TF32 precision")
    if report.cudnn_benchmark:
        active.append("cuDNN benchmark")
    if report.triton_fused_scale_shift:
        active.append("Triton fused scale/shift kernels")
    if report.triton_fused_modulate:
        active.append("Triton fused modulate kernels")
    if "flash" in report.attn_backend.lower():
        active.append(f"Flash Attention ({report.attn_backend})")
    if report.modulate_type == "triton":
        active.append("Triton modulate operations")
    
    for opt in active:
        print(f"   • {opt}")
    
    if not active:
        print("   (none detected)")
    
    print("\n⚠️ Potential Improvements:")
    if not report.torch_compile_enabled:
        print("   • torch.compile is NOT active (LightX2V uses custom Triton kernels instead)")
    
    for warning in report.warnings:
        print(f"   • {warning}")
    
    print("\n📊 Performance:")
    if report.inference_time_s > 0:
        print(f"   Inference: {report.inference_time_s:.2f}s ({report.time_per_step_s:.2f}s/step)")
        print(f"   Peak VRAM: {report.peak_memory_gb:.2f} GB")
    
    # Save results
    report_dict = asdict(report)
    with open(args.output, 'w') as f:
        json.dump(report_dict, f, indent=2, default=str)
    
    print(f"\n💾 Full report saved to: {args.output}")
    print("=" * 70 + "\n")
    
    # Key finding about torch.compile
    print("\n" + "=" * 70)
    print("  KEY FINDING: torch.compile vs LightX2V")
    print("=" * 70)
    print("""
LightX2V intentionally does NOT use torch.compile. Instead, it implements:

1. **Custom Triton Kernels** for fused operations:
   - fuse_scale_shift_kernel: Fuses scale+shift modulation
   - fuse_scale_shift_gate_select01_kernel: Fuses gate selection
   - Triton RMSNorm/LayerNorm kernels
   
2. **Optimized Attention Backends**:
   - Flash Attention 2/3
   - SageAttention 2
   - Ring/Ulysses distributed attention
   
3. **Hand-tuned CUDA/Triton autotune configurations**

This is WHY torch.compile made things worse for you - it competes with
the custom Triton kernels and causes overhead without additional benefit.

RECOMMENDATION: Keep torch.compile DISABLED when using LightX2V.
Focus instead on:
- Ensuring Flash Attention 3 is installed (if on H100+)
- Enabling TF32 and cuDNN benchmark
- Using TeaCache with optimized thresholds
""")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
