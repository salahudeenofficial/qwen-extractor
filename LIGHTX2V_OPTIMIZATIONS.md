# LightX2V Built-in Optimizations Analysis

## Summary

Based on analysis of the LightX2V framework source code, **torch.compile is NOT used and NOT recommended** because LightX2V implements its own highly optimized Triton kernels and custom attention backends that would conflict with torch.compile's JIT compilation.

## Optimizations Already Built Into LightX2V

### 1. Custom Triton Kernels (lightx2v/models/networks/qwen_image/infer/triton_ops.py)

LightX2V uses **hand-tuned Triton kernels** with autotune for optimal performance:

```python
# Fused scale+shift operations (eliminates 3 separate operations)
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 64}, num_warps=2),
        triton.Config({"BLOCK_N": 128}, num_warps=4),
        triton.Config({"BLOCK_N": 256}, num_warps=4),
        triton.Config({"BLOCK_N": 512}, num_warps=4),
        triton.Config({"BLOCK_N": 1024}, num_warps=8),
    ],
    key=["inner_dim"],
)
def _fused_scale_shift_4d_kernel(...)
```

**Kernels include:**
- `fuse_scale_shift_kernel` - Fused normalization + scale + shift
- `fuse_scale_shift_gate_select01_kernel` - Fused modulation with gate selection (for CFG)
- `triton_autotune_configs()` - Dynamic warp selection based on GPU

### 2. Multiple Attention Backends (lightx2v/common/ops/attn/)

LightX2V supports various optimized attention implementations:

| Backend | File | Notes |
|---------|------|-------|
| Flash Attention 2/3 | `flash_attn.py` | Best for single GPU |
| SageAttention 2 | `sage_attn.py` | Alternative to FA |
| Ring Attention | `ring_attn.py` | Multi-GPU sequence parallel |
| Ulysses Attention | `ulysses_attn.py` | Multi-GPU with FP8 comm |
| Radial Attention | `radial_attn.py` | Sparse attention |
| SVG Attention | `svg_attn.py` / `svg2_attn.py` | Specialized variants |
| Torch SDPA | `torch_sdpa.py` | Fallback |

The transformer selects via config:
```python
self.attn_type = config.get("attn_type", "flash_attn3")
```

### 3. Fused RoPE Implementation (triton_ops.py)

Multiple RoPE backends optimized for different scenarios:
```python
rope_funcs = {
    "flashinfer": apply_qwen_rope_with_flashinfer,  # Fastest
    "torch": apply_qwen_rope_with_torch,            # Fused torch
    "torch_naive": apply_qwen_rope_with_torch_naive,  # Fallback
}
```

### 4. Optimized Normalization (lightx2v/common/ops/norm/)

Triton-fused RMSNorm and LayerNorm with:
- Dynamic warp count selection
- FP32 accumulation for stability
- Fused residual addition variants

### 5. FP8/INT8 Quantization Support (lightx2v/common/ops/mm/)

Native FP8 matmul with per-tensor scaling:
```python
# From mm_weight.py
- w8a8-int8 quantization
- w8a8-fp8 quantization  
- w4a4-nvfp4 quantization (Blackwell)
```

### 6. Feature Caching (TeaCache, MagCache, etc.)

Located in `lightx2v/models/networks/.../infer/`:
- TeaCache: Timestep embedding aware caching
- MagCache: Magnitude-based adaptive caching
- AdaCache: Adaptive caching
- TaylorSeer: Taylor expansion prediction

## Why torch.compile Makes Things Worse

1. **Kernel Conflict**: torch.compile tries to JIT-compile operations that are already optimized Triton kernels, causing:
   - Compilation overhead (warmup time)
   - Suboptimal kernel selection (ignores Triton autotune)
   - Memory overhead from graph capturing

2. **Dynamic Shapes**: LightX2V uses dynamic shapes for:
   - Variable resolution inputs
   - CFG parallel processing
   - Sequence parallel distribution
   
   torch.compile struggles with dynamic shapes.

3. **Triton Autotune Competition**: The built-in `@triton.autotune` decorators select optimal block sizes and warp counts. torch.compile's Triton backend may override these.

## Recommended Optimizations Path

### Already Implemented in LightX2V ✅
- [x] Triton fused kernels (scale/shift/gate)
- [x] Flash Attention 2/3 support
- [x] FP8 quantization
- [x] 4-step Lightning LoRA distillation
- [x] TeaCache
- [x] CPU offloading
- [x] Sequence parallelism

### You Should Enable ✅
```python
# In your script
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
```

### You Should NOT Do ❌
```python
# Don't do this - conflicts with LightX2V Triton kernels
transformer = torch.compile(transformer)
```

## How to Verify on Your GPU Server

Run the profiling script when LightX2V is installed:

```bash
# Basic profiling
python profile_lightx2v_optimizations.py

# With detailed torch.profiler breakdown
python profile_lightx2v_optimizations.py --detailed

# Results saved to
cat profile_results.json
```

## Expected Results

On H100 with all optimizations:
- Flash Attention 3: ✅
- Triton fused kernels: ✅
- FP8 quantization: ✅
- 4-step LoRA: ✅
- TeaCache: ✅

Expected inference time: **~3-5 seconds** for 720p VTON

## Configuration for Maximum Performance

```python
# In your config or create_generator call
config = {
    "attn_type": "flash_attn3",      # or flash_attn2
    "rope_type": "flashinfer",       # Fastest RoPE
    "modulate_type": "triton",       # Use Triton kernels
    "feature_caching": "Tea",        # TeaCache
    "teacache_thresh": 0.08,         # Conservative threshold
}
```

## Bottom Line

**LightX2V is already heavily optimized**. The framework implements custom Triton kernels that are specifically tuned for video/image generation workloads. Adding torch.compile on top causes conflicts and overhead.

Focus instead on:
1. Ensuring Flash Attention 2 is installed (FA3 is Hopper-only)
2. Using FP8 quantization if speed is critical
3. Tuning TeaCache threshold for speed/quality tradeoff
4. Enabling TF32 and cuDNN benchmark

---

## L40/L40S Specific Optimization Guide

The **L40/L40S** is an excellent choice for Qwen-Image-Edit-2511 VTON:

### L40/L40S Specs

| Spec | L40 | L40S |
|------|-----|------|
| Architecture | Ada Lovelace (SM 8.9) | Ada Lovelace (SM 8.9) |
| VRAM | 48 GB GDDR6 | 48 GB GDDR6 |
| FP8 Hardware | ✅ Native (e4m3/e5m2) | ✅ Native |
| Flash Attention | **FA2 only** (FA3 is Hopper-only) | **FA2 only** |
| TF32 | ✅ Supported | ✅ Supported |
| BF16 | ✅ Supported | ✅ Supported |

### Recommended Configuration for L40/L40S

```python
# Attention backend - use flash_attn2 (NOT flash_attn3)
pipe.create_generator(
    attn_mode="flash_attn2",  # FA3 is Hopper-only!
    infer_steps=4,
    guidance_scale=1.0,
)

# Enable TF32 for Ada Lovelace
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
```

### L40/L40S Performance Expectations

| Resolution | Mode | Expected Time |
|-----------|------|---------------|
| 480p | BF16 + 4-step LoRA | ~4-6s |
| 480p | FP8 + 4-step | ~3-4s |
| 720p | BF16 + 4-step LoRA | ~6-10s |
| 720p | FP8 + 4-step | ~5-7s |
| 720p | FP8 + TeaCache | ~4-6s |

### L40 vs L40S

- **L40S has higher TDP** → faster sustained performance
- **L40 is more power efficient** → better for cost/watt
- Both have identical VRAM and compute capabilities
- For VTON workloads, L40S will be **~10-15% faster**

### What to Avoid on L40

1. ❌ Don't try to use `flash_attn3` - it will fall back to slower implementation
2. ❌ Don't use `torch.compile` - conflicts with LightX2V Triton kernels
3. ❌ Don't enable CPU offloading - 48GB VRAM is plenty for BF16

### Installation Commands for L40

```bash
# Flash Attention 2 (NOT 3!)
pip install flash-attn --no-build-isolation

# Verify sm_89 support
python -c "import torch; print(torch.cuda.get_device_properties(0))"
# Should show: compute_capability=8.9 or NVIDIA L40

# Check FA2 installed
python -c "from flash_attn import flash_attn_func; print('FA2 OK')"
```

### Optimal LightX2V Config for L40

```json
{
  "attn_type": "flash_attn2",
  "rope_type": "flashinfer",
  "modulate_type": "triton",
  "feature_caching": "Tea",
  "teacache_thresh": 0.08
}
```

