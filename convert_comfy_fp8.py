#!/usr/bin/env python3
"""
ComfyUI FP8 to Diffusers Weight Converter

This script attempts to convert ComfyUI FP8 scaled weights to a format
that can be loaded into diffusers QwenImageEditPlusPipeline.

IMPORTANT: This is experimental. ComfyUI FP8 weights have different:
1. Tensor naming conventions
2. Scaling factor storage (per-tensor vs per-channel)
3. Architecture assumptions

Use this as a starting point for custom conversion.
"""

import argparse
import os
import sys
from collections import OrderedDict
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def analyze_comfy_weights(weight_path: str):
    """Analyze ComfyUI FP8 weight file structure."""
    print(f"\n📊 Analyzing: {weight_path}")
    print("=" * 70)
    
    state_dict = load_file(weight_path)
    
    # Analyze structure
    total_params = 0
    fp8_tensors = 0
    scale_tensors = 0
    other_tensors = 0
    
    prefix_counts = {}
    dtype_counts = {}
    
    for key, tensor in state_dict.items():
        total_params += tensor.numel()
        
        # Count by dtype
        dtype_str = str(tensor.dtype)
        dtype_counts[dtype_str] = dtype_counts.get(dtype_str, 0) + 1
        
        # Check for FP8 or scale tensors
        if "float8" in dtype_str or "e4m3" in dtype_str.lower():
            fp8_tensors += 1
        elif "scale" in key.lower():
            scale_tensors += 1
        else:
            other_tensors += 1
        
        # Count by prefix
        prefix = key.split(".")[0]
        prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1
    
    print(f"\n📦 Total tensors: {len(state_dict)}")
    print(f"📊 Total parameters: {total_params:,}")
    print(f"💾 Estimated size: {total_params * 4 / (1024**3):.2f} GB (FP32)")
    
    print(f"\n🔢 Tensor types:")
    print(f"   FP8 tensors: {fp8_tensors}")
    print(f"   Scale tensors: {scale_tensors}")
    print(f"   Other tensors: {other_tensors}")
    
    print(f"\n📊 Dtypes found:")
    for dtype, count in sorted(dtype_counts.items()):
        print(f"   {dtype}: {count}")
    
    print(f"\n🏷️ Prefixes found:")
    for prefix, count in sorted(prefix_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"   {prefix}: {count}")
    
    # Sample some key names
    print(f"\n📝 Sample keys (first 20):")
    for key in list(state_dict.keys())[:20]:
        tensor = state_dict[key]
        print(f"   {key}: {tensor.dtype} {list(tensor.shape)}")
    
    return state_dict


def get_diffusers_weight_mapping():
    """
    Return a mapping from ComfyUI weight names to diffusers weight names.
    
    This is model-specific and needs to be built based on the architecture.
    """
    # Common mapping patterns (this needs to be expanded based on actual models)
    mappings = {
        # ComfyUI pattern -> Diffusers pattern
        "model.diffusion_model.": "transformer.",
        "diffusion_model.": "transformer.",
        
        # Attention blocks
        ".to_q.": ".attn.to_q.",
        ".to_k.": ".attn.to_k.", 
        ".to_v.": ".attn.to_v.",
        ".to_out.0.": ".attn.to_out.0.",
        
        # MLP/FFN
        ".ff.net.0.proj.": ".ff.net.0.",
        ".ff.net.2.": ".ff.net.2.",
        
        # Normalization
        ".norm_in.": ".norm1.",
        ".norm_out.": ".norm2.",
        
        # Time embedding
        "time_embed.": "time_embedding.",
    }
    
    return mappings


def convert_key(key: str, mappings: dict) -> str:
    """Convert a single key using the mapping."""
    converted = key
    for old, new in mappings.items():
        converted = converted.replace(old, new)
    return converted


def convert_comfy_to_diffusers(
    comfy_state_dict: dict,
    target_dtype: torch.dtype = torch.bfloat16,
    handle_scales: bool = True
):
    """
    Convert ComfyUI FP8 state dict to diffusers format.
    
    Args:
        comfy_state_dict: State dict from ComfyUI FP8 safetensors
        target_dtype: Target dtype (usually bfloat16 for compute)
        handle_scales: Whether to apply scaling factors to dequantize
    """
    print("\n🔄 Converting weights...")
    
    mappings = get_diffusers_weight_mapping()
    converted = OrderedDict()
    scales = {}
    
    # First pass: collect scales
    for key, tensor in comfy_state_dict.items():
        if "scale" in key.lower():
            # Associate scale with its weight
            weight_key = key.replace("_scale", "").replace(".scale", "")
            scales[weight_key] = tensor
    
    # Second pass: convert weights
    for key, tensor in comfy_state_dict.items():
        # Skip scale tensors (they're applied separately)
        if "scale" in key.lower():
            continue
        
        # Convert key
        new_key = convert_key(key, mappings)
        
        # Handle FP8 tensors
        if "float8" in str(tensor.dtype):
            if handle_scales and key in scales:
                # Dequantize: FP8 * scale -> BF16
                scale = scales[key]
                tensor = tensor.to(torch.float32) * scale.to(torch.float32)
                tensor = tensor.to(target_dtype)
            else:
                # Direct cast (may lose precision)
                tensor = tensor.to(target_dtype)
        else:
            # Already in higher precision
            tensor = tensor.to(target_dtype)
        
        converted[new_key] = tensor
    
    print(f"✅ Converted {len(converted)} tensors")
    return converted


def compare_with_diffusers_model(
    model_id: str = "Qwen/Qwen-Image-Edit-2511"
):
    """
    Load diffusers model and print expected weight names for comparison.
    """
    print(f"\n📋 Loading diffusers model to compare weight names...")
    print("=" * 70)
    
    try:
        from diffusers import QwenImageEditPlusPipeline
        
        # Just load the transformer to save memory
        pipeline = QwenImageEditPlusPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="cpu",  # Load to CPU for comparison
        )
        
        transformer_keys = list(pipeline.transformer.state_dict().keys())
        
        print(f"\n📝 Diffusers transformer weight keys (first 30):")
        for key in transformer_keys[:30]:
            print(f"   {key}")
        
        print(f"\n📊 Total transformer keys: {len(transformer_keys)}")
        
        # Group by prefix
        prefixes = {}
        for key in transformer_keys:
            prefix = ".".join(key.split(".")[:2])
            prefixes[prefix] = prefixes.get(prefix, 0) + 1
        
        print(f"\n🏷️ Diffusers key prefixes:")
        for prefix, count in sorted(prefixes.items(), key=lambda x: -x[1])[:15]:
            print(f"   {prefix}: {count}")
        
        return transformer_keys
        
    except Exception as e:
        print(f"❌ Could not load diffusers model: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(
        description="Convert ComfyUI FP8 weights for diffusers"
    )
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Path to ComfyUI FP8 safetensors file")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output path for converted weights")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Only analyze the input file without converting")
    parser.add_argument("--compare-diffusers", action="store_true",
                        help="Load diffusers model to compare weight names")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("  ComfyUI FP8 to Diffusers Weight Converter")
    print("=" * 70)
    
    # Analyze input
    state_dict = analyze_comfy_weights(args.input)
    
    if args.compare_diffusers:
        compare_with_diffusers_model()
    
    if args.analyze_only:
        print("\n✅ Analysis complete (--analyze-only mode)")
        return 0
    
    # Convert
    converted = convert_comfy_to_diffusers(state_dict)
    
    # Save
    if args.output:
        output_path = args.output
    else:
        input_path = Path(args.input)
        output_path = str(input_path.parent / f"{input_path.stem}_diffusers.safetensors")
    
    print(f"\n💾 Saving converted weights to: {output_path}")
    save_file(converted, output_path)
    print("✅ Done!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
