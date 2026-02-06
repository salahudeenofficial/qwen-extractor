"""
Pipeline Manager for GPU Server

Manages the LightX2V inference pipeline for VTON tasks.
This is the core inference engine that interfaces with the model.
"""

import os
import sys
import gc
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import torch
from PIL import Image

# Add parent directory to path for imports
SCRIPT_DIR = Path(__file__).parent.parent.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


# Virtual Try-On Prompts
# Virtual Try-On Prompts
VTON_PROMPT_CN = """
执行图像编辑操作：删除并移除人物、人体区域、配饰及非目标服装。

将人物、人体区域以及所有配饰完全擦除、清空、不可见。

严禁出现任何人体或人体部位。
包括但不限于：
人、模特、身体、皮肤、腿、脚、脚踝、手、手臂、手指、头部、脸部、头发、颈部、肩膀、躯干、人体轮廓、人体阴影。

严禁出现任何配饰或附属物。
包括但不限于：
鞋子、袜子、帽子、眼镜、墨镜、围巾、手套、腰带、首饰、项链、耳环、戒指、手表、包、背包、手提包、钱包、钥匙、耳机、发饰。

仅保留图像中【可见像素面积最大的单一服装】。
如图像中存在多件服装：
- 必须仅保留占据像素面积最多的一件服装
- 所有其他服装必须完全删除，不可见

不允许同时保留多件服装。
最终结果中只能存在一件服装。

该服装必须以无人体支撑的形式展示（悬浮或平铺）。

颜色与输入图像必须 100% 完全一致。
不允许任何颜色变化、色相偏移、饱和度变化或亮度变化。
必须保留原始面料纹理、材质和图案。

背景必须为纯白色。

如检测到任何人体、配饰或非目标服装残留，必须继续删除，仅保留单一最大面积服装。

"""

VTON_PROMPT_EN = VTON_PROMPT_CN


class PipelineManager:
    """
    Manages the LightX2V inference pipeline.
    
    Handles:
    - Model loading and initialization
    - Inference execution
    - Memory management
    """
    
    def __init__(self, config):
        self.config = config
        self.pipe = None
        self.model_loaded = False
        self.loading = False
        
        # Model paths
        self.model_path = None
        self.lora_path = None
        self.fp8_path = None
        
        # Inference settings
        self.mode = config.model.default_mode
        self.steps = config.model.default_steps
        self.enable_teacache = config.model.enable_teacache
        self.teacache_thresh = config.model.teacache_thresh
    
    def find_model_paths(self) -> Optional[str]:
        """Find model paths in common locations."""
        possible_paths = [
            # Local models directory (relative to gpu_server)
            "models/Qwen-Image-Edit-2511",
            "../models/Qwen-Image-Edit-2511",
            # Project root
            str(SCRIPT_DIR / "models" / "Qwen-Image-Edit-2511"),
            # Workspace paths (Vast.ai)
            "/workspace/try_og_pipeline/models/Qwen-Image-Edit-2511",
            "/workspace/models/Qwen-Image-Edit-2511",
            # Absolute paths
            "/models/Qwen-Image-Edit-2511",
            "./Qwen-Image-Edit-2511",
            # HuggingFace cache - check for snapshots directory
            os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen-Image-Edit-2511"),
            "/root/.cache/huggingface/hub/models--Qwen--Qwen-Image-Edit-2511",
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                # For HuggingFace cache, need to find the actual snapshot
                if "models--Qwen--Qwen-Image-Edit-2511" in path:
                    snapshots_dir = os.path.join(path, "snapshots")
                    if os.path.exists(snapshots_dir):
                        # Get the latest snapshot
                        snapshots = os.listdir(snapshots_dir)
                        if snapshots:
                            snapshot_path = os.path.join(snapshots_dir, snapshots[0])
                            print(f"📂 Found HF cache snapshot: {snapshot_path}")
                            return snapshot_path
                else:
                    return path
        
        return None
    
    def find_lora_path(self) -> Optional[str]:
        """Find LoRA weights path."""
        possible_paths = [
            # Relative paths
            "models/Qwen-Image-Edit-2511-Lightning/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-fp32.safetensors",
            "models/Qwen-Image-Edit-2511-Lightning/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
            "../models/Qwen-Image-Edit-2511-Lightning/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
            "lora_weights/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
            # Project root
            str(SCRIPT_DIR / "models" / "Qwen-Image-Edit-2511-Lightning" / "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"),
            # Workspace paths (Vast.ai)
            "/workspace/try_og_pipeline/models/Qwen-Image-Edit-2511-Lightning/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
            "/workspace/models/Qwen-Image-Edit-2511-Lightning/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def find_fp8_path(self) -> Optional[str]:
        """Find FP8 quantized weights path."""
        possible_paths = [
            # Correct filename with version suffix
            "models/Qwen-Image-Edit-2511-Lightning/qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning_4steps_v1.0.safetensors",
            "../models/Qwen-Image-Edit-2511-Lightning/qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning_4steps_v1.0.safetensors",
            # Workspace paths (Vast.ai)
            "/workspace/try_og_pipeline/models/Qwen-Image-Edit-2511-Lightning/qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning_4steps_v1.0.safetensors",
            "/workspace/models/Qwen-Image-Edit-2511-Lightning/qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning_4steps_v1.0.safetensors",
            # Project root
            str(SCRIPT_DIR / "models" / "Qwen-Image-Edit-2511-Lightning" / "qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning_4steps_v1.0.safetensors"),
            # Old filename (fallback)
            "models/Qwen-Image-Edit-2511-Lightning/qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning.safetensors",
            "../models/Qwen-Image-Edit-2511-Lightning/qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning.safetensors",
            "models/qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning.safetensors",
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def get_attention_mode(self) -> str:
        """Determine the best attention mode for the current GPU."""
        attn_mode = "torch_sdpa"  # Safe default
        
        if not torch.cuda.is_available():
            return attn_mode
        
        gpu_name = torch.cuda.get_device_name(0).lower()
        
        try:
            import importlib.util
            if importlib.util.find_spec("flash_attn") is not None:
                if "l40" in gpu_name or "4090" in gpu_name or "4080" in gpu_name:
                    attn_mode = "flash_attn2"
                    print(f"🔧 L40/Ada GPU detected → Using Flash Attention 2")
                elif "h100" in gpu_name or "h200" in gpu_name:
                    try:
                        from flash_attn_interface import flash_attn_func
                        attn_mode = "flash_attn3"
                        print(f"🔧 Hopper GPU detected → Using Flash Attention 3")
                    except ImportError:
                        attn_mode = "flash_attn2"
                        print(f"🔧 Hopper GPU but FA3 not installed → Using Flash Attention 2")
                else:
                    attn_mode = "flash_attn2"
                    print(f"🔧 Using Flash Attention 2")
        except Exception as e:
            print(f"🔧 Flash Attention unavailable ({type(e).__name__}), using PyTorch SDPA")
        
        return attn_mode
    
    def load_models(self):
        """Load the inference models into GPU memory."""
        if self.model_loaded or self.loading:
            return
        
        self.loading = True
        print("=" * 60)
        print("🚀 Loading LightX2V Pipeline")
        print("=" * 60)
        
        try:
            from lightx2v import LightX2VPipeline
            
            # Find model path
            self.model_path = self.find_model_paths()
            if self.model_path is None:
                raise RuntimeError("Could not find Qwen-Image-Edit-2511 model!")
            
            print(f"📂 Model path: {self.model_path}")
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            # Initialize pipeline
            self.pipe = LightX2VPipeline(
                model_path=self.model_path,
                model_cls="qwen-image-edit-2511",
                task="i2i",
            )
            
            # Configure based on mode
            if self.mode == "fp8":
                self.fp8_path = self.find_fp8_path()
                if self.fp8_path:
                    print(f"🔧 Enabling FP8 quantization: {self.fp8_path}")
                    self.pipe.enable_quantize(
                        dit_quantized=True,
                        dit_quantized_ckpt=self.fp8_path,
                        quant_scheme="fp8-sgl"
                    )
                    self.steps = self.config.model.default_steps
                else:
                    raise RuntimeError(
                        "FP8 weights not found! Please download with:\n"
                        "huggingface-cli download lightx2v/Qwen-Image-Edit-2511-Lightning --local-dir models/Qwen-Image-Edit-2511-Lightning\n"
                        "Expected file: qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning.safetensors"
                    )
            
            if self.mode == "lora":
                self.lora_path = self.find_lora_path()
                if self.lora_path:
                    print(f"🔧 Loading 4-step Lightning LoRA: {self.lora_path}")
                    self.pipe.enable_lora([
                        {"path": self.lora_path, "strength": 1.0},
                    ])
                    self.steps = self.config.model.default_steps
                else:
                    raise RuntimeError("LoRA weights not found!")
            
            if self.mode == "base":
                self.steps = 40
            
            # Get attention mode
            self.attn_mode = self.get_attention_mode()
            
            # Create generator ONCE with default 720p portrait dimensions
            # The monkey-patch in run_inference handles aspect ratio dynamically
            print("🔧 Creating generator (720p portrait default)...")
            self.pipe.create_generator(
                attn_mode=self.attn_mode,
                infer_steps=self.steps,
                guidance_scale=1.0,
                width=768,
                height=1024,
                aspect_ratio="3:4",
            )
            
            # Enable TeaCache if configured
            self.teacache_infer = None
            if self.enable_teacache:
                print(f"⚡ Enabling TeaCache (threshold={self.teacache_thresh})...")
                try:
                    # Import TeaCache implementation
                    teacache_path = Path(__file__).parent.parent.parent / "teacache_transformer_infer.py"
                    if teacache_path.exists():
                        import importlib.util
                        spec = importlib.util.spec_from_file_location("teacache_transformer_infer", teacache_path)
                        teacache_module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(teacache_module)
                        QwenImageTeaCacheTransformerInfer = teacache_module.QwenImageTeaCacheTransformerInfer
                    else:
                        from teacache_transformer_infer import QwenImageTeaCacheTransformerInfer
                    
                    # Get the current transformer_infer
                    orig_transformer_infer = self.pipe.runner.model.transformer_infer
                    
                    # Create new TeaCache infer instance with same config
                    teacache_config = orig_transformer_infer.config.copy()
                    teacache_config["teacache_thresh"] = self.teacache_thresh
                    teacache_config["coefficients"] = [0.5]  # Conservative for quality
                    teacache_config["infer_steps"] = self.steps
                    
                    # Create TeaCache wrapper
                    self.teacache_infer = QwenImageTeaCacheTransformerInfer(teacache_config)
                    self.teacache_infer.scheduler = orig_transformer_infer.scheduler
                    self.teacache_infer.infer_func = orig_transformer_infer.infer_func
                    
                    # Replace the transformer_infer
                    self.pipe.runner.model.transformer_infer = self.teacache_infer
                    
                    print("✅ TeaCache enabled")
                    print(f"   Threshold: {self.teacache_thresh}")
                except Exception as e:
                    print(f"⚠️ TeaCache setup failed: {e}")
                    print("   Continuing without TeaCache...")
            
            self.model_loaded = True
            print("✅ Pipeline loaded successfully!")
            
            # Memory stats
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                print(f"📊 GPU Memory: {allocated:.2f} GB / {total:.2f} GB")
        
        except Exception as e:
            self.loading = False
            raise RuntimeError(f"Failed to load models: {e}")
        
        finally:
            self.loading = False
        
        print("=" * 60)
    
    def warmup(self):
        """Run warmup inference for consistent timing."""
        if not self.model_loaded:
            raise RuntimeError("Models not loaded")
        
        print("🔥 Running warmup inference...")
        
        # Create dummy images
        import tempfile
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Create small dummy images
            input_img = Image.new("RGB", (512, 768), color=(0, 255, 0))
            
            input_path = os.path.join(temp_dir, "warmup_input.png")
            output_path = os.path.join(temp_dir, "warmup_output.png")
            
            input_img.save(input_path)
            
            # Run inference
            self.run_inference(
                input_image_path=input_path,
                output_path=output_path,
                seed=42,
                steps=self.steps,
                cfg=1.0,
            )
            
            print("✅ Warmup complete!")
        
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def run_inference(
        self,
        input_image_path: str,
        output_path: str,
        seed: int = 42,
        steps: Optional[int] = None,
        cfg: float = 1.0,
        target_width: Optional[int] = None,
        target_height: Optional[int] = None,
        prompt: Optional[str] = None,
    ):
        """Run VTON inference."""
        if not self.model_loaded:
            raise RuntimeError("Models not loaded")
        
        if steps is None:
            steps = self.steps
        
        if prompt is None:
            prompt = VTON_PROMPT_CN
        
        # Detect aspect ratio from input image
        input_img = Image.open(input_image_path)
        orig_w, orig_h = input_img.size
        orig_ratio = orig_w / orig_h
        input_img.close()
        
        # Determine aspect_ratio for LightX2V
        if orig_ratio < 0.8:  # Portrait
            target_aspect_ratio = "3:4"
        elif orig_ratio > 1.2:  # Landscape
            target_aspect_ratio = "4:3"
        else:  # Square
            target_aspect_ratio = "1:1"
        
        print(f"📐 Input: {orig_w}x{orig_h} (aspect: {target_aspect_ratio})")
        
        # Clear TeaCache state before each inference for consistent performance
        if self.teacache_infer is not None:
            self.teacache_infer.clear()
        
        # Prepare image paths (comma-separated for LightX2V)
        image_paths = input_image_path
        
        # Monkey-patch run_pipeline to inject aspect_ratio into input_info
        # This is needed because LightX2V's get_custom_shape() checks input_info.aspect_ratio
        original_run_pipeline = self.pipe.runner.run_pipeline
        
        def patched_run_pipeline(input_info):
            input_info.aspect_ratio = target_aspect_ratio
            # Also need to set _auto_resize in config (get_custom_shape checks it)
            self.pipe.runner.config["_auto_resize"] = False
            return original_run_pipeline(input_info)
        
        self.pipe.runner.run_pipeline = patched_run_pipeline
        
        try:
            # Generate
            self.pipe.generate(
                seed=seed,
                image_path=image_paths,
                prompt=prompt,
                negative_prompt="",
                save_result_path=output_path,
            )
        finally:
            # Restore original method
            self.pipe.runner.run_pipeline = original_run_pipeline
    
    def get_status(self) -> Dict[str, Any]:
        """Get pipeline status."""
        return {
            "model_loaded": self.model_loaded,
            "loading": self.loading,
            "mode": self.mode,
            "model_path": self.model_path,
        }
    
    def unload_models(self):
        """Unload models from GPU memory."""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        
        self.model_loaded = False
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        print("🔌 Models unloaded")
