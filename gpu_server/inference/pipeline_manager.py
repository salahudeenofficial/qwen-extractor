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
VTON_PROMPT_CN = """将图片 1 中的绿色遮罩区域仅用于判断服装属于上半身或下半身，不要将服装限制在遮罩范围内。

将图片 2 中的服装自然地穿戴到图片 1 中的人物身上，保持图片 2 中服装的完整形状、袖长和轮廓。无论图片 2 是单独的服装图还是人物穿着该服装的图，都应准确地转移服装，同时保留其原始面料质感、材质细节和颜色准确性。

确保图片 1 中人物的面部、头发和皮肤完全保持不变。光照与阴影应自然匹配图片 1 的环境，但服装的材质外观必须忠实于图片 2。

保持边缘平滑融合、阴影逼真，整体效果自然且不改变人物的身份特征。"""

VTON_PROMPT_EN = """Use the green mask area in image 1 only to determine if the garment belongs to upper or lower body, do not restrict the garment to the mask area.

Naturally dress the person in image 1 with the garment from image 2, maintaining the complete shape, sleeve length, and silhouette of the garment from image 2. Whether image 2 shows the garment alone or on a model, accurately transfer the garment while preserving its original fabric texture, material details, and color accuracy.

Ensure the face, hair, and skin of the person in image 1 remain completely unchanged. Lighting and shadows should naturally match the environment of image 1, but the material appearance of the garment must stay faithful to image 2.

Keep edges smoothly blended, shadows realistic, and the overall effect natural without altering the person's identity features."""


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
            # Relative paths
            "models/Qwen-Image-Edit-2511-Lightning/qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning.safetensors",
            "models/qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning.safetensors",
            "../models/Qwen-Image-Edit-2511-Lightning/qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning.safetensors",
            # Project root
            str(SCRIPT_DIR / "models" / "Qwen-Image-Edit-2511-Lightning" / "qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning.safetensors"),
            # Workspace paths (Vast.ai)
            "/workspace/try_og_pipeline/models/Qwen-Image-Edit-2511-Lightning/qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning.safetensors",
            "/workspace/models/Qwen-Image-Edit-2511-Lightning/qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning.safetensors",
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
                    self.steps = 4
                else:
                    print("⚠️ FP8 weights not found, falling back to LoRA mode")
                    self.mode = "lora"
            
            if self.mode == "lora":
                self.lora_path = self.find_lora_path()
                if self.lora_path:
                    print(f"🔧 Loading 4-step Lightning LoRA: {self.lora_path}")
                    self.pipe.enable_lora([
                        {"path": self.lora_path, "strength": 1.0},
                    ])
                    self.steps = 4
                else:
                    raise RuntimeError("LoRA weights not found!")
            
            if self.mode == "base":
                self.steps = 40
            
            # Get attention mode
            attn_mode = self.get_attention_mode()
            
            # Create generator with default settings
            # Using 720p 3:4 portrait as default
            self.pipe.create_generator(
                attn_mode=attn_mode,
                infer_steps=self.steps,
                guidance_scale=1.0,
                width=768,
                height=1024,
                aspect_ratio="3:4",
            )
            
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
            person = Image.new("RGB", (512, 768), color=(0, 255, 0))
            garment = Image.new("RGB", (512, 512), color=(255, 255, 255))
            
            person_path = os.path.join(temp_dir, "warmup_person.png")
            garment_path = os.path.join(temp_dir, "warmup_garment.png")
            output_path = os.path.join(temp_dir, "warmup_output.png")
            
            person.save(person_path)
            garment.save(garment_path)
            
            # Run inference
            self.run_inference(
                person_image_path=person_path,
                garment_image_path=garment_path,
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
        person_image_path: str,
        garment_image_path: str,
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
        
        # Detect aspect ratio and set resolution
        person_img = Image.open(person_image_path)
        orig_w, orig_h = person_img.size
        orig_ratio = orig_w / orig_h
        
        # Determine aspect_ratio for LightX2V
        if orig_ratio < 0.8:  # Portrait
            target_aspect_ratio = "3:4"
        elif orig_ratio > 1.2:  # Landscape
            target_aspect_ratio = "4:3"
        else:  # Square
            target_aspect_ratio = "1:1"
        
        # Prepare image paths
        image_paths = f"{person_image_path},{garment_image_path}"
        
        # Monkey-patch to inject aspect_ratio
        original_run_pipeline = self.pipe.runner.run_pipeline
        def patched_run_pipeline(input_info):
            input_info.aspect_ratio = target_aspect_ratio
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
