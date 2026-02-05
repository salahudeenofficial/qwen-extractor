"""
Mock Pipeline Manager for Testing

A mock implementation that doesn't require GPU or models.
Useful for testing the server API without actual inference.
"""

import os
import time
from typing import Optional, Dict, Any
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


class MockPipelineManager:
    """
    Mock pipeline manager for testing without GPU.
    
    Creates a simple composite image instead of actual inference.
    """
    
    def __init__(self, config):
        self.config = config
        self.model_loaded = False
        self.loading = False
        self.mode = config.model.default_mode
        self.steps = config.model.default_steps
    
    def load_models(self):
        """Simulate model loading."""
        self.loading = True
        print("🎭 Mock: Simulating model loading...")
        time.sleep(1)  # Simulate loading time
        self.model_loaded = True
        self.loading = False
        print("✅ Mock: Models 'loaded' (mock mode)")
    
    def warmup(self):
        """Simulate warmup."""
        print("🔥 Mock: Simulating warmup...")
        time.sleep(0.5)
        print("✅ Mock: Warmup complete (mock mode)")
    
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
        """
        Create a mock result image.
        
        Shows the input image with text overlay.
        """
        print(f"🎭 Mock: Running mock inference...")
        print(f"   Input: {input_image_path}")
        print(f"   Seed: {seed}, Steps: {steps or self.steps}")
        
        # Simulate processing time
        time.sleep(0.5)
        
        # Load images
        input_img = Image.open(input_image_path).convert("RGB")
        
        # Get dimensions
        width = target_width or input_img.width
        height = target_height or input_img.height
        
        # Create output image
        result = Image.new("RGB", (width, height), color=(30, 30, 40))
        
        # Resize and paste input image
        input_resized = input_img.resize((width, height), Image.LANCZOS)
        result.paste(input_resized, (0, 0))
        
        # Add text overlay
        draw = ImageDraw.Draw(result)
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except:
            font = ImageFont.load_default()
            font_small = font
        
        # Title
        draw.text((width // 2 - 80, height // 2 - 50), "MOCK EXTRACTED RESULT", fill=(255, 100, 100), font=font)
        
        # Info
        info_y = height - 100
        draw.text((10, info_y), f"Seed: {seed}", fill=(200, 200, 200), font=font_small)
        draw.text((10, info_y + 20), f"Steps: {steps or self.steps}", fill=(200, 200, 200), font=font_small)
        draw.text((10, info_y + 40), f"Mode: {self.mode}", fill=(200, 200, 200), font=font_small)
        draw.text((10, info_y + 60), "⚠️ Mock mode - not real inference", fill=(255, 200, 100), font=font_small)
        
        # Save result
        result.save(output_path)
        print(f"✅ Mock: Saved mock result to {output_path}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get pipeline status."""
        return {
            "model_loaded": self.model_loaded,
            "loading": self.loading,
            "mode": f"{self.mode} (mock)",
            "model_path": None,
        }
    
    def unload_models(self):
        """Simulate unloading models."""
        self.model_loaded = False
        print("🔌 Mock: Models 'unloaded'")


# Environment variable to enable mock mode
def get_pipeline_manager_class():
    """Return the appropriate pipeline manager class."""
    if os.environ.get("GPU_SERVER_MOCK", "").lower() in ("1", "true", "yes"):
        return MockPipelineManager
    else:
        from .pipeline_manager import PipelineManager
        return PipelineManager
