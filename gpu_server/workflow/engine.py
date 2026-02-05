"""
Workflow Engine for GPU Server

This module provides an easily editable workflow system for the VTON inference pipeline.
Workflows can be customized by modifying the config or extending the base classes.
"""

import os
import gc
import time
import tempfile
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

import torch
from PIL import Image


@dataclass
class JobContext:
    """Context object passed through the workflow pipeline."""
    job_id: str
    user_id: str
    session_id: str
    provider: str
    
    # Input data
    input_image_data: bytes
    
    # Config from request
    seed: int = 42
    steps: int = 4
    cfg: float = 1.0
    prompt: Optional[str] = None
    
    # Processing state
    temp_dir: Optional[str] = None
    input_image_path: Optional[str] = None
    output_image_path: Optional[str] = None
    
    # Results
    output_image_data: Optional[bytes] = None
    inference_time_ms: float = 0
    error: Optional[str] = None
    
    # Metadata
    start_time: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def cleanup(self):
        """Clean up temporary files."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            self.temp_dir = None


class WorkflowStep(ABC):
    """Base class for workflow steps."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Step name for logging."""
        pass
    
    @abstractmethod
    def execute(self, context: JobContext) -> JobContext:
        """Execute the workflow step."""
        pass
    
    def should_skip(self, context: JobContext) -> bool:
        """Check if this step should be skipped."""
        return False


class PrepareInputsStep(WorkflowStep):
    """Step 1: Prepare input images from binary data."""
    
    @property
    def name(self) -> str:
        return "PrepareInputs"
    
    def execute(self, context: JobContext) -> JobContext:
        data_len = len(context.input_image_data)
        print(f"DEBUG: PrepareInputsStep received {data_len} bytes")
        
        # Log first 50 bytes to identify file signature
        header = context.input_image_data[:50]
        print(f"DEBUG: Header (Hex): {header.hex(' ')}")
        try:
            print(f"DEBUG: Header (Text): {header.decode('utf-8', errors='replace')}")
        except:
            pass

        # Check for AVIF signature
        is_avif = b'ftypavif' in context.input_image_data[:20] or b'ftypmif1' in context.input_image_data[:20]

        # Validate image data before saving
        try:
            import io
            # Register AVIF plugin if installed (just by importing it)
            try:
                import pillow_avif
            except ImportError:
                pass
            
            # Try opening from bytes to verify it's a valid image
            img = Image.open(io.BytesIO(context.input_image_data))
            img.verify()  # Check if it's broken
            print(f"DEBUG: Image verified. Format: {img.format}, Size: {img.size}, Mode: {img.mode}")
        except Exception as e:
            print(f"ERROR: Invalid image data: {e}")
            
            if is_avif:
                 error_msg = "AVIF image format detected but could not be decoded. Please install 'pillow-avif-plugin' (via pip) or convert the image to PNG/JPG before uploading."
                 print(f"ERROR: {error_msg}")
                 raise ValueError(error_msg)

            if data_len < 1000:
                try:
                    print(f"DEBUG: Content as text: {context.input_image_data.decode('utf-8')}")
                except:
                    pass
            raise ValueError(f"Uploaded file is not a valid image: {e}")

        # Create temp directory
        context.temp_dir = tempfile.mkdtemp(prefix=f"vton_{context.job_id}_")
        
        # Save input image
        # Use the original format if detected, or generic extension
        ext = ".png"
        try:
             # Re-open stream since verify() consumes it
             img = Image.open(io.BytesIO(context.input_image_data))
             if img.format:
                 ext = f".{img.format.lower()}"
        except:
             if is_avif:
                 ext = ".avif"
             pass

        input_path = os.path.join(context.temp_dir, f"input{ext}")
        with open(input_path, 'wb') as f:
            f.write(context.input_image_data)
        context.input_image_path = input_path
        
        # Set output path
        context.output_image_path = os.path.join(context.temp_dir, "output.png")
        
        return context


class ResizeImagesStep(WorkflowStep):
    """Step 2: Resize images to target resolution."""
    
    def __init__(self, resolution: str = "720p", maintain_aspect_ratio: bool = True):
        self.resolution = resolution
        self.maintain_aspect_ratio = maintain_aspect_ratio
        
        # Resolution presets (shorter side)
        self.res_map = {"480p": 480, "720p": 720, "1080p": 1080}
    
    @property
    def name(self) -> str:
        return "ResizeImages"
    
    def execute(self, context: JobContext) -> JobContext:
        target_short = self.res_map.get(self.resolution, 720)
        
        # Load and resize input image
        input_img = Image.open(context.input_image_path)
        w, h = input_img.size
        aspect_ratio = w / h
        
        if w >= h:
            target_height = target_short
            target_width = int(target_short * aspect_ratio)
        else:
            target_width = target_short
            target_height = int(target_short / aspect_ratio)
        
        # Ensure divisible by 16
        target_width = (target_width // 16) * 16
        target_height = (target_height // 16) * 16
        
        # Resize and save
        input_resized = input_img.resize((target_width, target_height), Image.LANCZOS)
        input_resized.save(context.input_image_path)
        
        # Store resolution in metadata
        context.metadata["target_width"] = target_width
        context.metadata["target_height"] = target_height
        context.metadata["aspect_ratio"] = aspect_ratio
        
        return context


class InferenceStep(WorkflowStep):
    """Step 3: Run the VTON inference."""
    
    def __init__(self, pipeline_manager):
        self.pipeline_manager = pipeline_manager
    
    @property
    def name(self) -> str:
        return "Inference"
    
    def execute(self, context: JobContext) -> JobContext:
        infer_start = time.time()
        
        try:
            # Run inference using the pipeline manager
            self.pipeline_manager.run_inference(
                input_image_path=context.input_image_path,
                output_path=context.output_image_path,
                seed=context.seed,
                steps=context.steps,
                cfg=context.cfg,
                target_width=context.metadata.get("target_width"),
                target_height=context.metadata.get("target_height"),
                prompt=context.prompt,
            )
            
            context.inference_time_ms = (time.time() - infer_start) * 1000
            
        except Exception as e:
            context.error = str(e)
            raise
        
        return context


class CollectResultStep(WorkflowStep):
    """Step 4: Read the output image."""
    
    @property
    def name(self) -> str:
        return "CollectResult"
    
    def execute(self, context: JobContext) -> JobContext:
        if context.output_image_path and os.path.exists(context.output_image_path):
            with open(context.output_image_path, 'rb') as f:
                context.output_image_data = f.read()
        return context


class ClearGPUCacheStep(WorkflowStep):
    """Step 5: Clear GPU cache."""
    
    @property
    def name(self) -> str:
        return "ClearGPUCache"
    
    def execute(self, context: JobContext) -> JobContext:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        return context


class WorkflowEngine:
    """
    Workflow engine that orchestrates the VTON inference pipeline.
    
    Steps can be added, removed, or reordered for flexibility.
    """
    
    def __init__(self, steps: Optional[List[WorkflowStep]] = None):
        self.steps: List[WorkflowStep] = steps or []
    
    def add_step(self, step: WorkflowStep):
        """Add a step to the workflow."""
        self.steps.append(step)
    
    def insert_step(self, index: int, step: WorkflowStep):
        """Insert a step at a specific position."""
        self.steps.insert(index, step)
    
    def remove_step(self, name: str):
        """Remove a step by name."""
        self.steps = [s for s in self.steps if s.name != name]
    
    def execute(self, context: JobContext) -> JobContext:
        """Execute all workflow steps."""
        print(f"🔄 Starting workflow for job {context.job_id}")
        
        try:
            for step in self.steps:
                if step.should_skip(context):
                    print(f"  ⏭️  Skipping step: {step.name}")
                    continue
                
                print(f"  ▶️  Executing step: {step.name}")
                step_start = time.time()
                
                context = step.execute(context)
                
                step_time = (time.time() - step_start) * 1000
                print(f"  ✅ Step {step.name} completed in {step_time:.0f}ms")
            
            print(f"✅ Workflow completed for job {context.job_id}")
            
        except Exception as e:
            context.error = str(e)
            print(f"❌ Workflow failed for job {context.job_id}: {e}")
            raise
        
        finally:
            # Always cleanup temp files
            context.cleanup()
        
        return context


def create_default_workflow(pipeline_manager, config) -> WorkflowEngine:
    """
    Create the default VTON workflow based on configuration.
    
    This function is the main customization point for the workflow.
    Modify this to change the default pipeline behavior.
    """
    workflow = WorkflowEngine()
    
    # Step 1: Prepare inputs (always needed)
    workflow.add_step(PrepareInputsStep())
    
    # Step 2: Resize images (configurable)
    if config.workflow.preprocess.resize_to_resolution:
        workflow.add_step(ResizeImagesStep(
            resolution=config.model.default_resolution,
            maintain_aspect_ratio=config.workflow.preprocess.maintain_aspect_ratio,
        ))
    
    # Step 3: Run inference (always needed)
    workflow.add_step(InferenceStep(pipeline_manager))
    
    # Step 4: Collect result (always needed)
    workflow.add_step(CollectResultStep())
    
    # Step 5: Clear GPU cache (configurable)
    if config.workflow.advanced.clear_cache_between_jobs:
        workflow.add_step(ClearGPUCacheStep())
    
    return workflow
