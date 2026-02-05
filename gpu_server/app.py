"""
GPU Server - FastAPI Application

Main FastAPI application for the GPU inference server.
Implements all endpoints from the GPU_SERVER_GUIDE.md specification.
"""

import asyncio
import os
import time
import uuid
import json
import logging
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any

import httpx
from fastapi import FastAPI, HTTPException, Header, UploadFile, File, Form, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core.config import GPUServerConfig, load_config, get_config, set_config
from inference import get_pipeline_manager
from workflow.engine import JobContext, create_default_workflow


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Global state
class ServerState:
    def __init__(self):
        self.config: Optional[GPUServerConfig] = None
        self.pipeline_manager: Optional[PipelineManager] = None
        self.workflow = None
        
        # Job tracking
        self.busy = False
        self.current_job_id: Optional[str] = None
        self.jobs_processed = 0
        
        # Metrics
        self.inference_count = 0
        self.inference_latency_sum = 0.0
        self.inference_errors = 0
        
        # Startup state
        self.model_loaded = False
        self.git_commit = os.environ.get("GIT_COMMIT", "unknown")


state = ServerState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler - load models on startup."""
    # Startup
    logger.info("🚀 GPU Server starting up...")
    
    # Load configuration
    config_path = os.environ.get("CONFIG_PATH", "configs/config.yaml")
    state.config = load_config(config_path)
    set_config(state.config)
    
    # Configure logging level
    logging.getLogger().setLevel(getattr(logging, state.config.logging.level))
    
    logger.info(f"📋 Node ID: {state.config.server.node_id}")
    logger.info(f"🔧 Model type: {state.config.model.model_type}")
    logger.info(f"🔧 Mode: {state.config.model.default_mode}")
    
    # Initialize pipeline manager (uses mock mode if GPU_SERVER_MOCK=1)
    state.pipeline_manager = get_pipeline_manager(state.config)
    
    # Load models (this may take a while)
    try:
        logger.info("📥 Loading models...")
        state.pipeline_manager.load_models()
        state.model_loaded = True
        
        # Warmup if configured
        if state.config.workflow.advanced.warmup_on_startup:
            logger.info("🔥 Running warmup inference...")
            state.pipeline_manager.warmup()
        
        # Create workflow
        state.workflow = create_default_workflow(state.pipeline_manager, state.config)
        
        logger.info("✅ GPU Server ready to accept jobs!")
        
    except Exception as e:
        logger.error(f"❌ Failed to load models: {e}")
        state.model_loaded = False
    
    yield
    
    # Shutdown
    logger.info("🔌 GPU Server shutting down...")
    if state.pipeline_manager:
        state.pipeline_manager.unload_models()


# Create FastAPI app
app = FastAPI(
    title="GPU Inference Server",
    description="Virtual Try-On GPU inference server with LightX2V",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Authentication
# ============================================================================

def verify_auth(x_internal_auth: Optional[str] = None) -> bool:
    """Verify the internal auth token."""
    if not state.config.security.require_auth:
        return True
    
    if not x_internal_auth:
        return False
    
    return x_internal_auth == state.config.security.internal_auth_token


# ============================================================================
# Response Models
# ============================================================================

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    node_id: str


class TestResponse(BaseModel):
    status: str
    model_loaded: bool
    node_id: str
    model_type: Optional[str] = None


class GPUStatusResponse(BaseModel):
    node_id: str
    busy: bool
    current_job_id: Optional[str]
    queue_length: int


class VersionResponse(BaseModel):
    model_type: str
    model_version: str
    backend: str
    git_commit: str
    node_id: str


class MetricsResponse(BaseModel):
    vton_inference_count: int
    vton_inference_latency_ms: float
    vton_inference_errors_total: int
    gpu_memory_used_bytes: int
    gpu_utilization_percent: float


class TryonAcceptedResponse(BaseModel):
    status: str
    job_id: str
    message: str


class TryonBusyResponse(BaseModel):
    status: str
    message: str
    retry_after: int


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health():
    """
    Liveness probe for load balancer.
    Indicates if the server is running.
    Must respond within 300ms.
    """
    return HealthResponse(
        status="ok",
        model_loaded=state.model_loaded,
        node_id=state.config.server.node_id,
    )


@app.get("/test", response_model=TestResponse)
async def test():
    """
    Readiness probe for load balancer.
    Indicates if the server is ready to accept jobs.
    Returns "hot" when ready, "loading" when models are still loading.
    """
    if state.model_loaded:
        return TestResponse(
            status="hot",
            model_loaded=True,
            node_id=state.config.server.node_id,
            model_type=state.config.model.model_type,
        )
    else:
        return TestResponse(
            status="loading",
            model_loaded=False,
            node_id=state.config.server.node_id,
        )


@app.get("/gpu/status", response_model=GPUStatusResponse)
async def gpu_status(x_internal_auth: Optional[str] = Header(None)):
    """
    Get GPU busy state for load balancer routing decisions.
    Requires X-Internal-Auth header.
    """
    if not verify_auth(x_internal_auth):
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    return GPUStatusResponse(
        node_id=state.config.server.node_id,
        busy=state.busy,
        current_job_id=state.current_job_id,
        queue_length=state.jobs_processed,
    )


@app.get("/version", response_model=VersionResponse)
async def version():
    """
    Get version and model information.
    No authentication required.
    """
    return VersionResponse(
        model_type=state.config.model.model_type,
        model_version=state.config.model.model_version,
        backend="lightx2v",
        git_commit=state.git_commit,
        node_id=state.config.server.node_id,
    )


@app.get("/metrics", response_model=MetricsResponse)
async def metrics():
    """
    Prometheus-style metrics for monitoring.
    No authentication required.
    """
    import torch
    
    gpu_memory = 0
    gpu_util = 0.0
    
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated()
        # Note: GPU utilization would need nvidia-smi or pynvml
        # This is a placeholder
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = util.gpu
        except:
            pass
    
    avg_latency = 0.0
    if state.inference_count > 0:
        avg_latency = state.inference_latency_sum / state.inference_count
    
    return MetricsResponse(
        vton_inference_count=state.inference_count,
        vton_inference_latency_ms=avg_latency,
        vton_inference_errors_total=state.inference_errors,
        gpu_memory_used_bytes=gpu_memory,
        gpu_utilization_percent=gpu_util,
    )


async def send_asset_callback(context: JobContext):
    """Send result to Asset Service callback."""
    callback_url = state.config.asset_service.callback_url
    if not callback_url:
        logger.info("No callback URL configured, skipping callback")
        return
    
    logger.info(f"📤 Sending callback to {callback_url}")
    
    retries = state.config.asset_service.retries
    backoff = state.config.asset_service.retry_backoff
    timeout = state.config.asset_service.timeout
    
    for attempt in range(retries):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                files = {}
                data = {
                    "job_id": context.job_id,
                    "user_id": context.user_id,
                    "session_id": context.session_id,
                    "provider": context.provider,
                    "node_id": state.config.server.node_id,
                    "model_version": state.config.model.model_version,
                    "inference_time_ms": str(int(context.inference_time_ms)),
                }
                
                if context.error:
                    data["error"] = context.error
                elif context.output_image_data:
                    files["output_image"] = ("result.png", context.output_image_data, "image/png")
                
                headers = {
                    "X-Internal-Auth": state.config.asset_service.internal_auth_token,
                }
                
                response = await client.post(
                    callback_url,
                    data=data,
                    files=files,
                    headers=headers,
                )
                
                if response.status_code == 200:
                    logger.info(f"✅ Callback successful for job {context.job_id}")
                    return
                else:
                    logger.warning(f"Callback returned {response.status_code}: {response.text}")
        
        except Exception as e:
            logger.warning(f"Callback attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                await asyncio.sleep(backoff[min(attempt, len(backoff) - 1)])
    
    logger.error(f"❌ All callback retries failed for job {context.job_id}")


async def send_lb_notification(job_id: str):
    """Notify Load Balancer that job is complete."""
    lb_url = state.config.load_balancer.url
    if not lb_url:
        return
    
    url = f"{lb_url}/job_complete"
    
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.post(
                url,
                json={
                    "node_id": state.config.server.node_id,
                    "job_id": job_id,
                    "metadata": {},
                },
                headers={
                    "Content-Type": "application/json",
                    "X-Internal-Auth": state.config.load_balancer.internal_auth_token,
                },
            )
            logger.info(f"LB notification sent: {response.status_code}")
    except Exception as e:
        logger.warning(f"LB notification failed: {e}")


async def process_job(context: JobContext):
    """Process a VTON job in the background."""
    try:
        # Run the workflow
        context = state.workflow.execute(context)
        
        # Update metrics
        state.inference_count += 1
        state.inference_latency_sum += context.inference_time_ms
        
        logger.info(f"✅ Job {context.job_id} completed in {context.inference_time_ms:.0f}ms")
        
    except Exception as e:
        context.error = str(e)
        state.inference_errors += 1
        logger.error(f"❌ Job {context.job_id} failed: {e}")
    
    finally:
        # Send callbacks
        await send_asset_callback(context)
        await send_lb_notification(context.job_id)
        
        # Mark GPU as free
        state.busy = False
        state.current_job_id = None
        state.jobs_processed += 1


@app.post("/tryon")
async def tryon(
    background_tasks: BackgroundTasks,
    job_id: str = Form(...),
    user_id: str = Form(...),
    session_id: str = Form(...),
    provider: str = Form(...),
    image: UploadFile = File(...),
    config: str = Form(...),
    x_internal_auth: Optional[str] = Header(None),
):
    """
    Main inference endpoint.
    Accepts a job and processes it asynchronously.
    
    Returns 202 Accepted immediately, processes in background.
    Returns 429 if GPU is busy.
    """
    # Auth check
    if not verify_auth(x_internal_auth):
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    # Validate provider
    if provider not in ["qwen", "lightxv"]:
        raise HTTPException(status_code=400, detail=f"Invalid provider: {provider}")
    
    # Check model loaded
    if not state.model_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded yet")
    
    # Check if GPU is busy
    if state.busy:
        return JSONResponse(
            status_code=429,
            content={
                "status": "busy",
                "message": "GPU server is currently busy",
                "retry_after": 5,
            },
            headers={"Retry-After": "5"},
        )
    
    # Mark GPU as busy
    state.busy = True
    state.current_job_id = job_id
    
    try:
        # Parse config
        job_config = json.loads(config)
        seed = job_config.get("seed", state.config.model.default_seed)
        steps = job_config.get("steps", state.config.model.default_steps)
        cfg = job_config.get("cfg", state.config.model.default_cfg)
        prompt = job_config.get("prompt", None)  # Optional prompt from config
        
        # Read image data
        image_data = await image.read()
        
        # Create job context
        context = JobContext(
            job_id=job_id,
            user_id=user_id,
            session_id=session_id,
            provider=provider,
            input_image_data=image_data,
            seed=seed,
            steps=steps,
            cfg=cfg,
            prompt=prompt,
        )
        
        # Add background task
        background_tasks.add_task(process_job, context)
        
        return JSONResponse(
            status_code=202,
            content={
                "status": "accepted",
                "job_id": job_id,
                "message": "Job queued for processing",
            },
        )
    
    except Exception as e:
        # Release GPU on error
        state.busy = False
        state.current_job_id = None
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# Direct Inference Endpoint (for Frontend/Testing)
# ============================================================================

@app.post("/infer")
async def infer(
    image: UploadFile = File(...),
    seed: int = Form(42),
    steps: int = Form(4),
    cfg: float = Form(1.0),
    prompt: Optional[str] = Form(None),
    x_internal_auth: Optional[str] = Header(None),
):
    """
    Direct inference endpoint for frontend/testing.
    Processes synchronously and returns the result image directly.
    
    This is simpler than /tryon for development and testing.
    """
    # Auth check (optional for this endpoint)
    if state.config.security.require_auth and not verify_auth(x_internal_auth):
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    # Check model loaded
    if not state.model_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded yet")
    
    # Check if GPU is busy
    if state.busy:
        raise HTTPException(status_code=429, detail="GPU is busy")
    
    # Mark GPU as busy
    state.busy = True
    job_id = str(uuid.uuid4())
    state.current_job_id = job_id
    
    try:
        # Read image data
        image_data = await image.read()
        
        # Create job context
        context = JobContext(
            job_id=job_id,
            user_id="frontend",
            session_id="direct",
            provider="qwen",
            input_image_data=image_data,
            seed=seed,
            steps=steps,
            cfg=cfg,
            prompt=prompt,
        )
        
        # Run workflow synchronously
        context = state.workflow.execute(context)
        
        # Update metrics
        state.inference_count += 1
        state.inference_latency_sum += context.inference_time_ms
        
        if context.error:
            raise HTTPException(status_code=500, detail=context.error)
        
        from fastapi.responses import Response
        return Response(
            content=context.output_image_data,
            media_type="image/png",
            headers={
                "X-Inference-Time-Ms": str(int(context.inference_time_ms)),
                "X-Job-Id": job_id,
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        state.inference_errors += 1
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        state.busy = False
        state.current_job_id = None
        state.jobs_processed += 1


# ============================================================================
# Main entry point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    config = load_config()
    
    uvicorn.run(
        "app:app",
        host=config.server.host,
        port=config.server.port,
        workers=config.server.workers,
        reload=False,
    )
