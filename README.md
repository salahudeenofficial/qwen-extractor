# LightX2V Virtual Try-On Pipeline

Fast VTON inference using LightX2V with Qwen-Image-Edit-2511.

## Performance

| Mode | Steps | Time (L40) | VRAM | Output |
|------|-------|------------|------|--------|
| FP8 | 4 | ~12-13s | ~35GB | 768x1024 |
| FP8+TeaCache | 4 | ~10s | ~35GB | 768x1024 |

## Quick Start

### Using the Startup Script (Recommended)

The easiest way to get started is using the startup script:

```bash
cd gpu_server
# Edit the three variables at the top of start_server.sh:
# - RESULT_CALLBACK_URL
# - JOB_COMPLETE_CALLBACK_URL  
# - SERVER_PORT
./start_server.sh
```

The script will:
1. Configure callbacks and port
2. Setup LightX2V (clone, checkout compatible version, install)
3. Download required models
4. Install all dependencies
5. Start the server on port 8000

### Manual Setup

```bash
# Docker: lightx2v/lightx2v:25101501-cu124

git clone https://github.com/salahudeenofficial/try_og_pipeline.git
cd try_og_pipeline

# Setup LightX2V and download models
cd gpu_server
./start_server.sh

# Or for standalone inference:
python ../test_lightx2v_vton.py --mode fp8 --person person.jpg --cloth cloth.png
```

## GPU Server

A production-ready HTTP API for the VTON pipeline.

### Configuration

Edit `gpu_server/start_server.sh` and set these variables at the top:

```bash
RESULT_CALLBACK_URL="http://your-backend:9009/v1/vton/result"
JOB_COMPLETE_CALLBACK_URL="http://your-lb:9005"
SERVER_PORT=8000
```

### Key Endpoints

- `POST /tryon` - Async inference (for backend integration)
- `POST /infer` - Sync inference (for frontend/testing)
- `GET /health` - Liveness probe
- `GET /test` - Readiness probe
- `GET /metrics` - Prometheus metrics

### Callbacks

**Result Callback (Asset Service)**
- Sends inference results (image + metadata) to your backend
- Configure via `RESULT_CALLBACK_URL` in `start_server.sh`
- POSTs to the full endpoint URL with multipart/form-data

**Job Complete Callback (Load Balancer)**
- Notifies load balancer when job completes
- Configure via `JOB_COMPLETE_CALLBACK_URL` in `start_server.sh`
- Server automatically appends `/job_complete` to the base URL

## Frontend Tester

A modern web UI for testing the GPU server:

```bash
cd vton_frontend
python serve.py
# Open http://localhost:3000
```

Features:
- 🎨 Premium dark theme with glassmorphism
- 📤 Drag & drop image upload
- ⚙️ Configurable inference parameters
- 📊 Real-time server status

## Project Structure

```
try_og_pipeline/
├── gpu_server/               # HTTP API Server
│   ├── start_server.sh       # Main startup script (configurable)
│   ├── app.py                # FastAPI application
│   ├── configs/              # YAML configuration
│   ├── inference/            # Pipeline manager
│   └── workflow/             # Editable workflow engine
│
├── vton_frontend/            # Frontend Tester
│   ├── index.html            # Main page
│   ├── styles.css            # Premium styling
│   └── app.js                # Application logic
│
├── test_lightx2v_vton.py     # Standalone inference script
├── README.md                 # This file
└── PROBLEMS_FACED.txt        # Known issues & fixes
```

## Documentation

- `PROBLEMS_FACED.txt` - Known issues & fixes
