#!/bin/bash
# GPU Server Startup Script
# Configures, installs dependencies, and starts the server

set -e

# ============================================================================
# CONFIGURATION VARIABLES - Edit these values as needed
# ============================================================================

# Result callback URL - where to send inference results (full endpoint URL)
# This is the complete URL where inference results (images + metadata) will be sent
RESULT_CALLBACK_URL="http://13.201.18.255:9009/v1/vton/result"

# Load balancer base URL - for job completion notifications (base URL only, NOT including /job_complete)
# The server automatically appends /job_complete to this URL
# Example: If you set "http://13.201.18.255:9005", the server will POST to "http://13.201.18.255:9005/job_complete"
JOB_COMPLETE_CALLBACK_URL="http://13.201.18.255:9005"

# Server port
SERVER_PORT=8000

# ============================================================================
# SCRIPT START - Do not edit below unless you know what you're doing
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "🚀 GPU Server Startup"
echo "============================================================"
echo ""

# Step 1: Validate we're in the right directory
if [ ! -f "app.py" ] || [ ! -d "configs" ]; then
    echo "❌ Error: Must run this script from the gpu_server directory"
    echo "   Current directory: $(pwd)"
    exit 1
fi

# Step 2: Update configuration file
echo "📝 Step 1: Configuring server..."
CONFIG_FILE="configs/config.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Create backup of config file
cp "$CONFIG_FILE" "${CONFIG_FILE}.backup"
echo "   ✓ Created backup: ${CONFIG_FILE}.backup"

# Update port (match line starting with "  port:")
sed -i "s|^  port:.*|  port: ${SERVER_PORT}                      # Port to listen on|" "$CONFIG_FILE"

# Update result callback URL (asset_service.callback_url)
# Match line that has "callback_url:" under asset_service section (2 spaces indent)
sed -i "s|^  callback_url:.*|  callback_url: \"${RESULT_CALLBACK_URL}\"|" "$CONFIG_FILE"

# Update job complete callback URL (load_balancer.url)
# Match line that has "url:" under load_balancer section (2 spaces indent)
sed -i "/^load_balancer:/,/^[^ ]/s|^  url:.*|  url: \"${JOB_COMPLETE_CALLBACK_URL}\"    # Load balancer URL|" "$CONFIG_FILE"

echo "   ✓ Updated port: ${SERVER_PORT}"
echo "   ✓ Updated result callback: ${RESULT_CALLBACK_URL}"
echo "   ✓ Updated load balancer URL: ${JOB_COMPLETE_CALLBACK_URL} (will POST to ${JOB_COMPLETE_CALLBACK_URL}/job_complete)"
echo ""

# Step 3: Setup LightX2V (pinned to compatible version)
echo "📦 Step 2: Setting up LightX2V..."
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
if [ ! -d "${PARENT_DIR}/LightX2V" ]; then
    echo "   Cloning LightX2V..."
    cd "$PARENT_DIR"
    git clone https://github.com/ModelTC/LightX2V.git
else
    echo "   ✓ LightX2V directory already exists"
fi

cd "${PARENT_DIR}/LightX2V"
# IMPORTANT: Pin to commit before breaking changes
echo "   Checking out compatible version (7651b0f)..."
git fetch origin 2>/dev/null || true
git checkout 7651b0f 2>/dev/null || git checkout -f 7651b0f
pip install -q -e . 2>/dev/null || pip install -e .
cd "$SCRIPT_DIR"
echo "   ✅ LightX2V ready"
echo ""

# Step 3.5: Download models
echo "📥 Step 2.5: Downloading models..."
cd "$PARENT_DIR"

# Ensure huggingface-cli is available
if ! command -v huggingface-cli &> /dev/null; then
    echo "   Installing huggingface_hub for model downloads..."
    pip install -q "huggingface_hub[cli]" || pip install "huggingface_hub[cli]"
fi

mkdir -p models

# Download base model if not present
if [ ! -d "models/Qwen-Image-Edit-2511" ] || [ -z "$(ls -A models/Qwen-Image-Edit-2511 2>/dev/null)" ]; then
    echo "   Downloading Qwen-Image-Edit-2511 base model..."
    huggingface-cli download Qwen/Qwen-Image-Edit-2511 --local-dir models/Qwen-Image-Edit-2511
else
    echo "   ✅ Base model already downloaded"
fi

# Download Lightning/FP8 weights if not present
if [ ! -d "models/Qwen-Image-Edit-2511-Lightning" ] || [ -z "$(ls -A models/Qwen-Image-Edit-2511-Lightning 2>/dev/null)" ]; then
    echo "   Downloading Lightning LoRA and FP8 weights..."
    huggingface-cli download lightx2v/Qwen-Image-Edit-2511-Lightning --local-dir models/Qwen-Image-Edit-2511-Lightning
else
    echo "   ✅ Lightning weights already downloaded"
fi

# Verify FP8 weights exist
FP8_FILE="models/Qwen-Image-Edit-2511-Lightning/qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning_4steps_v1.0.safetensors"
if [ -f "$FP8_FILE" ]; then
    echo "   ✅ FP8 weights found"
else
    echo "   ⚠️  FP8 weights not found at expected path!"
    echo "      Looking for: $FP8_FILE"
    echo "      Available files:"
    ls -la models/Qwen-Image-Edit-2511-Lightning/ 2>/dev/null || echo "   (folder not found)"
fi

cd "$SCRIPT_DIR"
echo ""

# Step 4: Install parent requirements
echo "📦 Step 3: Installing parent requirements (PyTorch, transformers, etc.)..."
if [ -f "${PARENT_DIR}/requirements.txt" ]; then
    echo "   Installing from: ${PARENT_DIR}/requirements.txt"
    pip install -q -r "${PARENT_DIR}/requirements.txt"
    echo "   ✓ Parent requirements installed"
else
    echo "   ⚠️  Parent requirements.txt not found, skipping..."
fi
echo ""

# Step 5: Install GPU server requirements
echo "📦 Step 4: Installing GPU server requirements..."
if [ -f "requirements.txt" ]; then
    pip install -q -r requirements.txt
    echo "   ✓ GPU server requirements installed"
else
    echo "   ❌ Error: requirements.txt not found in gpu_server directory"
    exit 1
fi
echo ""

# Step 6: Set environment variables
export CONFIG_PATH="${CONFIG_PATH:-configs/config.yaml}"
export PYTHONPATH="${SCRIPT_DIR}:${SCRIPT_DIR}/..:${PYTHONPATH}"

# Step 7: Verify configuration
echo "📋 Step 5: Verifying configuration..."
echo ""
echo "   Configuration Summary:"
echo "   ────────────────────────────────────────────"
echo "   Server Port:              ${SERVER_PORT}"
echo "   Result Callback:          ${RESULT_CALLBACK_URL}"
echo "   Load Balancer Base URL:   ${JOB_COMPLETE_CALLBACK_URL}"
echo "   (LB POST endpoint):       ${JOB_COMPLETE_CALLBACK_URL}/job_complete"
echo "   Config File:              ${CONFIG_PATH}"
echo "   ────────────────────────────────────────────"
echo ""

# Step 8: Start the server
echo "============================================================"
echo "🚀 Starting GPU Server..."
echo "============================================================"
echo ""
echo "   Server will be available at: http://0.0.0.0:${SERVER_PORT}"
echo "   Health check: http://localhost:${SERVER_PORT}/health"
echo "   API docs: http://localhost:${SERVER_PORT}/docs"
echo ""
echo "   Press Ctrl+C to stop the server"
echo ""
echo "============================================================"
echo ""

# Start uvicorn server
python -m uvicorn app:app --host 0.0.0.0 --port "${SERVER_PORT}" --workers 1
