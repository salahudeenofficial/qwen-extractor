#!/bin/bash
# GPU Server Startup Script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=" * 60
echo "🚀 Starting GPU Server"
echo "=" * 60

# Check if config exists
if [ ! -f "configs/config.yaml" ]; then
    echo "⚠️  No config found, using defaults"
fi

# Set environment variables
export CONFIG_PATH="${CONFIG_PATH:-configs/config.yaml}"
export PYTHONPATH="${SCRIPT_DIR}:${SCRIPT_DIR}/..:${PYTHONPATH}"

# Install dependencies if needed
if ! python -c "import fastapi" 2>/dev/null; then
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
fi

# Get host and port from config or use defaults
HOST="${GPU_SERVER_HOST:-0.0.0.0}"
PORT="${GPU_SERVER_PORT:-8000}"

echo ""
echo "📋 Configuration:"
echo "   Host: $HOST"
echo "   Port: $PORT"
echo "   Config: $CONFIG_PATH"
echo ""

# Start the server
python -m uvicorn app:app --host "$HOST" --port "$PORT" --workers 1
