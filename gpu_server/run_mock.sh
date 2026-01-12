#!/bin/bash
# Run GPU Server in Mock Mode (for testing without GPU)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=" * 60
echo "🎭 Starting GPU Server (MOCK MODE)"
echo "=" * 60
echo ""
echo "⚠️  This mode does NOT run actual inference!"
echo "    It creates mock composite images for API testing."
echo ""

# Enable mock mode
export GPU_SERVER_MOCK=1
export CONFIG_PATH="${CONFIG_PATH:-configs/config.dev.yaml}"
export PYTHONPATH="${SCRIPT_DIR}:${SCRIPT_DIR}/..:${PYTHONPATH}"

# Get host and port
HOST="${GPU_SERVER_HOST:-0.0.0.0}"
PORT="${GPU_SERVER_PORT:-8000}"

echo "📋 Configuration:"
echo "   Mock Mode: ENABLED"
echo "   Host: $HOST"
echo "   Port: $PORT"
echo "   Config: $CONFIG_PATH"
echo ""

# Start the server
python -m uvicorn app:app --host "$HOST" --port "$PORT" --reload
