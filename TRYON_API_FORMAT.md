# /tryon API Request Format

## Endpoint
```
POST /tryon
```

## Authentication
**Required Header:**
```
X-Internal-Auth: <your-auth-token>
```
- Token must match `INCOMING_AUTH_TOKEN` configured in `start_server.sh`
- If `require_auth: false` in config.yaml, this header is optional

## Request Format
**Content-Type:** `multipart/form-data`

## Required Form Fields

### 1. `job_id` (string, required)
- Unique identifier for this inference job
- Example: `"job-12345"` or `"vton-2024-01-21-001"`

### 2. `user_id` (string, required)
- User identifier
- Example: `"user-abc123"`

### 3. `session_id` (string, required)
- Session identifier
- Example: `"session-xyz789"`

### 4. `provider` (string, required)
- Model provider name
- Valid values: `"qwen"` or `"lightxv"`
- Example: `"qwen"`

### 5. `image` (file, required)
- Input image file (PNG or JPEG)
- Must be uploaded as a file in multipart form
- Example: `File("input.png", image_bytes, "image/png")`

### 7. `config` (string, required - JSON format)
- Inference configuration as JSON string
- Fields: `seed`, `steps`, `cfg`, `prompt` (all optional)
- Example: `'{"seed": 42, "steps": 4, "cfg": 1.0}'`

**Config JSON Fields:**
- `seed` (int, optional): Random seed (default: 42)
- `steps` (int, optional): Inference steps (default: 4)
- `cfg` (float, optional): Guidance scale (default: 1.0)
- `prompt` (string, optional): Custom prompt text (default: uses built-in VTON prompt)

## Example Request (cURL)

```bash
curl -X POST "http://localhost:8000/tryon" \
  -H "X-Internal-Auth: dev-secret-token-change-in-production" \
  -F "job_id=job-12345" \
  -F "user_id=user-abc123" \
  -F "session_id=session-xyz789" \
  -F "provider=qwen" \
  -F "config={\"seed\": 42, \"steps\": 4, \"cfg\": 1.0}" \
  -F "image=@input.png"
```

## Example Request (Python)

```python
import requests

url = "http://localhost:8000/tryon"
headers = {
    "X-Internal-Auth": "dev-secret-token-change-in-production"
}

data = {
    "job_id": "job-12345",
    "user_id": "user-abc123",
    "session_id": "session-xyz789",
    "provider": "qwen",
    "config": '{"seed": 42, "steps": 4, "cfg": 1.0}'
}

files = {
    "image": ("input.png", open("input.png", "rb"), "image/png")
}

response = requests.post(url, headers=headers, data=data, files=files)
print(response.status_code)  # 202, 429, 401, 400, 503
print(response.json())
```

## Response Codes

### 202 Accepted
Job accepted and queued for processing.
```json
{
  "status": "accepted",
  "job_id": "job-12345",
  "message": "Job queued for processing"
}
```

### 429 Too Many Requests
GPU server is currently busy processing another job.
```json
{
  "status": "busy",
  "message": "GPU server is currently busy",
  "retry_after": 5
}
```
- Check `Retry-After` header for suggested retry delay

### 401 Unauthorized
Authentication failed or missing.
```json
{
  "detail": "Unauthorized"
}
```

### 400 Bad Request
Invalid request (missing fields, invalid provider, etc.)
```json
{
  "detail": "Invalid provider: invalid_provider"
}
```

### 503 Service Unavailable
Models not loaded yet (server still starting up).
```json
{
  "detail": "Models not loaded yet"
}
```

## Important Notes

1. **Asynchronous Processing**: This endpoint returns immediately (202 Accepted) and processes the job in the background
2. **Results via Callback**: Results are NOT returned in the HTTP response. They are sent to the `RESULT_CALLBACK_URL` configured in `start_server.sh`
3. **Single Job at a Time**: The server processes one job at a time. If busy, you'll get a 429 response
4. **Job Tracking**: Use the `job_id` to track your job. The same `job_id` will be included in the callback response

## Callback Response Format

When inference completes, the server sends a POST request to your `RESULT_CALLBACK_URL`:

**Headers:**
```
X-Internal-Auth: <RESULT_CALLBACK_AUTH_TOKEN>
Content-Type: multipart/form-data
```

**Form Data:**
- `job_id` - Original job identifier
- `user_id` - User identifier
- `session_id` - Session identifier
- `provider` - Model provider (e.g., "qwen")
- `node_id` - GPU node identifier
- `model_version` - Model version used
- `inference_time_ms` - Inference duration in milliseconds
- `output_image` - Result PNG image file (on success)
- `error` - Error message (on failure)

**Expected Response:** `200 OK`
