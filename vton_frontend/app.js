/**
 * VTON Inference Tester - Application Logic
 * Handles interaction with the GPU inference server
 */

// ============================================
// State Management
// ============================================
const state = {
    serverUrl: 'http://localhost:8080',
    authToken: '',
    inputFile: null,
    resultBlob: null,
    isProcessing: false,
    serverStatus: 'unknown',
    progressTimer: null,
    startTime: 0,
};


// ============================================
// Initialization
// ============================================
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
});

function initializeApp() {
    // Load saved config from localStorage
    const savedUrl = localStorage.getItem('gpu_server_url');
    const savedToken = localStorage.getItem('gpu_server_token');

    if (savedUrl) {
        document.getElementById('serverUrl').value = savedUrl;
        state.serverUrl = savedUrl;
    }
    if (savedToken) {
        document.getElementById('authToken').value = savedToken;
        state.authToken = savedToken;
    }

    // Set up drag and drop
    setupDragAndDrop('inputUploadZone', 'input');

    // Check server status
    checkServerStatus();

    // Update config on input change
    document.getElementById('serverUrl').addEventListener('change', saveConfig);
    document.getElementById('authToken').addEventListener('change', saveConfig);
}

function saveConfig() {
    state.serverUrl = document.getElementById('serverUrl').value;
    state.authToken = document.getElementById('authToken').value;
    localStorage.setItem('gpu_server_url', state.serverUrl);
    localStorage.setItem('gpu_server_token', state.authToken);
}


// ============================================
// Server Status
// ============================================
async function checkServerStatus() {
    const statusEl = document.getElementById('serverStatus');
    const dotEl = statusEl.querySelector('.status-dot');
    const textEl = statusEl.querySelector('.status-text');

    dotEl.className = 'status-dot status-checking';
    textEl.textContent = 'Checking server...';

    try {
        // Check /test endpoint for readiness
        const response = await fetch(`${state.serverUrl}/test`, {
            method: 'GET',
            mode: 'cors',
            signal: AbortSignal.timeout(5000),
        });

        if (response.ok) {
            const data = await response.json();
            state.serverStatus = data.status;

            if (data.status === 'hot') {
                dotEl.className = 'status-dot status-hot';
                textEl.textContent = `Connected - Ready (${data.model_type || 'Unknown'})`;
            } else if (data.status === 'loading') {
                dotEl.className = 'status-dot status-checking';
                textEl.textContent = 'Connected - Loading models...';
            } else {
                dotEl.className = 'status-dot status-connected';
                textEl.textContent = `Connected - ${data.status}`;
            }
        } else {
            throw new Error(`HTTP ${response.status}`);
        }
    } catch (error) {
        state.serverStatus = 'disconnected';
        dotEl.className = 'status-dot status-disconnected';
        textEl.textContent = `Disconnected: ${error.message}`;
    }
}


// ============================================
// File Upload Handling
// ============================================
function setupDragAndDrop(zoneId, type) {
    const zone = document.getElementById(zoneId);

    zone.addEventListener('dragover', (e) => {
        e.preventDefault();
        zone.classList.add('drag-over');
    });

    zone.addEventListener('dragleave', () => {
        zone.classList.remove('drag-over');
    });

    zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.classList.remove('drag-over');

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            processFile(files[0], type);
        }
    });
}

function handleFileUpload(event, type) {
    const file = event.target.files[0];
    if (file) {
        processFile(file, type);
    }
}

function processFile(file, type) {
    // Validate file type
    if (!file.type.startsWith('image/')) {
        showError('Please upload an image file (PNG, JPG, etc.)');
        return;
    }

    // Validate file size (10MB max)
    if (file.size > 10 * 1024 * 1024) {
        showError('File size must be less than 10MB');
        return;
    }

    // Store file
    state.inputFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        const previewEl = document.getElementById(`${type}Preview`);
        const placeholderEl = document.getElementById(`${type}Placeholder`);
        const clearBtn = document.getElementById(`${type}Clear`);

        previewEl.src = e.target.result;
        previewEl.classList.remove('hidden');
        placeholderEl.classList.add('hidden');
        clearBtn.classList.remove('hidden');
    };
    reader.readAsDataURL(file);
}

function clearUpload(type) {
    state.inputFile = null;
    document.getElementById('inputInput').value = '';

    const previewEl = document.getElementById(`${type}Preview`);
    const placeholderEl = document.getElementById(`${type}Placeholder`);
    const clearBtn = document.getElementById(`${type}Clear`);

    previewEl.src = '';
    previewEl.classList.add('hidden');
    placeholderEl.classList.remove('hidden');
    clearBtn.classList.add('hidden');
}


// ============================================
// Inference
// ============================================
function randomizeSeed() {
    const seed = Math.floor(Math.random() * 2147483647);
    document.getElementById('seed').value = seed;
}

async function runInference() {
    // Validate inputs
    if (!state.inputFile) {
        showError('Please upload an input image');
        return;
    }

    if (state.serverStatus !== 'hot') {
        showError('Server is not ready. Please check connection.');
        return;
    }

    if (state.isProcessing) {
        return;
    }

    state.isProcessing = true;
    hideError();
    hideResult();
    showProgress();

    // Get parameters
    const seed = parseInt(document.getElementById('seed').value) || 42;
    const steps = parseInt(document.getElementById('steps').value) || 4;
    const cfg = parseFloat(document.getElementById('cfg').value) || 1.0;
    const prompt = document.getElementById('prompt').value.trim();

    // Create form data
    const formData = new FormData();
    formData.append('image', state.inputFile);
    formData.append('seed', seed);
    formData.append('steps', steps);
    formData.append('cfg', cfg);
    
    // Add prompt only if provided
    if (prompt) {
        formData.append('prompt', prompt);
    }

    // Prepare headers
    const headers = {};
    if (state.authToken) {
        headers['X-Internal-Auth'] = state.authToken;
    }

    try {
        updateProgress(10, 'Uploading images...');

        const response = await fetch(`${state.serverUrl}/infer`, {
            method: 'POST',
            headers: headers,
            body: formData,
        });

        updateProgress(30, 'Processing on GPU...');

        if (response.ok) {
            const blob = await response.blob();
            state.resultBlob = blob;

            const inferenceTime = response.headers.get('X-Inference-Time-Ms');
            const jobId = response.headers.get('X-Job-Id');

            hideProgress();
            showResult(blob, inferenceTime, jobId);
        } else if (response.status === 429) {
            throw new Error('GPU server is busy. Please try again in a few seconds.');
        } else if (response.status === 503) {
            throw new Error('Models are still loading. Please wait and try again.');
        } else {
            const text = await response.text();
            throw new Error(`Server error: ${text}`);
        }
    } catch (error) {
        hideProgress();
        showError(error.message);
    } finally {
        state.isProcessing = false;
    }
}


// ============================================
// Progress UI
// ============================================
function showProgress() {
    const section = document.getElementById('progressSection');
    section.classList.remove('hidden');

    state.startTime = Date.now();
    updateProgressTimer();

    state.progressTimer = setInterval(updateProgressTimer, 100);
}

function hideProgress() {
    const section = document.getElementById('progressSection');
    section.classList.add('hidden');

    if (state.progressTimer) {
        clearInterval(state.progressTimer);
        state.progressTimer = null;
    }
}

function updateProgressTimer() {
    const elapsed = (Date.now() - state.startTime) / 1000;
    document.getElementById('progressTime').textContent = `${elapsed.toFixed(1)}s`;
}

function updateProgress(percent, status) {
    document.getElementById('progressFill').style.width = `${percent}%`;
    document.getElementById('progressStatus').textContent = status;
}


// ============================================
// Result UI
// ============================================
function showResult(blob, inferenceTime, jobId) {
    const section = document.getElementById('resultSection');
    const resultImage = document.getElementById('resultImage');
    const inferenceTimeEl = document.getElementById('inferenceTime');
    const jobIdEl = document.getElementById('jobId');

    // Create object URL for the image
    const imageUrl = URL.createObjectURL(blob);
    resultImage.src = imageUrl;

    // Display stats
    if (inferenceTime) {
        const seconds = (parseFloat(inferenceTime) / 1000).toFixed(2);
        inferenceTimeEl.textContent = `${seconds}s`;
    } else {
        inferenceTimeEl.textContent = 'N/A';
    }

    jobIdEl.textContent = jobId || 'N/A';

    section.classList.remove('hidden');
    section.scrollIntoView({ behavior: 'smooth' });
}

function hideResult() {
    const section = document.getElementById('resultSection');
    section.classList.add('hidden');
    state.resultBlob = null;
}

function clearResult() {
    hideResult();
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

function downloadResult() {
    if (!state.resultBlob) {
        return;
    }

    const url = URL.createObjectURL(state.resultBlob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `vton_result_${Date.now()}.png`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}


// ============================================
// Error UI
// ============================================
function showError(message) {
    const section = document.getElementById('errorSection');
    const messageEl = document.getElementById('errorMessage');

    messageEl.textContent = message;
    section.classList.remove('hidden');
}

function hideError() {
    const section = document.getElementById('errorSection');
    section.classList.add('hidden');
}
