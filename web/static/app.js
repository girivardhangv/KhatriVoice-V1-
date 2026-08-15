/**
 * KhatriVoice Web Playground - Frontend Logic
 */

// State
let isGenerating = false;
let modelLoaded = false;

// DOM Elements
const chatMessages = document.getElementById('chat-messages');
const promptInput = document.getElementById('prompt-input');
const generateBtn = document.getElementById('generate-btn');
const statusDot = document.getElementById('status-dot');
const statusText = document.getElementById('status-text');
const errorContainer = document.getElementById('error-container');
const errorMessage = document.getElementById('error-message');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    checkStatus();

    // Enter key to submit (Shift+Enter for new line)
    promptInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            generate();
        }
    });

    // Focus input
    promptInput.focus();
});

/**
 * Update slider value display
 */
function updateSliderValue(id) {
    const slider = document.getElementById(id);
    const valueDisplay = document.getElementById(`${id}-value`);
    valueDisplay.textContent = slider.value;
}

/**
 * Check model status
 */
async function checkStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();

        modelLoaded = data.model_loaded;
        updateStatus(modelLoaded ? 'ready' : 'error');

        if (data.model_loaded) {
            statusText.textContent = `Model Ready (${data.device})`;
            hideError();
        } else {
            statusText.textContent = 'Model Not Loaded';
            if (data.error) {
                showError(`Failed to load model: ${data.error}`);
            }
        }
    } catch (error) {
        updateStatus('error');
        statusText.textContent = 'Server Error';
        showError(`Cannot connect to server: ${error.message}`);
    }
}

/**
 * Update status indicator
 */
function updateStatus(status) {
    statusDot.className = `status-dot ${status}`;
}

/**
 * Show error message
 */
function showError(message) {
    errorContainer.style.display = 'block';
    errorMessage.textContent = message;
}

/**
 * Hide error message
 */
function hideError() {
    errorContainer.style.display = 'none';
}

/**
 * Add message to chat
 */
function addMessage(role, content) {
    // Remove welcome message if present
    const welcome = chatMessages.querySelector('.welcome-message');
    if (welcome) {
        welcome.remove();
    }

    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;

    const label = document.createElement('div');
    label.className = 'message-label';
    label.textContent = role === 'user' ? 'You' : 'KhatriVoice';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.textContent = content;

    messageDiv.appendChild(label);
    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);

    // Scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;

    return messageDiv;
}

/**
 * Add loading indicator
 */
function addLoadingIndicator() {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';
    messageDiv.id = 'loading-message';

    const label = document.createElement('div');
    label.className = 'message-label';
    label.textContent = 'KhatriVoice';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';

    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'loading-indicator';
    loadingDiv.innerHTML = '<span></span><span></span><span></span>';

    contentDiv.appendChild(loadingDiv);
    messageDiv.appendChild(label);
    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);

    chatMessages.scrollTop = chatMessages.scrollHeight;

    return messageDiv;
}

/**
 * Remove loading indicator
 */
function removeLoadingIndicator() {
    const loading = document.getElementById('loading-message');
    if (loading) {
        loading.remove();
    }
}

/**
 * Generate text
 */
async function generate() {
    if (isGenerating) return;

    const prompt = promptInput.value.trim();

    if (!prompt) {
        showError('Please enter a prompt');
        return;
    }

    if (!modelLoaded) {
        showError('Model is not loaded. Please wait or check the server logs.');
        return;
    }

    hideError();
    isGenerating = true;
    generateBtn.disabled = true;
    generateBtn.innerHTML = '<span class="btn-icon">⏳</span> Generating...';

    // Add user message
    addMessage('user', prompt);

    // Clear input
    promptInput.value = '';

    // Add loading indicator
    addLoadingIndicator();

    try {
        // Get parameters
        const temperature = parseFloat(document.getElementById('temperature').value);
        const maxTokens = parseInt(document.getElementById('max_tokens').value);
        const topP = parseFloat(document.getElementById('top_p').value);
        const topK = parseInt(document.getElementById('top_k').value);

        // Make request
        const response = await fetch('/api/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                prompt: prompt,
                max_new_tokens: maxTokens,
                temperature: temperature,
                top_p: topP,
                top_k: topK,
            }),
        });

        removeLoadingIndicator();

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `Server error: ${response.status}`);
        }

        const data = await response.json();

        // Add assistant message
        addMessage('assistant', data.response);

    } catch (error) {
        removeLoadingIndicator();
        showError(`Generation failed: ${error.message}`);
        console.error('Generation error:', error);
    } finally {
        isGenerating = false;
        generateBtn.disabled = false;
        generateBtn.innerHTML = '<span class="btn-icon">⚡</span> Generate';
        promptInput.focus();
    }
}

/**
 * Clear chat history
 */
function clearChat() {
    chatMessages.innerHTML = `
        <div class="welcome-message">
            <div class="welcome-icon">🎤</div>
            <h2>Welcome to KhatriVoice</h2>
            <p>Type a prompt below and press Generate to test the model.</p>
            <p class="hint">The model runs locally on your machine.</p>
        </div>
    `;
    hideError();
    promptInput.focus();
}

/**
 * Reload model (for debugging)
 */
async function reloadModel() {
    try {
        const response = await fetch('/api/reload', { method: 'POST' });
        const data = await response.json();

        if (response.ok) {
            await checkStatus();
        } else {
            throw new Error(data.detail || 'Failed to reload');
        }
    } catch (error) {
        showError(`Failed to reload model: ${error.message}`);
    }
}
