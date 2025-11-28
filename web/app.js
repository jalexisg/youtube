const API_URL = '/api';

// DOM Elements
const tabs = document.querySelectorAll('.tab-btn');
const tabContents = document.querySelectorAll('.tab-content');
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const fileNameDisplay = document.getElementById('file-name');
const uploadBtn = document.getElementById('upload-btn');
const urlInput = document.getElementById('url-input');
const urlBtn = document.getElementById('url-btn');
const statusContainer = document.getElementById('status-container');
const statusTitle = document.getElementById('status-title');
const statusMessage = document.getElementById('status-message');
const resultContainer = document.getElementById('result-container');
const resTabs = document.querySelectorAll('.res-tab');
const resContents = document.querySelectorAll('.res-content');
const copyBtn = document.getElementById('copy-btn');
const downloadBtn = document.getElementById('download-btn');

let currentTaskId = null;
let pollInterval = null;
let currentResult = null;

// Tab Switching
tabs.forEach(tab => {
    tab.addEventListener('click', () => {
        tabs.forEach(t => t.classList.remove('active'));
        tabContents.forEach(c => c.classList.remove('active'));
        tab.classList.add('active');
        document.getElementById(`${tab.dataset.tab}-content`).classList.add('active');
    });
});

// Result Tab Switching
resTabs.forEach(tab => {
    tab.addEventListener('click', () => {
        resTabs.forEach(t => t.classList.remove('active'));
        resContents.forEach(c => c.classList.remove('active'));
        tab.classList.add('active');
        document.getElementById(`res-${tab.dataset.res}`).classList.add('active');
    });
});

// File Handling
dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
        handleFile(e.dataTransfer.files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) {
        handleFile(e.target.files[0]);
    }
});

function handleFile(file) {
    fileNameDisplay.textContent = file.name;
    uploadBtn.disabled = false;
    // Store file in element for access
    fileInput.file = file;
}

// Upload Action
uploadBtn.addEventListener('click', async () => {
    const file = fileInput.file || fileInput.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    setLoading(true, 'Subiendo archivo...');

    try {
        const res = await fetch(`${API_URL}/process-file`, {
            method: 'POST',
            body: formData
        });
        const data = await res.json();
        startPolling(data.task_id);
    } catch (error) {
        showError(error.message);
    }
});

// URL Action
urlInput.addEventListener('input', (e) => {
    urlBtn.disabled = !e.target.value.trim();
});

urlBtn.addEventListener('click', async () => {
    const url = urlInput.value.trim();
    if (!url) return;

    setLoading(true, 'Enviando URL...');

    try {
        const res = await fetch(`${API_URL}/process-url`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url })
        });
        const data = await res.json();
        startPolling(data.task_id);
    } catch (error) {
        showError(error.message);
    }
});

// Polling
function startPolling(taskId) {
    currentTaskId = taskId;
    if (pollInterval) clearInterval(pollInterval);

    pollInterval = setInterval(async () => {
        try {
            const res = await fetch(`${API_URL}/tasks/${taskId}`);
            const data = await res.json();

            updateStatus(data);

            if (data.status === 'completed') {
                clearInterval(pollInterval);
                showResult(data.result);
            } else if (data.status === 'failed') {
                clearInterval(pollInterval);
                showError(data.message);
            }
        } catch (error) {
            clearInterval(pollInterval);
            showError('Error de conexión');
        }
    }, 2000);
}

// UI Helpers
function setLoading(isLoading, message = 'Procesando...') {
    if (isLoading) {
        statusContainer.classList.remove('hidden');
        resultContainer.classList.add('hidden');
        statusTitle.textContent = message;
        statusMessage.textContent = 'Por favor espera...';
        uploadBtn.disabled = true;
        urlBtn.disabled = true;
    } else {
        statusContainer.classList.add('hidden');
        uploadBtn.disabled = false;
        urlBtn.disabled = false;
    }
}

function updateStatus(data) {
    statusTitle.textContent = capitalize(data.status);
    statusMessage.textContent = data.message || '';
}

function showError(msg) {
    setLoading(false);
    statusContainer.classList.remove('hidden');
    statusTitle.textContent = 'Error';
    statusMessage.textContent = msg;
    statusContainer.querySelector('.loader').style.borderColor = 'var(--error)';
}

function showResult(result) {
    setLoading(false);
    resultContainer.classList.remove('hidden');
    currentResult = result;

    // Populate contents
    document.getElementById('res-summary').innerHTML = formatText(result.analysis.extractive_summary);
    document.getElementById('res-transcription').innerHTML = formatText(result.transcription.full_text);
    document.getElementById('res-keywords').innerHTML = result.analysis.keywords.map(k => `<span class="keyword-tag">${k}</span>`).join(', ');

    // Topic summary if available
    if (result.analysis.topic_summary) {
        let topicHtml = '<h3>Resumen por Temas</h3>';
        for (const [topic, text] of Object.entries(result.analysis.topic_summary)) {
            topicHtml += `<h4>${topic}</h4><p>${text}</p>`;
        }
        document.getElementById('res-summary').innerHTML += `<hr>${topicHtml}`;
    }

    // Handle Downloads
    const downloadSection = document.getElementById('download-section');
    const dlMediaBtn = document.getElementById('dl-media-btn');
    const dlAudioBtn = document.getElementById('dl-audio-btn');

    if (result.media_url || result.audio_url) {
        downloadSection.classList.remove('hidden');

        const title = result.title || 'download';
        const safeTitle = title.replace(/[^a-z0-9]/gi, '_').toLowerCase();

        if (result.media_url) {
            dlMediaBtn.href = result.media_url;
            dlMediaBtn.download = `${safeTitle}_original${result.media_url.substring(result.media_url.lastIndexOf('.'))}`;
            dlMediaBtn.classList.remove('hidden');
        } else {
            dlMediaBtn.classList.add('hidden');
        }

        if (result.audio_url) {
            dlAudioBtn.href = result.audio_url;
            dlAudioBtn.download = `${safeTitle}_audio.wav`;
            dlAudioBtn.classList.remove('hidden');
        } else {
            dlAudioBtn.classList.add('hidden');
        }
    } else {
        downloadSection.classList.add('hidden');
    }
}

function formatText(text) {
    return text.replace(/\n/g, '<br>');
}

function capitalize(s) {
    return s.charAt(0).toUpperCase() + s.slice(1);
}

// Copy & Download
copyBtn.addEventListener('click', () => {
    if (!currentResult) return;
    const activeTab = document.querySelector('.res-tab.active').dataset.res;
    let text = '';
    if (activeTab === 'summary') text = currentResult.analysis.extractive_summary;
    else if (activeTab === 'transcription') text = currentResult.transcription.full_text;
    else if (activeTab === 'keywords') text = currentResult.analysis.keywords.join(', ');

    navigator.clipboard.writeText(text);
    // Could add a toast notification here
});

downloadBtn.addEventListener('click', () => {
    if (!currentResult) return;
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(currentResult, null, 2));
    const downloadAnchorNode = document.createElement('a');
    downloadAnchorNode.setAttribute("href", dataStr);
    downloadAnchorNode.setAttribute("download", "analysis_result.json");
    document.body.appendChild(downloadAnchorNode);
    downloadAnchorNode.click();
    downloadAnchorNode.remove();
});
