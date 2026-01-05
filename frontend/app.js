// ==========================================
// MENTAL HEALTH CRISIS SUPPORT CHATBOT
// JavaScript Frontend Logic
// ==========================================

const API_BASE_URL = 'http://localhost:5000/api';
let mediaRecorder;
let audioChunks = [];
let recordingStartTime = null;
let recordingInterval = null;
let conversationHistory = [];
let crisisChart = null;

// ==================== INITIALIZATION ====================

document.addEventListener('DOMContentLoaded', () => {
    initializeEventListeners();
    loadConversationHistory();
    displayWelcomeMessage();
});

function initializeEventListeners() {
    // Tab Navigation
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.addEventListener('click', switchTab);
    });

    // Chat
    document.getElementById('sendBtn').addEventListener('click', sendMessage);
    document.getElementById('userInput').addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Audio
    document.getElementById('recordBtn').addEventListener('click', startRecording);
    document.getElementById('stopBtn').addEventListener('click', stopRecording);
    document.getElementById('uploadAudioBtn').addEventListener('click', uploadAudio);

    // Image
    document.getElementById('uploadBox').addEventListener('click', () => {
        document.getElementById('imageInput').click();
    });
    document.getElementById('imageInput').addEventListener('change', handleImageSelect);
    document.getElementById('analyzeImageBtn').addEventListener('click', uploadImage);

    // Utilities
    document.getElementById('clearHistoryBtn').addEventListener('click', clearHistory);
    document.querySelector('.close-btn').addEventListener('click', closeModal);

    // Drag and drop for images
    setupDragAndDrop();
}

// ==================== TAB SWITCHING ====================

function switchTab(e) {
    const tabName = e.currentTarget.getAttribute('data-tab');
    
    // Remove active class from all buttons and sections
    document.querySelectorAll('.nav-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(section => section.classList.remove('active'));
    
    // Add active class to clicked button and corresponding section
    e.currentTarget.classList.add('active');
    document.getElementById(`${tabName}-tab`).classList.add('active');

    // Initialize chart if dashboard tab
    if (tabName === 'dashboard') {
        setTimeout(initializeDashboard, 100);
    }
}

// ==================== CHAT FUNCTIONALITY ====================

function displayWelcomeMessage() {
    const chatMessages = document.getElementById('chatMessages');
    const welcomeMsg = document.createElement('div');
    welcomeMsg.className = 'message bot';
    welcomeMsg.innerHTML = `
        <div class="message-content">
            <p>👋 Hello! I'm MindCare, your mental health support assistant. I'm here to listen and help. How are you feeling today?</p>
        </div>
    `;
    chatMessages.appendChild(welcomeMsg);
}

async function sendMessage() {
    const userInput = document.getElementById('userInput');
    const text = userInput.value.trim();

    if (!text) return;

    // Add user message to chat
    addMessageToChat(text, 'user');
    userInput.value = '';

    // Show typing indicator
    showTypingIndicator();

    try {
        // Analyze text
        const response = await fetch(`${API_BASE_URL}/analyze-text`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: text })
        });

        const data = await response.json();

        if (data.success) {
            // Store in conversation history
            conversationHistory.push({
                type: 'text',
                text: text,
                crisis_type: data.crisis_type,
                confidence: data.confidence,
                timestamp: new Date()
            });

            // Display bot response
            setTimeout(() => {
                hideTypingIndicator();
                displayBotResponse(data);
                saveConversationHistory();
            }, 500);
        }
    } catch (error) {
        console.error('Error:', error);
        hideTypingIndicator();
        addMessageToChat('Sorry, I encountered an error. Please try again.', 'bot');
    }
}

function displayBotResponse(data) {
    const message = data.response;
    let botMessage = `<strong>${message.message}</strong>\n\n`;

    if (message.suggestions && message.suggestions.length > 0) {
        botMessage += '💡 <strong>Suggestions:</strong>\n';
        message.suggestions.forEach(suggestion => {
            botMessage += `• ${suggestion}\n`;
        });
    }

    botMessage += `\n🆘 <strong>${message.hotline}</strong>`;

    // Add confidence indicator
    const confidenceBar = createConfidenceBar(data.confidence);
    
    const msgElement = document.createElement('div');
    msgElement.className = 'message bot';
    msgElement.innerHTML = `
        <div class="message-content">
            <p>${botMessage.replace(/\n/g, '<br>')}</p>
            ${confidenceBar}
        </div>
    `;

    document.getElementById('chatMessages').appendChild(msgElement);
    document.getElementById('chatMessages').scrollTop = document.getElementById('chatMessages').scrollHeight;
}

function addMessageToChat(text, sender) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${sender}`;
    msgDiv.innerHTML = `<div class="message-content"><p>${escapeHtml(text)}</p></div>`;
    document.getElementById('chatMessages').appendChild(msgDiv);
    document.getElementById('chatMessages').scrollTop = document.getElementById('chatMessages').scrollHeight;
}

function showTypingIndicator() {
    document.getElementById('typingIndicator').classList.remove('hidden');
}

function hideTypingIndicator() {
    document.getElementById('typingIndicator').classList.add('hidden');
}

function createConfidenceBar(confidence) {
    const percentage = Math.round(confidence * 100);
    const color = confidence > 0.7 ? '#4caf50' : confidence > 0.4 ? '#ff9800' : '#f44336';
    return `<div style="margin-top: 10px; font-size: 12px;">
        <p>Detection Confidence: ${percentage}%</p>
        <div style="background: #e0e0e0; height: 6px; border-radius: 3px; overflow: hidden;">
            <div style="background: ${color}; height: 100%; width: ${percentage}%; border-radius: 3px;"></div>
        </div>
    </div>`;
}

// ==================== AUDIO FUNCTIONALITY ====================

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];

        mediaRecorder.ondataavailable = (event) => {
            audioChunks.push(event.data);
        };

        mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            const audioUrl = URL.createObjectURL(audioBlob);
            document.getElementById('audioPlayback').src = audioUrl;
            document.getElementById('audioPreview').classList.remove('hidden');
        };

        mediaRecorder.start();
        recordingStartTime = Date.now();
        recordingInterval = setInterval(updateRecordingTime, 100);

        document.getElementById('recordingIndicator').classList.remove('hidden');
        document.getElementById('recordBtn').disabled = true;
        document.getElementById('stopBtn').disabled = false;

    } catch (error) {
        console.error('Error accessing microphone:', error);
        showAlert('Unable to access microphone. Please check permissions.', 'error');
    }
}

function stopRecording() {
    mediaRecorder.stop();
    clearInterval(recordingInterval);

    document.getElementById('recordingIndicator').classList.add('hidden');
    document.getElementById('recordBtn').disabled = false;
    document.getElementById('stopBtn').disabled = true;
}

function updateRecordingTime() {
    const elapsed = Math.floor((Date.now() - recordingStartTime) / 1000);
    const minutes = Math.floor(elapsed / 60);
    const seconds = elapsed % 60;
    document.getElementById('recordingTime').textContent = 
        `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
}

async function uploadAudio() {
    try {
        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
        const formData = new FormData();
        formData.append('audio', audioBlob, 'recording.wav');
        formData.append('use_whisper', 'true');

        showAlert('Processing audio... This may take a moment.', 'info');

        const response = await fetch(`${API_BASE_URL}/process-audio`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            displayAudioResults(data);
            conversationHistory.push({
                type: 'audio',
                transcription: data.transcription,
                crisis_type: data.crisis_type,
                timestamp: new Date()
            });
            saveConversationHistory();
        }
    } catch (error) {
        console.error('Error uploading audio:', error);
        showAlert('Error processing audio. Please try again.', 'error');
    }
}

function displayAudioResults(data) {
    document.getElementById('transcriptionText').textContent = data.transcription;
    document.getElementById('crisisInfo').innerHTML = `
        <p><strong>Crisis Type:</strong> ${data.crisis_type}</p>
        <p><strong>Confidence:</strong> ${Math.round(data.confidence * 100)}%</p>
        <p><strong>Audio Features:</strong></p>
        <ul>
            <li>Pitch: ${data.audio_features.pitch.toFixed(2)}</li>
            <li>Energy: ${data.audio_features.energy.toFixed(2)}</li>
            <li>MFCC Mean: ${data.audio_features.mfcc_mean.toFixed(2)}</li>
        </ul>
    `;
    document.getElementById('audioResults').classList.remove('hidden');
}

// ==================== IMAGE FUNCTIONALITY ====================

function handleImageSelect(e) {
    const file = e.target.files[0];
    if (file) {
        displayImagePreview(file);
    }
}

function displayImagePreview(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        document.getElementById('previewImage').src = e.target.result;
        document.getElementById('imagePreview').classList.remove('hidden');
    };
    reader.readAsDataURL(file);
}

async function uploadImage() {
    try {
        const file = document.getElementById('imageInput').files[0];
        if (!file) return;

        const formData = new FormData();
        formData.append('image', file);

        showAlert('Analyzing image... Please wait.', 'info');

        const response = await fetch(`${API_BASE_URL}/analyze-image`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            displayEmotionResults(data);
            conversationHistory.push({
                type: 'image',
                emotions: data.emotions,
                timestamp: new Date()
            });
            saveConversationHistory();
        }
    } catch (error) {
        console.error('Error uploading image:', error);
        showAlert('Error analyzing image. Please try again.', 'error');
    }
}

function displayEmotionResults(data) {
    const emotionBars = document.getElementById('emotionBars');
    emotionBars.innerHTML = '';

    Object.entries(data.emotions).forEach(([emotion, score]) => {
        const percentage = Math.round(score * 100);
        emotionBars.innerHTML += `
            <div class="emotion-bar">
                <div class="emotion-bar-label">
                    <span>${emotion}</span>
                    <span>${percentage}%</span>
                </div>
                <div class="emotion-bar-container">
                    <div class="emotion-bar-fill" style="width: ${percentage}%"></div>
                </div>
            </div>
        `;
    });

    document.getElementById('emotionMessage').innerHTML = `
        <p><strong>Dominant Emotion:</strong> ${data.dominant_emotion}</p>
        <p>${data.response.message}</p>
    `;

    document.getElementById('emotionResults').classList.remove('hidden');
}

function setupDragAndDrop() {
    const uploadBox = document.getElementById('uploadBox');

    uploadBox.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadBox.style.borderColor = '#1b5e20';
        uploadBox.style.backgroundColor = '#e8f5e9';
    });

    uploadBox.addEventListener('dragleave', () => {
        uploadBox.style.borderColor = '#2e7d32';
        uploadBox.style.backgroundColor = '#f0f7f0';
    });

    uploadBox.addEventListener('drop', (e) => {
        e.preventDefault();
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            document.getElementById('imageInput').files = files;
            displayImagePreview(files[0]);
        }
    });
}

// ==================== DASHBOARD ====================

function initializeDashboard() {
    updateDashboardStats();
    updateCrisisChart();
    updateEmotionTimeline();
}

function updateDashboardStats() {
    const stats = calculateConversationStats();
    document.getElementById('totalMessages').textContent = stats.totalMessages;
    document.getElementById('crisisCount').textContent = stats.crisisCount;
    document.getElementById('sessionDuration').textContent = stats.duration + ' min';
    document.getElementById('avgConfidence').textContent = stats.avgConfidence + '%';
}

function calculateConversationStats() {
    let totalMessages = conversationHistory.length;
    let crisisCount = conversationHistory.filter(h => h.crisis_type && h.crisis_type !== 'out_of_topic').length;
    let avgConfidence = 0;

    if (conversationHistory.length > 0) {
        const confidences = conversationHistory
            .filter(h => h.confidence)
            .map(h => h.confidence);
        avgConfidence = confidences.length > 0 
            ? Math.round((confidences.reduce((a, b) => a + b, 0) / confidences.length) * 100)
            : 0;
    }

    const duration = conversationHistory.length > 0 ? 
        Math.floor((Date.now() - new Date(conversationHistory[0].timestamp)) / 60000) : 0;

    return { totalMessages, crisisCount, avgConfidence, duration };
}

function updateCrisisChart() {
    const crisisCounts = {};
    conversationHistory.forEach(h => {
        if (h.crisis_type) {
            crisisCounts[h.crisis_type] = (crisisCounts[h.crisis_type] || 0) + 1;
        }
    });

    const ctx = document.getElementById('crisisChart');
    if (crisisChart) {
        crisisChart.destroy();
    }

    crisisChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: Object.keys(crisisCounts),
            datasets: [{
                data: Object.values(crisisCounts),
                backgroundColor: [
                    '#2e7d32',
                    '#1976d2',
                    '#d32f2f',
                    '#f57c00',
                    '#7b1fa2'
                ]
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'bottom'
                }
            }
        }
    });
}

function updateEmotionTimeline() {
    // Placeholder for emotion timeline visualization
    const timelineContainer = document.getElementById('timelineContainer');
    timelineContainer.innerHTML = '<p>Emotion tracking will update as you interact with the chatbot.</p>';
}

// ==================== UTILITIES ====================

function clearHistory() {
    if (confirm('Are you sure you want to clear all conversation history?')) {
        conversationHistory = [];
        localStorage.removeItem('chatbotHistory');
        document.getElementById('chatMessages').innerHTML = '';
        displayWelcomeMessage();
        updateDashboardStats();
        showAlert('History cleared successfully!', 'success');
    }
}

function saveConversationHistory() {
    localStorage.setItem('chatbotHistory', JSON.stringify(conversationHistory));
}

function loadConversationHistory() {
    const saved = localStorage.getItem('chatbotHistory');
    conversationHistory = saved ? JSON.parse(saved) : [];
}

function showAlert(message, type = 'info') {
    const modal = document.getElementById('modal');
    const modalBody = document.getElementById('modalBody');
    
    const icon = type === 'error' ? '❌' : type === 'success' ? '✅' : 'ℹ️';
    modalBody.innerHTML = `<p>${icon} ${message}</p>`;
    modal.style.display = 'flex';
    
    setTimeout(() => {
        modal.style.display = 'none';
    }, 3000);
}

function closeModal() {
    document.getElementById('modal').style.display = 'none';
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Close modal on outside click
window.addEventListener('click', (e) => {
    const modal = document.getElementById('modal');
    if (e.target === modal) {
        closeModal();
    }
});
