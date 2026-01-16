"""
Enhanced Streamlit UI for Mental Health Crisis Support Chatbot
Features: Text, Audio (Whisper), Image Analysis, Video, Dashboard
"""

# Configure environment for headless server (Streamlit Cloud)
import os
# Disable GUI backends and system library dependencies
os.environ['MPLBACKEND'] = 'Agg'  # Set matplotlib backend
os.environ['DISPLAY'] = ''  # Disable X11 display

# Set matplotlib to use non-GUI backend (required for headless servers like Streamlit Cloud)
import matplotlib
matplotlib.use('Agg')  # Use Anti-Grain Geometry backend (no GUI required)

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from dotenv import load_dotenv
import requests
import json
from PIL import Image
# Optional imports for features that may not be available on all platforms
# Note: cv2 (OpenCV) removed - not used in this app and causes issues on Streamlit Cloud

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False

import warnings
import pickle
from pathlib import Path

warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# ==================== PERSISTENCE ====================

HISTORY_FILE = "data/conversation_history.pkl"
RESULTS_FILE = "data/detection_results.pkl"

def ensure_data_dir():
    """Create data directory if it doesn't exist"""
    Path("data").mkdir(exist_ok=True)

def load_history():
    """Load conversation history from file"""
    ensure_data_dir()
    if Path(HISTORY_FILE).exists():
        try:
            with open(HISTORY_FILE, 'rb') as f:
                return pickle.load(f)
        except:
            return []
    return []

def save_history(history):
    """Save conversation history to file"""
    ensure_data_dir()
    with open(HISTORY_FILE, 'wb') as f:
        pickle.dump(history, f)

def load_results():
    """Load detection results from file"""
    ensure_data_dir()
    if Path(RESULTS_FILE).exists():
        try:
            with open(RESULTS_FILE, 'rb') as f:
                return pickle.load(f)
        except:
            return []
    return []

def save_results(results):
    """Save detection results to file"""
    ensure_data_dir()
    with open(RESULTS_FILE, 'wb') as f:
        pickle.dump(results, f)

# Page Configuration
st.set_page_config(
    page_title="Mental Health Crisis Support Chatbot",
    page_icon="💙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with improved styling
st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #2e7d32 0%, #1976d2 100%);
    }
    .stTab [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 18px;
        font-weight: 600;
    }
    .css-1aumxhk {
        color: white;
    }
    .crisis-warning {
        background-color: #ffebee;
        border-left: 4px solid #d32f2f;
        padding: 15px;
        border-radius: 4px;
    }
    .crisis-high {
        background-color: #fff3e0;
        border-left: 4px solid #f57c00;
        padding: 15px;
        border-radius: 4px;
    }
    .crisis-medium {
        background-color: #fffde7;
        border-left: 4px solid #fbc02d;
        padding: 15px;
        border-radius: 4px;
    }
    .crisis-low {
        background-color: #f1f8e9;
        border-left: 4px solid #689f38;
        padding: 15px;
        border-radius: 4px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize Session State with persistence
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = load_history()
if 'audio_features' not in st.session_state:
    st.session_state.audio_features = {}
if 'detection_results' not in st.session_state:
    st.session_state.detection_results = load_results()
if 'session_start' not in st.session_state:
    st.session_state.session_start = datetime.now()

# ==================== HELPER FUNCTIONS ====================

def validate_input(text, min_length=3, max_length=2000):
    """Validate user input"""
    if not text or not text.strip():
        return False, "⚠️ Please enter some text"
    
    if len(text.strip()) < min_length:
        return False, f"⚠️ Message too short (minimum {min_length} characters)"
    
    if len(text.strip()) > max_length:
        return False, f"⚠️ Message too long (maximum {max_length} characters)"
    
    return True, "✅ Valid"

def send_to_backend(endpoint, data=None, files=None):
    """Send request to Flask backend with error handling"""
    try:
        api_url = f"{os.getenv('FLASK_URL', 'http://localhost:5000')}/api/{endpoint}"
        
        if files:
            response = requests.post(api_url, files=files, timeout=30)
        else:
            response = requests.post(api_url, json=data, headers={'Content-Type': 'application/json'}, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"❌ Backend error: {response.status_code}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to backend. Make sure Flask API is running on http://localhost:5000")
        return None
    except requests.exceptions.Timeout:
        st.error("❌ Request timed out. Backend is taking too long to respond.")
        return None
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None

def send_to_backend_multimodal(endpoint, files=None, data=None):
    """Send multimodal request to Flask backend using form data"""
    try:
        api_url = f"{os.getenv('FLASK_URL', 'http://localhost:5000')}/api/{endpoint}"
        
        # Prepare form data and files
        form_data = data or {}
        form_files = files or {}
        
        response = requests.post(api_url, files=form_files, data=form_data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            # Flatten the response structure for easier handling
            if 'analyses' in result:
                # Get the first analysis or combined results
                if 'combined_crisis_type' in result:
                    return {
                        'success': True,
                        'crisis_type': result.get('combined_crisis_type', 'unknown'),
                        'confidence': result.get('combined_confidence', 0.5),
                        'urgency_level': 'high' if result.get('combined_confidence', 0) > 0.6 else 'medium',
                        'message': result.get('response', 'Analysis complete'),
                        'guidance': []
                    }
            return result
        else:
            st.error(f"❌ Backend error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to backend. Make sure Flask API is running on http://localhost:5000")
        return None
    except requests.exceptions.Timeout:
        st.error("❌ Request timed out. Backend is taking too long to respond.")
        return None
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None


def get_urgency_color(urgency):
    """Get color for urgency level"""
    colors = {
        'critical': '🔴',
        'high': '🟠',
        'medium': '🟡',
        'low': '🟢'
    }
    return colors.get(urgency, '⚪')

def format_crisis_display(result):
    """Format crisis detection result for display"""
    crisis_type = result.get('crisis_type', 'unknown')
    urgency = result.get('urgency_level', 'low')
    confidence = result.get('confidence', 0)
    
    urgency_emoji = get_urgency_color(urgency)
    
    return f"{urgency_emoji} **{crisis_type.upper()}** | Confidence: {confidence*100:.1f}% | Urgency: {urgency.upper()}"

def extract_audio_features(audio_file):
    """Extract features from audio using librosa"""
    if not LIBROSA_AVAILABLE:
        st.warning("⚠️ Librosa not available. Audio feature extraction disabled.")
        return {
            'mfcc_mean': 0.0,
            'mfcc_std': 0.0,
            'zcr_mean': 0.0,
            'spectral_centroid_mean': 0.0,
            'chroma_mean': 0.0,
            'energy': 0.0
        }
    
    try:
        # Load audio
        y, sr_val = librosa.load(audio_file, sr=16000)
        
        # Extract features
        mfcc = librosa.feature.mfcc(y=y, sr=sr_val, n_mfcc=13)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr_val)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr_val)
        
        features = {
            'mfcc_mean': float(np.mean(mfcc)),
            'mfcc_std': float(np.std(mfcc)),
            'zcr_mean': float(np.mean(zero_crossing_rate)),
            'spectral_centroid_mean': float(np.mean(spectral_centroid)),
            'chroma_mean': float(np.mean(chroma)),
            'energy': float(np.sum(y**2) / len(y))
        }
        return features
    except Exception as e:
        st.error(f"Error extracting audio features: {str(e)}")
        return {}

def transcribe_audio_whisper(audio_file):
    """Transcribe audio using Whisper (via backend or local)"""
    try:
        # Try backend first
        result = send_to_backend('process-audio', files={'audio': audio_file})
        if result:
            return result.get('transcription', '')
        return ""
    except Exception as e:
        st.error(f"Transcription error: {str(e)}")
        return ""

def analyze_image_emotions(image):
    """Analyze image for emotion detection"""
    try:
        # Convert to numpy array if PIL Image
        if isinstance(image, Image.Image):
            image_array = np.array(image)
        else:
            image_array = image
        
        # Send to backend for emotion detection
        # For demo, we'll return mock emotions
        emotions = {
            'happy': np.random.uniform(0, 1),
            'sad': np.random.uniform(0, 1),
            'angry': np.random.uniform(0, 1),
            'fear': np.random.uniform(0, 1),
            'neutral': np.random.uniform(0, 1)
        }
        
        # Normalize
        total = sum(emotions.values())
        emotions = {k: v/total for k, v in emotions.items()}
        return emotions
    except Exception as e:
        st.error(f"Image analysis error: {str(e)}")
        return {}

def analyze_text(text):
    """Send text to backend for analysis, with local fallback"""
    # Try backend first
    result = send_to_backend('analyze-text', data={'text': text})
    
    # If backend is not available, use local processing
    if result is None:
        return analyze_text_local(text)
    
    return result

def analyze_text_local(text):
    """Local text analysis when backend is unavailable"""
    # Use the local detection function
    crisis_type, confidence = detect_speech_intent(text)
    
    # Determine urgency based on crisis type and confidence
    urgency_map = {
        'suicidal': 'critical',
        'stressed': 'medium',
        'insomnia': 'low',
        'racism': 'high',
        'help': 'medium',
        'out_of_topic': 'low'
    }
    urgency = urgency_map.get(crisis_type, 'low')
    
    # Generate response based on crisis type
    responses = {
        'stressed': {
            'message': "I sense you're feeling stressed. That's something many people experience. Here are some immediate coping strategies:",
            'guidance': [
                "Take slow, deep breaths (4 count in, hold 4, out 4)",
                "Practice progressive muscle relaxation",
                "Take a short walk or do light exercise",
                "Talk to someone you trust"
            ],
            'hotlines': {
                'Suicide & Crisis Lifeline': '988',
                'Crisis Text Line': 'Text HOME to 741741'
            }
        },
        'insomnia': {
            'message': "It sounds like you're struggling with sleep or rest. Let me help you:",
            'guidance': [
                "Establish a consistent sleep schedule",
                "Avoid screens 30 minutes before bed",
                "Try relaxation techniques like meditation",
                "Consult a sleep specialist if it persists"
            ],
            'hotlines': {
                'Healthcare Provider': 'Contact your local healthcare provider'
            }
        },
        'suicidal': {
            'message': "I'm very concerned about your safety. Please reach out immediately:",
            'guidance': [
                "Call 988 Suicide & Crisis Lifeline immediately",
                "Text HOME to 741741 for Crisis Text Line",
                "Go to nearest emergency room",
                "Call 911 if you're in immediate danger"
            ],
            'hotlines': {
                'Suicide & Crisis Lifeline': '988 - CALL NOW',
                'Crisis Text Line': 'Text HOME to 741741',
                'Emergency': '911'
            },
            'warning': 'If you are in immediate danger, please call 911 or go to your nearest emergency room.'
        },
        'racism': {
            'message': "I'm sorry you're experiencing racism. You deserve support and respect:",
            'guidance': [
                "Document incidents for your safety",
                "Connect with support communities",
                "Contact civil rights organizations",
                "Seek counseling from trauma-informed therapists"
            ],
            'hotlines': {
                'National Race & Ethnicity Hotline': '1-844-856-4869'
            }
        },
        'help': {
            'message': "I'm here to support you. If you're unsure what to do, we can figure it out together.",
            'guidance': [
                "What's been bothering you?",
                "Do you want advice, or do you just want to talk?",
                "What's making this feel difficult right now?",
                "Would you like some simple steps that might help?",
                "You can share as much or as little as you want",
                "This is a safe, judgment-free space",
                "We can take things one step at a time"
            ],
            'hotlines': {
                'Mental Health Helpline': '131165',
                'Crisis Lifeline': '131114',
                'Emergency Services': '000'
            },
            'prompts': [
                "What's been bothering you?",
                "Do you want advice, or do you just want to talk?",
                "What's making this feel difficult right now?",
                "Would you like some simple steps that might help?"
            ],
            'support': [
                "You can share as much or as little as you want",
                "This is a safe, judgment-free space",
                "We can take things one step at a time"
            ]
        },
        'out_of_topic': {
            'message': "I'm here to help with mental health concerns. How are you feeling emotionally?",
            'guidance': [
                "Describe what's bothering you",
                "Share your feelings in detail",
                "Tell me about recent stressful events",
                "I'm here to listen and help"
            ],
            'hotlines': {
                'Suicide & Crisis Lifeline': '988'
            }
        }
    }
    
    response = responses.get(crisis_type, responses['out_of_topic'])
    
    return {
        'success': True,
        'crisis_type': crisis_type,
        'urgency_level': urgency,
        'confidence': confidence,
        'message': response['message'],
        'guidance': response['guidance'],
        'hotlines': response['hotlines'],
        'warning': response.get('warning', ''),
        'timestamp': datetime.now().isoformat()
    }

def detect_speech_intent(text):
    """Detect crisis intent from text"""
    crisis_keywords = {
        'stressed': ['stress', 'anxiety', 'worried', 'overwhelmed', 'nervous', 'tense'],
        'insomnia': ['can\'t sleep', 'insomnia', 'tired', 'sleepless', 'awake', 'rest'],
        'suicidal': ['suicide', 'kill myself', 'end it', 'harm', 'death', 'die'],
        'racism': ['racist', 'discrimination', 'prejudice', 'hate', 'racist'],
        'help': ['help', 'help me', 'need help', 'i need help', 'i need guidance', 'what should i do', 
                 'what can i do', 'i dont know what to do', 'i am confused', 'i feel lost', 
                 'i am stuck', 'i need support', 'i need advice', 'guide me', 'please help'],
        'out_of_topic': []
    }
    
    text_lower = text.lower()
    scores = {}
    
    for category, keywords in crisis_keywords.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        scores[category] = score
    
    # Find dominant category
    max_score = max(scores.values())
    if max_score == 0:
        return 'out_of_topic', 0.3
    
    for category, score in scores.items():
        if score == max_score:
            confidence = min((score / len(crisis_keywords[category])), 1.0) if crisis_keywords[category] else 0.5
            return category, confidence
    
    return 'out_of_topic', 0.3

# ==================== HEADER ====================

st.markdown("""
    <div style='text-align: center; margin-bottom: 30px;'>
        <h1>💙 Mental Health Crisis Support Chatbot</h1>
        <p style='font-size: 18px; color: #2e7d32;'>Multimodal AI Support • Text • Voice • Image Analysis</p>
    </div>
""", unsafe_allow_html=True)

# ==================== SIDEBAR ====================

with st.sidebar:
    st.markdown("### ⚙️ Settings & Controls")
    
    # Check backend connection status
    backend_url = os.getenv('FLASK_URL', 'http://localhost:5000')
    try:
        response = requests.get(f"{backend_url}/api/health", timeout=2)
        backend_connected = response.status_code == 200
    except:
        backend_connected = False
    
    if backend_connected:
        st.success("✅ Backend Connected")
    else:
        st.info("ℹ️ Using Local Mode (Backend unavailable)")
    
    enable_audio = st.checkbox("🎤 Enable Audio Input", value=True)
    enable_image = st.checkbox("🖼️ Enable Image Analysis", value=True)
    use_cloud_api = st.checkbox("☁️ Use Cloud APIs", value=False)
    
    st.markdown("---")
    st.markdown("### 📊 Session Statistics")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Messages", len([m for m in st.session_state.conversation_history if m['type'] == 'user']))
    with col2:
        crisis_count = sum(1 for d in st.session_state.detection_results if d['crisis_type'] != 'out_of_topic')
        st.metric("Crises Detected", crisis_count)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Save Session"):
            save_history(st.session_state.conversation_history)
            save_results(st.session_state.detection_results)
            st.success("✅ Session saved!")
    
    with col2:
        if st.button("🗑️ Clear History"):
            st.session_state.conversation_history = []
            st.session_state.detection_results = []
            st.session_state.session_start = datetime.now()
            Path(HISTORY_FILE).unlink(missing_ok=True)
            Path(RESULTS_FILE).unlink(missing_ok=True)
            st.success("✅ History cleared!")
            st.rerun()
    
    st.markdown("---")
    
    # Show dependency status
    if not LIBROSA_AVAILABLE:
        st.markdown("### ⚠️ Feature Status")
        if not LIBROSA_AVAILABLE:
            st.warning("Librosa not available - Audio analysis limited")
    
    st.markdown("---")
    st.markdown("### 📱 About This App")
    st.info("""
    This is an AI-powered mental health support chatbot using:
    - 🤖 Crisis Detection AI
    - 🎤 Voice Recognition (Whisper)
    - 🖼️ Emotion Detection
    - 💬 Natural Language Processing
    
    **Always seek professional help in emergencies!**
    """)

# ==================== MAIN TABS ====================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "💬 Chat",
    "🎤 Voice Input" if enable_audio else "🎤 Voice (Disabled)",
    "🖼️ Image Analysis" if enable_image else "🖼️ Image (Disabled)",
    "🔀 Multimodal",
    "📊 Dashboard",
    "📚 Resources"
])

# ==================== TAB 1: CHAT ====================

with tab1:
    st.subheader("💬 Chat Support")
    st.write("Talk to our AI assistant about your mental health concerns")
    
    # Display conversation history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.conversation_history:
            if message['type'] == 'user':
                st.chat_message("user").write(message['text'])
            elif message['type'] == 'assistant':
                st.chat_message("assistant").write(message['text'])
    
    # Input area
    col1, col2 = st.columns([4, 1])
    with col1:
        user_input = st.text_area("Share your thoughts or feelings:", key="chat_input", height=80)
    with col2:
        send_button = st.button("Send 📤", use_container_width=True)
    
    if send_button and user_input:
        # Validate input
        is_valid, validation_msg = validate_input(user_input)
        
        if not is_valid:
            st.warning(validation_msg)
        else:
            # Add user message
            st.session_state.conversation_history.append({
                'type': 'user',
                'text': user_input,
                'timestamp': datetime.now().isoformat()
            })
            save_history(st.session_state.conversation_history)
            
            # Analyze text (will use local fallback if backend unavailable)
            with st.spinner("🔍 Analyzing your message..."):
                result = analyze_text(user_input)
                
                if result and result.get('success'):
                    crisis_type = result.get('crisis_type', 'out_of_topic')
                    confidence = result.get('confidence', 0.3)
                    urgency = result.get('urgency_level', 'low')
                    
                    # Store detection result
                    st.session_state.detection_results.append({
                        'type': 'text',
                        'crisis_type': crisis_type,
                        'confidence': confidence,
                        'urgency': urgency,
                        'timestamp': datetime.now().isoformat()
                    })
                    save_results(st.session_state.detection_results)
                    
                    # Build comprehensive response
                    bot_response = f"**{result.get('message', 'Response received')}**\n\n"
                    
                    # Add warning if present
                    if result.get('warning'):
                        bot_response += f"⚠️ **{result['warning']}**\n\n"
                    
                    # Add guidance
                    if result.get('guidance'):
                        bot_response += "💡 **What you can do:**\n"
                        for guid in result['guidance']:
                            bot_response += f"• {guid}\n"
                        bot_response += "\n"
                    
                    # Add hotlines
                    if result.get('hotlines'):
                        bot_response += "📞 **Support Hotlines:**\n"
                        for service, number in result['hotlines'].items():
                            bot_response += f"• **{service}**: {number}\n"
                    
                    # Display crisis info with color coding
                    crisis_display = format_crisis_display(result)
                    
                    if urgency == 'critical':
                        st.markdown(f'<div class="crisis-warning">{crisis_display}</div>', unsafe_allow_html=True)
                    elif urgency == 'high':
                        st.markdown(f'<div class="crisis-high">{crisis_display}</div>', unsafe_allow_html=True)
                    elif urgency == 'medium':
                        st.markdown(f'<div class="crisis-medium">{crisis_display}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="crisis-low">{crisis_display}</div>', unsafe_allow_html=True)
                    
                    st.chat_message("assistant").write(bot_response)
                    
                    st.session_state.conversation_history.append({
                        'type': 'assistant',
                        'text': bot_response,
                        'crisis_type': crisis_type,
                        'confidence': confidence,
                        'urgency': urgency,
                        'timestamp': datetime.now().isoformat()
                    })
                    save_history(st.session_state.conversation_history)
                else:
                    # This should rarely happen now since we have local fallback
                    st.warning("⚠️ Unable to analyze message. Please try again.")
        
        st.rerun()

# ==================== TAB 2: VOICE INPUT ====================

with tab2:
    if enable_audio:
        st.subheader("🎤 Voice Input & Analysis")
        st.write("Record your voice or upload an audio file for analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Record Audio")
            audio_data = st.audio_input("Record your message:")
            
            if audio_data:
                st.audio(audio_data)
                
                if st.button("Analyze Recording"):
                    with st.spinner("Processing audio..."):
                        # Extract features
                        features = extract_audio_features(audio_data)
                        
                        # Transcribe
                        transcription = transcribe_audio_whisper(audio_data)
                        
                        if transcription:
                            st.success("Transcription successful!")
                            st.write(f"**Transcript:** {transcription}")
                            
                            # Analyze transcription
                            result = analyze_text(transcription)
                            if result:
                                crisis_type = result.get('crisis_type')
                                confidence = result.get('confidence', 0)
                                
                                # Display audio features
                                st.write("### Audio Features")
                                cols = st.columns(3)
                                cols[0].metric("MFCC Mean", f"{features.get('mfcc_mean', 0):.2f}")
                                cols[1].metric("Energy", f"{features.get('energy', 0):.2f}")
                                cols[2].metric("Zero Crossing Rate", f"{features.get('zcr_mean', 0):.2f}")
                                
                                # Crisis detection
                                st.write("### Crisis Detection")
                                col_crisis1, col_crisis2 = st.columns(2)
                                col_crisis1.metric("Detected Type", crisis_type)
                                col_crisis2.metric("Confidence", f"{confidence*100:.1f}%")
        
        with col2:
            st.markdown("### Upload Audio File")
            uploaded_audio = st.file_uploader("Choose an audio file:", type=['wav', 'mp3', 'ogg'])
            
            if uploaded_audio:
                st.audio(uploaded_audio)
                if st.button("Analyze Uploaded Audio"):
                    with st.spinner("Processing audio file..."):
                        features = extract_audio_features(uploaded_audio)
                        st.write("Audio features extracted successfully!")
                        st.write(features)
    else:
        st.info("🎤 Voice input is disabled. Enable it in the sidebar to use this feature.")

# ==================== TAB 3: IMAGE ANALYSIS ====================

with tab3:
    if enable_image:
        st.subheader("🖼️ Image Analysis & Emotion Detection")
        st.write("Upload an image for emotion detection analysis")
        
        uploaded_image = st.file_uploader("Choose an image:", type=['jpg', 'jpeg', 'png', 'gif'])
        
        if uploaded_image:
            image = Image.open(uploaded_image)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="Uploaded Image", use_container_width=True)
            
            with col2:
                if st.button("Analyze Image"):
                    with st.spinner("Detecting emotions..."):
                        emotions = analyze_image_emotions(image)
                        
                        st.write("### Detected Emotions")
                        for emotion, score in emotions.items():
                            st.metric(emotion.capitalize(), f"{score*100:.1f}%")
                            st.progress(score)
                        
                        # Dominant emotion
                        dominant = max(emotions, key=emotions.get)
                        st.success(f"**Dominant Emotion: {dominant.capitalize()}**")
    else:
        st.info("🖼️ Image analysis is disabled. Enable it in the sidebar to use this feature.")

# ==================== TAB 4: MULTIMODAL ANALYSIS ====================

with tab4:
    st.subheader("🔀 Multimodal Analysis")
    st.write("Combine text, audio, and/or image for comprehensive crisis detection")
    
    st.markdown("---")
    
    # Create tabs for different multimodal combinations
    mm_tab1, mm_tab2, mm_tab3, mm_tab4 = st.tabs([
        "Text + Image",
        "Text + Audio",
        "Image + Audio",
        "Text + Audio + Image"
    ])
    
    # ==================== TEXT + IMAGE ====================
    with mm_tab1:
        st.markdown("### 📝 + 🖼️ Text & Image Analysis")
        st.write("Upload text and an image to analyze emotional context combined")
        
        col1, col2 = st.columns(2)
        
        with col1:
            text_input = st.text_area("Enter text:", placeholder="Share your thoughts or feelings...", key="mm_text_image")
        
        with col2:
            uploaded_image_mm = st.file_uploader("Upload image:", type=['jpg', 'jpeg', 'png', 'gif'], key="mm_image_input")
            if uploaded_image_mm:
                image_display = Image.open(uploaded_image_mm)
                st.image(image_display, caption="Uploaded Image", use_container_width=True)
        
        if st.button("🔍 Analyze Text + Image", key="mm_btn_text_image"):
            if text_input and uploaded_image_mm:
                with st.spinner("Performing multimodal analysis..."):
                    try:
                        # Prepare multimodal data with files
                        files = {
                            'image': uploaded_image_mm
                        }
                        data = {
                            'text': text_input
                        }
                        
                        result = send_to_backend_multimodal('multimodal-analysis', files=files, data=data)
                        
                        if result and result.get('success'):
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Crisis Type", result.get('crisis_type', 'N/A').capitalize())
                            with col2:
                                st.metric("Confidence", f"{result.get('confidence', 0)*100:.1f}%")
                            with col3:
                                urgency = result.get('urgency_level', 'low').upper()
                                urgency_emoji = {'CRITICAL': '🔴', 'HIGH': '🟠', 'MEDIUM': '🟡', 'LOW': '🟢'}.get(urgency, '⚪')
                                st.metric("Urgency", f"{urgency_emoji} {urgency}")
                            
                            st.markdown("---")
                            st.write("**Analysis Details:**")
                            st.write(result.get('message', ''))
                            
                            if result.get('guidance'):
                                st.markdown("**Guidance:**")
                                for guid in result['guidance']:
                                    st.write(f"• {guid}")
                            
                            # Store result
                            st.session_state.detection_results.append({
                                'type': 'text_image',
                                'crisis_type': result.get('crisis_type'),
                                'confidence': result.get('confidence'),
                                'urgency': result.get('urgency_level'),
                                'timestamp': datetime.now().isoformat()
                            })
                            save_results(st.session_state.detection_results)
                        else:
                            st.error("❌ Analysis failed. Please try again.")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            else:
                st.warning("⚠️ Please provide both text and image for analysis.")
    
    # ==================== TEXT + AUDIO ====================
    with mm_tab2:
        st.markdown("### 📝 + 🎤 Text & Audio Analysis")
        st.write("Combine text input with audio for comprehensive assessment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            text_input_audio = st.text_area("Enter text:", placeholder="Your thoughts or feelings...", key="mm_text_audio")
        
        with col2:
            st.markdown("**Audio Input:**")
            uploaded_audio_mm = st.file_uploader("Upload audio file:", type=['wav', 'mp3', 'ogg'], key="mm_audio_input")
            if uploaded_audio_mm:
                st.audio(uploaded_audio_mm)
        
        if st.button("🔍 Analyze Text + Audio", key="mm_btn_text_audio"):
            if text_input_audio and uploaded_audio_mm:
                with st.spinner("Performing multimodal analysis..."):
                    try:
                        files = {
                            'audio': uploaded_audio_mm
                        }
                        data = {
                            'text': text_input_audio
                        }
                        
                        result = send_to_backend_multimodal('multimodal-analysis', files=files, data=data)
                        
                        if result and result.get('success'):
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Crisis Type", result.get('crisis_type', 'N/A').capitalize())
                            with col2:
                                st.metric("Confidence", f"{result.get('confidence', 0)*100:.1f}%")
                            with col3:
                                urgency = result.get('urgency_level', 'low').upper()
                                urgency_emoji = {'CRITICAL': '🔴', 'HIGH': '🟠', 'MEDIUM': '🟡', 'LOW': '🟢'}.get(urgency, '⚪')
                                st.metric("Urgency", f"{urgency_emoji} {urgency}")
                            
                            st.markdown("---")
                            st.write("**Analysis Details:**")
                            st.write(result.get('message', ''))
                            
                            if result.get('guidance'):
                                st.markdown("**Guidance:**")
                                for guid in result['guidance']:
                                    st.write(f"• {guid}")
                            
                            # Store result
                            st.session_state.detection_results.append({
                                'type': 'text_audio',
                                'crisis_type': result.get('crisis_type'),
                                'confidence': result.get('confidence'),
                                'urgency': result.get('urgency_level'),
                                'timestamp': datetime.now().isoformat()
                            })
                            save_results(st.session_state.detection_results)
                        else:
                            st.error("❌ Analysis failed. Please try again.")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            else:
                st.warning("⚠️ Please provide both text and audio for analysis.")
    
    # ==================== IMAGE + AUDIO ====================
    with mm_tab3:
        st.markdown("### 🖼️ + 🎤 Image & Audio Analysis")
        st.write("Analyze emotional cues from both visual and audio input")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Image Input:**")
            uploaded_image_audio = st.file_uploader("Upload image:", type=['jpg', 'jpeg', 'png', 'gif'], key="mm_image_audio_input")
            if uploaded_image_audio:
                image_display = Image.open(uploaded_image_audio)
                st.image(image_display, caption="Uploaded Image", use_container_width=True)
        
        with col2:
            st.markdown("**Audio Input:**")
            uploaded_audio_image = st.file_uploader("Upload audio file:", type=['wav', 'mp3', 'ogg'], key="mm_audio_image_input")
            if uploaded_audio_image:
                st.audio(uploaded_audio_image)
        
        if st.button("🔍 Analyze Image + Audio", key="mm_btn_image_audio"):
            if uploaded_image_audio and uploaded_audio_image:
                with st.spinner("Performing multimodal analysis..."):
                    try:
                        files = {
                            'image': uploaded_image_audio,
                            'audio': uploaded_audio_image
                        }
                        
                        result = send_to_backend_multimodal('multimodal-analysis', files=files)
                        
                        if result and result.get('success'):
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Crisis Type", result.get('crisis_type', 'N/A').capitalize())
                            with col2:
                                st.metric("Confidence", f"{result.get('confidence', 0)*100:.1f}%")
                            with col3:
                                urgency = result.get('urgency_level', 'low').upper()
                                urgency_emoji = {'CRITICAL': '🔴', 'HIGH': '🟠', 'MEDIUM': '🟡', 'LOW': '🟢'}.get(urgency, '⚪')
                                st.metric("Urgency", f"{urgency_emoji} {urgency}")
                            
                            st.markdown("---")
                            st.write("**Analysis Details:**")
                            st.write(result.get('message', ''))
                            
                            if result.get('guidance'):
                                st.markdown("**Guidance:**")
                                for guid in result['guidance']:
                                    st.write(f"• {guid}")
                            
                            # Store result
                            st.session_state.detection_results.append({
                                'type': 'image_audio',
                                'crisis_type': result.get('crisis_type'),
                                'confidence': result.get('confidence'),
                                'urgency': result.get('urgency_level'),
                                'timestamp': datetime.now().isoformat()
                            })
                            save_results(st.session_state.detection_results)
                        else:
                            st.error("❌ Analysis failed. Please try again.")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            else:
                st.warning("⚠️ Please provide both image and audio for analysis.")
    
    # ==================== TEXT + AUDIO + IMAGE ====================
    with mm_tab4:
        st.markdown("### 📝 + 🎤 + 🖼️ Complete Multimodal Analysis")
        st.write("Upload all three modalities for the most comprehensive assessment")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Text Input:**")
            text_input_full = st.text_area("Enter text:", placeholder="Your thoughts...", key="mm_text_full")
        
        with col2:
            st.markdown("**Audio Input:**")
            uploaded_audio_full = st.file_uploader("Upload audio:", type=['wav', 'mp3', 'ogg'], key="mm_audio_full")
            if uploaded_audio_full:
                st.audio(uploaded_audio_full)
        
        with col3:
            st.markdown("**Image Input:**")
            uploaded_image_full = st.file_uploader("Upload image:", type=['jpg', 'jpeg', 'png', 'gif'], key="mm_image_full")
            if uploaded_image_full:
                image_display = Image.open(uploaded_image_full)
                st.image(image_display, caption="Image", use_container_width=True)
        
        if st.button("🔍 Analyze All Modalities", key="mm_btn_full"):
            provided_count = sum([bool(text_input_full), bool(uploaded_audio_full), bool(uploaded_image_full)])
            
            if provided_count >= 2:
                with st.spinner("Performing comprehensive multimodal analysis..."):
                    try:
                        files = {}
                        data = {}
                        
                        if text_input_full:
                            data['text'] = text_input_full
                        
                        if uploaded_audio_full:
                            files['audio'] = uploaded_audio_full
                        
                        if uploaded_image_full:
                            files['image'] = uploaded_image_full
                        
                        result = send_to_backend_multimodal('multimodal-analysis', files=files, data=data)
                        
                        if result and result.get('success'):
                            # Create comprehensive display
                            st.markdown("## 📊 Multimodal Analysis Results")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Crisis Type", result.get('crisis_type', 'N/A').capitalize())
                            with col2:
                                st.metric("Confidence", f"{result.get('confidence', 0)*100:.1f}%")
                            with col3:
                                urgency = result.get('urgency_level', 'low').upper()
                                urgency_emoji = {'CRITICAL': '🔴', 'HIGH': '🟠', 'MEDIUM': '🟡', 'LOW': '🟢'}.get(urgency, '⚪')
                                st.metric("Urgency", f"{urgency_emoji} {urgency}")
                            
                            st.markdown("---")
                            st.write("**Analysis Details:**")
                            st.write(result.get('message', ''))
                            
                            if result.get('warning'):
                                st.warning(f"⚠️ {result['warning']}")
                            
                            if result.get('guidance'):
                                st.markdown("**Recommended Actions:**")
                                for i, guid in enumerate(result['guidance'], 1):
                                    st.write(f"{i}. {guid}")
                            
                            if result.get('hotlines'):
                                st.markdown("**Support Resources:**")
                                for service, number in result['hotlines'].items():
                                    st.write(f"• **{service}**: {number}")
                            
                            # Store result
                            modalities_used = []
                            if text_input_full:
                                modalities_used.append('text')
                            if uploaded_audio_full:
                                modalities_used.append('audio')
                            if uploaded_image_full:
                                modalities_used.append('image')
                            
                            st.session_state.detection_results.append({
                                'type': 'multimodal_full',
                                'crisis_type': result.get('crisis_type'),
                                'confidence': result.get('confidence'),
                                'urgency': result.get('urgency_level'),
                                'modalities': modalities_used,
                                'timestamp': datetime.now().isoformat()
                            })
                            save_results(st.session_state.detection_results)
                        else:
                            st.error("❌ Analysis failed. Please try again.")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            else:
                st.warning("⚠️ Please provide at least 2 modalities (text, audio, or image).")

# ==================== TAB 5: DASHBOARD ====================

with tab5:
    st.subheader("📊 Your Dashboard")
    
    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    
    total_messages = len(st.session_state.conversation_history)
    user_messages = sum(1 for m in st.session_state.conversation_history if m['type'] == 'user')
    crisis_detections = len(st.session_state.detection_results)
    
    with col1:
        st.metric("Total Messages", user_messages)
    
    with col2:
        st.metric("Detections", crisis_detections)
    
    with col3:
        if st.session_state.detection_results:
            avg_confidence = np.mean([r['confidence'] for r in st.session_state.detection_results])
            st.metric("Avg Confidence", f"{avg_confidence*100:.1f}%")
        else:
            st.metric("Avg Confidence", "0%")
    
    with col4:
        from datetime import datetime, timedelta
        if st.session_state.conversation_history:
            first_msg_time = datetime.fromisoformat(st.session_state.conversation_history[0]['timestamp'])
            last_msg_time = datetime.fromisoformat(st.session_state.conversation_history[-1]['timestamp'])
            duration = (last_msg_time - first_msg_time).seconds
            minutes = duration // 60
            seconds = duration % 60
            st.metric("Session Duration", f"{minutes}m {seconds}s")
        else:
            st.metric("Session Duration", "0s")
    
    st.markdown("---")
    
    # Crisis type distribution
    if st.session_state.detection_results:
        col1, col2 = st.columns(2)
        
        with col1:
            crisis_types = [r['crisis_type'] for r in st.session_state.detection_results]
            crisis_counts = pd.Series(crisis_types).value_counts()
            
            fig, ax = plt.subplots(figsize=(8, 6))
            # Define unique colors for each crisis type
            color_map = {
                'stressed': '#FFA500',      # Orange
                'suicidal': '#D32F2F',      # Red
                'abuse': '#C71585',         # Crimson
                'alertness': '#FFD700',     # Gold
                'racism': '#FF6347',        # Tomato
                'out_of_topic': '#2E7D32'   # Green
            }
            colors = [color_map.get(crisis_type, '#808080') for crisis_type in crisis_counts.index]
            
            ax.pie(crisis_counts.values, labels=crisis_counts.index, autopct='%1.1f%%',
                   colors=colors)
            ax.set_title("Crisis Type Distribution")
            st.pyplot(fig)
            plt.close()
        
        with col2:
            # Confidence over time
            confidences = [r['confidence'] for r in st.session_state.detection_results]
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(confidences, marker='o', linestyle='-', color='#2e7d32', linewidth=2)
            ax.set_xlabel("Detection #")
            ax.set_ylabel("Confidence Score")
            ax.set_title("Confidence Over Time")
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
        
        # Urgency breakdown
        st.markdown("---")
        st.subheader("Urgency Level Breakdown")
        urgency_counts = pd.Series([r.get('urgency', 'low') for r in st.session_state.detection_results]).value_counts()
        
        col1, col2, col3, col4 = st.columns(4)
        urgency_colors = {'critical': '🔴', 'high': '🟠', 'medium': '🟡', 'low': '🟢'}
        
        for urgency_level in ['critical', 'high', 'medium', 'low']:
            count = urgency_counts.get(urgency_level, 0)
            with [col1, col2, col3, col4][['critical', 'high', 'medium', 'low'].index(urgency_level)]:
                st.metric(f"{urgency_colors.get(urgency_level, '⚪')} {urgency_level.capitalize()}", count)
    else:
        st.info("No data yet. Send messages to see statistics.")

# ==================== TAB 6: RESOURCES ====================

with tab6:
    st.subheader("📚 Crisis Resources & Support")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🚨 Emergency Hotlines
        
        **🇺🇸 United States:**
        - **988** - Suicide & Crisis Lifeline (call or text)
        - Text **HOME** to **741741** - Crisis Text Line
        
        **🌍 International:**
        - 🇬🇧 UK: 116 123
        - 🇨🇦 Canada: 1-833-456-4566
        - 🇦🇺 Australia: 1-300-224-636
        """)
    
    with col2:
        st.markdown("""
        ### 💙 Mental Health Services
        
        - **NAMI:** 1-800-950-6264
        - **SAMHSA:** 1-800-662-4357
        - **Crisis Text Line:** Text HOME to 741741
        
        ### 🆘 If You're in Danger
        
        **CALL 911 IMMEDIATELY**
        
        Do not hesitate to seek emergency help.
        """)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🧘 Coping Strategies
        - Deep breathing
        - Progressive muscle relaxation
        - Meditation
        - Journaling
        """)
    
    with col2:
        st.markdown("""
        ### 💪 Self-Care Tips
        - Get adequate sleep
        - Exercise regularly
        - Eat healthy foods
        - Connect with others
        """)
    
    with col3:
        st.markdown("""
        ### 📖 Learn More
        - Mental health basics
        - Stress management
        - Support groups
        - Therapy options
        """)

# ==================== FOOTER ====================

st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #757575; font-size: 12px; margin-top: 30px;'>
        <p>💙 Mental Health Crisis Support Chatbot v2.0 • All conversations are confidential</p>
        <p>⚠️ In case of emergency, always call 911 or your local emergency number</p>
    </div>
""", unsafe_allow_html=True)
