"""
Flask API Backend for Mental Health Crisis Support Chatbot
Handles all backend processing: text, audio, image, and video analysis
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import sys
import tempfile
import numpy as np
from datetime import datetime
import logging
from dotenv import load_dotenv
from collections import Counter

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import processing modules
from preprocessing.text_preprocessing import TextPreprocessor, CrisisIntentAnalyzer
from preprocessing.audio_preprocessing_enhanced import AudioProcessor, WhisperSTT
from preprocessing.image_preprocessing import ImageAnalyzer, EmotionDetector

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO'))
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize processors
text_processor = TextPreprocessor()
crisis_analyzer = CrisisIntentAnalyzer()
audio_processor = AudioProcessor()
whisper_stt = WhisperSTT()
image_analyzer = ImageAnalyzer()
emotion_detector = EmotionDetector()


# ==================== TEXT ENDPOINTS ====================

@app.route('/api/analyze-text', methods=['POST'])
def analyze_text():
    """Analyze text for crisis indicators"""
    try:
        data = request.json
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Full analysis using improved algorithm
        analysis = crisis_analyzer.analyze_full(text)
        
        return jsonify({
            'success': True,
            'crisis_type': analysis['crisis_type'],
            'urgency_level': analysis['urgency_level'],
            'confidence': float(analysis['confidence']),
            'message': analysis['message'],
            'guidance': analysis['guidance'],
            'hotlines': analysis['hotlines'],
            'warning': analysis.get('warning', ''),
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error analyzing text: {str(e)}")
        return jsonify({'error': str(e)}), 500


# ==================== AUDIO ENDPOINTS ====================

@app.route('/api/process-audio', methods=['POST'])
def process_audio():
    """Process audio file - convert speech to text and analyze"""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        use_whisper = request.form.get('use_whisper', 'true').lower() == 'true'
        
        # Save temporary audio file using cross-platform tempfile
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            audio_path = tmp.name
        audio_file.save(audio_path)
        
        # Convert speech to text
        if use_whisper:
            transcription = whisper_stt.transcribe(audio_path)
        else:
            transcription = audio_processor.recognize_speech(audio_path)
        
        # Extract audio features
        audio_features = audio_processor.extract_features(audio_path)
        
        # Analyze transcribed text
        crisis_type, confidence, intensity = crisis_analyzer.analyze(transcription)
        
        # Generate response
        response_text = generate_crisis_response(crisis_type, confidence, intensity)
        
        # Clean up
        if os.path.exists(audio_path):
            os.remove(audio_path)
        
        return jsonify({
            'success': True,
            'transcription': transcription,
            'crisis_type': crisis_type,
            'confidence': float(confidence),
            'intensity': float(intensity),
            'audio_features': {
                'mfcc_mean': float(np.mean(audio_features.get('mfcc', [0]))),
                'pitch': float(audio_features.get('pitch', 0)),
                'energy': float(audio_features.get('energy', 0))
            },
            'response': response_text,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        return jsonify({'error': str(e)}), 500


# ==================== IMAGE ENDPOINTS ====================

@app.route('/api/analyze-image', methods=['POST'])
def analyze_image():
    """Analyze image for emotion detection"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        image_file = request.files['image']
        
        # Save temporary image file
        image_path = f"/tmp/image_{datetime.now().timestamp()}.jpg"
        image_file.save(image_path)
        
        # Detect emotions in image
        emotions = emotion_detector.detect_emotions(image_path)
        
        # Analyze image content
        analysis = image_analyzer.analyze_image(image_path)
        
        # Generate response based on detected emotions
        dominant_emotion = max(emotions, key=emotions.get) if emotions else 'neutral'
        response_text = generate_emotion_response(dominant_emotion, emotions)
        
        # Clean up
        if os.path.exists(image_path):
            os.remove(image_path)
        
        return jsonify({
            'success': True,
            'emotions': emotions,
            'dominant_emotion': dominant_emotion,
            'analysis': analysis,
            'response': response_text,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error analyzing image: {str(e)}")
        return jsonify({'error': str(e)}), 500


# ==================== MULTIMODAL ENDPOINTS ====================

@app.route('/api/multimodal-analysis', methods=['POST'])
def multimodal_analysis():
    """Analyze combination of text, audio, and/or image"""
    try:
        results = {'success': True, 'analyses': {}}
        
        # Text analysis
        if 'text' in request.form:
            text_result = analyze_text_request(request.form['text'])
            results['analyses']['text'] = text_result
        
        # Audio analysis
        if 'audio' in request.files:
            audio_result = analyze_audio_request(request.files['audio'])
            results['analyses']['audio'] = audio_result
        
        # Image analysis
        if 'image' in request.files:
            image_result = analyze_image_request(request.files['image'])
            results['analyses']['image'] = image_result
        
        # Combine results
        combined_crisis_type, combined_confidence = combine_analyses(results['analyses'])
        response_text = generate_crisis_response(combined_crisis_type, combined_confidence, 0.7)
        
        results['combined_crisis_type'] = combined_crisis_type
        results['combined_confidence'] = combined_confidence
        results['response'] = response_text
        results['timestamp'] = datetime.now().isoformat()
        
        return jsonify(results)
    
    except Exception as e:
        logger.error(f"Error in multimodal analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500


# ==================== HELPER FUNCTIONS ====================

def generate_crisis_response(crisis_type, confidence, intensity):
    """Generate personalized response based on crisis type"""
    
    responses = {
        'stressed': {
            'message': "I sense you're feeling stressed. That's something many people experience. Here are some immediate coping strategies:",
            'suggestions': [
                "Take slow, deep breaths (4 count in, hold 4, out 4)",
                "Practice progressive muscle relaxation",
                "Take a short walk or do light exercise",
                "Talk to someone you trust"
            ],
            'hotline': "988 - Suicide & Crisis Lifeline (US)"
        },
        'insomnia': {
            'message': "It sounds like you're struggling with sleep or rest. Let me help you:",
            'suggestions': [
                "Establish a consistent sleep schedule",
                "Avoid screens 30 minutes before bed",
                "Try relaxation techniques like meditation",
                "Consult a sleep specialist if it persists"
            ],
            'hotline': "Call your healthcare provider"
        },
        'suicidal': {
            'message': "I'm very concerned about your safety. Please reach out immediately:",
            'suggestions': [
                "Call 988 Suicide & Crisis Lifeline immediately",
                "Text HOME to 741741 for Crisis Text Line",
                "Go to nearest emergency room",
                "Call 911 if you're in immediate danger"
            ],
            'hotline': "988 - CALL NOW (Suicide & Crisis Lifeline)"
        },
        'racism': {
            'message': "I'm sorry you're experiencing racism. You deserve support and respect:",
            'suggestions': [
                "Document incidents for your safety",
                "Connect with support communities",
                "Contact civil rights organizations",
                "Seek counseling from trauma-informed therapists"
            ],
            'hotline': "National Race & Ethnicity Hotline: 1-844-856-4869"
        },
        'out_of_topic': {
            'message': "I'm here to help with mental health concerns. How are you feeling emotionally?",
            'suggestions': [
                "Describe what's bothering you",
                "Share your feelings in detail",
                "Tell me about recent stressful events",
                "I'm here to listen and help"
            ],
            'hotline': "988 - Suicide & Crisis Lifeline"
        }
    }
    
    response = responses.get(crisis_type, responses['out_of_topic'])
    
    return {
        'message': response['message'],
        'suggestions': response['suggestions'],
        'hotline': response['hotline'],
        'confidence': float(confidence),
        'intensity': float(intensity)
    }


def generate_emotion_response(emotion, emotion_scores):
    """Generate response based on detected emotion"""
    
    emotion_messages = {
        'happy': "Great! I'm glad you're feeling positive. Keep nurturing those good feelings!",
        'sad': "I see you're feeling sad. It's okay to feel this way. Would you like to talk about what's bothering you?",
        'angry': "I notice some anger in your expression. Let's talk about what's making you feel this way.",
        'fear': "You seem worried or fearful. I'm here to help you work through this.",
        'neutral': "How are you really feeling? I'm here to listen.",
        'surprise': "Something seems surprising or unexpected. Tell me more about it.",
        'disgust': "Something seems troubling you. What's on your mind?"
    }
    
    message = emotion_messages.get(emotion, "I'm here to listen. How can I help?")
    
    return {
        'message': message,
        'detected_emotion': emotion,
        'emotion_scores': emotion_scores
    }


def combine_analyses(analyses):
    """Combine results from multiple modalities"""
    crisis_types = []
    confidences = []
    
    for modality, result in analyses.items():
        if result and 'crisis_type' in result:
            crisis_types.append(result['crisis_type'])
            confidences.append(result.get('confidence', 0.5))
    
    if not crisis_types:
        return 'out_of_topic', 0.3
    
    # Weight by confidence
    weighted_types = Counter()
    for ctype, conf in zip(crisis_types, confidences):
        weighted_types[ctype] += conf
    
    most_common = weighted_types.most_common(1)[0]
    return most_common[0], most_common[1] / len(analyses)


def analyze_text_request(text):
    """Helper to analyze text from multimodal request"""
    cleaned = text_processor.clean_text(text)
    analysis = crisis_analyzer.analyze_full(cleaned)
    return {
        'crisis_type': analysis['crisis_type'],
        'confidence': float(analysis['confidence']),
        'text': text
    }


def analyze_audio_request(audio_file):
    """Helper to analyze audio from multimodal request"""
    # Save temporary audio file using cross-platform tempfile
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        audio_path = tmp.name
    audio_file.save(audio_path)
    
    try:
        transcription = whisper_stt.transcribe(audio_path)
        analysis = crisis_analyzer.analyze_full(transcription)
        return {
            'crisis_type': analysis['crisis_type'],
            'confidence': float(analysis['confidence']),
            'transcription': transcription
        }
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)


def analyze_image_request(image_file):
    """Helper to analyze image from multimodal request"""
    # Save temporary image file using cross-platform tempfile
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        image_path = tmp.name
    image_file.save(image_path)
    
    try:
        emotions = emotion_detector.detect_emotions(image_path)
        return {
            'crisis_type': 'emotion_detected',
            'confidence': max(emotions.values()) if emotions else 0.5,
            'emotions': emotions
        }
    finally:
        if os.path.exists(image_path):
            os.remove(image_path)


# ==================== HEALTH CHECK ====================

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'features': {
            'text_analysis': True,
            'audio_processing': os.getenv('ENABLE_AUDIO_INPUT', 'true').lower() == 'true',
            'image_analysis': os.getenv('ENABLE_IMAGE_ANALYSIS', 'true').lower() == 'true',
            'whisper_stt': True
        }
    })


if __name__ == '__main__':
    port = int(os.getenv('FLASK_PORT', 5000))
    debug = os.getenv('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug)
