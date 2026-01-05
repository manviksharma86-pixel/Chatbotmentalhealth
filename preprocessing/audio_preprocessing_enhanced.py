"""
Audio preprocessing module with Librosa feature extraction and Whisper transcription
Comprehensive audio analysis for mental health crisis detection
"""

import numpy as np
import os
import librosa
import soundfile as sf
import speech_recognition as sr
from pathlib import Path
import logging
from typing import Dict, Tuple, List
import warnings

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioProcessor:
    """Process and analyze audio files"""
    
    def __init__(self, sample_rate: int = 16000):
        """
        Initialize audio processor
        
        Args:
            sample_rate: Target sample rate (Hz)
        """
        self.sr = sample_rate
        self.recognizer = sr.Recognizer()
    
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file"""
        try:
            y, sr_val = librosa.load(audio_path, sr=self.sr)
            return y, sr_val
        except Exception as e:
            logger.error(f"Error loading audio: {str(e)}")
            return None, None
    
    def extract_features(self, audio_path: str) -> Dict[str, float]:
        """
        Extract comprehensive audio features using librosa
        
        Features:
        - MFCC (Mel-frequency cepstral coefficients)
        - Spectral features
        - Temporal features
        - Energy-based features
        """
        y, sr_val = self.load_audio(audio_path)
        if y is None:
            return {}
        
        try:
            # MFCC features
            mfcc = librosa.feature.mfcc(y=y, sr=sr_val, n_mfcc=13)
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr_val)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr_val)[0]
            
            # Chroma features (pitch-based)
            chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr_val)
            
            # Zero crossing rate (voice activity detection)
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            
            # RMS energy
            rms = librosa.feature.rms(y=y)[0]
            
            # Chromagram
            chromagram = librosa.feature.chroma_cqt(y=y, sr=sr_val)
            
            # Tempogram (rhythm analysis)
            onset_env = librosa.onset.onset_strength(y=y, sr=sr_val)
            tempogram = librosa.feature.tempogram(onset_env=onset_env, sr=sr_val)
            
            features = {
                # MFCC statistics
                'mfcc_mean': float(np.mean(mfcc)),
                'mfcc_std': float(np.std(mfcc)),
                'mfcc_min': float(np.min(mfcc)),
                'mfcc_max': float(np.max(mfcc)),
                
                # Spectral features
                'spectral_centroid_mean': float(np.mean(spectral_centroids)),
                'spectral_centroid_std': float(np.std(spectral_centroids)),
                'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
                'spectral_rolloff_std': float(np.std(spectral_rolloff)),
                
                # Chroma features
                'chroma_mean': float(np.mean(chroma_stft)),
                'chroma_std': float(np.std(chroma_stft)),
                
                # Zero crossing rate
                'zcr_mean': float(np.mean(zcr)),
                'zcr_std': float(np.std(zcr)),
                
                # Energy features
                'energy_mean': float(np.mean(rms)),
                'energy_std': float(np.std(rms)),
                'energy_max': float(np.max(rms)),
                
                # Chromagram
                'chromagram_mean': float(np.mean(chromagram)),
                
                # Tempogram
                'tempo_mean': float(np.mean(tempogram)),
                
                # Derived features
                'pitch': self.estimate_pitch(y, sr_val),
                'energy': float(np.sum(y**2) / len(y)),
                'duration': float(len(y) / sr_val)
            }
            
            return features
        
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            return {}
    
    def estimate_pitch(self, y: np.ndarray, sr: int) -> float:
        """Estimate fundamental frequency (pitch)"""
        try:
            # Use pyin for better pitch estimation
            f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=80, fmax=400, sr=sr)
            # Get median pitch (ignore silence and NaN values)
            valid_f0 = f0[~np.isnan(f0)]
            if len(valid_f0) > 0:
                return float(np.median(valid_f0))
            return 0.0
        except Exception as e:
            logger.warning(f"Could not estimate pitch: {str(e)}")
            try:
                # Fallback: use spectral centroid as pitch proxy
                spectral_cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
                return float(np.median(spectral_cent))
            except:
                return 0.0
    
    def recognize_speech(self, audio_path: str) -> str:
        """
        Recognize speech using SpeechRecognition library
        
        Fallback method if Whisper is not available
        """
        try:
            # Check if file exists
            if not os.path.exists(audio_path):
                logger.error(f"Audio file not found: {audio_path}")
                return "[Audio file not found]"
            
            with sr.AudioFile(audio_path) as source:
                audio = self.recognizer.record(source)
                logger.info("Audio loaded successfully")
            
            # Try Google Speech Recognition
            text = self.recognizer.recognize_google(audio)
            logger.info("Speech recognition successful")
            return text
        except sr.UnknownValueError:
            logger.warning("Could not understand audio")
            return "[Audio could not be understood]"
        except sr.RequestError as e:
            logger.error(f"Recognition service error: {str(e)}")
            return "[Recognition error]"
        except Exception as e:
            logger.error(f"Speech recognition error: {str(e)}")
            return f"[Error: {type(e).__name__}]"
    
    def detect_emotion_from_audio(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Detect emotional state from audio features
        Based on acoustic characteristics:
        - Low energy, high pitch variation → anxiety
        - Slow speech, low energy → depression
        - High energy, sharp peaks → anger
        - Stable features → calm
        """
        emotions = {
            'calm': 0.0,
            'stressed': 0.0,
            'sad': 0.0,
            'angry': 0.0,
            'neutral': 0.0
        }
        
        try:
            energy = features.get('energy_mean', 0)
            zcr = features.get('zcr_mean', 0)
            spectral_cent = features.get('spectral_centroid_mean', 0)
            duration = features.get('duration', 0)
            
            # Stressed: high energy, high ZCR, fast speech
            if energy > 0.05 and zcr > 0.1:
                emotions['stressed'] = 0.7
            
            # Sad: low energy, low ZCR, slow speech
            elif energy < 0.02 and duration > 10:
                emotions['sad'] = 0.7
            
            # Angry: high energy, sharp spectral features
            elif energy > 0.08 and spectral_cent > 2000:
                emotions['angry'] = 0.6
            
            # Calm: balanced features
            else:
                emotions['calm'] = 0.6
            
            # Normalize
            total = sum(emotions.values()) or 1
            emotions = {k: v/total for k, v in emotions.items()}
            
            return emotions
        
        except Exception as e:
            logger.error(f"Error detecting emotion: {str(e)}")
            emotions['neutral'] = 1.0
            return emotions
    
    def detect_stress_indicators(self, features: Dict[str, float]) -> Dict[str, bool]:
        """
        Detect stress indicators from audio
        
        Indicators:
        - Elevated pitch (pitch > 150 Hz)
        - High energy variability
        - Rapid speech (high ZCR)
        - Inconsistent rhythm (high tempo variation)
        """
        indicators = {
            'elevated_pitch': features.get('pitch', 0) > 150,
            'high_energy': features.get('energy_mean', 0) > 0.05,
            'rapid_speech': features.get('zcr_mean', 0) > 0.1,
            'variable_energy': features.get('energy_std', 0) > 0.02,
            'high_tempo': features.get('tempo_mean', 0) > 120
        }
        
        return indicators


class WhisperSTT:
    """Whisper-based Speech-to-Text transcription"""
    
    def __init__(self, model_size: str = 'base'):
        """
        Initialize Whisper model
        
        Args:
            model_size: Model size ('tiny', 'base', 'small', 'medium', 'large')
        """
        self.model_size = model_size
        try:
            import whisper
            self.whisper = whisper
            self.model = None
        except ImportError:
            logger.warning("Whisper not installed. Fallback to SpeechRecognition.")
            self.whisper = None
            self.model = None
    
    def transcribe(self, audio_path: str) -> str:
        """
        Transcribe audio using Whisper
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            Transcribed text
        """
        try:
            if self.whisper is None:
                logger.warning("Whisper not available. Using SpeechRecognition fallback.")
                processor = AudioProcessor()
                return processor.recognize_speech(audio_path)
            
            # Load model if not already loaded
            if self.model is None:
                self.model = self.whisper.load_model(self.model_size)
            
            # Transcribe
            result = self.model.transcribe(audio_path)
            return result.get('text', '').strip()
        
        except Exception as e:
            logger.error(f"Whisper transcription error: {str(e)}")
            # Fallback
            processor = AudioProcessor()
            return processor.recognize_speech(audio_path)
    
    def transcribe_with_timestamps(self, audio_path: str) -> List[Dict]:
        """
        Transcribe audio with timing information
        
        Returns:
            List of segments with text and timing
        """
        try:
            if self.whisper is None or self.model is None:
                return []
            
            result = self.model.transcribe(audio_path)
            segments = []
            
            for segment in result.get('segments', []):
                segments.append({
                    'text': segment['text'],
                    'start': segment['start'],
                    'end': segment['end']
                })
            
            return segments
        
        except Exception as e:
            logger.error(f"Error transcribing with timestamps: {str(e)}")
            return []
