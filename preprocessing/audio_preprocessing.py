"""
Audio Preprocessing Module for Mental Health Crisis Detection
Simplified version without heavy dependencies
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


class AudioPreprocessor:
    """Preprocess audio data for crisis detection"""
    
    def __init__(self, sr=16000, n_mfcc=13):
        self.sr = sr  # Sample rate
        self.n_mfcc = n_mfcc  # Number of MFCC coefficients
        self.duration = 30  # Max audio duration in seconds
        
    def load_audio(self, filepath):
        """Load audio file"""
        try:
            audio, sr = librosa.load(filepath, sr=self.sr, duration=self.duration)
            return audio, sr
        except Exception as e:
            print(f"Error loading audio {filepath}: {e}")
            return None, None
    
    def normalize_audio(self, audio):
        """Normalize audio to [-1, 1] range"""
        audio = audio.astype(np.float32)
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        return audio
    
    def extract_mfcc(self, audio, sr):
        """Extract MFCC features"""
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc)
        return mfcc
    
    def extract_mel_spectrogram(self, audio, sr, n_mels=128):
        """Extract mel-spectrogram features"""
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db
    
    def extract_zero_crossing_rate(self, audio):
        """Extract zero-crossing rate"""
        zcr = librosa.feature.zero_crossing_rate(audio)
        return zcr
    
    def extract_spectral_centroid(self, audio, sr):
        """Extract spectral centroid"""
        centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        return centroid
    
    def extract_chroma_features(self, audio, sr):
        """Extract chroma features"""
        chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
        return chroma
    
    def extract_temporal_features(self, audio, sr):
        """Extract temporal features like energy and RMS"""
        energy = np.sqrt(np.sum(audio**2)) / len(audio)
        rms = librosa.feature.rms(y=audio)
        
        return {
            'energy': energy,
            'rms_mean': np.mean(rms),
            'rms_std': np.std(rms),
        }
    
    def extract_all_features(self, filepath):
        """Extract all audio features"""
        audio, sr = self.load_audio(filepath)
        if audio is None:
            return None
        
        # Normalize
        audio = self.normalize_audio(audio)
        
        features = {
            'mfcc': self.extract_mfcc(audio, sr),
            'mel_spectrogram': self.extract_mel_spectrogram(audio, sr),
            'zcr': self.extract_zero_crossing_rate(audio),
            'spectral_centroid': self.extract_spectral_centroid(audio, sr),
            'chroma': self.extract_chroma_features(audio, sr),
        }
        
        # Add temporal features
        temporal = self.extract_temporal_features(audio, sr)
        features.update(temporal)
        
        return features
    
    def pad_or_truncate_sequence(self, feature, target_length=100):
        """Pad or truncate feature sequence to target length"""
        current_length = feature.shape[1]
        
        if current_length >= target_length:
            return feature[:, :target_length]
        else:
            padding = np.zeros((feature.shape[0], target_length - current_length))
            return np.hstack([feature, padding])
    
    def normalize_features(self, features):
        """Normalize features using StandardScaler"""
        scaler = StandardScaler()
        normalized = scaler.fit_transform(features.T).T
        return normalized


class SpeechAnalyzer:
    """Analyze speech characteristics for crisis detection"""
    
    STRESS_INDICATORS = {
        'fast_speech': {'zcr_threshold': 0.15, 'energy_threshold': 0.5},
        'slow_speech': {'zcr_threshold': 0.05, 'energy_threshold': 0.2},
        'shaky_voice': {'spectral_variance_threshold': 0.7},
        'emotional_distress': {'energy_variance_threshold': 0.6}
    }
    
    @staticmethod
    def analyze_speech_rate(audio, sr):
        """Analyze speech rate (tempo)"""
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        return tempo
    
    @staticmethod
    def analyze_pitch_variation(audio, sr):
        """Estimate pitch variation"""
        f0 = librosa.yin(audio, fmin=80, fmax=400, sr=sr)
        pitch_variation = np.std(f0[f0 > 0]) if np.any(f0 > 0) else 0
        return pitch_variation
    
    @staticmethod
    def analyze_voice_quality(mfcc):
        """Analyze voice quality from MFCC"""
        # Calculate statistics
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        mfcc_delta = np.mean(np.abs(np.diff(mfcc, axis=1)))
        
        return {
            'mfcc_mean': mfcc_mean,
            'mfcc_std': mfcc_std,
            'mfcc_delta': mfcc_delta
        }
    
    @staticmethod
    def detect_distress_signals(audio, sr, mfcc):
        """Detect audio-based distress signals"""
        signals = {}
        
        # Speech rate
        tempo = SpeechAnalyzer.analyze_speech_rate(audio, sr)
        signals['speech_rate'] = tempo
        signals['rapid_speech'] = tempo > 130  # BPM threshold
        
        # Pitch variation
        pitch_var = SpeechAnalyzer.analyze_pitch_variation(audio, sr)
        signals['pitch_variation'] = pitch_var
        signals['unstable_pitch'] = pitch_var > 50  # Hz threshold
        
        # Energy
        energy = np.sqrt(np.sum(audio**2)) / len(audio)
        signals['energy'] = energy
        signals['high_energy'] = energy > 0.5
        
        # Voice tremor (rapid variation in amplitude)
        rms = librosa.feature.rms(y=audio)[0]
        tremor = np.std(np.diff(rms))
        signals['tremor'] = tremor
        signals['voice_tremor'] = tremor > 0.1
        
        return signals


def visualize_audio_features(audio, sr, mfcc, mel_spec, output_path='data/processed/audio_features.png'):
    """Visualize audio features"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Waveform
    axes[0, 0].plot(audio[:sr*3])  # First 3 seconds
    axes[0, 0].set_title('Audio Waveform')
    axes[0, 0].set_ylabel('Amplitude')
    
    # MFCC
    img1 = axes[0, 1].imshow(mfcc, aspect='auto', origin='lower')
    axes[0, 1].set_title('MFCC Features')
    axes[0, 1].set_ylabel('MFCC Coefficient')
    plt.colorbar(img1, ax=axes[0, 1])
    
    # Mel Spectrogram
    img2 = axes[1, 0].imshow(mel_spec, aspect='auto', origin='lower')
    axes[1, 0].set_title('Mel-Spectrogram')
    axes[1, 0].set_ylabel('Frequency')
    plt.colorbar(img2, ax=axes[1, 0])
    
    # Spectral Centroid
    cent = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    axes[1, 1].plot(cent, label='Spectral Centroid')
    axes[1, 1].set_title('Spectral Centroid Over Time')
    axes[1, 1].set_ylabel('Hz')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Audio features visualization saved to {output_path}")
    plt.close()


def main():
    print("Audio Preprocessing Module")
    print("="*60)
    
    # Create example audio for testing
    print("\nGenerating synthetic audio samples...")
    
    sr = 16000
    duration = 5
    
    # Sample emotional speech patterns
    for category in ['normal', 'stressed', 'rapid']:
        t = np.linspace(0, duration, sr * duration)
        
        if category == 'normal':
            # Normal frequency
            frequency = 200
            audio = 0.3 * np.sin(2 * np.pi * frequency * t)
            
        elif category == 'stressed':
            # Varying frequency (anxious pattern)
            frequency = 200 + 50 * np.sin(2 * np.pi * 0.5 * t)
            audio = 0.5 * np.sin(2 * np.pi * frequency * t)
            
        elif category == 'rapid':
            # Higher frequency (rapid speech)
            frequency = 250
            audio = 0.4 * np.sin(2 * np.pi * frequency * t)
        
        # Add noise
        audio += 0.05 * np.random.randn(len(audio))
        
        filepath = f'data/raw/audio_{category}.wav'
        Path('data/raw').mkdir(parents=True, exist_ok=True)
        sf.write(filepath, audio, sr)
        print(f"✓ Created {filepath}")
    
    # Process audio files
    preprocessor = AudioPreprocessor(sr=sr)
    
    audio_files = list(Path('data/raw').glob('audio_*.wav'))
    
    for audio_file in audio_files:
        print(f"\nProcessing {audio_file.name}...")
        features = preprocessor.extract_all_features(str(audio_file))
        
        if features:
            print(f"  MFCC shape: {features['mfcc'].shape}")
            print(f"  Mel-spectrogram shape: {features['mel_spectrogram'].shape}")
            print(f"  Energy: {features['energy']:.4f}")
            print(f"  RMS mean: {features['rms_mean']:.4f}")
            
            # Analyze distress signals
            audio, sr_loaded = librosa.load(str(audio_file), sr=sr)
            signals = SpeechAnalyzer.detect_distress_signals(audio, sr_loaded, features['mfcc'])
            
            print(f"  Speech Rate (BPM): {signals['speech_rate']:.1f}")
            print(f"  Pitch Variation: {signals['pitch_variation']:.2f}")
            print(f"  Voice Tremor: {signals['voice_tremor']}")
    
    print("\n✓ Audio preprocessing completed!")


if __name__ == "__main__":
    main()
