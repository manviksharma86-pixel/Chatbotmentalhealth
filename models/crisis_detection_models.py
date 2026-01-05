"""
Model Design Module for Mental Health Crisis Detection
Implements text classification, speech transcription, and multimodal fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, pipeline
import numpy as np
from pathlib import Path
import json


class TextClassificationModel(nn.Module):
    """BERT-based text classification for crisis detection"""
    
    def __init__(self, model_name='distilbert-base-uncased', num_classes=5, dropout=0.2):
        super(TextClassificationModel, self).__init__()
        
        self.transformer = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.transformer.config.hidden_size
        self.num_classes = num_classes
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(self.hidden_size, 256)
        self.dense_out = nn.Linear(256, num_classes)
        
        # For crisis intensity prediction
        self.intensity_head = nn.Linear(256, 1)
        
    def forward(self, input_ids, attention_mask):
        """Forward pass"""
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # Classification head
        hidden = self.dropout(cls_output)
        hidden = F.relu(self.dense(hidden))
        hidden = self.dropout(hidden)
        
        logits = self.dense_out(hidden)
        intensity = torch.sigmoid(self.intensity_head(hidden))
        
        return {
            'logits': logits,
            'intensity': intensity,
            'embeddings': hidden
        }
    
    def get_embeddings(self, input_ids, attention_mask):
        """Get text embeddings for multimodal fusion"""
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        cls_output = outputs.last_hidden_state[:, 0, :]
        hidden = F.relu(self.dense(cls_output))
        
        return hidden


class AudioClassificationModel(nn.Module):
    """CNN-LSTM model for audio-based crisis detection"""
    
    def __init__(self, num_classes=5, input_size=13, dropout=0.2):
        super(AudioClassificationModel, self).__init__()
        
        self.input_size = input_size  # MFCC coefficients
        self.num_classes = num_classes
        self.embedding_size = 128
        
        # CNN feature extraction
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.Linear(256, 1)
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(256, self.embedding_size)
        self.dense_out = nn.Linear(self.embedding_size, num_classes)
        
        # Intensity prediction
        self.intensity_head = nn.Linear(self.embedding_size, 1)
        
    def forward(self, x):
        """Forward pass
        
        Args:
            x: Input of shape (batch_size, num_mfcc, time_steps)
        """
        # CNN feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        # Reshape for LSTM: (batch_size, time_steps, channels)
        x = x.transpose(1, 2)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Attention
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        attended = torch.sum(lstm_out * attention_weights, dim=1)
        
        # Classification head
        hidden = self.dropout(F.relu(self.dense(attended)))
        logits = self.dense_out(hidden)
        intensity = torch.sigmoid(self.intensity_head(hidden))
        
        return {
            'logits': logits,
            'intensity': intensity,
            'embeddings': hidden,
            'attention': attention_weights
        }
    
    def get_embeddings(self, x):
        """Get audio embeddings for multimodal fusion"""
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        attended = torch.sum(lstm_out * attention_weights, dim=1)
        
        hidden = F.relu(self.dense(attended))
        return hidden


class MultimodalFusionModel(nn.Module):
    """Multimodal fusion combining text and audio for crisis detection"""
    
    def __init__(self, text_model, audio_model, fusion_dim=256, num_classes=5, dropout=0.2):
        super(MultimodalFusionModel, self).__init__()
        
        self.text_model = text_model
        self.audio_model = audio_model
        
        # Embedding projection layers
        self.text_projector = nn.Linear(256, fusion_dim)
        self.audio_projector = nn.Linear(128, fusion_dim)
        
        # Fusion layers
        self.fusion_dense = nn.Linear(fusion_dim * 2, 256)
        self.dropout = nn.Dropout(dropout)
        self.output_dense = nn.Linear(256, num_classes)
        
        # Confidence and intensity heads
        self.confidence_head = nn.Linear(256, 1)
        self.intensity_head = nn.Linear(256, 1)
        
    def forward(self, input_ids, attention_mask, audio_mfcc):
        """Forward pass with both modalities
        
        Args:
            input_ids: Text input IDs
            attention_mask: Attention mask for text
            audio_mfcc: Audio MFCC features
        """
        # Get embeddings from both modalities
        text_embedding = self.text_model.get_embeddings(input_ids, attention_mask)
        audio_embedding = self.audio_model.get_embeddings(audio_mfcc)
        
        # Project embeddings
        text_proj = F.relu(self.text_projector(text_embedding))
        audio_proj = F.relu(self.audio_projector(audio_embedding))
        
        # Concatenate and fuse
        fused = torch.cat([text_proj, audio_proj], dim=1)
        
        # Fusion layers
        hidden = self.dropout(F.relu(self.fusion_dense(fused)))
        logits = self.output_dense(hidden)
        
        # Additional outputs
        confidence = torch.sigmoid(self.confidence_head(hidden))
        intensity = torch.sigmoid(self.intensity_head(hidden))
        
        return {
            'logits': logits,
            'confidence': confidence,
            'intensity': intensity,
            'text_embedding': text_embedding,
            'audio_embedding': audio_embedding,
            'fused_embedding': hidden
        }


class SpeechToTextProcessor:
    """Handle speech-to-text conversion using Whisper"""
    
    def __init__(self, model_name='base'):
        """Initialize Whisper model for transcription"""
        try:
            import whisper
            self.model = whisper.load_model(model_name)
            self.whisper_available = True
        except:
            print("Whisper not available. Using fallback transcription.")
            self.whisper_available = False
            # Fallback to SpeechRecognition
            self.recognizer = None
    
    def transcribe_audio(self, audio_path):
        """Transcribe audio to text"""
        try:
            if self.whisper_available:
                result = self.model.transcribe(audio_path, language='en')
                return result['text'], result['language']
            else:
                return self._fallback_transcribe(audio_path)
        except Exception as e:
            print(f"Transcription error: {e}")
            return "", "en"
    
    def _fallback_transcribe(self, audio_path):
        """Fallback transcription method"""
        try:
            import speech_recognition as sr
            recognizer = sr.Recognizer()
            
            with sr.AudioFile(audio_path) as source:
                audio = recognizer.record(source)
            
            try:
                text = recognizer.recognize_google(audio)
                return text, "en"
            except sr.UnknownValueError:
                return "", "en"
        except:
            return "", "en"


class CrisisResponseGenerator:
    """Generate appropriate responses based on detected crisis type"""
    
    # Response templates
    RESPONSE_TEMPLATES = {
        'stressed': {
            'response': "I understand you're feeling stressed. This is a difficult time, and your feelings are valid.",
            'suggestions': [
                'Try deep breathing exercises (4-7-8 technique)',
                'Engage in physical activity or stretching',
                'Talk to someone you trust',
                'Consider reaching out to a mental health professional'
            ],
            'hotline': '988 Suicide & Crisis Lifeline (call or text 988)',
            'resources_link': 'https://www.samhsa.gov/find-help'
        },
        'alertness': {
            'response': "I hear that you're experiencing sleep difficulties. This can be very challenging.",
            'suggestions': [
                'Establish a consistent sleep schedule',
                'Avoid screens before bedtime',
                'Try relaxation techniques like meditation',
                'Consider consulting a sleep specialist'
            ],
            'hotline': '1-833-CALM-NOW',
            'resources_link': 'https://www.sleepfoundation.org'
        },
        'suicidal': {
            'response': "I'm deeply concerned about what you're sharing. Please know that help is available and recovery is possible.",
            'suggestions': [
                'CALL 988 IMMEDIATELY if you are in crisis',
                'Text "HOME" to 741741 to reach the Crisis Text Line',
                'Go to your nearest emergency room if in immediate danger',
                'Tell someone you trust how you\'re feeling'
            ],
            'hotline': '988 Suicide & Crisis Lifeline - CALL NOW',
            'resources_link': 'https://suicidepreventionlifeline.org',
            'urgency': 'CRITICAL'
        },
        'racism': {
            'response': "I'm sorry you experienced discrimination. Your experience is valid, and support is available.",
            'suggestions': [
                'Document the incident for records',
                'Reach out to support organizations',
                'Consider consulting with legal aid if needed',
                'Seek counseling to process the experience'
            ],
            'hotline': 'NAACP Crisis Response: 1-844-7-NAACP',
            'resources_link': 'https://www.naacp.org'
        },
        'out_of_topic': {
            'response': "Thank you for your message. I specialize in mental health support.",
            'suggestions': [
                'I\'m here to help with mental health and emotional concerns',
                'If you have a mental health question, please share it',
                'For general questions, please try a general-purpose AI assistant'
            ],
            'hotline': None,
            'resources_link': None
        }
    }
    
    @staticmethod
    def generate_response(category, confidence, intensity):
        """Generate crisis response"""
        template = CrisisResponseGenerator.RESPONSE_TEMPLATES.get(
            category, 
            CrisisResponseGenerator.RESPONSE_TEMPLATES['out_of_topic']
        )
        
        response = {
            'category': category,
            'confidence': float(confidence),
            'intensity': float(intensity),
            'main_response': template['response'],
            'suggestions': template['suggestions'],
            'hotline': template['hotline'],
            'resources_link': template.get('resources_link'),
            'urgency': template.get('urgency', 'normal')
        }
        
        return response


class ModelFactory:
    """Factory for creating and managing models"""
    
    @staticmethod
    def create_text_model(model_name='distilbert-base-uncased', num_classes=5, device='cpu'):
        """Create text classification model"""
        model = TextClassificationModel(model_name=model_name, num_classes=num_classes)
        return model.to(device)
    
    @staticmethod
    def create_audio_model(num_classes=5, device='cpu'):
        """Create audio classification model"""
        model = AudioClassificationModel(num_classes=num_classes)
        return model.to(device)
    
    @staticmethod
    def create_multimodal_model(text_model, audio_model, num_classes=5, device='cpu'):
        """Create multimodal fusion model"""
        model = MultimodalFusionModel(text_model, audio_model, num_classes=num_classes)
        return model.to(device)
    
    @staticmethod
    def save_model(model, filepath):
        """Save model checkpoint"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), filepath)
        print(f"Model saved to {filepath}")
    
    @staticmethod
    def load_model(model, filepath, device='cpu'):
        """Load model checkpoint"""
        model.load_state_dict(torch.load(filepath, map_location=device))
        return model


def main():
    print("Model Design Module")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create models
    print("\nCreating models...")
    text_model = ModelFactory.create_text_model(device=device)
    audio_model = ModelFactory.create_audio_model(device=device)
    multimodal_model = ModelFactory.create_multimodal_model(text_model, audio_model, device=device)
    
    print(f"✓ Text Classification Model: {text_model}")
    print(f"✓ Audio Classification Model: {audio_model}")
    print(f"✓ Multimodal Fusion Model: {multimodal_model}")
    
    # Test forward pass
    print("\n" + "="*60)
    print("TESTING MODEL FORWARD PASSES")
    print("="*60)
    
    # Text model test
    print("\nTesting Text Model...")
    batch_size = 4
    input_ids = torch.randint(0, 30522, (batch_size, 128))
    attention_mask = torch.ones((batch_size, 128))
    
    with torch.no_grad():
        text_output = text_model(input_ids, attention_mask)
        print(f"  Logits shape: {text_output['logits'].shape}")
        print(f"  Intensity shape: {text_output['intensity'].shape}")
        print(f"  Embeddings shape: {text_output['embeddings'].shape}")
    
    # Audio model test
    print("\nTesting Audio Model...")
    audio_input = torch.randn(batch_size, 13, 100)  # (batch, mfcc, time)
    
    with torch.no_grad():
        audio_output = audio_model(audio_input)
        print(f"  Logits shape: {audio_output['logits'].shape}")
        print(f"  Intensity shape: {audio_output['intensity'].shape}")
        print(f"  Embeddings shape: {audio_output['embeddings'].shape}")
    
    # Multimodal model test
    print("\nTesting Multimodal Model...")
    with torch.no_grad():
        multimodal_output = multimodal_model(input_ids, attention_mask, audio_input)
        print(f"  Logits shape: {multimodal_output['logits'].shape}")
        print(f"  Confidence shape: {multimodal_output['confidence'].shape}")
        print(f"  Intensity shape: {multimodal_output['intensity'].shape}")
        print(f"  Fused embedding shape: {multimodal_output['fused_embedding'].shape}")
    
    # Test response generation
    print("\n" + "="*60)
    print("TESTING CRISIS RESPONSE GENERATION")
    print("="*60)
    
    categories = ['stressed', 'alertness', 'suicidal', 'racism', 'out_of_topic']
    for category in categories:
        response = CrisisResponseGenerator.generate_response(
            category, 
            confidence=0.85, 
            intensity=0.7
        )
        print(f"\n{category.upper()}:")
        print(f"  Response: {response['main_response']}")
        print(f"  Hotline: {response['hotline']}")
    
    print("\n✓ Model design testing completed!")


if __name__ == "__main__":
    main()
