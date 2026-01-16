"""
Text Preprocessing Module for Mental Health Crisis Detection
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import torch

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


class TextPreprocessor:
    """Preprocess text data for mental health crisis classification"""
    
    def __init__(self, model_name='distilbert-base-uncased'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.max_length = 128  # Maximum sequence length
        
    def clean_text(self, text):
        """Clean and normalize text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters but keep punctuation for context
        text = re.sub(r'[^a-zA-Z0-9\s\.\!\?\-]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_and_lemmatize(self, text):
        """Tokenize and lemmatize text"""
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens 
            if token not in self.stop_words and len(token) > 2
        ]
        
        return ' '.join(tokens)
    
    def preprocess(self, text):
        """Complete preprocessing pipeline"""
        text = self.clean_text(text)
        text = self.tokenize_and_lemmatize(text)
        return text
    
    def tokenize_for_model(self, text, add_special_tokens=True, return_tensors='pt'):
        """Tokenize text for transformer model"""
        encoding = self.tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors=return_tensors
        )
        
        return encoding
    
    def batch_preprocess(self, texts):
        """Preprocess batch of texts"""
        return [self.preprocess(text) for text in texts]


class CrisisIntentAnalyzer:
    """Analyze intent and urgency from text with response generation"""
    
    # Keywords for each crisis category - EXPANDED
    CRISIS_KEYWORDS = {
        'stressed': [
            'stressed', 'pressure', 'overwhelming', 'anxious', 'panic', 'anxiety',
            'drowning', 'unbearable', 'tense', 'worried', 'overwhelm', 'overwhelmed',
            'burden', 'weight', 'crushing', 'suffocating', 'nervous', 'pressured',
            'stressed out', 'freaking out', 'falling apart', 'breaking down',
            'can\'t cope', 'too much', 'intense', 'exhausted', 'burnt out',
            'overwhelm', 'panic attack', 'anxious', 'nervous breakdown'
        ],
        'alertness': [
            'insomnia', "can't sleep", 'sleep', 'restless', 'racing mind',
            'jumpy', 'hypervigilant', 'jittery', 'on edge', 'alert', 'awake',
            'tired', 'exhausted', 'fatigue', 'sleepless', 'sleeping', 'rest',
            'days without sleep', 'no sleep', 'sleep deprived', 'can\'t rest',
            'wired', 'cannot sleep', 'insomnia', 'restless nights', 'tossing',
            'turning', 'exhaustion', 'drained', 'worn out'
        ],
        'suicidal': [
            'suicide', 'kill myself', 'end it all', 'hopeless', 'want to die',
            'harm myself', 'self harm', 'no point', 'end my life', 'give up',
            'better off dead', 'end everything', 'take my life', 'thoughts',
            'suicidal', 'killing myself', 'kill myself', 'slash', 'cut myself',
            'not worth living', 'should be dead', 'everyone better off', 'end it',
            'this pain', 'unbearable', 'can\'t live', 'don\'t want to live'
        ],
        'racism': [
            'racism', 'racist', 'discrimination', 'discriminated', 'racial',
            'race', 'skin color', 'targeted', 'prejudice', 'biased', 'bias',
            'hate crime', 'racial slur', 'ethnic', 'ethnicity', 'unfairly',
            'treated unfairly', 'color', 'minority', 'cultural', 'stereotype',
            'hate', 'intolerance', 'bigotry', 'racial abuse', 'ethnic cleansing'
        ],
        'abuse': [
            'abuse', 'abused', 'abuse', 'violent', 'violence', 'hurt', 'pain',
            'hit', 'beaten', 'hurt me', 'domestic violence', 'assault',
            'sexual assault', 'rape', 'harassed', 'bullied', 'bullying',
            'mistreated', 'abusive', 'toxic', 'manipulation', 'control'
        ] ,
        'help': [
            'help', 'help me', 'need help', 'please help', 'can you help me',
            'what should i do', 'what can i do', 'what do i do', 'what now',
            'i need help', 'i need guidance', 'guide me', 'show me the way',
            'i need advice', 'give me advice', 'i need direction',
            'i don’t know what to do', 'i dont know what to do',
            'i am confused', 'i feel lost', 'i am stuck', 'i feel stuck',
            'i need support', 'support me', 'i need someone',
            'i need to talk', 'i need to talk to someone',
            'how can i fix this', 'how do i fix this',
            'how do i handle this', 'how do i deal with this',
            'how can i cope', 'how can i manage this',
            'what are my options', 'what can help me',
            'what is the solution', 'any solution',
            'what is the next step', 'next steps',
            'i need guidance now', 'i need immediate help',
            'i am overwhelmed', 'i feel overwhelmed',
            'i am struggling', 'i am not okay',
            'can someone help me', 'is anyone there',
            'i need reassurance', 'i need clarity',
            'please guide me', 'please advise'
        ]

    }
    
    # Crisis responses with hotline numbers and guidance
    CRISIS_RESPONSES = {
        'stressed': {
            'message': "I understand you're feeling stressed and overwhelmed. Your feelings are valid and important.",
            'guidance': [
                "Take deep breaths - try the 4-7-8 breathing technique",
                "Break tasks into smaller, manageable steps",
                "Reach out to trusted friends or family for support",
                "Consider speaking with a mental health professional"
            ],
            'hotlines': {
                'Police Emergency': '000',
                'Lifeline': '131114',
                'Mental Health Crisis': '131165'
            }
        },
        'alertness': {
            'message': "Sleep and rest are crucial for mental health. I'm here to support you through this.",
            'guidance': [
                "Establish a consistent sleep schedule",
                "Avoid caffeine and screens before bedtime",
                "Try relaxation techniques like meditation or progressive muscle relaxation",
                "Consult a healthcare provider if insomnia persists"
            ],
            'hotlines': {
                'Sleep Health Hotline': '131114',
                'Mental Health Support': '131165',
                'Emergency Services': '000'
            }
        },
        'suicidal': {
            'message': "I'm truly concerned about your safety. Please reach out for immediate help. You matter, and there is always hope.",
            'guidance': [
                "CALL EMERGENCY SERVICES IMMEDIATELY: 000",
                "Contact a suicide prevention hotline right now",
                "Tell someone you trust - a friend, family member, or counselor",
                "Go to the nearest emergency room",
                "Remove access to means of self-harm if possible"
            ],
            'hotlines': {
                'SUICIDE & CRISIS LIFELINE (URGENT)': '131114',
                'Mental Health Crisis (24/7)': '131165',
                'Emergency Services (CALL NOW)': '000',
                'International Crisis Text Line': 'Text HOME to 741741'
            }
        },
        'racism': {
            'message': "I'm sorry you're experiencing racism or discrimination. This is not acceptable, and your experiences matter.",
            'guidance': [
                "Document incidents if safe to do so",
                "Report to appropriate authorities if you feel safe",
                "Seek support from community organizations",
                "Connect with support groups for shared experiences",
                "Consider counseling to process emotional impact"
            ],
            'hotlines': {
                'Police (if in danger)': '000',
                'Mental Health Support': '131165',
                'Crisis Lifeline': '131114',
                'Civil Rights Hotline': '1800 969 622'
            }
        },
        'abuse': {
            'message': "I'm deeply concerned about your safety. Abuse is never your fault, and you deserve support and protection.",
            'guidance': [
                "Call emergency services immediately if in immediate danger: 000",
                "Reach out to a domestic violence hotline for safety planning",
                "Get to a safe location away from the abuser",
                "Tell someone you trust - a friend, family, or counselor",
                "Document injuries and incidents for evidence if safe",
                "Create a safety plan with professional support"
            ],
            'hotlines': {
                'POLICE (EMERGENCY)': '000',
                'Domestic Violence Hotline': '1800 799 7233',
                'Crisis Lifeline (24/7)': '131114',
                'Mental Health Support': '131165',
                'Sexual Assault Support': '1800 211 211'
            }
        },
        'out_of_topic': {
            'message': "Hello! I'm here to assist you. I specialize in mental health support and crisis intervention.",
            'guidance': [
                "Feel free to share what's on your mind",
                "I can help with stress, anxiety, sleep issues, and crisis support",
                "All conversations are confidential and judgment-free",
                "If you need emergency help, please contact the numbers below"
            ],
            'hotlines': {
                'Mental Health Helpline': '131165',
                'Crisis Lifeline': '131114',
                'Emergency Services': '000'
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
        }

    }
    
    URGENCY_INDICATORS = {
        'critical': [
            'suicide', 'suicidal', 'kill', 'kill myself', 'harm myself', 'end my life', 
            'right now', 'immediately', 'now', 'abused', 'abuse', 'violence', 'violent',
            'emergency', 'danger', 'urgent help', 'please help', 'end it all',
            'thoughts', 'harm', 'hurt', 'assault', 'rape', 'beating', 'beaten'
        ],
        'high': [
            'hopeless', "can't take it", 'unbearable', 'emergency', 'urgent',
            'danger', 'severe', 'extreme', 'intense', 'panic attack', 'racism',
            'panic', 'severe anxiety', 'overwhelming', 'racist', 'discrimination',
            'discriminated', 'abuse', 'abused', 'discrimination', 'prejudice'
        ],
        'medium': [
            'stressed', 'stress', 'anxious', 'anxiety', 'overwhelmed', 'overwhelm',
            'worried', 'worry', 'depressed', 'depression', 'struggling',
            'struggling with', 'difficulty', 'hard time', 'sleep', 'insomnia',
            'can\'t sleep', 'tired', 'exhausted', 'fatigue', 'sleepless'
        ],
        'low': [
            'feeling', 'wondering', 'thinking about', 'curious', 'hello',
            'how', 'what', 'help'
        ]
    }
    
    @staticmethod
    def detect_intent(text):
        """Detect intent category from text using improved scoring"""
        text_lower = text.lower()
        intent_scores = {}
        
        for category, keywords in CrisisIntentAnalyzer.CRISIS_KEYWORDS.items():
            # Count keyword matches with word boundary consideration
            score = 0
            for keyword in keywords:
                # Check for whole word match to avoid false positives
                if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
                    score += 1
            intent_scores[category] = score
        
        if max(intent_scores.values()) == 0:
            return 'out_of_topic', 0.1
        
        detected_intent = max(intent_scores, key=intent_scores.get)
        # Better confidence calculation
        keyword_matches = intent_scores[detected_intent]
        total_keywords = len(CrisisIntentAnalyzer.CRISIS_KEYWORDS[detected_intent])
        confidence = min((keyword_matches / total_keywords) + 0.3, 1.0)
        
        return detected_intent, confidence
    
    @staticmethod
    def detect_urgency(text):
        """Detect urgency level of the message (checks from most to least urgent)"""
        text_lower = text.lower()
        
        # Check in order of urgency: critical → high → medium → low
        urgency_order = ['critical', 'high', 'medium', 'low']
        
        for urgency_level in urgency_order:
            indicators = CrisisIntentAnalyzer.URGENCY_INDICATORS[urgency_level]
            if any(re.search(r'\b' + re.escape(indicator) + r'\b', text_lower) for indicator in indicators):
                return urgency_level
        
        return 'low'
    
    @staticmethod
    def generate_response(intent, urgency):
        """Generate appropriate response based on intent and urgency"""
        if intent not in CrisisIntentAnalyzer.CRISIS_RESPONSES:
            intent = 'out_of_topic'
        
        response_template = CrisisIntentAnalyzer.CRISIS_RESPONSES[intent]
        
        response = {
            'crisis_type': intent,
            'urgency_level': urgency,
            'message': response_template['message'],
            'guidance': response_template.get('guidance', []),
            'hotlines': response_template.get('hotlines', {}),
            'confidence': 0.85 if intent != 'out_of_topic' else 0.5
        }
        
        # Add optional fields if they exist (for 'help' intent)
        if 'prompts' in response_template:
            response['prompts'] = response_template['prompts']
        if 'support' in response_template:
            response['support'] = response_template['support']
        
        # Add urgency-based warnings
        if urgency == 'critical' or intent == 'suicidal':
            response['warning'] = "⚠️ URGENT: This appears to be a crisis situation. Please contact emergency services immediately."
        elif urgency == 'high':
            response['warning'] = "⚠️ HIGH PRIORITY: Please consider reaching out to a professional or hotline."
        
        return response
    
    @staticmethod
    def analyze_full(text):
        """Complete analysis returning formatted response"""
        intent, confidence = CrisisIntentAnalyzer.detect_intent(text)
        urgency = CrisisIntentAnalyzer.detect_urgency(text)
        response = CrisisIntentAnalyzer.generate_response(intent, urgency)
        response['confidence'] = confidence
        return response
    
    @staticmethod
    def extract_features(text):
        """Extract text-based features for analysis"""
        features = {
            'char_count': len(text),
            'word_count': len(text.split()),
            'sentence_count': len(re.split(r'[.!?]+', text)),
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'ellipsis_count': text.count('...'),
            'has_caps': any(c.isupper() for c in text),
            'avg_word_length': np.mean([len(word) for word in text.split()]) if text.split() else 0,
        }
        
        return features


def preprocess_dataset(df, output_path='data/processed/preprocessed_data.csv'):
    """Preprocess entire dataset"""
    
    preprocessor = TextPreprocessor()
    analyzer = CrisisIntentAnalyzer()
    
    # Clean text
    df['text_cleaned'] = df['text'].apply(preprocessor.preprocess)
    
    # Detect intent and urgency
    intent_results = df['text'].apply(analyzer.detect_intent)
    df['detected_intent'] = intent_results.apply(lambda x: x[0])
    df['intent_confidence'] = intent_results.apply(lambda x: x[1])
    
    df['urgency'] = df['text'].apply(analyzer.detect_urgency)
    
    # Extract features
    features_list = df['text'].apply(analyzer.extract_features).apply(pd.Series)
    df = pd.concat([df, features_list], axis=1)
    
    # Save preprocessed data
    df.to_csv(output_path, index=False)
    print(f"Preprocessed dataset saved to {output_path}")
    
    return df


def create_data_loaders(df, batch_size=16, test_size=0.2, val_size=0.1):
    """Create train/validation/test splits"""
    
    # Split into train and temp
    train_data, temp_data = train_test_split(
        df, test_size=test_size + val_size, random_state=42
    )
    
    # Split temp into validation and test
    val_data, test_data = train_test_split(
        temp_data, test_size=test_size / (test_size + val_size), random_state=42
    )
    
    print(f"\nData Split Statistics:")
    print(f"Training samples: {len(train_data)} ({len(train_data)/len(df)*100:.1f}%)")
    print(f"Validation samples: {len(val_data)} ({len(val_data)/len(df)*100:.1f}%)")
    print(f"Test samples: {len(test_data)} ({len(test_data)/len(df)*100:.1f}%)")
    
    return {
        'train': train_data,
        'validation': val_data,
        'test': test_data
    }


def main():
    print("Text Preprocessing Module")
    print("="*60)
    
    # Load dataset
    df = pd.read_csv('data/raw/crisis_dataset.csv')
    
    print(f"\nOriginal dataset shape: {df.shape}")
    print(f"Sample text: {df['text'].iloc[0][:100]}...")
    
    # Preprocess
    df_processed = preprocess_dataset(df)
    
    print(f"\nProcessed dataset shape: {df_processed.shape}")
    print(f"Cleaned text: {df_processed['text_cleaned'].iloc[0][:100]}...")
    
    # Show intent detection
    print("\n" + "="*60)
    print("INTENT DETECTION ANALYSIS")
    print("="*60)
    intent_dist = df_processed['detected_intent'].value_counts()
    print("\nDetected Intent Distribution:")
    print(intent_dist)
    
    # Show urgency distribution
    print("\nUrgency Distribution:")
    print(df_processed['urgency'].value_counts())
    
    # Create data splits
    data_splits = create_data_loaders(df_processed)
    
    # Save splits
    data_splits['train'].to_csv('data/processed/train_data.csv', index=False)
    data_splits['validation'].to_csv('data/processed/val_data.csv', index=False)
    data_splits['test'].to_csv('data/processed/test_data.csv', index=False)
    
    print("\n✓ Text preprocessing completed!")


if __name__ == "__main__":
    main()
