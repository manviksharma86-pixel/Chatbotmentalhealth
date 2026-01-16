"""
Standalone sample inputs - Tests API functions without requiring server running
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.text_preprocessing import TextPreprocessor, CrisisIntentAnalyzer

# Initialize processors
text_processor = TextPreprocessor()
crisis_analyzer = CrisisIntentAnalyzer()

text_samples = {
    "suicidal_crisis": "I can't take this anymore. I've been thinking about ending it all. Everything hurts and I don't see any point in living. I have a plan and I'm ready to do it.",
    "severe_stress": "I'm completely overwhelmed. Work is killing me, my family is falling apart, and I can't sleep. I don't know how much longer I can handle this pressure. I feel like I'm breaking down.",
    "insomnia": "I haven't slept properly in weeks. Every night I lie awake for hours, my mind racing. I'm exhausted but can't fall asleep no matter what I try. This is affecting everything.",
    "anxiety_disorder": "I have constant panic attacks. My heart races, I can't breathe, and I feel like something terrible is about to happen. I'm terrified and I can't go anywhere without feeling this way.",
    "depression": "I feel empty and hopeless. Nothing interests me anymore. I've lost all motivation and energy. Even getting out of bed feels impossible. I don't know why I'm still here.",
    "racism_trauma": "I experienced racism today at work. They made hateful comments and excluded me from meetings. This has been ongoing and I feel isolated, angry, and hurt. I don't know how to cope with this anymore.",
    "grief_loss": "I lost my best friend last month and I can't stop crying. Everything reminds me of them. I feel lost without them and I don't know how to move forward.",
    "healthy_check_in": "I've been having a good week. My anxiety has been manageable and I've been able to enjoy time with my friends. I'm feeling hopeful about my future.",
    "help_request": "I need help. I don't know what to do. I'm confused and feel stuck. Can someone help me?",
    "out_of_topic": "What's the capital of France? I need help with my homework."
}

print("\n" + "="*70)
print("MENTAL HEALTH CRISIS CHATBOT - STANDALONE TEXT ANALYSIS TEST".center(70))
print("="*70)

for scenario, text in text_samples.items():
    print(f"\n📝 Testing: {scenario.upper()}")
    print(f"Input: {text[:75]}...")
    
    try:
        # Test with full analysis
        analysis = crisis_analyzer.analyze_full(text)
        
        print(f"✅ Crisis Type: {analysis['crisis_type']}")
        print(f"⚠️  Urgency Level: {analysis['urgency_level']}")
        print(f"📊 Confidence: {analysis['confidence']:.2%}")
        print(f"💬 Message: {analysis['message'][:60]}...")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")

print("\n" + "="*70)
print("✅ Standalone tests completed!".center(70))
print("="*70 + "\n")
