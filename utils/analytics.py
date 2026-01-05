"""
Analytics utilities for the chatbot
"""

import pandas as pd
from datetime import datetime
from collections import Counter

class Analytics:
    """Analytics for conversation and detection data"""
    
    @staticmethod
    def get_session_stats(conversation_history, detection_results):
        """Calculate session statistics"""
        total_messages = sum(1 for m in conversation_history if m['type'] == 'user')
        total_responses = sum(1 for m in conversation_history if m['type'] == 'assistant')
        
        if detection_results:
            avg_confidence = sum(d['confidence'] for d in detection_results) / len(detection_results)
            crisis_detections = sum(1 for d in detection_results if d['crisis_type'] != 'out_of_topic')
        else:
            avg_confidence = 0
            crisis_detections = 0
        
        return {
            'total_messages': total_messages,
            'total_responses': total_responses,
            'avg_confidence': avg_confidence,
            'crisis_detections': crisis_detections,
            'detection_rate': crisis_detections / total_messages if total_messages > 0 else 0
        }
    
    @staticmethod
    def get_crisis_distribution(detection_results):
        """Get crisis type distribution"""
        if not detection_results:
            return {}
        
        crisis_types = [d['crisis_type'] for d in detection_results]
        return dict(Counter(crisis_types))
    
    @staticmethod
    def get_urgency_distribution(detection_results):
        """Get urgency level distribution"""
        if not detection_results:
            return {}
        
        urgency_levels = [d.get('urgency', 'low') for d in detection_results]
        return dict(Counter(urgency_levels))
    
    @staticmethod
    def get_confidence_trend(detection_results):
        """Get confidence scores over time"""
        if not detection_results:
            return []
        
        return [d['confidence'] for d in detection_results]
    
    @staticmethod
    def get_most_common_crisis(detection_results):
        """Get the most frequently detected crisis type"""
        if not detection_results:
            return None
        
        crisis_types = [d['crisis_type'] for d in detection_results if d['crisis_type'] != 'out_of_topic']
        if not crisis_types:
            return None
        
        return max(set(crisis_types), key=crisis_types.count)
    
    @staticmethod
    def get_highest_confidence_detection(detection_results):
        """Get the highest confidence detection"""
        if not detection_results:
            return None
        
        return max(detection_results, key=lambda x: x['confidence'])
    
    @staticmethod
    def get_session_duration(conversation_history):
        """Calculate session duration"""
        if len(conversation_history) < 2:
            return 0
        
        first_time = datetime.fromisoformat(conversation_history[0]['timestamp'])
        last_time = datetime.fromisoformat(conversation_history[-1]['timestamp'])
        
        duration = (last_time - first_time).total_seconds()
        return duration
    
    @staticmethod
    def get_critical_alerts(detection_results):
        """Get critical urgency detections"""
        return [d for d in detection_results if d.get('urgency') == 'critical']
    
    @staticmethod
    def get_session_summary(conversation_history, detection_results):
        """Get complete session summary"""
        stats = Analytics.get_session_stats(conversation_history, detection_results)
        duration = Analytics.get_session_duration(conversation_history)
        critical_alerts = Analytics.get_critical_alerts(detection_results)
        most_common = Analytics.get_most_common_crisis(detection_results)
        highest_confidence = Analytics.get_highest_confidence_detection(detection_results)
        
        return {
            'stats': stats,
            'duration': duration,
            'critical_alerts': len(critical_alerts),
            'most_common_crisis': most_common,
            'highest_confidence_detection': highest_confidence,
            'crisis_distribution': Analytics.get_crisis_distribution(detection_results),
            'urgency_distribution': Analytics.get_urgency_distribution(detection_results)
        }
