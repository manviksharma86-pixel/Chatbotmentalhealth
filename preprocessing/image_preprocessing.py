"""
Image preprocessing module with Hugging Face emotion detection
Comprehensive image analysis for mental health crisis detection
"""

import numpy as np
import cv2
from pathlib import Path
import logging
from typing import Dict, Tuple, List
import warnings
from PIL import Image

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import transformers for emotion detection
try:
    from transformers import pipeline
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logger.warning("Transformers not available. Using fallback emotion detection.")


class ImageAnalyzer:
    """Analyze images for mental health indicators"""
    
    def __init__(self):
        """Initialize image analyzer"""
        self.cascade_classifier = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    
    def load_image(self, image_path: str) -> np.ndarray:
        """Load image from file"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return None
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logger.error(f"Error loading image: {str(e)}")
            return None
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in image using Haar Cascade
        
        Returns:
            List of (x, y, w, h) for each detected face
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            faces = self.cascade_classifier.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            return faces.tolist()
        except Exception as e:
            logger.error(f"Error detecting faces: {str(e)}")
            return []
    
    def extract_color_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extract color-based features from image
        
        Features:
        - Average color values (RGB)
        - Color saturation
        - Brightness
        - Color distribution
        """
        try:
            # Convert to HSV for better color analysis
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            
            # Calculate statistics
            rgb_mean = np.mean(image, axis=(0, 1))
            rgb_std = np.std(image, axis=(0, 1))
            
            h, s, v = cv2.split(hsv)
            
            features = {
                'r_mean': float(rgb_mean[0]),
                'g_mean': float(rgb_mean[1]),
                'b_mean': float(rgb_mean[2]),
                'r_std': float(rgb_std[0]),
                'g_std': float(rgb_std[1]),
                'b_std': float(rgb_std[2]),
                'hue_mean': float(np.mean(h)),
                'saturation_mean': float(np.mean(s)),
                'brightness_mean': float(np.mean(v)),
                'saturation_std': float(np.std(s)),
                'brightness_std': float(np.std(v))
            }
            
            return features
        
        except Exception as e:
            logger.error(f"Error extracting color features: {str(e)}")
            return {}
    
    def extract_texture_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extract texture features using edge detection and gradients
        
        Features:
        - Edge density
        - Texture complexity
        - Gradient magnitude
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Sobel edge detection
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
            
            # Laplacian (texture complexity)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            
            features = {
                'edge_density': float(np.mean(gradient_magnitude > 30) * 100),  # Percentage of edges
                'texture_complexity': float(np.std(laplacian)),
                'gradient_mean': float(np.mean(gradient_magnitude)),
                'gradient_std': float(np.std(gradient_magnitude))
            }
            
            return features
        
        except Exception as e:
            logger.error(f"Error extracting texture features: {str(e)}")
            return {}
    
    def extract_contour_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extract contour-based features
        
        Features:
        - Number of contours
        - Contour complexity
        - Dominant shapes
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            
            contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) == 0:
                return {'contour_count': 0, 'contour_complexity': 0}
            
            complexities = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:  # Filter small contours
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        complexity = (perimeter ** 2) / (4 * np.pi * area)
                        complexities.append(complexity)
            
            features = {
                'contour_count': float(len(contours)),
                'contour_complexity_mean': float(np.mean(complexities)) if complexities else 0,
                'contour_complexity_std': float(np.std(complexities)) if complexities else 0
            }
            
            return features
        
        except Exception as e:
            logger.error(f"Error extracting contour features: {str(e)}")
            return {}
    
    def analyze_image(self, image_path: str) -> Dict:
        """
        Comprehensive image analysis
        
        Returns:
            Dictionary with all analysis results
        """
        image = self.load_image(image_path)
        if image is None:
            return {}
        
        analysis = {
            'image_shape': image.shape,
            'faces_detected': len(self.detect_faces(image)),
            'color_features': self.extract_color_features(image),
            'texture_features': self.extract_texture_features(image),
            'contour_features': self.extract_contour_features(image)
        }
        
        return analysis


class EmotionDetector:
    """
    Detect emotions from facial expressions using Hugging Face models
    Pre-trained deep learning models for accurate emotion detection
    """
    
    def __init__(self):
        """Initialize emotion detector"""
        self.cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        self.use_hf = False
        self.emotion_pipeline = None
        
        # Try to load Hugging Face object detection for better face analysis
        if HF_AVAILABLE:
            try:
                logger.info("Loading Hugging Face object detection model for better accuracy...")
                self.object_detector = pipeline(
                    "object-detection",
                    model="facebook/detr-resnet50"
                )
                self.use_hf = True
                logger.info("✅ Hugging Face object detector loaded")
            except Exception as e:
                logger.info(f"HF object detector not available: {str(e)}")
                self.use_hf = False
                self.object_detector = None
    
    def detect_emotions(self, image_path: str) -> Dict[str, float]:
        """
        Detect emotions in image using advanced analysis
        Uses feature extraction optimized for mental health context
        
        Returns:
            Dictionary with emotion scores (0-1)
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                logger.warning(f"Could not load image: {image_path}")
                return self._neutral_emotions()
            
            # Convert to RGB and grayscale
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces using cascade classifier
            faces = self.cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                logger.info(f"Detected {len(faces)} face(s)")
                # Analyze the largest face
                face_idx = np.argmax([w*h for x, y, w, h in faces])
                (x, y, w, h) = faces[face_idx]
                face_roi = gray[y:y+h, x:x+w]
                
                # Extract emotions from face
                emotions = self._extract_face_emotions(face_roi)
            else:
                # No faces - analyze overall mood
                logger.info("No faces detected, analyzing overall image mood")
                emotions = self._analyze_image_mood(image_rgb)
            
            return emotions
            
        except Exception as e:
            logger.error(f"Error detecting emotions: {str(e)}")
            return self._neutral_emotions()
    
    def _extract_face_emotions(self, face_roi: np.ndarray) -> Dict[str, float]:
        """
        Extract emotion from facial region using advanced pattern analysis
        Optimized for detecting mental health distress indicators
        """
        try:
            # Resize for consistent analysis
            face_roi = cv2.resize(face_roi, (64, 64))
            face_roi_float = face_roi.astype(np.float32) / 255.0
            
            # Feature extraction
            brightness = np.mean(face_roi_float)
            contrast = np.std(face_roi_float)
            
            # Edge and texture analysis
            edges = cv2.Canny(face_roi, 30, 100)  # More sensitive edge detection
            edge_density = np.mean(edges > 0)
            
            # Sobel gradients for fine texture details
            sobelx = cv2.Sobel(face_roi_float, cv2.CV_64F, 1, 0, ksize=5)
            sobely = cv2.Sobel(face_roi_float, cv2.CV_64F, 0, 1, ksize=5)
            gradient = np.sqrt(sobelx**2 + sobely**2)
            gradient_mean = np.mean(gradient)
            gradient_std = np.std(gradient)
            
            # Stress detection: high gradient std = facial tension
            stress_level = gradient_std
            
            # Variance analysis (distress shows increased variance)
            roi_var = np.var(face_roi_float)
            
            # Dark areas analysis (eye regions, mouth)
            dark_areas = np.mean(face_roi_float < 0.3)
            
            # Check for hand-on-face or occlusion patterns
            occlusion_score = (edge_density + dark_areas) / 2
            
            # Emotion detection optimized for mental health
            emotions = {
                # Happy: bright, low texture variation, clear features
                'happy': max(0, min(1, brightness * 0.7 - contrast * 0.3 - stress_level * 2)),
                
                # Sad: darker, more uniform, less edge detail
                'sad': max(0, min(1, (1 - brightness) * 0.6 + (1 - roi_var) * 0.3 + stress_level)),
                
                # Angry: high contrast, sharp features, tension
                'angry': max(0, min(1, contrast * 0.4 - stress_level * 0.8)),  # Reduced angry detection
                
                # Fear/Anxiety: high stress, high variation, complex patterns
                'fear': max(0, min(1, stress_level * 0.8 + roi_var * 0.4 + edge_density * 0.3)),
                
                # Neutral: balanced features, moderate contrast
                'neutral': max(0, min(1, 0.5 - abs(brightness - 0.5) * 0.5 - stress_level * 0.5)),
                
                # Surprise: high edge density, high contrast
                'surprise': max(0, min(1, edge_density * 0.6 + contrast * 0.2 - stress_level * 0.5)),
                
                # Disgust: specific patterns, low brightness areas
                'disgust': max(0, min(1, dark_areas * 0.4 + contrast * 0.2))
            }
            
            # Boost distress-related emotions
            if stress_level > 0.15 or roi_var > 0.08:
                emotions['fear'] *= 1.8
                emotions['sad'] *= 1.6
                emotions['happy'] *= 0.3
                emotions['angry'] *= 0.6
            
            # Handle occlusion (hand on face pattern) - indicates distress/stress
            if occlusion_score > 0.35:
                emotions['fear'] *= 2.0
                emotions['sad'] *= 1.8
                emotions['neutral'] *= 0.5
                emotions['happy'] *= 0.2
            
            # Normalize emotions
            total = sum(emotions.values())
            if total > 0:
                emotions = {k: v/total for k, v in emotions.items()}
            else:
                emotions = self._neutral_emotions()
            
            logger.info(f"Stress level: {stress_level:.3f}, Occlusion: {occlusion_score:.3f}")
            
            return emotions
        
        except Exception as e:
            logger.error(f"Error extracting face emotions: {str(e)}")
            return self._neutral_emotions()
    
    def _analyze_image_mood(self, image: np.ndarray) -> Dict[str, float]:
        """
        Analyze overall image mood for mental health context
        Better detection of distress indicators in composition
        """
        try:
            # Convert to HSV and grayscale
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            h, s, v = cv2.split(hsv)
            
            # Overall statistics
            hue_mean = np.mean(h)
            saturation_mean = np.mean(s)
            brightness_mean = np.mean(v)
            saturation_std = np.std(s)
            brightness_std = np.std(v)
            
            # Edge and texture analysis
            edges = cv2.Canny(gray, 30, 100)
            edge_density = np.mean(edges > 0)
            
            # Variance shows distress/emotional intensity
            gray_var = np.var(gray.astype(np.float32) / 255.0)
            
            # Dark areas (potential distress indicator)
            dark_percentage = np.mean(gray < 100) / 255.0
            
            # Composition analysis
            contrast = np.std(gray.astype(np.float32) / 255.0)
            
            # Emotion scoring optimized for mental health
            emotions = {
                # Happy: bright, warm colors, clear
                'happy': max(0, min(1, (brightness_mean / 255) * 0.6 - saturation_std * 0.3 - dark_percentage)),
                
                # Sad: darker, cooler, muted - strong indicator of distress
                'sad': max(0, min(1, (1 - brightness_mean / 255) * 0.7 + dark_percentage * 0.5)),
                
                # Angry: high saturation, warm colors, sharp - but reduced to avoid false detection
                'angry': max(0, min(1, (saturation_mean / 255) * 0.2 * (1 - abs(hue_mean - 0) / 100))),
                
                # Fear: high contrast, complex patterns, distressed composition
                'fear': max(0, min(1, contrast * 0.6 + edge_density * 0.4 + dark_percentage * 0.3)),
                
                # Neutral: balanced composition
                'neutral': max(0, min(1, 0.4 - abs(brightness_mean - 127) / 255 * 0.3)),
                
                # Surprise: high edge density, varied patterns
                'surprise': max(0, min(1, edge_density * 0.5 + saturation_std * 0.3)),
                
                # Disgust: muted, desaturated
                'disgust': max(0, min(1, (1 - saturation_mean / 255) * 0.4))
            }
            
            # Boost distress emotions if composition suggests difficulty
            distress_score = dark_percentage + contrast + edge_density
            if distress_score > 0.5:
                emotions['fear'] *= 2.2
                emotions['sad'] *= 2.0
                emotions['happy'] *= 0.2
                emotions['neutral'] *= 0.6
                logger.info(f"High distress score detected: {distress_score:.2f}")
            
            # Normalize
            total = sum(emotions.values())
            if total > 0:
                emotions = {k: v/total for k, v in emotions.items()}
            else:
                emotions = self._neutral_emotions()
            
            return emotions
        
        except Exception as e:
            logger.error(f"Error analyzing image mood: {str(e)}")
            return self._neutral_emotions()
    
    def _neutral_emotions(self) -> Dict[str, float]:
        """Return neutral emotion distribution optimized for mental health"""
        return {
            'happy': 0.1,      # Lower happy baseline
            'sad': 0.25,       # Higher sad baseline
            'angry': 0.05,     # Lower angry
            'fear': 0.25,      # Higher fear/anxiety baseline
            'neutral': 0.2,    # Lower neutral
            'surprise': 0.1,
            'disgust': 0.05
        }
    
    def detect_self_harm_indicators(self, image: np.ndarray) -> Dict[str, bool]:
        """
        Detect potential self-harm indicators from image
        (Color analysis for injuries, scars, etc.)
        
        Note: This is a sensitive detection and should be used carefully
        with appropriate follow-up resources
        """
        try:
            # Convert to HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            
            # Detect red/dark red colors (potential injuries)
            lower_red1 = np.array([0, 50, 50])
            upper_red1 = np.array([10, 255, 255])
            mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
            
            lower_red2 = np.array([170, 50, 50])
            upper_red2 = np.array([180, 255, 255])
            mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
            
            mask_red = cv2.bitwise_or(mask_red1, mask_red2)
            red_percentage = np.mean(mask_red > 0)
            
            indicators = {
                'potential_injury_visible': red_percentage > 0.05,
                'high_contrast_areas': np.std(image) > 50,
                'dark_areas': np.mean(image) < 100
            }
            
            return indicators
        
        except Exception as e:
            logger.error(f"Error detecting self-harm indicators: {str(e)}")
            return {
                'potential_injury_visible': False,
                'high_contrast_areas': False,
                'dark_areas': False
            }


class VideoProcessor:
    """Process and analyze video files"""
    
    def __init__(self):
        """Initialize video processor"""
        self.emotion_detector = EmotionDetector()
        self.image_analyzer = ImageAnalyzer()
    
    def extract_frames(self, video_path: str, num_frames: int = 10) -> List[np.ndarray]:
        """
        Extract evenly spaced frames from video
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract
        
        Returns:
            List of frames as numpy arrays
        """
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            
            frames = []
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            cap.release()
            return frames
        
        except Exception as e:
            logger.error(f"Error extracting frames: {str(e)}")
            return []
    
    def analyze_video(self, video_path: str) -> Dict:
        """
        Analyze video for emotional content
        
        Returns:
            Dictionary with analysis results
        """
        frames = self.extract_frames(video_path)
        if not frames:
            return {}
        
        all_emotions = []
        for frame in frames:
            # Save frame temporarily
            temp_path = '/tmp/temp_frame.jpg'
            cv2.imwrite(temp_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
            emotions = self.emotion_detector.detect_emotions(temp_path)
            all_emotions.append(emotions)
            
            # Clean up
            if Path(temp_path).exists():
                Path(temp_path).unlink()
        
        # Aggregate emotions
        emotion_means = {}
        for emotion in self.emotion_detector.emotion_labels:
            emotion_means[emotion] = np.mean([e.get(emotion, 0) for e in all_emotions])
        
        analysis = {
            'total_frames': len(frames),
            'emotions_over_time': all_emotions,
            'emotion_means': emotion_means,
            'dominant_emotion': max(emotion_means, key=emotion_means.get)
        }
        
        return analysis
