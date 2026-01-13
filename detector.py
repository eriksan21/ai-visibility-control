"""
Face bounding box detector using OpenCV Haar Cascade.
NO face recognition. NO identity matching. Only bbox detection.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional

class FaceDetector:
    """Detects face bounding boxes using classical CV (non-biometric)."""
    
    def __init__(self):
        # Load OpenCV's pre-trained Haar Cascade (classical, not deep learning)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
    
    def detect_faces(self, img: np.ndarray) -> List[dict]:
        """
        Detect face regions (bbox only, no recognition).
        
        Returns list of face regions with sub-zones:
        [{
            'bbox': (x, y, w, h),
            'eyes': [(x, y, w, h), ...],
            'nose_bridge': (x, y, w, h)
        }]
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect face bboxes
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            return []
        
        results = []
        for (x, y, w, h) in faces:
            face_region = {
                'bbox': (x, y, w, h),
                'eyes': [],
                'nose_bridge': None
            }
            
            # Detect eyes within face region
            face_gray = gray[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(
                face_gray,
                scaleFactor=1.1,
                minNeighbors=10,
                minSize=(15, 15)
            )
            
            # Convert eye coordinates to global image space
            for (ex, ey, ew, eh) in eyes:
                face_region['eyes'].append((x + ex, y + ey, ew, eh))
            
            # Estimate nose bridge (upper-middle region of face)
            # Simple heuristic: 30-60% down from top, centered horizontally
            nose_y = y + int(h * 0.3)
            nose_h = int(h * 0.3)
            nose_x = x + int(w * 0.35)
            nose_w = int(w * 0.3)
            face_region['nose_bridge'] = (nose_x, nose_y, nose_w, nose_h)
            
            results.append(face_region)
        
        return results
    
    def get_processing_zones(self, face: dict) -> List[Tuple[str, Tuple[int, int, int, int]]]:
        """
        Extract prioritized zones for processing.
        Returns: [(zone_name, (x, y, w, h)), ...]
        """
        zones = []
        
        # Priority 1: Eyes (highest impact on AI recognition)
        for eye_bbox in face['eyes']:
            zones.append(('eye', eye_bbox))
        
        # Priority 2: Nose bridge
        if face['nose_bridge']:
            zones.append(('nose_bridge', face['nose_bridge']))
        
        return zones
