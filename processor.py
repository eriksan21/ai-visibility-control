"""
Core image processing for AI visibility control.
Applies localized transformations to face sub-zones.
"""

import cv2
import numpy as np
from typing import Tuple, List
from dataclasses import dataclass

@dataclass
class ProcessingMode:
    """Privacy mode configuration."""
    name: str
    blur_radius: int  # Gaussian blur kernel size (odd number)
    noise_strength: float  # Luminance noise intensity (0-1)
    asymmetry_shift: int  # Pixel shift for asymmetry
    
    @staticmethod
    def get_preset(mode: str) -> 'ProcessingMode':
        presets = {
            'social_safe': ProcessingMode(
                name='Social Safe',
                blur_radius=3,
                noise_strength=0.02,
                asymmetry_shift=1
            ),
            'genai_safe': ProcessingMode(
                name='GenAI Safe',
                blur_radius=5,
                noise_strength=0.05,
                asymmetry_shift=2
            ),
            'max_privacy': ProcessingMode(
                name='Max Privacy',
                blur_radius=7,
                noise_strength=0.08,
                asymmetry_shift=3
            )
        }
        return presets.get(mode, presets['genai_safe'])


class ImageProcessor:
    """Processes face zones to reduce AI recognition consistency."""
    
    def __init__(self, mode: ProcessingMode):
        self.mode = mode
    
    def process_image(self, img: np.ndarray, zones: List[Tuple[str, Tuple[int, int, int, int]]]) -> np.ndarray:
        """
        Apply transformations to face zones.
        
        Args:
            img: Input image (BGR format)
            zones: List of (zone_name, bbox) tuples
        
        Returns:
            Processed image
        """
        # Work on a copy
        result = img.copy()
        
        for zone_name, (x, y, w, h) in zones:
            # Extract zone with padding for smooth blending
            pad = 5
            x_start = max(0, x - pad)
            y_start = max(0, y - pad)
            x_end = min(img.shape[1], x + w + pad)
            y_end = min(img.shape[0], y + h + pad)
            
            zone_img = result[y_start:y_end, x_start:x_end].copy()
            
            if zone_img.size == 0:
                continue
            
            # Apply transformations
            zone_processed = self._apply_blur(zone_img)
            zone_processed = self._apply_luminance_noise(zone_processed)
            zone_processed = self._apply_asymmetry(zone_processed, zone_name)
            
            # Blend back with feathered edges
            zone_blended = self._feather_edges(zone_img, zone_processed, feather_size=3)
            
            # Place back into result
            result[y_start:y_end, x_start:x_end] = zone_blended
        
        return result
    
    def _apply_blur(self, zone: np.ndarray) -> np.ndarray:
        """Apply neutral Gaussian blur (no color shift)."""
        # Ensure odd kernel size
        k = self.mode.blur_radius
        if k % 2 == 0:
            k += 1
        
        # Use bilateral filter for edge-preserving blur
        # This keeps human perception better while confusing AI
        blurred = cv2.bilateralFilter(zone, k, sigmaColor=75, sigmaSpace=75)
        return blurred
    
    def _apply_luminance_noise(self, zone: np.ndarray) -> np.ndarray:
        """Add subtle luminance-only noise (no hue/saturation change)."""
        # Convert to LAB color space (L = luminance)
        lab = cv2.cvtColor(zone, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0].astype(np.float32)
        
        # Generate noise pattern
        noise = np.random.normal(0, self.mode.noise_strength * 255, l_channel.shape)
        
        # Apply noise only to luminance
        l_channel = np.clip(l_channel + noise, 0, 255).astype(np.uint8)
        lab[:, :, 0] = l_channel
        
        # Convert back to BGR
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return result
    
    def _apply_asymmetry(self, zone: np.ndarray, zone_name: str) -> np.ndarray:
        """Apply slight asymmetric shift to break pattern matching."""
        h, w = zone.shape[:2]
        shift = self.mode.asymmetry_shift
        
        # Different shift direction based on zone
        if 'eye' in zone_name:
            # Horizontal shift for eyes
            M = np.float32([[1, 0, shift], [0, 1, 0]])
        else:
            # Vertical shift for nose
            M = np.float32([[1, 0, 0], [0, 1, shift]])
        
        shifted = cv2.warpAffine(zone, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        return shifted
    
    def _feather_edges(self, original: np.ndarray, processed: np.ndarray, 
                       feather_size: int) -> np.ndarray:
        """Blend processed zone with original using gradient mask."""
        h, w = original.shape[:2]
        
        # Create gradient mask (center = 1, edges = 0)
        mask = np.ones((h, w), dtype=np.float32)
        
        # Apply gradient falloff on edges
        for i in range(feather_size):
            alpha = i / feather_size
            mask[i, :] *= alpha
            mask[h-1-i, :] *= alpha
            mask[:, i] *= alpha
            mask[:, w-1-i] *= alpha
        
        # Expand mask to 3 channels
        mask = np.expand_dims(mask, axis=2)
        
        # Blend
        blended = (processed * mask + original * (1 - mask)).astype(np.uint8)
        return blended


def process_face_image(img_bytes: bytes, mode_name: str = 'genai_safe') -> Tuple[bytes, dict]:
    """
    Main processing function.
    
    Args:
        img_bytes: Input image as bytes
        mode_name: Privacy mode ('social_safe', 'genai_safe', 'max_privacy')
    
    Returns:
        (processed_image_bytes, metadata_dict)
    """
    from detector import FaceDetector
    
    # Decode image
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Invalid image format")
    
    # Detect faces
    detector = FaceDetector()
    faces = detector.detect_faces(img)
    
    if len(faces) == 0:
        raise ValueError("No faces detected in image")
    
    # Collect all processing zones
    all_zones = []
    for face in faces:
        zones = detector.get_processing_zones(face)
        all_zones.extend(zones)
    
    # Process image
    mode = ProcessingMode.get_preset(mode_name)
    processor = ImageProcessor(mode)
    result_img = processor.process_image(img, all_zones)
    
    # Encode back to bytes
    _, buffer = cv2.imencode('.jpg', result_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    result_bytes = buffer.tobytes()
    
    metadata = {
        'faces_processed': len(faces),
        'zones_modified': len(all_zones),
        'mode': mode.name,
        'settings': {
            'blur_radius': mode.blur_radius,
            'noise_strength': mode.noise_strength,
            'asymmetry_shift': mode.asymmetry_shift
        }
    }
    
    return result_bytes, metadata
