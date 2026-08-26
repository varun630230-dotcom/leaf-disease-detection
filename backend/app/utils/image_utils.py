import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

def load_image(path: str | Path) -> np.ndarray:
    """Load image from path and convert to RGB numpy array."""
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Failed to load image from {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def save_image(path: str | Path, image_array: np.ndarray) -> bool:
    """Save RGB image array to path."""
    bgr_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    return cv2.imwrite(str(path), bgr_image)

def compute_blur_score(image: np.ndarray) -> float:
    """Compute blur score using Laplacian variance."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def compute_brightness(image: np.ndarray) -> float:
    """Compute average brightness of the image."""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    return float(np.mean(hsv[:, :, 2]))

def compute_contrast(image: np.ndarray) -> float:
    """Compute contrast of the image based on standard deviation of intensity."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return float(gray.std())

def resize_image(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize image to given size (width, height)."""
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

def create_overlay(original: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int] = (255, 0, 0), alpha: float = 0.4) -> np.ndarray:
    """Create a colored overlay on the original image where mask is > 0."""
    overlay = original.copy()
    colored_mask = np.zeros_like(original)
    colored_mask[mask > 0] = color
    
    cv2.addWeighted(colored_mask, alpha, overlay, 1 - alpha, 0, overlay)
    
    # Only keep the blended parts where the mask was active
    result = original.copy()
    result[mask > 0] = overlay[mask > 0]
    return result

def create_heatmap_overlay(original: np.ndarray, heatmap: np.ndarray, colormap: int = cv2.COLORMAP_JET, alpha: float = 0.4) -> np.ndarray:
    """Create a heatmap overlay on the original image."""
    # Normalize heatmap to 0-255
    heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap_norm, colormap)
    heatmap_colored_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Blend images
    blended = cv2.addWeighted(heatmap_colored_rgb, alpha, original, 1 - alpha, 0)
    return blended
