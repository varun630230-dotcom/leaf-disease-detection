"""LeafGuard AI — Preprocessing and Image Normalization."""

from pathlib import Path
from typing import Tuple, Union
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms


class ImagePreprocessor:
    """Standardizes input images to normalized ImageNet tensors and display arrays."""

    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

    def load_image(self, image_source: Union[str, Path, np.ndarray, Image.Image]) -> Image.Image:
        if isinstance(image_source, (str, Path)):
            return Image.open(image_source).convert("RGB")
        elif isinstance(image_source, np.ndarray):
            return Image.fromarray(image_source).convert("RGB")
        elif isinstance(image_source, Image.Image):
            return image_source.convert("RGB")
        else:
            raise ValueError(f"Unsupported image source type: {type(image_source)}")

    def preprocess(self, image_source: Union[str, Path, np.ndarray, Image.Image]) -> Tuple[np.ndarray, torch.Tensor]:
        """
        Preprocesses an image for inference and Grad-CAM.
        Returns:
            raw_numpy: (H, W, 3) float32 in [0, 1] range for visualization.
            tensor: (1, 3, H, W) normalized float32 tensor for PyTorch models.
        """
        img = self.load_image(image_source)

        # Raw resized RGB numpy array for display and Grad-CAM overlays
        raw_img = img.resize(self.target_size)
        raw_numpy = np.array(raw_img, dtype=np.float32) / 255.0

        # Normalized inference tensor
        tensor = self.transform(img).unsqueeze(0)

        return raw_numpy, tensor
