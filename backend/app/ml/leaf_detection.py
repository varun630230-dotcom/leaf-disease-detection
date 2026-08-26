"""LeafGuard AI — Dedicated Leaf & Plant Detection Module.

Accurately distinguishes plant leaves from:
- Vehicles (Cars, Trucks, Bikes)
- Domestic Animals & Wildlife (Dogs, Cats, Birds)
- Human Faces, People & Apparel
- Buildings, Furniture & Indoor Rooms
- Electronics, Screens & Web Screenshots
- Food, Tools & Random Man-Made Objects

Uses a combination of:
1. ImageNet Semantic Pretrained Network (MobileNetV3)
2. Central ROI Botanical Feature Extraction (Excess Green, Hue Distribution)
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class LeafDetectionResult:
    leaf_detected: bool
    confidence: float
    reason: Optional[str] = None
    detected_category: Optional[str] = None


# Explicitly non-plant ImageNet keywords
NON_PLANT_KEYWORDS = {
    # Vehicles & parts
    "car", "automobile", "cab", "convertible", "coupe", "jeep", "limousine", "minivan",
    "racer", "sports car", "station wagon", "pickup", "trailer", "truck", "van",
    "bus", "ambulance", "fire engine", "police van", "recreational vehicle", "tow truck",
    "motorcycle", "moped", "scooter", "bicycle", "tricycle", "unicycle",
    "airplane", "airliner", "wing", "helicopter", "balloon", "airship",
    "boat", "canoe", "gondola", "speedboat", "lifeboat", "submarine", "ship",
    "grille", "car wheel", "bumper", "odometer", "seat belt",
    
    # Animals
    "dog", "retriever", "terrier", "hound", "shepherd", "bulldog", "poodle", "pug",
    "cat", "tabby", "siamese", "persian", "cougar", "lynx", "leopard", "jaguar", "lion", "tiger", "cheetah",
    "bird", "robin", "finch", "jay", "magpie", "sparrow", "eagle", "vulture", "parrot", "penguin",
    "horse", "zebra", "elephant", "bear", "panda", "fox", "wolf", "coyote",
    "monkey", "gorilla", "chimpanzee", "baboon", "koala", "kangaroo",
    "fish", "shark", "whale", "dolphin", "turtle", "frog", "snake", "lizard",

    # Electronics, Digital & Screen
    "web site", "website", "screen", "monitor", "television", "laptop", "notebook",
    "cellular telephone", "hand-held computer", "ipod", "mouse", "keyboard", "printer",
    "modem", "hard disc", "cassette", "cd player", "loudspeaker", "microphone",

    # Furniture & Indoor
    "desk", "dining table", "table", "chair", "armchair", "sofa", "couch", "bed", "wardrobe",
    "cabinet", "bookcase", "refrigerator", "microwave", "oven", "toaster", "dishwasher",
    "lamp", "lampshade", "candle", "curtain", "pillow", "quilt", "rug", "doormat",

    # Buildings & Structures
    "building", "house", "palace", "church", "monastery", "castle", "bridge", "viaduct",
    "dam", "pier", "lighthouse", "beacon", "fountain", "monument", "steel arch bridge",

    # Apparel & Accessories
    "suit", "dress", "gown", "jersey", "t-shirt", "sweatshirt", "jacket", "coat",
    "jean", "pants", "shorts", "skirt", "shoe", "boot", "sandal", "sneaker",
    "hat", "cap", "helmet", "sunglasses", "glasses", "tie", "bow tie", "watch",
    "backpack", "handbag", "purse", "wallet", "umbrella",

    # Miscellaneous Objects
    "guitar", "piano", "violin", "drum", "hammer", "screwdriver", "wrench", "pliers",
    "plate", "cup", "mug", "bottle", "can", "bowl", "fork", "knife", "spoon",
    "ball", "racket", "dumbbell", "barbell", "balloon", "toy", "envelope", "binder",
}


class LeafDetector:
    """Detects whether an image contains a supported plant leaf."""

    def __init__(self):
        self.device = torch.device("cpu")
        self._model = None
        self._categories = []
        self._transform = None
        self._init_detector()

    def _init_detector(self):
        try:
            weights = MobileNet_V3_Small_Weights.DEFAULT
            self._model = mobilenet_v3_small(weights=weights).eval()
            self._categories = weights.meta["categories"]
            self._transform = weights.transforms()
            logger.info("Semantic Leaf Detector initialized successfully.")
        except Exception as e:
            logger.warning(f"Could not initialize semantic detector model: {e}")
            self._model = None

    def _check_central_roi_botanical(self, img_np: np.ndarray) -> Tuple[bool, float, str]:
        """
        Verify that the central primary subject of the image is leaf tissue.
        Prevents cars/people with background trees from being classified as leaves.
        """
        h, w = img_np.shape[:2]
        # Central 50% bounding box
        y1, y2 = int(h * 0.25), int(h * 0.75)
        x1, x2 = int(w * 0.25), int(w * 0.75)
        center_roi = img_np[y1:y2, x1:x2]

        hsv_center = cv2.cvtColor(center_roi, cv2.COLOR_RGB2HSV)

        # Vegetation hue range: Hue 18 to 85, Saturation > 25, Value > 25
        green_mask = cv2.inRange(hsv_center, (18, 25, 25), (85, 255, 255))
        # Necrotic / yellow-brown lesion range inside leaf: Hue 8 to 25, Saturation > 35
        lesion_mask = cv2.inRange(hsv_center, (8, 35, 25), (25, 255, 220))
        
        leaf_mask_center = (green_mask > 0) | (lesion_mask > 0)
        center_leaf_ratio = float(np.count_nonzero(leaf_mask_center) / (center_roi.shape[0] * center_roi.shape[1]))

        # Non-organic color checks in central ROI:
        # Metallic blue / red paint, dark tarmac, grey synthetic surfaces
        r = center_roi[:, :, 0].astype(np.float32)
        g = center_roi[:, :, 1].astype(np.float32)
        b = center_roi[:, :, 2].astype(np.float32)

        # Excess Blue (cars, sky, screens) or Excess Red (red cars, bricks)
        exb = b - (r + g) / 2.0
        exr = r - (g + b) / 2.0
        exg = 2.0 * g - r - b

        mean_exg = float(np.mean(exg))
        mean_exb = float(np.mean(exb))
        mean_exr = float(np.mean(exr))

        # If central subject is strongly blue (like a blue car) or has minimal central vegetation
        if mean_exb > 15.0 or (center_leaf_ratio < 0.18 and mean_exg < 5.0):
            return False, center_leaf_ratio, "center_roi_not_vegetation"

        if center_leaf_ratio < 0.12:
            return False, center_leaf_ratio, "insufficient_central_leaf_tissue"

        return True, center_leaf_ratio, "valid_central_leaf"

    def detect_leaf(self, image_path: str) -> LeafDetectionResult:
        """
        Determine if the uploaded image contains a real supported plant leaf.
        Returns LeafDetectionResult with rejection details if non-leaf.
        """
        try:
            pil_img = Image.open(image_path).convert("RGB")
            img_np = np.array(pil_img)
        except Exception as e:
            return LeafDetectionResult(
                leaf_detected=False,
                confidence=0.0,
                reason="corrupted_or_unreadable",
            )

        # ── 1. Semantic ImageNet Classifier Check ──────────────────────
        detected_subject = "unknown"
        if self._model is not None and self._transform is not None:
            try:
                tensor = self._transform(pil_img).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    logits = self._model(tensor)
                    probs = torch.softmax(logits, dim=1)[0]
                    top_probs, top_indices = torch.topk(probs, 5)

                top_category = self._categories[top_indices[0].item()].lower()
                top_prob = top_probs[0].item()
                detected_subject = f"{top_category} ({top_prob*100:.1f}%)"

                # Check if any top prediction is a non-plant category
                for p, idx in zip(top_probs, top_indices):
                    cat_name = self._categories[idx.item()].lower()
                    prob_val = p.item()

                    # Match keyword
                    is_non_plant = any(kw in cat_name for kw in NON_PLANT_KEYWORDS)

                    # If a non-plant category (car, dog, website, building, screen, etc.) has high score
                    if is_non_plant and prob_val > 0.15:
                        logger.info(
                            f"Non-leaf object detected: '{cat_name}' with {prob_val*100:.1f}% confidence."
                        )
                        return LeafDetectionResult(
                            leaf_detected=False,
                            confidence=prob_val,
                            reason="no_supported_leaf_detected",
                            detected_category=cat_name,
                        )
            except Exception as e:
                logger.warning(f"Semantic filter check skipped due to error: {e}")

        # ── 2. Central Botanical ROI Analysis ──────────────────────────
        is_botanical, leaf_ratio, roi_reason = self._check_central_roi_botanical(img_np)
        if not is_botanical:
            logger.info(
                f"Image failed central ROI botanical check: {roi_reason} (ratio: {leaf_ratio:.2f})"
            )
            return LeafDetectionResult(
                leaf_detected=False,
                confidence=float(1.0 - leaf_ratio),
                reason="no_supported_leaf_detected",
                detected_category=detected_subject,
            )

        # Passed all leaf detection stages
        return LeafDetectionResult(
            leaf_detected=True,
            confidence=max(0.85, float(leaf_ratio)),
            reason=None,
            detected_category="plant_leaf",
        )
