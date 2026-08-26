"""LeafGuard AI — Dedicated Leaf & Plant Detection Module."""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

logger = logging.getLogger(__name__)


@dataclass
class LeafDetectionResult:
    leaf_detected: bool
    confidence: float
    reason: Optional[str] = None
    detected_category: Optional[str] = None


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

    # Electronics & Digital
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

    # Apparel & Items
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
    """Accurately verifies whether the uploaded image contains a supported plant leaf."""

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
            logger.info("Semantic Leaf Detector initialized.")
        except Exception as e:
            logger.warning(f"Could not initialize MobileNetV3 detector: {e}")
            self._model = None

    def _check_central_roi_botanical(self, img_np: np.ndarray) -> Tuple[bool, float, str]:
        h, w = img_np.shape[:2]
        # Central 50% bounding box
        y1, y2 = int(h * 0.25), int(h * 0.75)
        x1, x2 = int(w * 0.25), int(w * 0.75)
        center_roi = img_np[y1:y2, x1:x2]

        hsv_center = cv2.cvtColor(center_roi, cv2.COLOR_RGB2HSV)

        # Green vegetation mask: Hue 18 to 85, Saturation > 25, Value > 25
        green_mask = cv2.inRange(hsv_center, (18, 25, 25), (85, 255, 255))
        # Necrotic/yellow-brown lesion mask: Hue 8 to 25, Saturation > 35
        lesion_mask = cv2.inRange(hsv_center, (8, 35, 25), (25, 255, 220))
        
        leaf_mask_center = (green_mask > 0) | (lesion_mask > 0)
        center_leaf_ratio = float(np.count_nonzero(leaf_mask_center) / (center_roi.shape[0] * center_roi.shape[1]))

        # Color balance checks
        r = center_roi[:, :, 0].astype(np.float32)
        g = center_roi[:, :, 1].astype(np.float32)
        b = center_roi[:, :, 2].astype(np.float32)

        exb = b - (r + g) / 2.0
        exg = 2.0 * g - r - b

        mean_exg = float(np.mean(exg))
        mean_exb = float(np.mean(exb))

        # Reject dominant metallic blue or non-vegetative surfaces
        if mean_exb > 15.0 or (center_leaf_ratio < 0.18 and mean_exg < 5.0):
            return False, center_leaf_ratio, "center_roi_not_vegetation"

        if center_leaf_ratio < 0.12:
            return False, center_leaf_ratio, "insufficient_central_leaf_tissue"

        return True, center_leaf_ratio, "valid_central_leaf"

    def detect_leaf(self, image_path: str) -> LeafDetectionResult:
        try:
            pil_img = Image.open(image_path).convert("RGB")
            img_np = np.array(pil_img)
        except Exception:
            return LeafDetectionResult(
                leaf_detected=False,
                confidence=0.0,
                reason="corrupted_or_unreadable",
            )

        # 1. Semantic ImageNet Category Filter
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

                for p, idx in zip(top_probs, top_indices):
                    cat_name = self._categories[idx.item()].lower()
                    prob_val = p.item()
                    is_non_plant = any(kw in cat_name for kw in NON_PLANT_KEYWORDS)

                    if is_non_plant and prob_val > 0.15:
                        logger.info(f"Non-leaf object identified: {cat_name} ({prob_val*100:.1f}%)")
                        return LeafDetectionResult(
                            leaf_detected=False,
                            confidence=prob_val,
                            reason="no_supported_leaf_detected",
                            detected_category=cat_name,
                        )
            except Exception as e:
                logger.warning(f"Semantic filter check skipped: {e}")

        # 2. Central Botanical ROI Analysis
        is_botanical, leaf_ratio, roi_reason = self._check_central_roi_botanical(img_np)
        if not is_botanical:
            logger.info(f"Failed botanical ROI check: {roi_reason} (ratio: {leaf_ratio:.2f})")
            return LeafDetectionResult(
                leaf_detected=False,
                confidence=float(1.0 - leaf_ratio),
                reason="no_supported_leaf_detected",
                detected_category=detected_subject,
            )

        return LeafDetectionResult(
            leaf_detected=True,
            confidence=max(0.85, float(leaf_ratio)),
            reason=None,
            detected_category="plant_leaf",
        )
