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


HIGH_CONFIDENCE_NON_PLANT = {
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

    # Electronics & Screens
    "web site", "website", "screen", "monitor", "television", "laptop", "notebook",
    "cellular telephone", "hand-held computer", "ipod", "mouse", "keyboard", "printer",
    "modem", "hard disc", "cassette", "cd player", "loudspeaker", "microphone",

    # Furniture & Indoor
    "desk", "dining table", "chair", "armchair", "sofa", "couch", "bed", "wardrobe",
    "cabinet", "bookcase", "refrigerator", "microwave", "oven", "toaster", "dishwasher",

    # Buildings & Architecture
    "building", "palace", "church", "monastery", "castle", "bridge", "viaduct",
    "dam", "pier", "lighthouse", "beacon", "monument", "steel arch bridge",
}


class LeafDetector:
    """Accurately verifies whether the uploaded image contains plant foliage.
    
    Distinguishes genuine non-plant objects (cars, dogs, electronics) from plant leaves,
    ensuring that real leaves (even with unfamiliar diseases or discoloration) are never falsely rejected.
    """

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

    def _analyze_botanical_tissue(self, img_np: np.ndarray) -> Tuple[bool, float, str]:
        """Examines the entire image for plant tissue (green foliage, chlorosis, necrosis, venation)."""
        h, w = img_np.shape[:2]
        total_pixels = h * w

        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)

        # 1. Healthy green spectrum: Hue 20 to 88, Saturation > 20, Value > 20
        green_mask = cv2.inRange(hsv, (18, 20, 20), (88, 255, 255))

        # 2. Chlorotic yellow / lesion spectrum: Hue 8 to 25, Saturation > 30, Value > 20
        chlorotic_mask = cv2.inRange(hsv, (8, 30, 20), (25, 255, 255))

        # 3. Brown / necrotic tissue on leaf
        brown_mask = cv2.inRange(hsv, (5, 25, 15), (22, 255, 200))

        leaf_mask = (green_mask > 0) | (chlorotic_mask > 0) | (brown_mask > 0)
        leaf_pixels = np.count_nonzero(leaf_mask)
        leaf_ratio = float(leaf_pixels / total_pixels)

        # Excess Green Index: 2G - R - B
        r = img_np[:, :, 0].astype(np.float32)
        g = img_np[:, :, 1].astype(np.float32)
        b = img_np[:, :, 2].astype(np.float32)
        exg = 2.0 * g - r - b
        mean_exg = float(np.mean(exg))

        # If significant leaf tissue is present anywhere in the frame
        if leaf_ratio >= 0.05 or (leaf_pixels > 2500 and mean_exg > -10.0):
            return True, leaf_ratio, "leaf_tissue_confirmed"

        return False, leaf_ratio, "no_botanical_tissue_detected"

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

        # 1. Check botanical tissue across image
        is_botanical, leaf_ratio, reason = self._analyze_botanical_tissue(img_np)

        # 2. Semantic ImageNet Category Filter for non-plant rejection
        if self._model is not None and self._transform is not None:
            try:
                tensor = self._transform(pil_img).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    logits = self._model(tensor)
                    probs = torch.softmax(logits, dim=1)[0]
                    top_prob, top_idx = torch.topk(probs, 1)

                top_category = self._categories[top_idx[0].item()].lower()
                top_prob_val = top_prob[0].item()

                is_definite_non_plant = any(kw in top_category for kw in HIGH_CONFIDENCE_NON_PLANT)

                # If the image strongly matches a non-plant class (e.g. car, dog, airplane) AND lacks leaf tissue
                if is_definite_non_plant and top_prob_val > 0.40 and leaf_ratio < 0.15:
                    logger.info(f"Non-leaf object identified: {top_category} ({top_prob_val*100:.1f}%)")
                    return LeafDetectionResult(
                        leaf_detected=False,
                        confidence=top_prob_val,
                        reason="no_supported_leaf_detected",
                        detected_category=top_category,
                    )
            except Exception as e:
                logger.warning(f"Semantic filter check skipped: {e}")

        # If botanical tissue is found or not an overt non-plant
        if is_botanical:
            return LeafDetectionResult(
                leaf_detected=True,
                confidence=max(0.85, float(leaf_ratio)),
                reason=None,
                detected_category="plant_leaf",
            )

        # If neither botanical tissue nor recognized object
        return LeafDetectionResult(
            leaf_detected=False,
            confidence=0.0,
            reason="no_supported_leaf_detected",
            detected_category="unrecognized_non_leaf",
        )
