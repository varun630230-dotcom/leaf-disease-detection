"""LeafGuard AI — Dynamic Plant & Disease Class Taxonomy Registry."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Base path for models
MODELS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "models" / "classifier"


@dataclass
class ClassInfo:
    class_index: int
    class_name: str
    plant: str
    disease: Optional[str]
    is_healthy: bool
    disease_type: str = "fungal"  # "fungal" | "bacterial" | "viral" | "pest_mite" | "healthy"
    pathogen: Optional[str] = None
    synonyms: List[str] = None

    @property
    def display_name(self) -> str:
        if self.is_healthy:
            return f"{self.plant} (Healthy)"
        return f"{self.plant} — {self.disease}"


def _load_taxonomy_registry() -> Tuple[Dict[int, ClassInfo], Dict[str, ClassInfo], List[str]]:
    taxonomy_path = MODELS_DIR / "taxonomy.json"
    mapping_path = MODELS_DIR / "class_mapping.json"

    class_index: Dict[int, ClassInfo] = {}
    class_name_index: Dict[str, ClassInfo] = {}
    crops: List[str] = []

    if taxonomy_path.exists():
        try:
            with open(taxonomy_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                crops = data.get("crops", [])
                for item in data.get("classes", []):
                    info = ClassInfo(
                        class_index=int(item["class_index"]),
                        class_name=item["class_name"],
                        plant=item["plant"],
                        disease=None if item["is_healthy"] else item["disease"],
                        is_healthy=bool(item["is_healthy"]),
                        disease_type=item.get("disease_type", "fungal"),
                        pathogen=item.get("pathogen"),
                        synonyms=item.get("synonyms", []),
                    )
                    class_index[info.class_index] = info
                    class_name_index[info.class_name] = info
        except Exception as e:
            logger.error(f"Error reading taxonomy.json: {e}")

    # Fallback to class_mapping.json if needed
    if not class_index and mapping_path.exists():
        try:
            with open(mapping_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
                for idx_str, item in raw.items():
                    idx = int(idx_str)
                    is_h = bool(item["is_healthy"])
                    info = ClassInfo(
                        class_index=idx,
                        class_name=item["class_name"],
                        plant=item["plant"],
                        disease=None if is_h else item["disease"],
                        is_healthy=is_h,
                        disease_type=item.get("disease_type", "healthy" if is_h else "fungal"),
                    )
                    class_index[idx] = info
                    class_name_index[info.class_name] = info
                    if info.plant not in crops:
                        crops.append(info.plant)
        except Exception as e:
            logger.error(f"Error reading class_mapping.json: {e}")

    return class_index, class_name_index, crops


CLASS_INDEX, CLASS_NAME_INDEX, SUPPORTED_PLANTS = _load_taxonomy_registry()
CLASS_NAME_TO_INFO = CLASS_NAME_INDEX
PLANTVILLAGE_CLASSES = list(CLASS_NAME_INDEX.keys())
NUM_CLASSES = len(CLASS_INDEX) if CLASS_INDEX else 38


def get_supported_plants() -> List[str]:
    return list(SUPPORTED_PLANTS)


def get_class_info(index: int) -> Optional[ClassInfo]:
    return CLASS_INDEX.get(index)


def get_class_info_by_name(name: str) -> Optional[ClassInfo]:
    return CLASS_NAME_INDEX.get(name)


def get_classes_for_plant(plant_name: str) -> List[ClassInfo]:
    """Returns all supported disease and healthy classes for a specific crop."""
    target = plant_name.lower()
    return [c for c in CLASS_INDEX.values() if c.plant.lower() == target]


def get_healthy_class_for_plant(plant_name: str) -> Optional[ClassInfo]:
    """Returns the healthy ClassInfo for a given crop."""
    target = plant_name.lower()
    for c in CLASS_INDEX.values():
        if c.plant.lower() == target and c.is_healthy:
            return c
    return None
