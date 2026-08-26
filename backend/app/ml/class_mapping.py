"""LeafGuard AI — Plant and Disease Class Mapping."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from app.config import settings


@dataclass
class ClassInfo:
    class_index: int
    original_name: str
    class_name: str
    plant: str
    disease: str
    is_healthy: bool


DEFAULT_CLASSES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy", "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight",
    "Potato___Late_blight", "Potato___healthy", "Raspberry___healthy", "Soybean___healthy",
    "Squash___Powdery_mildew", "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]


class ClassMapping:
    """Manages class index to plant/disease name mappings."""

    def __init__(self, mapping_path: Optional[str | Path] = None):
        self.mapping_path = (
            Path(mapping_path) if mapping_path
            else settings.classifier_dir / "class_mapping.json"
        )
        self.index_to_info: Dict[int, ClassInfo] = {}
        self.name_to_index: Dict[str, int] = {}
        self._load_mapping()

    def _parse_class_name(self, name: str) -> Tuple[str, str, bool]:
        parts = name.split("___")
        if len(parts) != 2:
            return name, "Unknown", False
        plant_raw = parts[0].replace("_", " ").strip()
        # Clean specific plant names
        plant = plant_raw.replace("(maize)", "").replace("(including sour)", "").strip().title()
        disease_part = parts[1]
        is_healthy = "healthy" in disease_part.lower()
        disease = "Healthy" if is_healthy else disease_part.replace("_", " ").strip().title()
        return plant, disease, is_healthy

    def _load_mapping(self):
        loaded = False
        if self.mapping_path.exists():
            try:
                with open(self.mapping_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if isinstance(data, dict):
                    for k, v in data.items():
                        idx = int(k)
                        cls_name = v.get("class_name", "")
                        plant = v.get("plant", "")
                        disease = v.get("disease", "")
                        is_healthy = v.get("is_healthy", False)
                        info = ClassInfo(
                            class_index=idx,
                            original_name=cls_name,
                            class_name=cls_name,
                            plant=plant,
                            disease=disease,
                            is_healthy=is_healthy,
                        )
                        self.index_to_info[idx] = info
                        self.name_to_index[cls_name] = idx
                    loaded = True
                elif isinstance(data, list):
                    for idx, cls_name in enumerate(data):
                        plant, disease, is_healthy = self._parse_class_name(cls_name)
                        info = ClassInfo(
                            class_index=idx,
                            original_name=cls_name,
                            class_name=cls_name,
                            plant=plant,
                            disease=disease,
                            is_healthy=is_healthy,
                        )
                        self.index_to_info[idx] = info
                        self.name_to_index[cls_name] = idx
                    loaded = True
            except Exception as e:
                print(f"Error reading class mapping from {self.mapping_path}: {e}")

        if not loaded:
            for idx, cls_name in enumerate(DEFAULT_CLASSES):
                plant, disease, is_healthy = self._parse_class_name(cls_name)
                info = ClassInfo(
                    class_index=idx,
                    original_name=cls_name,
                    class_name=cls_name,
                    plant=plant,
                    disease=disease,
                    is_healthy=is_healthy,
                )
                self.index_to_info[idx] = info
                self.name_to_index[cls_name] = idx

    def get_info(self, index: int) -> Optional[ClassInfo]:
        return self.index_to_info.get(index)

    def get_supported_plants(self) -> List[str]:
        plants = set(info.plant for info in self.index_to_info.values())
        return sorted(list(plants))

    def get_classes_for_plant(self, plant: str) -> List[ClassInfo]:
        return [
            info for info in self.index_to_info.values()
            if info.plant.lower() == plant.lower()
        ]

    def get_num_classes(self) -> int:
        return len(self.index_to_info)
