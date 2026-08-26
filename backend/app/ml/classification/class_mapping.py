"""LeafGuard AI — PlantVillage 38-class dataset mapping and metadata."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ClassInfo:
    index: int
    class_name: str
    plant: str
    disease: Optional[str]
    is_healthy: bool
    display_name: str


PLANTVILLAGE_CLASSES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]


def parse_class_name(index: int, raw_name: str) -> ClassInfo:
    parts = raw_name.split("___")
    if len(parts) == 2:
        raw_plant, raw_disease = parts
    else:
        raw_plant = parts[0]
        raw_disease = "unknown"

    plant = (
        raw_plant.replace("_", " ")
        .replace("(including sour)", "")
        .replace("(maize)", "")
        .replace(", bell", " Bell")
        .strip()
    )

    is_healthy = raw_disease.lower() == "healthy"

    if is_healthy:
        disease = None
        display_name = f"{plant} (Healthy)"
    else:
        disease = (
            raw_disease.replace("_", " ")
            .replace("  ", " ")
            .replace("Haunglongbing (Citrus greening)", "Citrus Greening")
            .strip()
        )
        display_name = f"{plant} — {disease}"

    return ClassInfo(
        index=index,
        class_name=raw_name,
        plant=plant,
        disease=disease,
        is_healthy=is_healthy,
        display_name=display_name,
    )


CLASS_INDEX: Dict[int, ClassInfo] = {
    i: parse_class_name(i, name) for i, name in enumerate(PLANTVILLAGE_CLASSES)
}

CLASS_NAME_TO_INFO: Dict[str, ClassInfo] = {
    info.class_name: info for info in CLASS_INDEX.values()
}

NUM_CLASSES = len(PLANTVILLAGE_CLASSES)


def get_class_info(index: int) -> Optional[ClassInfo]:
    return CLASS_INDEX.get(index)


def get_class_info_by_name(name: str) -> Optional[ClassInfo]:
    return CLASS_NAME_TO_INFO.get(name)


def get_supported_plants() -> List[str]:
    plants = sorted(list({info.plant for info in CLASS_INDEX.values()}))
    return plants
