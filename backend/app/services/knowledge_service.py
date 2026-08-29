"""LeafGuard AI — Verified Disease Knowledge Service."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel

logger = logging.getLogger(__name__)

KNOWLEDGE_BASE_DIR = Path(__file__).resolve().parent.parent / "knowledge_base"


class DiseaseKnowledge(BaseModel):
    plant: str
    disease: str
    class_name: str
    disease_type: str  # "fungal" | "bacterial" | "viral" | "pest_mite"
    pathogen: str
    symptoms: List[str]
    risk_factors: List[str]
    prevention: List[str]
    recommended_actions: List[str]


class KnowledgeService:
    """Provides verified agronomic pathology information for predicted diseases."""

    def __init__(self, kb_dir: Optional[Path] = None):
        self.kb_dir = kb_dir or KNOWLEDGE_BASE_DIR
        self._cache: Dict[str, DiseaseKnowledge] = {}
        self._load_all()

    def _load_all(self):
        if not self.kb_dir.exists():
            logger.warning(f"Knowledge base directory not found at {self.kb_dir}")
            return

        for json_file in self.kb_dir.rglob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    rec = DiseaseKnowledge(**data)
                    # Key by class_name (e.g. "Tomato___Late_blight") and key by (plant, disease)
                    self._cache[rec.class_name] = rec
                    self._cache[f"{rec.plant}_{rec.disease}".lower()] = rec
            except Exception as e:
                logger.error(f"Error loading disease knowledge from {json_file}: {e}")

        logger.info(f"Loaded {len(self._cache)//2} verified disease knowledge records.")

    def get_knowledge(self, class_name: Optional[str] = None, plant: Optional[str] = None, disease: Optional[str] = None) -> Optional[DiseaseKnowledge]:
        """Retrieves verified disease information. Returns None for healthy/unknown/rejected."""
        if not class_name and not (plant and disease):
            return None

        if class_name and class_name in self._cache:
            return self._cache[class_name]

        if plant and disease:
            key = f"{plant}_{disease}".lower()
            if key in self._cache:
                return self._cache[key]

        return None
