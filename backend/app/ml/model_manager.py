import os
import json
import threading
from pathlib import Path
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

try:
    from app.config import settings
except ImportError:
    class DummySettings:
        models_path = "models"
        device = "cuda" if torch.cuda.is_available() else "cpu"
    settings = DummySettings()

from .class_mapping import ClassMapping

class ModelManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ModelManager, cls).__new__(cls)
                cls._instance._initialize()
            return cls._instance

    def _initialize(self):
        self.device = torch.device(getattr(settings, "device", "cpu"))
        self.models_path = Path(getattr(settings, "models_path", "models"))
        self.classifier_dir = self.models_path / "classifier"
        
        self.model = None
        self.class_mapping = ClassMapping(self.classifier_dir / "class_mapping.json")
        self.model_info = {}
        self.calibration_config = {}
        self.ood_config = {}
        self.model_loaded = False
        
        self._load_configs()
        self._load_model()

    def _load_configs(self):
        info_path = self.classifier_dir / "model_info.json"
        if info_path.exists():
            try:
                with open(info_path, "r") as f:
                    self.model_info = json.load(f)
            except Exception:
                pass

        calib_path = self.classifier_dir / "calibration_config.json"
        if calib_path.exists():
            try:
                with open(calib_path, "r") as f:
                    self.calibration_config = json.load(f)
            except Exception:
                pass

        ood_path = self.classifier_dir / "ood_config.json"
        if ood_path.exists():
            try:
                with open(ood_path, "r") as f:
                    self.ood_config = json.load(f)
            except Exception:
                pass

    def _load_model(self):
        num_classes = self.class_mapping.get_num_classes()
        model_path = self.classifier_dir / "best_model.pth"

        try:
            self.model = efficientnet_b0(weights=None)
            
            in_features = self.model.classifier[1].in_features
            self.model.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(in_features, num_classes),
            )

            if model_path.exists():
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model_loaded = True
            else:
                print(f"Warning: Model weights not found at {model_path}. Using mock mode.")
                self.model_loaded = False

            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"Error loading model: {e}. Running in mock mode.")
            self.model_loaded = False
            self.model = None

    def get_model(self) -> Optional[nn.Module]:
        return self.model

    def is_loaded(self) -> bool:
        return self.model_loaded

    def get_class_mapping(self) -> ClassMapping:
        return self.class_mapping

    def get_version(self) -> str:
        return self.model_info.get("version", "leafguard-v1.0")

    def get_calibration_config(self) -> Dict[str, Any]:
        return self.calibration_config

    def get_ood_config(self) -> Dict[str, Any]:
        return self.ood_config
