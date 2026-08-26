"""LeafGuard AI - Application Configuration."""

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Environment
    environment: str = "development"
    host: str = "0.0.0.0"
    port: int = 8000

    # Paths
    base_dir: Path = Path(__file__).resolve().parent.parent
    upload_dir: str = "uploads"
    results_dir: str = "results"
    models_dir: str = "models"

    # Model
    model_version: str = "leafguard-v1.0"

    # Image validation
    max_file_size_mb: float = 25.0
    min_image_dim: int = 64
    max_image_dim: int = 4096
    blur_threshold: float = 15.0
    brightness_min: float = 20.0
    brightness_max: float = 245.0
    contrast_min: float = 10.0

    # Inference & OOD
    device: str = "cpu"
    confidence_high_threshold: float = 0.80
    confidence_low_threshold: float = 0.45
    ood_energy_threshold: float = -8.5

    # Case-insensitive aliases
    @property
    def CONFIDENCE_HIGH_THRESHOLD(self) -> float:
        return self.confidence_high_threshold

    @property
    def CONFIDENCE_LOW_THRESHOLD(self) -> float:
        return self.confidence_low_threshold

    @property
    def OOD_ENERGY_THRESHOLD(self) -> float:
        return self.ood_energy_threshold

    # CORS
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000", "*"]

    @property
    def upload_path(self) -> Path:
        path = self.base_dir / self.upload_dir
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def results_path(self) -> Path:
        path = self.base_dir / self.results_dir
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def RESULTS_DIR(self) -> Path:
        return self.results_path

    @property
    def models_path(self) -> Path:
        return self.base_dir / self.models_dir

    @property
    def classifier_dir(self) -> Path:
        return self.models_path / "classifier"

    @property
    def model_weights_path(self) -> str:
        return str(self.classifier_dir / "best_model.pth")

    @property
    def MODEL_WEIGHTS_PATH(self) -> str:
        return self.model_weights_path

    @property
    def evaluation_dir(self) -> Path:
        return self.models_path / "evaluation"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
