import os
from typing import Dict, Any, Optional
from pathlib import Path
import yaml
from pydantic import BaseModel, Field


class FeaturesConfig(BaseModel):
    numerical: list = Field(default_factory=list)
    categorical: list = Field(default_factory=list)


class ThresholdsConfig(BaseModel):
    psi: Dict[str, float] = Field(default_factory=lambda: {"warning": 0.1, "critical": 0.2})
    p_value: float = 0.05
    performance_drop: float = 0.05


class NotificationsConfig(BaseModel):
    terminal: Dict[str, Any] = Field(default_factory=lambda: {"enabled": True})
    webhook: Dict[str, Any] = Field(
        default_factory=lambda: {"enabled": False, "url": ""}
    )
    log_file: Dict[str, Any] = Field(
        default_factory=lambda: {
            "enabled": True,
            "path": "drift_alerts.log",
            "max_bytes": 10485760,
            "backup_count": 5,
        }
    )


class VisualizationConfig(BaseModel):
    output_dir: str = "plots"
    dpi: int = 100
    figsize: list = Field(default_factory=lambda: [12, 6])


class MonitoringConfig(BaseModel):
    prometheus_port: int = 8000
    slide_window_days: int = 7


class SchedulingConfig(BaseModel):
    enabled: bool = False
    interval: str = "daily"
    time: str = "08:00"


class AppConfig(BaseModel):
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    thresholds: ThresholdsConfig = Field(default_factory=ThresholdsConfig)
    notifications: NotificationsConfig = Field(default_factory=NotificationsConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    scheduling: SchedulingConfig = Field(default_factory=SchedulingConfig)


class ConfigParser:
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config.yaml"
        self.config: Optional[AppConfig] = None

    def load(self) -> AppConfig:
        if not os.path.exists(self.config_path):
            default_config = AppConfig()
            self.save(default_config)
            self.config = default_config
            return default_config

        with open(self.config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}

        self.config = AppConfig(**config_data)
        return self.config

    def save(self, config: AppConfig, path: Optional[str] = None) -> None:
        save_path = path or self.config_path
        config_dict = config.model_dump()

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)

    def get_config(self) -> AppConfig:
        if self.config is None:
            self.load()
        return self.config

    def get_numerical_features(self) -> list:
        return self.get_config().features.numerical

    def get_categorical_features(self) -> list:
        return self.get_config().features.categorical

    def get_all_features(self) -> list:
        return self.get_numerical_features() + self.get_categorical_features()

    def get_psi_thresholds(self) -> Dict[str, float]:
        return self.get_config().thresholds.psi

    def get_p_value_threshold(self) -> float:
        return self.get_config().thresholds.p_value

    def get_performance_drop_threshold(self) -> float:
        return self.get_config().thresholds.performance_drop
