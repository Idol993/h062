from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
from pathlib import Path


class BaselineLoader:
    def __init__(self, filepath: Optional[str] = None):
        self.filepath = filepath
        self.data: Optional[pd.DataFrame] = None
        self.feature_stats: Dict[str, Any] = {}

    def load(self, filepath: Optional[str] = None) -> pd.DataFrame:
        load_path = filepath or self.filepath
        if load_path is None:
            raise ValueError("No filepath provided for baseline data")

        path = Path(load_path)
        if not path.exists():
            raise FileNotFoundError(f"Baseline file not found: {load_path}")

        if path.suffix == ".csv":
            self.data = pd.read_csv(load_path)
        elif path.suffix in [".parquet", ".pq"]:
            self.data = pd.read_parquet(load_path)
        elif path.suffix == ".json":
            self.data = pd.read_json(load_path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        self._compute_feature_stats()
        return self.data

    def _compute_feature_stats(self) -> None:
        if self.data is None:
            return

        for column in self.data.columns:
            series = self.data[column]
            if pd.api.types.is_numeric_dtype(series):
                self.feature_stats[column] = {
                    "type": "numerical",
                    "mean": float(series.mean()),
                    "std": float(series.std()),
                    "min": float(series.min()),
                    "max": float(series.max()),
                    "median": float(series.median()),
                    "q25": float(series.quantile(0.25)),
                    "q75": float(series.quantile(0.75)),
                    "missing": int(series.isnull().sum()),
                    "missing_ratio": float(series.isnull().mean()),
                    "unique_count": int(series.nunique()),
                }
            else:
                value_counts = series.value_counts(normalize=True)
                self.feature_stats[column] = {
                    "type": "categorical",
                    "categories": list(value_counts.index.astype(str)),
                    "category_ratios": {str(k): float(v) for k, v in value_counts.items()},
                    "missing": int(series.isnull().sum()),
                    "missing_ratio": float(series.isnull().mean()),
                    "unique_count": int(series.nunique()),
                }

    def get_feature_data(self, feature_name: str) -> np.ndarray:
        if self.data is None:
            raise ValueError("Baseline data not loaded")
        if feature_name not in self.data.columns:
            raise ValueError(f"Feature '{feature_name}' not found in baseline data")
        return self.data[feature_name].dropna().values

    def get_feature_stats(self, feature_name: str) -> Dict[str, Any]:
        if feature_name not in self.feature_stats:
            raise ValueError(f"Feature '{feature_name}' stats not computed")
        return self.feature_stats[feature_name]

    def get_all_feature_stats(self) -> Dict[str, Any]:
        return self.feature_stats

    def get_numerical_features(self) -> list:
        return [
            name
            for name, stats in self.feature_stats.items()
            if stats["type"] == "numerical"
        ]

    def get_categorical_features(self) -> list:
        return [
            name
            for name, stats in self.feature_stats.items()
            if stats["type"] == "categorical"
        ]

    def get_sample_size(self) -> int:
        if self.data is None:
            return 0
        return len(self.data)
