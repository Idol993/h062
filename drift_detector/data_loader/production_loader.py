from typing import Optional, List, Dict, Any
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path

from ..utils.time_utils import TimeUtils


class ProductionLoader:
    def __init__(
        self,
        filepath: Optional[str] = None,
        date_column: Optional[str] = None,
        window_days: int = 7,
    ):
        self.filepath = filepath
        self.date_column = date_column
        self.window_days = window_days
        self.data: Optional[pd.DataFrame] = None
        self.filtered_data: Optional[pd.DataFrame] = None

    def load(
        self,
        filepath: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        load_path = filepath or self.filepath
        if load_path is None:
            raise ValueError("No filepath provided for production data")

        path = Path(load_path)
        if not path.exists():
            raise FileNotFoundError(f"Production file not found: {load_path}")

        if path.suffix == ".csv":
            self.data = pd.read_csv(load_path)
        elif path.suffix in [".parquet", ".pq"]:
            self.data = pd.read_parquet(load_path)
        elif path.suffix == ".json":
            self.data = pd.read_json(load_path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        self._parse_dates()
        self.filtered_data = self._filter_by_date_window(start_date, end_date)
        return self.filtered_data

    def _parse_dates(self) -> None:
        if self.date_column and self.date_column in self.data.columns:
            self.data[self.date_column] = pd.to_datetime(self.data[self.date_column])

    def _filter_by_date_window(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        if self.date_column is None or self.date_column not in self.data.columns:
            return self.data.copy()

        if start_date is None or end_date is None:
            start_date, end_date = TimeUtils.get_sliding_window(self.window_days)

        mask = (self.data[self.date_column] >= start_date) & (
            self.data[self.date_column] <= end_date
        )
        filtered = self.data[mask].copy()

        if len(filtered) == 0:
            filtered = self.data.copy()

        return filtered

    def get_feature_data(self, feature_name: str) -> np.ndarray:
        data = self.filtered_data if self.filtered_data is not None else self.data
        if data is None:
            raise ValueError("Production data not loaded")
        if feature_name not in data.columns:
            raise ValueError(f"Feature '{feature_name}' not found in production data")
        return data[feature_name].dropna().values

    def get_data(self) -> pd.DataFrame:
        if self.filtered_data is not None:
            return self.filtered_data
        if self.data is not None:
            return self.data
        raise ValueError("Production data not loaded")

    def get_sample_size(self) -> int:
        data = self.filtered_data if self.filtered_data is not None else self.data
        if data is None:
            return 0
        return len(data)

    def get_date_range(self) -> Optional[tuple]:
        if self.date_column is None or self.filtered_data is None:
            return None
        dates = self.filtered_data[self.date_column]
        return (dates.min(), dates.max())

    def get_feature_names(self) -> List[str]:
        data = self.filtered_data if self.filtered_data is not None else self.data
        if data is None:
            return []
        return list(data.columns)

    def filter_features(
        self,
        include_features: Optional[List[str]] = None,
        exclude_features: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        data = self.filtered_data if self.filtered_data is not None else self.data
        if data is None:
            raise ValueError("Production data not loaded")

        result = data.copy()

        if include_features:
            valid_features = [f for f in include_features if f in result.columns]
            result = result[valid_features]

        if exclude_features:
            result = result.drop(columns=exclude_features, errors="ignore")

        return result

    def get_summary(self) -> Dict[str, Any]:
        data = self.filtered_data if self.filtered_data is not None else self.data
        if data is None:
            return {}

        summary = {
            "total_rows": len(data),
            "total_columns": len(data.columns),
            "columns": list(data.columns),
            "date_range": self.get_date_range(),
            "missing_values": data.isnull().sum().to_dict(),
        }
        return summary
