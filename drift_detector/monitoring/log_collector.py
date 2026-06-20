from typing import Optional, Dict, Any, List, Callable
import os
import time
import json
import threading
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

from ..utils.time_utils import TimeUtils


class LogCollector:
    def __init__(
        self,
        log_dir: Optional[str] = None,
        log_file_pattern: str = "*.log",
        date_column: Optional[str] = None,
        window_days: int = 7,
    ):
        self.log_dir = log_dir
        self.log_file_pattern = log_file_pattern
        self.date_column = date_column
        self.window_days = window_days
        self._collected_data: Optional[pd.DataFrame] = None
        self._watching = False
        self._watch_thread: Optional[threading.Thread] = None
        self._callbacks: List[Callable] = []
        self._last_modified: Dict[str, float] = {}

    def collect(self) -> pd.DataFrame:
        if not self.log_dir:
            raise ValueError("No log directory specified")

        log_path = Path(self.log_dir)
        if not log_path.exists():
            raise FileNotFoundError(f"Log directory not found: {self.log_dir}")

        log_files = list(log_path.glob(self.log_file_pattern))
        data_frames = []

        for log_file in log_files:
            df = self._parse_log_file(log_file)
            if df is not None and len(df) > 0:
                data_frames.append(df)

        if not data_frames:
            self._collected_data = pd.DataFrame()
            return self._collected_data

        combined = pd.concat(data_frames, ignore_index=True)
        combined = self._filter_by_date_window(combined)

        self._collected_data = combined
        return combined

    def _parse_log_file(self, filepath: Path) -> Optional[pd.DataFrame]:
        try:
            if filepath.suffix == ".csv":
                return pd.read_csv(filepath)
            elif filepath.suffix in [".parquet", ".pq"]:
                return pd.read_parquet(filepath)
            elif filepath.suffix == ".json":
                return pd.read_json(filepath)
            elif filepath.suffix == ".log":
                return self._parse_text_log(filepath)
            else:
                return None
        except Exception as e:
            print(f"Error parsing {filepath}: {str(e)}")
            return None

    def _parse_text_log(self, filepath: Path) -> Optional[pd.DataFrame]:
        records = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError:
                    parsed = self._parse_unstructured_line(line)
                    if parsed:
                        records.append(parsed)

        if records:
            return pd.DataFrame(records)
        return None

    def _parse_unstructured_line(self, line: str) -> Optional[Dict[str, Any]]:
        try:
            parts = line.split(" - ", 2)
            if len(parts) >= 3:
                timestamp_str, level, message = parts[0], parts[1], parts[2]
                try:
                    timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                    return {
                        "timestamp": timestamp,
                        "level": level,
                        "message": message,
                    }
                except ValueError:
                    pass
        except Exception:
            pass
        return None

    def _filter_by_date_window(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.date_column is None or self.date_column not in df.columns:
            return df

        df[self.date_column] = pd.to_datetime(df[self.date_column])
        start_date, end_date = TimeUtils.get_sliding_window(self.window_days)

        mask = (df[self.date_column] >= start_date) & (
            df[self.date_column] <= end_date
        )
        filtered = df[mask].copy()

        if len(filtered) == 0:
            return df

        return filtered

    def start_watching(
        self,
        callback: Optional[Callable[[pd.DataFrame], None]] = None,
        interval: int = 60,
    ) -> None:
        if self._watching:
            return

        if callback:
            self._callbacks.append(callback)

        self._watching = True
        self._watch_thread = threading.Thread(
            target=self._watch_loop,
            args=(interval,),
            daemon=True,
        )
        self._watch_thread.start()

    def _watch_loop(self, interval: int) -> None:
        while self._watching:
            try:
                changed = self._check_file_changes()
                if changed:
                    new_data = self.collect()
                    for callback in self._callbacks:
                        try:
                            callback(new_data)
                        except Exception as e:
                            print(f"Error in watch callback: {str(e)}")
            except Exception as e:
                print(f"Error in watch loop: {str(e)}")

            time.sleep(interval)

    def _check_file_changes(self) -> bool:
        if not self.log_dir:
            return False

        log_path = Path(self.log_dir)
        log_files = list(log_path.glob(self.log_file_pattern))

        changed = False
        for log_file in log_files:
            mtime = log_file.stat().st_mtime
            if (
                log_file not in self._last_modified
                or self._last_modified[log_file] < mtime
            ):
                changed = True
                self._last_modified[log_file] = mtime

        return changed

    def stop_watching(self) -> None:
        self._watching = False
        if self._watch_thread:
            self._watch_thread.join(timeout=5)

    def get_data(self) -> Optional[pd.DataFrame]:
        return self._collected_data

    def get_feature_data(self, feature_name: str) -> Optional[np.ndarray]:
        if self._collected_data is None:
            return None
        if feature_name not in self._collected_data.columns:
            return None
        return self._collected_data[feature_name].dropna().values

    def is_watching(self) -> bool:
        return self._watching
