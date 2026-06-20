from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd


class PerformanceTracker:
    def __init__(self, baseline_accuracy: Optional[float] = None, threshold: float = 0.05):
        self.baseline_accuracy = baseline_accuracy
        self.baseline_f1: Optional[float] = None
        self.baseline_precision: Optional[float] = None
        self.baseline_recall: Optional[float] = None
        self.threshold = threshold
        self.history: List[Dict[str, Any]] = []

    def set_baseline(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        metrics = self._calculate_metrics(y_true, y_pred)
        self.baseline_accuracy = metrics["accuracy"]
        self.baseline_f1 = metrics["f1"]
        self.baseline_precision = metrics["precision"]
        self.baseline_recall = metrics["recall"]

        return metrics

    def set_baseline_metrics(
        self,
        accuracy: float,
        f1: Optional[float] = None,
        precision: Optional[float] = None,
        recall: Optional[float] = None,
    ) -> None:
        self.baseline_accuracy = accuracy
        self.baseline_f1 = f1
        self.baseline_precision = precision
        self.baseline_recall = recall

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        timestamp: Optional[str] = None,
    ) -> Dict[str, Any]:
        if self.baseline_accuracy is None:
            raise ValueError("Baseline metrics have not been set")

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        current_metrics = self._calculate_metrics(y_true, y_pred)

        accuracy_drop = self.baseline_accuracy - current_metrics["accuracy"]
        f1_drop = (self.baseline_f1 - current_metrics["f1"]) if self.baseline_f1 is not None else None
        precision_drop = (self.baseline_precision - current_metrics["precision"]) if self.baseline_precision is not None else None
        recall_drop = (self.baseline_recall - current_metrics["recall"]) if self.baseline_recall is not None else None

        is_degraded = accuracy_drop > self.threshold

        result = {
            "current": current_metrics,
            "baseline": {
                "accuracy": self.baseline_accuracy,
                "f1": self.baseline_f1,
                "precision": self.baseline_precision,
                "recall": self.baseline_recall,
            },
            "drops": {
                "accuracy": float(accuracy_drop),
                "f1": float(f1_drop) if f1_drop is not None else None,
                "precision": float(precision_drop) if precision_drop is not None else None,
                "recall": float(recall_drop) if recall_drop is not None else None,
            },
            "threshold": self.threshold,
            "is_degraded": bool(is_degraded),
            "severity": self._get_severity(accuracy_drop),
            "timestamp": timestamp,
            "sample_size": int(len(y_true)),
        }

        self.history.append(result)

        return result

    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
        if len(y_true) == 0:
            return {"accuracy": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}

        correct = (y_true == y_pred).sum()
        accuracy = correct / len(y_true)

        unique_labels = np.unique(np.concatenate([y_true, y_pred]))

        if len(unique_labels) == 2:
            tp = int(((y_pred == 1) & (y_true == 1)).sum())
            fp = int(((y_pred == 1) & (y_true == 0)).sum())
            fn = int(((y_pred == 0) & (y_true == 1)).sum())
            tn = int(((y_pred == 0) & (y_true == 0)).sum())

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
        else:
            precision, recall, f1 = self._multiclass_metrics(y_true, y_pred)

        return {
            "accuracy": float(accuracy),
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
        }

    @staticmethod
    def _multiclass_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Tuple[float, float, float]:
        labels = np.unique(np.concatenate([y_true, y_pred]))
        precisions = []
        recalls = []
        f1s = []

        for label in labels:
            y_true_bin = (y_true == label).astype(int)
            y_pred_bin = (y_pred == label).astype(int)

            tp = ((y_pred_bin == 1) & (y_true_bin == 1)).sum()
            fp = ((y_pred_bin == 1) & (y_true_bin == 0)).sum()
            fn = ((y_pred_bin == 0) & (y_true_bin == 1)).sum()

            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

            precisions.append(p)
            recalls.append(r)
            f1s.append(f)

        return (
            float(np.mean(precisions)),
            float(np.mean(recalls)),
            float(np.mean(f1s)),
        )

    def _get_severity(self, drop: float) -> str:
        if drop <= 0:
            return "improved"
        elif drop < self.threshold:
            return "minor"
        elif drop < 2 * self.threshold:
            return "moderate"
        else:
            return "severe"

    def get_trend(self, last_n: Optional[int] = None) -> Dict[str, Any]:
        if not self.history:
            return {"trend": "no_data", "slope": 0.0}

        data = self.history[-last_n:] if last_n else self.history
        accuracies = [item["current"]["accuracy"] for item in data]

        if len(accuracies) < 2:
            return {"trend": "insufficient_data", "slope": 0.0}

        x = np.arange(len(accuracies))
        slope, _ = np.polyfit(x, accuracies, 1)

        if slope > 0.001:
            trend = "improving"
        elif slope < -0.001:
            trend = "declining"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "slope": float(slope),
            "recent_accuracies": accuracies,
            "average_drop": float(np.mean([item["drops"]["accuracy"] for item in data])),
        }

    def get_history(self) -> List[Dict[str, Any]]:
        return self.history
