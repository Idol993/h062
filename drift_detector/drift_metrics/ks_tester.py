from typing import Dict, Any, Optional, Tuple
import numpy as np
from scipy import stats


class KSTester:
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.baseline_data: Optional[np.ndarray] = None

    def fit(self, baseline_data: np.ndarray) -> None:
        self.baseline_data = np.asarray(baseline_data).flatten()

    def test(self, production_data: np.ndarray) -> Dict[str, Any]:
        if self.baseline_data is None:
            raise ValueError("KSTester has not been fitted with baseline data")

        production_data = np.asarray(production_data).flatten()

        ks_statistic, p_value = stats.ks_2samp(
            self.baseline_data, production_data, alternative="two-sided"
        )

        is_significant = p_value < self.alpha

        return {
            "statistic": float(ks_statistic),
            "p_value": float(p_value),
            "alpha": self.alpha,
            "is_significant": bool(is_significant),
            "conclusion": (
                "Reject H0: distributions are significantly different"
                if is_significant
                else "Fail to reject H0: no significant difference detected"
            ),
            "baseline_size": int(len(self.baseline_data)),
            "production_size": int(len(production_data)),
        }

    def test_critical_value(
        self,
        production_data: np.ndarray,
    ) -> Dict[str, Any]:
        if self.baseline_data is None:
            raise ValueError("KSTester has not been fitted with baseline data")

        production_data = np.asarray(production_data).flatten()

        n1 = len(self.baseline_data)
        n2 = len(production_data)
        c_alpha = np.sqrt(-0.5 * np.log(self.alpha / 2))
        critical_value = c_alpha * np.sqrt((n1 + n2) / (n1 * n2))

        result = self.test(production_data)
        result["critical_value"] = float(critical_value)
        result["exceeds_critical"] = bool(result["statistic"] > critical_value)

        return result

    def calculate(
        self,
        baseline_data: np.ndarray,
        production_data: np.ndarray,
    ) -> Dict[str, Any]:
        self.fit(baseline_data)
        return self.test(production_data)

    def get_effect_size(self, production_data: np.ndarray) -> Dict[str, Any]:
        result = self.test(production_data)

        n1 = len(self.baseline_data)
        n2 = len(production_data)

        cohen_d = self._cohen_d(self.baseline_data, production_data)
        cliffs_delta = self._cliffs_delta(self.baseline_data, production_data)

        result["cohen_d"] = float(cohen_d)
        result["cliffs_delta"] = float(cliffs_delta)
        result["effect_size_interpretation"] = self._interpret_effect_size(cohen_d)

        return result

    @staticmethod
    def _cohen_d(sample1: np.ndarray, sample2: np.ndarray) -> float:
        mean1, mean2 = np.mean(sample1), np.mean(sample2)
        std1, std2 = np.std(sample1, ddof=1), np.std(sample2, ddof=1)

        n1, n2 = len(sample1), len(sample2)
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))

        if pooled_std == 0:
            return 0.0

        return (mean2 - mean1) / pooled_std

    @staticmethod
    def _cliffs_delta(sample1: np.ndarray, sample2: np.ndarray) -> float:
        n1, n2 = len(sample1), len(sample2)
        if n1 == 0 or n2 == 0:
            return 0.0

        comparisons = np.subtract.outer(sample2, sample1)
        greater = np.sum(comparisons > 0)
        less = np.sum(comparisons < 0)

        return (greater - less) / (n1 * n2)

    @staticmethod
    def _interpret_effect_size(cohen_d: float) -> str:
        abs_d = abs(cohen_d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
