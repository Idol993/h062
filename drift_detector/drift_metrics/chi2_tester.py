from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from scipy import stats


class Chi2Tester:
    def __init__(self, alpha: float = 0.05, min_expected_freq: float = 5.0):
        self.alpha = alpha
        self.min_expected_freq = min_expected_freq
        self.baseline_distribution: Optional[Dict[str, float]] = None
        self.baseline_categories: Optional[list] = None

    def fit(self, baseline_data: np.ndarray) -> None:
        baseline_data = np.asarray(baseline_data).astype(str)
        counts = pd.Series(baseline_data).value_counts()
        self.baseline_categories = counts.index.tolist()
        self.baseline_distribution = (counts / counts.sum()).to_dict()

    def test(self, production_data: np.ndarray) -> Dict[str, Any]:
        if self.baseline_distribution is None or self.baseline_categories is None:
            raise ValueError("Chi2Tester has not been fitted with baseline data")

        production_data = np.asarray(production_data).astype(str)

        all_categories = sorted(
            set(self.baseline_categories) | set(np.unique(production_data))
        )

        expected_probs = [
            self.baseline_distribution.get(cat, 0.01) for cat in all_categories
        ]
        total = sum(expected_probs)
        expected_probs = [p / total for p in expected_probs]

        production_counts = pd.Series(production_data).value_counts().reindex(all_categories, fill_value=0).values
        production_total = production_counts.sum()
        expected_counts = np.array(expected_probs) * production_total

        valid_mask = expected_counts >= self.min_expected_freq
        if valid_mask.sum() < 2:
            valid_mask = expected_counts >= 1.0
            if valid_mask.sum() < 2:
                return {
                    "statistic": 0.0,
                    "p_value": 1.0,
                    "alpha": self.alpha,
                    "is_significant": False,
                    "conclusion": "Insufficient data for chi-square test",
                    "categories": all_categories,
                    "observed_counts": production_counts.tolist(),
                    "expected_counts": expected_counts.tolist(),
                    "warning": "Many expected frequencies are too small",
                }

        production_counts_valid = production_counts[valid_mask]
        expected_counts_valid = expected_counts[valid_mask]

        if expected_counts_valid.sum() > 0:
            expected_counts_valid = (
                expected_counts_valid
                / expected_counts_valid.sum()
                * production_counts_valid.sum()
            )

        chi2_stat, p_value = stats.chisquare(
            f_obs=production_counts_valid, f_exp=expected_counts_valid
        )

        degrees_of_freedom = valid_mask.sum() - 1
        is_significant = p_value < self.alpha

        cramers_v = self._cramers_v(
            production_counts_valid, expected_counts_valid, degrees_of_freedom
        )

        return {
            "statistic": float(chi2_stat),
            "p_value": float(p_value),
            "alpha": self.alpha,
            "degrees_of_freedom": int(degrees_of_freedom),
            "is_significant": bool(is_significant),
            "conclusion": (
                "Reject H0: distributions are significantly different"
                if is_significant
                else "Fail to reject H0: no significant difference detected"
            ),
            "cramers_v": float(cramers_v),
            "effect_size": self._interpret_cramers_v(cramers_v),
            "categories": all_categories,
            "observed_counts": production_counts.tolist(),
            "expected_counts": expected_counts.tolist(),
            "baseline_size": int(sum(self.baseline_distribution.values()) * production_total) if production_total > 0 else 0,
            "production_size": int(production_total),
        }

    def calculate(
        self,
        baseline_data: np.ndarray,
        production_data: np.ndarray,
    ) -> Dict[str, Any]:
        self.fit(baseline_data)
        return self.test(production_data)

    @staticmethod
    def _cramers_v(
        observed: np.ndarray,
        expected: np.ndarray,
        degrees_of_freedom: int,
    ) -> float:
        total = observed.sum()
        if total == 0:
            return 0.0

        chi2 = np.sum((observed - expected) ** 2 / expected)
        k = len(observed)
        phi_squared = chi2 / total
        v = np.sqrt(phi_squared / min(k - 1, 1))

        return float(v)

    @staticmethod
    def _interpret_cramers_v(v: float) -> str:
        if v < 0.1:
            return "negligible"
        elif v < 0.3:
            return "small"
        elif v < 0.5:
            return "medium"
        else:
            return "large"
