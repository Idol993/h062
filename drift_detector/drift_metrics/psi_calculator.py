from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd

from ..utils.binning import Binning


class PSICalculator:
    def __init__(
        self,
        n_bins: int = 10,
        binning_method: str = "equal_frequency",
        epsilon: float = 1e-10,
    ):
        self.n_bins = n_bins
        self.binning_method = binning_method
        self.epsilon = epsilon
        self.baseline_edges: Optional[np.ndarray] = None
        self.baseline_ratios: Optional[np.ndarray] = None

    def fit(self, baseline_data: np.ndarray) -> None:
        baseline_data = np.asarray(baseline_data).flatten()

        if self.binning_method == "equal_width":
            bin_indices, edges = Binning.equal_width(baseline_data, self.n_bins)
        elif self.binning_method == "optimal":
            bin_indices, edges = Binning.optimal_binning(baseline_data, n_bins=self.n_bins)
        else:
            bin_indices, edges = Binning.equal_frequency(baseline_data, self.n_bins)

        self.baseline_edges = edges
        n_bins_actual = len(edges) - 1
        self.baseline_ratios = Binning.get_bin_ratios(bin_indices, n_bins_actual)

    def transform(self, production_data: np.ndarray) -> Dict[str, Any]:
        if self.baseline_edges is None or self.baseline_ratios is None:
            raise ValueError("PSICalculator has not been fitted with baseline data")

        production_data = np.asarray(production_data).flatten()

        bin_indices = np.digitize(production_data, self.baseline_edges, right=False)
        n_bins = len(self.baseline_edges) - 1
        bin_indices = np.clip(bin_indices, 1, n_bins) - 1

        production_ratios = Binning.get_bin_ratios(bin_indices, n_bins)

        psi_value, bin_contributions = self._calculate_psi(
            self.baseline_ratios, production_ratios
        )

        return {
            "psi": float(psi_value),
            "baseline_ratios": self.baseline_ratios.tolist(),
            "production_ratios": production_ratios.tolist(),
            "bin_contributions": bin_contributions.tolist(),
            "bin_edges": self.baseline_edges.tolist(),
            "n_bins": n_bins,
            "level": self._get_psi_level(psi_value),
        }

    def _calculate_psi(
        self,
        expected: np.ndarray,
        actual: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        expected_safe = np.clip(expected, self.epsilon, 1 - self.epsilon)
        actual_safe = np.clip(actual, self.epsilon, 1 - self.epsilon)

        bin_contributions = (actual_safe - expected_safe) * np.log(
            actual_safe / expected_safe
        )
        psi_value = np.sum(bin_contributions)

        return float(psi_value), bin_contributions

    def _get_psi_level(self, psi_value: float) -> str:
        if psi_value < 0.1:
            return "no_drift"
        elif psi_value < 0.2:
            return "slight_drift"
        else:
            return "severe_drift"

    def calculate(
        self,
        baseline_data: np.ndarray,
        production_data: np.ndarray,
    ) -> Dict[str, Any]:
        self.fit(baseline_data)
        return self.transform(production_data)

    def calculate_categorical(
        self,
        baseline_data: np.ndarray,
        production_data: np.ndarray,
    ) -> Dict[str, Any]:
        baseline_data = np.asarray(baseline_data).astype(str)
        production_data = np.asarray(production_data).astype(str)

        all_categories = np.union1d(np.unique(baseline_data), np.unique(production_data))

        baseline_counts = pd.Series(baseline_data).value_counts().reindex(all_categories, fill_value=0)
        production_counts = pd.Series(production_data).value_counts().reindex(all_categories, fill_value=0)

        baseline_ratios = (baseline_counts / baseline_counts.sum()).values
        production_ratios = (production_counts / production_counts.sum()).values

        psi_value, bin_contributions = self._calculate_psi(
            baseline_ratios, production_ratios
        )

        return {
            "psi": float(psi_value),
            "baseline_ratios": baseline_ratios.tolist(),
            "production_ratios": production_ratios.tolist(),
            "bin_contributions": bin_contributions.tolist(),
            "categories": all_categories.tolist(),
            "level": self._get_psi_level(psi_value),
        }
