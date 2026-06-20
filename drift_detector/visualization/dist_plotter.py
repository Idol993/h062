from typing import Optional, Dict, Any, List
import os
import base64
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path


class DistributionPlotter:
    def __init__(
        self,
        output_dir: str = "plots",
        dpi: int = 100,
        figsize: tuple = (12, 6),
    ):
        self.output_dir = output_dir
        self.dpi = dpi
        self.figsize = figsize
        self._ensure_output_dir()

    def _ensure_output_dir(self) -> None:
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def plot_numerical(
        self,
        feature_name: str,
        baseline_data: np.ndarray,
        production_data: np.ndarray,
        psi_result: Optional[Dict[str, Any]] = None,
        save_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        baseline_data = np.asarray(baseline_data).flatten()
        production_data = np.asarray(production_data).flatten()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize, dpi=self.dpi)

        all_data = np.concatenate([baseline_data, production_data])
        bins = 50
        hist_range = (np.min(all_data), np.max(all_data))

        ax1.hist(
            baseline_data,
            bins=bins,
            range=hist_range,
            alpha=0.7,
            label="Baseline",
            color="#1f77b4",
            density=True,
            edgecolor="black",
            linewidth=0.5,
        )
        ax1.hist(
            production_data,
            bins=bins,
            range=hist_range,
            alpha=0.5,
            label="Production",
            color="#ff7f0e",
            density=True,
            edgecolor="black",
            linewidth=0.5,
        )
        ax1.set_title(f"Distribution Comparison: {feature_name}", fontsize=14, fontweight="bold")
        ax1.set_xlabel("Value", fontsize=12)
        ax1.set_ylabel("Density", fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        if psi_result and "bin_edges" in psi_result:
            edges = np.array(psi_result["bin_edges"])
            baseline_ratios = np.array(psi_result["baseline_ratios"])
            production_ratios = np.array(psi_result["production_ratios"])
            bin_contributions = np.array(psi_result["bin_contributions"])

            bin_centers = (edges[:-1] + edges[1:]) / 2
            width = np.diff(edges)

            ax2.bar(
                bin_centers - width / 4,
                baseline_ratios,
                width=width / 2,
                alpha=0.7,
                label="Baseline",
                color="#1f77b4",
                edgecolor="black",
            )
            ax2.bar(
                bin_centers + width / 4,
                production_ratios,
                width=width / 2,
                alpha=0.7,
                label="Production",
                color="#ff7f0e",
                edgecolor="black",
            )

            ax2_twin = ax2.twinx()
            colors = []
            for contrib in bin_contributions:
                if contrib > 0.05:
                    colors.append("#d62728")
                elif contrib > 0.01:
                    colors.append("#ff7f0e")
                else:
                    colors.append("#2ca02c")

            ax2_twin.bar(
                bin_centers,
                bin_contributions,
                width=width * 0.3,
                alpha=0.4,
                color=colors,
                label="PSI Contribution",
            )
            ax2_twin.set_ylabel("PSI Contribution", fontsize=12)

            psi_value = psi_result.get("psi", 0.0)
            psi_level = psi_result.get("level", "unknown")
            level_colors = {
                "no_drift": "#2ca02c",
                "slight_drift": "#ff7f0e",
                "severe_drift": "#d62728",
            }
            color = level_colors.get(psi_level, "#1f77b4")

            ax2.set_title(
                f"PSI = {psi_value:.4f} ({psi_level.replace('_', ' ').title()})",
                fontsize=14,
                fontweight="bold",
                color=color,
            )

        ax2.set_xlabel("Value", fontsize=12)
        ax2.set_ylabel("Ratio", fontsize=12)
        ax2.legend(loc="upper left")
        ax2_twin.legend(loc="upper right")
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.output_dir, f"feature_{feature_name}_dist.png")

        fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        base64_img = self._fig_to_base64(fig)

        return {
            "path": save_path,
            "base64": base64_img,
            "feature": feature_name,
        }

    def plot_categorical(
        self,
        feature_name: str,
        baseline_data: np.ndarray,
        production_data: np.ndarray,
        chi2_result: Optional[Dict[str, Any]] = None,
        save_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        baseline_data = np.asarray(baseline_data).astype(str)
        production_data = np.asarray(production_data).astype(str)

        all_categories = sorted(
            set(np.unique(baseline_data)) | set(np.unique(production_data))
        )

        baseline_counts = (
            pd.Series(baseline_data)
            .value_counts()
            .reindex(all_categories, fill_value=0)
        )
        production_counts = (
            pd.Series(production_data)
            .value_counts()
            .reindex(all_categories, fill_value=0)
        )

        baseline_ratios = baseline_counts / baseline_counts.sum()
        production_ratios = production_counts / production_counts.sum()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize, dpi=self.dpi)

        x = np.arange(len(all_categories))
        width = 0.35

        bars1 = ax1.bar(
            x - width / 2,
            baseline_ratios.values,
            width,
            label="Baseline",
            color="#1f77b4",
            edgecolor="black",
        )
        bars2 = ax1.bar(
            x + width / 2,
            production_ratios.values,
            width,
            label="Production",
            color="#ff7f0e",
            edgecolor="black",
        )

        ax1.set_title(
            f"Category Distribution: {feature_name}",
            fontsize=14,
            fontweight="bold",
        )
        ax1.set_xlabel("Category", fontsize=12)
        ax1.set_ylabel("Ratio", fontsize=12)
        ax1.set_xticks(x)
        ax1.set_xticklabels(all_categories, rotation=45, ha="right")
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3, axis="y")

        diff = production_ratios.values - baseline_ratios.values
        colors = ["#2ca02c" if abs(d) < 0.05 else "#d62728" if abs(d) > 0.1 else "#ff7f0e"
        for d in diff]

        ax2.bar(x, diff, color=colors, edgecolor="black")
        ax2.axhline(y=0, color="black", linewidth=0.8)
        ax2.set_title("Difference (Production - Baseline)", fontsize=14, fontweight="bold")
        ax2.set_xlabel("Category", fontsize=12)
        ax2.set_ylabel("Ratio Difference", fontsize=12)
        ax2.set_xticks(x)
        ax2.set_xticklabels(all_categories, rotation=45, ha="right")
        ax2.grid(True, alpha=0.3, axis="y")

        if chi2_result:
            p_value = chi2_result.get("p_value", 1.0)
            is_significant = chi2_result.get("is_significant", False)
            color = "#d62728" if is_significant else "#2ca02c"
            ax2.set_title(
                f"Chi2 p-value = {p_value:.6f}\n{'Significant' if is_significant else 'Not Significant'}",
                fontsize=12,
                color=color,
            )

        fig.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.output_dir, f"feature_{feature_name}_dist.png")

        fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        base64_img = self._fig_to_base64(fig)

        return {
            "path": save_path,
            "base64": base64_img,
            "feature": feature_name,
        }

    def _fig_to_base64(self, fig) -> str:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=self.dpi, bbox_inches="tight")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        return img_base64
