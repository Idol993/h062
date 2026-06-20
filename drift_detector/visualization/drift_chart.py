from typing import Optional, Dict, Any, List
import os
import base64
import io
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


class DriftChart:
    def __init__(
        self,
        output_dir: str = "plots",
        dpi: int = 100,
        figsize: tuple = (14, 7),
    ):
        self.output_dir = output_dir
        self.dpi = dpi
        self.figsize = figsize
        self._ensure_output_dir()

    def _ensure_output_dir(self) -> None:
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def plot_psi_trend(
        self,
        feature_name: str,
        dates: List[datetime],
        psi_values: List[float],
        warning_threshold: float = 0.1,
        critical_threshold: float = 0.2,
        save_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        dates_sorted, psi_sorted = zip(*sorted(zip(dates, psi_values)))

        ax.plot(
            dates_sorted,
            psi_sorted,
            marker="o",
            linewidth=2,
            markersize=6,
            color="#1f77b4",
            label="PSI Value",
            zorder=5,
        )

        ax.axhline(
            y=warning_threshold,
            color="#ff7f0e",
            linestyle="--",
            linewidth=2,
            label=f"Warning ({warning_threshold})",
            alpha=0.8,
        )
        ax.axhline(
            y=critical_threshold,
            color="#d62728",
            linestyle="--",
            linewidth=2,
            label=f"Critical ({critical_threshold})",
            alpha=0.8,
        )

        ax.fill_between(
            dates_sorted,
            warning_threshold,
            critical_threshold,
            color="#ff7f0e",
            alpha=0.1,
        )
        ax.fill_between(
            dates_sorted,
            critical_threshold,
            max(critical_threshold, max(psi_sorted) * 1.1),
            color="#d62728",
            alpha=0.1,
        )

        for i, (date, psi) in enumerate(zip(dates_sorted, psi_sorted)):
            if psi >= critical_threshold:
                color = "#d62728"
            elif psi >= warning_threshold:
                color = "#ff7f0e"
            else:
                color = "#2ca02c"
            ax.scatter([date], [psi], color=color, s=100, zorder=10)

        ax.set_title(
            f"PSI Trend: {feature_name}",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("PSI Value", fontsize=12)
        ax.legend(fontsize=10, loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

        plt.xticks(rotation=45, ha="right")
        fig.autofmt_xdate()

        fig.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.output_dir, f"psi_trend_{feature_name}.png")

        fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        base64_img = self._fig_to_base64(fig)

        return {
            "path": save_path,
            "base64": base64_img,
            "feature": feature_name,
        }

    def plot_overall_dashboard(
        self,
        drift_results: Dict[str, Any],
        save_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        features = list(drift_results.keys())
        n_features = len(features)

        if n_features == 0:
            return {"path": None, "base64": None, "feature": "overview"}

        ncols = min(3, n_features)
        nrows = (n_features + ncols - 1) // ncols

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(self.figsize[0], 4 * nrows),
            dpi=self.dpi,
        )
        if nrows == 1 and ncols == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for i, feature in enumerate(features):
            ax = axes[i]
            result = drift_results[feature]
            psi_value = result.get("psi", {}).get("psi", 0.0) if "psi" in result else 0.0
            ks_p = result.get("ks", {}).get("p_value", 1.0) if "ks" in result else 1.0
            chi2_p = result.get("chi2", {}).get("p_value", 1.0) if "chi2" in result else 1.0

            metrics = ["PSI", "KS p-value", "Chi2 p-value"]
            values = [psi_value, -np.log10(ks_p), -np.log10(chi2_p)]
            colors = []
            for v in values:
                if v > 2:
                    colors.append("#d62728")
                elif v > 1:
                    colors.append("#ff7f0e")
                else:
                    colors.append("#2ca02c")

            bars = ax.bar(metrics, values, color=colors, edgecolor="black")
            ax.axhline(y=1.3, color="#ff7f0e", linestyle="--", alpha=0.7)
            ax.axhline(y=2, color="#d62728", linestyle="--", alpha=0.7)

            ax.set_title(feature, fontsize=12, fontweight="bold")
            ax.set_ylabel("Value", fontsize=10)
            ax.grid(True, alpha=0.3, axis="y")

            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                height,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                )

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(
            "Drift Detection Overview",
            fontsize=16,
            fontweight="bold",
            y=1.02,
        )
        fig.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.output_dir, "drift_overview.png")

        fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        base64_img = self._fig_to_base64(fig)

        return {
            "path": save_path,
            "base64": base64_img,
            "feature": "overview",
        }

    def plot_performance_trend(
        self,
        dates: List[datetime],
        accuracies: List[float],
        baseline_accuracy: float,
        threshold: float = 0.05,
        save_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        dates_sorted, acc_sorted = zip(*sorted(zip(dates, accuracies)))

        ax.plot(
            dates_sorted,
            acc_sorted,
            marker="o",
            linewidth=2,
            markersize=6,
            color="#1f77b4",
            label="Current Accuracy",
            zorder=5,
        )

        ax.axhline(
            y=baseline_accuracy,
            color="#2ca02c",
            linestyle="-",
            linewidth=2,
            label=f"Baseline ({baseline_accuracy:.2%})",
        )
        ax.axhline(
            y=baseline_accuracy - threshold,
            color="#d62728",
            linestyle="--",
            linewidth=2,
            label=f"Alert Threshold ({baseline_accuracy - threshold:.2%})",
        )

        ax.fill_between(
            dates_sorted,
            0,
            baseline_accuracy - threshold,
            color="#d62728",
            alpha=0.1,
        )

        for i, (date, acc) in enumerate(zip(dates_sorted, acc_sorted)):
            if acc < baseline_accuracy - threshold:
                color = "#d62728"
            elif acc < baseline_accuracy:
                color = "#ff7f0e"
            else:
                color = "#2ca02c"
            ax.scatter([date], [acc], color=color, s=100, zorder=10)

        ax.set_title(
            "Model Performance Trend",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.legend(fontsize=10, loc="lower left")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=min(0.5, min(acc_sorted) - 0.05))

        vals = ax.get_yticks()
        ax.set_yticklabels([f"{x:.0%}" for x in vals])

        plt.xticks(rotation=45, ha="right")
        fig.autofmt_xdate()

        fig.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.output_dir, "performance_trend.png")

        fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        base64_img = self._fig_to_base64(fig)

        return {
            "path": save_path,
            "base64": base64_img,
            "feature": "performance",
        }

    def _fig_to_base64(self, fig) -> str:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=self.dpi, bbox_inches="tight")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        return img_base64
