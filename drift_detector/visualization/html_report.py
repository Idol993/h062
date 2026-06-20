from typing import Dict, Any, Optional, List
import os
import json
import base64
from datetime import datetime
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..utils.time_utils import TimeUtils
from ..alerting.threshold_checker import AlertLevel


class HTMLReport:
    def __init__(
        self,
        template_dir: Optional[str] = None,
        output_dir: str = ".",
    ):
        if template_dir is None:
            template_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..",
                "templates",
            )
        self.template_dir = os.path.abspath(template_dir)
        self.output_dir = output_dir
        self.env = Environment(
            loader=FileSystemLoader(self.template_dir),
            autoescape=select_autoescape(["html", "xml"]),
        )

    def _get_level_color(self, level: str) -> str:
        color_map = {
            "no_drift": "#2ca02c",
            "slight_drift": "#ff7f0e",
            "severe_drift": "#d62728",
            "improved": "#2ca02c",
            "minor": "#ff7f0e",
            "moderate": "#ff7f0e",
            "severe": "#d62728",
            AlertLevel.CRITICAL: "#d62728",
            AlertLevel.WARNING: "#ff7f0e",
            AlertLevel.INFO: "#2ca02c",
        }
        return color_map.get(level, "#1f77b4")

    def _get_level_text(self, level: str) -> str:
        text_map = {
            "no_drift": "No Drift",
            "slight_drift": "Slight Drift",
            "severe_drift": "Severe Drift",
            "improved": "Improved",
            "minor": "Minor Degradation",
            "moderate": "Moderate Degradation",
            "severe": "Severe Degradation",
        }
        return text_map.get(str(level), str(level).replace("_", " ").title())

    def generate(
        self,
        drift_results: Dict[str, Any],
        alerts: List[Any],
        performance_result: Optional[Dict[str, Any]] = None,
        validation_result: Optional[Dict[str, Any]] = None,
        plots: Dict[str, Any] = None,
        filename: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        timestamp = TimeUtils.get_timestamp()

        template = self.env.get_template("report.html")

        feature_data = []
        for feature_name, results in drift_results.items():
            psi_result = results.get("psi", {})
            ks_result = results.get("ks", {})
            chi2_result = results.get("chi2", {})

            feature_entry = {
                "name": feature_name,
                "type": results.get("type", "unknown"),
                "psi": {
                    "value": psi_result.get("psi", 0.0),
                    "level": psi_result.get("level", "no_drift"),
                    "level_color": self._get_level_color(psi_result.get("level", "no_drift")),
                    "level_text": self._get_level_text(psi_result.get("level", "no_drift")),
                    "n_bins": psi_result.get("n_bins", 0),
                },
                "ks": {
                    "statistic": ks_result.get("statistic", 0.0),
                    "p_value": ks_result.get("p_value", 1.0),
                    "is_significant": ks_result.get("is_significant", False),
                } if ks_result else None,
                "chi2": {
                    "statistic": chi2_result.get("statistic", 0.0),
                    "p_value": chi2_result.get("p_value", 1.0),
                    "is_significant": chi2_result.get("is_significant", False),
                } if chi2_result else None,
                "plot": plots.get(feature_name, {}).get("base64", None) if plots else None,
            }
            feature_data.append(feature_entry)

        alert_data = []
        for alert in alerts:
            alert_data.append({
                "feature": alert.feature,
                "metric": alert.metric,
                "value": alert.value,
                "threshold": alert.threshold,
                "level": alert.level.value,
                "level_color": self._get_level_color(alert.level),
                "message": alert.message,
                "timestamp": alert.timestamp,
            })

        context = {
            "timestamp": timestamp,
            "generated_at": TimeUtils.get_iso_timestamp(),
            "features": feature_data,
            "alerts": alert_data,
            "alert_summary": {
                "total": len(alerts),
                "critical": sum(1 for a in alerts if a.level == AlertLevel.CRITICAL),
                "warning": sum(1 for a in alerts if a.level == AlertLevel.WARNING),
            },
            "performance": performance_result,
            "validation": validation_result,
            "overview_plot": plots.get("overview", {}).get("base64", None) if plots else None,
            "performance_plot": plots.get("performance", {}).get("base64", None) if plots else None,
            **kwargs,
        }

        html_content = template.render(context)

        if filename is None:
            filename = f"drift_report_{timestamp}.html"

        output_path = os.path.join(self.output_dir, filename)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        json_filename = f"drift_metrics_{timestamp}.json"
        json_path = os.path.join(self.output_dir, json_filename)

        json_data = {
            "timestamp": timestamp,
            "generated_at": TimeUtils.get_iso_timestamp(),
            "drift_results": drift_results,
            "alerts": [
                {
                    "feature": a.feature,
                    "metric": a.metric,
                    "value": a.value,
                    "threshold": a.threshold,
                    "level": a.level.value,
                    "message": a.message,
                    "timestamp": a.timestamp,
                    "details": a.details,
                }
                for a in alerts
            ],
            "performance": performance_result,
            "validation": validation_result,
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        return {
            "html_path": output_path,
            "json_path": json_path,
            "timestamp": timestamp,
        }

    def generate_summary(
        self,
        drift_results: Dict[str, Any],
        alerts: List[Any],
        **kwargs,
    ) -> Dict[str, Any]:
        return self.generate(
            drift_results=drift_results,
            alerts=alerts,
            **kwargs,
        )
