from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum


class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Alert:
    feature: str
    metric: str
    value: float
    threshold: float
    level: AlertLevel
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[str] = None


class ThresholdChecker:
    def __init__(
        self,
        psi_warning: float = 0.1,
        psi_critical: float = 0.2,
        p_value_threshold: float = 0.05,
        performance_drop_threshold: float = 0.05,
    ):
        self.psi_warning = psi_warning
        self.psi_critical = psi_critical
        self.p_value_threshold = p_value_threshold
        self.performance_drop_threshold = performance_drop_threshold

    def check_psi(self, feature: str, psi_value: float) -> Optional[Alert]:
        if psi_value >= self.psi_critical:
            return Alert(
                feature=feature,
                metric="PSI",
                value=psi_value,
                threshold=self.psi_critical,
                level=AlertLevel.CRITICAL,
                message=f"Severe drift detected in feature '{feature}': PSI = {psi_value:.4f}",
                details={
                    "psi_value": psi_value,
                    "threshold_warning": self.psi_warning,
                    "threshold_critical": self.psi_critical,
                    "level": "severe_drift",
                },
            )
        elif psi_value >= self.psi_warning:
            return Alert(
                feature=feature,
                metric="PSI",
                value=psi_value,
                threshold=self.psi_warning,
                level=AlertLevel.WARNING,
                message=f"Slight drift detected in feature '{feature}': PSI = {psi_value:.4f}",
                details={
                    "psi_value": psi_value,
                    "threshold_warning": self.psi_warning,
                    "threshold_critical": self.psi_critical,
                    "level": "slight_drift",
                },
            )
        return None

    def check_ks(self, feature: str, ks_result: Dict[str, Any]) -> Optional[Alert]:
        p_value = ks_result.get("p_value", 1.0)
        statistic = ks_result.get("statistic", 0.0)

        if p_value < self.p_value_threshold:
            level = AlertLevel.CRITICAL if statistic > 0.3 else AlertLevel.WARNING
            return Alert(
                feature=feature,
                metric="KS_test",
                value=p_value,
                threshold=self.p_value_threshold,
                level=level,
                message=(
                    f"Significant distribution change in '{feature}': "
                    f"KS statistic = {statistic:.4f}, p-value = {p_value:.6f}"
                ),
                details={
                    "statistic": statistic,
                    "p_value": p_value,
                    "threshold": self.p_value_threshold,
                    "is_significant": True,
                },
            )
        return None

    def check_chi2(self, feature: str, chi2_result: Dict[str, Any]) -> Optional[Alert]:
        p_value = chi2_result.get("p_value", 1.0)
        statistic = chi2_result.get("statistic", 0.0)

        if p_value < self.p_value_threshold:
            level = AlertLevel.CRITICAL if statistic > 10 else AlertLevel.WARNING
            return Alert(
                feature=feature,
                metric="Chi2_test",
                value=p_value,
                threshold=self.p_value_threshold,
                level=level,
                message=(
                    f"Significant category distribution change in '{feature}': "
                    f"Chi2 = {statistic:.4f}, p-value = {p_value:.6f}"
                ),
                details={
                    "statistic": statistic,
                    "p_value": p_value,
                    "threshold": self.p_value_threshold,
                    "is_significant": True,
                },
            )
        return None

    def check_performance(
        self,
        performance_result: Dict[str, Any],
    ) -> Optional[Alert]:
        accuracy_drop = performance_result.get("drops", {}).get("accuracy", 0.0)
        is_degraded = performance_result.get("is_degraded", False)
        severity = performance_result.get("severity", "minor")

        if is_degraded:
            level_map = {
                "minor": AlertLevel.WARNING,
                "moderate": AlertLevel.CRITICAL,
                "severe": AlertLevel.CRITICAL,
            }
            level = level_map.get(severity, AlertLevel.WARNING)

            return Alert(
                feature="model_performance",
                metric="accuracy_drop",
                value=accuracy_drop,
                threshold=self.performance_drop_threshold,
                level=level,
                message=(
                    f"Model performance {severity}ly degraded: "
                    f"accuracy dropped by {accuracy_drop:.2%} "
                    f"(threshold: {self.performance_drop_threshold:.0%})"
                ),
                details={
                    "accuracy_drop": accuracy_drop,
                    "threshold": self.performance_drop_threshold,
                    "severity": severity,
                    "current_metrics": performance_result.get("current", {}),
                    "baseline_metrics": performance_result.get("baseline", {}),
                },
            )
        return None

    def check_all(
        self,
        drift_results: Dict[str, Any],
        performance_result: Optional[Dict[str, Any]] = None,
    ) -> List[Alert]:
        alerts = []

        for feature, results in drift_results.items():
            if "psi" in results:
                psi_alert = self.check_psi(feature, results["psi"].get("psi", 0.0))
                if psi_alert:
                    alerts.append(psi_alert)

            if "ks" in results:
                ks_alert = self.check_ks(feature, results["ks"])
                if ks_alert:
                    alerts.append(ks_alert)

            if "chi2" in results:
                chi2_alert = self.check_chi2(feature, results["chi2"])
                if chi2_alert:
                    alerts.append(chi2_alert)

        if performance_result:
            perf_alert = self.check_performance(performance_result)
            if perf_alert:
                alerts.append(perf_alert)

        return alerts

    def get_alert_summary(self, alerts: List[Alert]) -> Dict[str, Any]:
        if not alerts:
            return {
                "total": 0,
                "critical": 0,
                "warning": 0,
                "info": 0,
                "has_alerts": False,
            }

        counts = {
            AlertLevel.CRITICAL: 0,
            AlertLevel.WARNING: 0,
            AlertLevel.INFO: 0,
        }
        for alert in alerts:
            counts[alert.level] += 1

        return {
            "total": len(alerts),
            "critical": counts[AlertLevel.CRITICAL],
            "warning": counts[AlertLevel.WARNING],
            "info": counts[AlertLevel.INFO],
            "has_alerts": len(alerts) > 0,
            "critical_features": [
                a.feature for a in alerts if a.level == AlertLevel.CRITICAL
            ],
            "warning_features": [
                a.feature for a in alerts if a.level == AlertLevel.WARNING
            ],
        }
