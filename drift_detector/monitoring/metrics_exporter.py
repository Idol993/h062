from typing import Dict, Any, Optional, List
from prometheus_client import (
    Gauge,
    Counter,
    Histogram,
    start_http_server,
    REGISTRY,
    CollectorRegistry,
)


class MetricsExporter:
    def __init__(
        self,
        port: int = 8000,
        registry: Optional[CollectorRegistry] = None,
    ):
        self.port = port
        self.registry = registry or REGISTRY
        self._gauges: Dict[str, Gauge] = {}
        self._counters: Dict[str, Counter] = {}
        self._histograms: Dict[str, Histogram] = {}
        self._server_started = False

        self._init_metrics()

    def _init_metrics(self) -> None:
        self._gauges["drift_psi"] = Gauge(
            "drift_psi",
            "PSI value for feature distribution drift",
            labelnames=["feature"],
            registry=self.registry,
        )

        self._gauges["drift_ks_statistic"] = Gauge(
            "drift_ks_statistic",
            "KS test statistic for feature distribution",
            labelnames=["feature"],
            registry=self.registry,
        )

        self._gauges["drift_ks_pvalue"] = Gauge(
            "drift_ks_pvalue",
            "KS test p-value for feature distribution",
            labelnames=["feature"],
            registry=self.registry,
        )

        self._gauges["drift_chi2_statistic"] = Gauge(
            "drift_chi2_statistic",
            "Chi-square test statistic",
            labelnames=["feature"],
            registry=self.registry,
        )

        self._gauges["drift_chi2_pvalue"] = Gauge(
            "drift_chi2_pvalue",
            "Chi-square test p-value",
            labelnames=["feature"],
            registry=self.registry,
        )

        self._gauges["model_accuracy"] = Gauge(
            "model_accuracy",
            "Current model accuracy",
            registry=self.registry,
        )

        self._gauges["model_accuracy_drop"] = Gauge(
            "model_accuracy_drop",
            "Model accuracy drop from baseline",
            registry=self.registry,
        )

        self._gauges["model_f1"] = Gauge(
            "model_f1",
            "Current model F1 score",
            registry=self.registry,
        )

        self._gauges["alert_count"] = Gauge(
            "drift_alert_count",
            "Number of drift alerts",
            labelnames=["level", "feature"],
            registry=self.registry,
        )

        self._counters["detection_runs"] = Counter(
            "drift_detection_runs_total",
            "Total number of drift detection runs",
            registry=self.registry,
        )

        self._histograms["detection_duration"] = Histogram(
            "drift_detection_duration_seconds",
            "Time taken for drift detection",
            registry=self.registry,
        )

    def start_server(self, port: Optional[int] = None) -> None:
        if self._server_started:
            return

        server_port = port or self.port
        start_http_server(server_port, registry=self.registry)
        self._server_started = True
        print(f"Prometheus metrics server started on port {server_port}")

    def update_psi(self, feature: str, psi_value: float) -> None:
        self._gauges["drift_psi"].labels(feature=feature).set(psi_value)

    def update_ks(self, feature: str, statistic: float, p_value: float) -> None:
        self._gauges["drift_ks_statistic"].labels(feature=feature).set(statistic)
        self._gauges["drift_ks_pvalue"].labels(feature=feature).set(p_value)

    def update_chi2(self, feature: str, statistic: float, p_value: float) -> None:
        self._gauges["drift_chi2_statistic"].labels(feature=feature).set(statistic)
        self._gauges["drift_chi2_pvalue"].labels(feature=feature).set(p_value)

    def update_performance(
        self,
        accuracy: float,
        accuracy_drop: float,
        f1: Optional[float] = None,
    ) -> None:
        self._gauges["model_accuracy"].set(accuracy)
        self._gauges["model_accuracy_drop"].set(accuracy_drop)
        if f1 is not None:
            self._gauges["model_f1"].set(f1)

    def update_alerts(self, alerts: List[Any]) -> None:
        self._gauges["alert_count"].clear()
        for alert in alerts:
            self._gauges["alert_count"].labels(
                level=alert.level.value,
                feature=alert.feature,
            ).inc()

    def update_drift_results(self, drift_results: Dict[str, Any]) -> None:
        for feature, results in drift_results.items():
            if "psi" in results:
                self.update_psi(feature, results["psi"].get("psi", 0.0))

            if "ks" in results:
                self.update_ks(
                    feature,
                    results["ks"].get("statistic", 0.0),
                    results["ks"].get("p_value", 1.0),
                )

            if "chi2" in results:
                self.update_chi2(
                    feature,
                    results["chi2"].get("statistic", 0.0),
                    results["chi2"].get("p_value", 1.0),
                )

    def increment_detection_runs(self) -> None:
        self._counters["detection_runs"].inc()

    def observe_duration(self, duration: float) -> None:
        self._histograms["detection_duration"].observe(duration)

    def is_server_started(self) -> bool:
        return self._server_started
