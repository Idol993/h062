from typing import Dict, Any, Optional, List
import json
import os
import threading
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from prometheus_client import (
    Gauge,
    Counter,
    Histogram,
    start_http_server,
    REGISTRY,
    CollectorRegistry,
    generate_latest,
)


class MetricsExporter:
    def __init__(
        self,
        port: int = 8000,
        registry: Optional[CollectorRegistry] = None,
        data_dir: str = ".",
    ):
        self.port = port
        self.registry = registry or REGISTRY
        self.data_dir = data_dir
        self._gauges: Dict[str, Gauge] = {}
        self._counters: Dict[str, Counter] = {}
        self._histograms: Dict[str, Histogram] = {}
        self._server_started = False
        self._latest_json: Optional[Dict[str, Any]] = None
        self._trend_data: List[Dict[str, Any]] = []

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
        self._load_latest_json()
        self._load_trend_data()

        exporter_ref = self

        class DashboardHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/metrics":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(generate_latest(exporter_ref.registry))
                elif self.path == "/" or self.path == "/dashboard":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.end_headers()
                    html = exporter_ref._render_dashboard()
                    self.wfile.write(html.encode("utf-8"))
                elif self.path == "/api/latest":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.end_headers()
                    data = exporter_ref._latest_json or {}
                    self.wfile.write(json.dumps(data, ensure_ascii=False, default=str).encode("utf-8"))
                elif self.path == "/api/trend":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(json.dumps(exporter_ref._trend_data, ensure_ascii=False, default=str).encode("utf-8"))
                else:
                    self.send_response(404)
                    self.send_header("Content-Type", "text/plain")
                    self.end_headers()
                    self.wfile.write(b"404 Not Found")

            def log_message(self, format, *args):
                pass

        server = HTTPServer(("0.0.0.0", server_port), DashboardHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self._server_started = True
        print(f"Monitoring server started on port {server_port}")
        print(f"  Dashboard:    http://localhost:{server_port}/")
        print(f"  Prometheus:   http://localhost:{server_port}/metrics")
        print(f"  Latest JSON:  http://localhost:{server_port}/api/latest")
        print(f"  Trend data:   http://localhost:{server_port}/api/trend")

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

    def update_latest_result(self, result: Dict[str, Any]) -> None:
        self._latest_json = result
        self._append_trend_point(result)

    def _load_latest_json(self) -> None:
        json_files = sorted(
            [f for f in os.listdir(self.data_dir) if f.startswith("drift_metrics_") and f.endswith(".json")],
            reverse=True,
        )
        if not json_files:
            return
        try:
            with open(os.path.join(self.data_dir, json_files[0]), "r", encoding="utf-8") as f:
                self._latest_json = json.load(f)
        except Exception:
            pass

    def _load_trend_data(self) -> None:
        json_files = sorted(
            [f for f in os.listdir(self.data_dir) if f.startswith("drift_metrics_") and f.endswith(".json")],
        )
        now = datetime.now()
        for jf in json_files[-60:]:
            try:
                with open(os.path.join(self.data_dir, jf), "r", encoding="utf-8") as f:
                    data = json.load(f)
                ts_str = data.get("generated_at", "")
                try:
                    ts = datetime.fromisoformat(ts_str)
                except (ValueError, TypeError):
                    continue
                if (now - ts).days <= 30:
                    self._append_trend_point(data)
            except Exception:
                continue

    def _append_trend_point(self, data: Dict[str, Any]) -> None:
        drift_results = data.get("drift_results", {})
        point = {
            "timestamp": data.get("generated_at", ""),
        }
        for feature, results in drift_results.items():
            psi_info = results.get("psi", {})
            if isinstance(psi_info, dict) and "psi" in psi_info:
                point[f"psi_{feature}"] = psi_info["psi"]
        perf = data.get("performance")
        if perf:
            point["accuracy"] = perf.get("current", {}).get("accuracy")
            point["accuracy_drop"] = perf.get("drops", {}).get("accuracy")
        self._trend_data.append(point)

    def _render_dashboard(self) -> str:
        latest = self._latest_json or {}
        trend = self._trend_data[-30:] if self._trend_data else []

        feature_rows = ""
        drift_results = latest.get("drift_results", {})
        for feature, results in drift_results.items():
            psi_info = results.get("psi", {})
            psi_val = psi_info.get("psi", 0) if isinstance(psi_info, dict) else 0
            level = psi_info.get("level", "unknown") if isinstance(psi_info, dict) else "unknown"
            color = "#d62728" if level == "severe_drift" else "#ff7f0e" if level == "slight_drift" else "#2ca02c"

            ks_info = results.get("ks", {})
            chi2_info = results.get("chi2", {})
            test_val = ""
            if isinstance(ks_info, dict) and ks_info:
                test_val = f"KS stat={ks_info.get('statistic', 0):.4f}, p={ks_info.get('p_value', 1):.6f}"
            elif isinstance(chi2_info, dict) and chi2_info:
                test_val = f"Chi2={chi2_info.get('statistic', 0):.4f}, p={chi2_info.get('p_value', 1):.6f}"

            feature_rows += f"""
            <tr>
                <td><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:{color};margin-right:8px;"></span>{feature}</td>
                <td style="color:{color};font-weight:bold;">{psi_val:.4f}</td>
                <td>{level.replace('_', ' ').title()}</td>
                <td style="font-size:12px;">{test_val}</td>
            </tr>"""

        perf_section = ""
        perf = latest.get("performance")
        if perf:
            ba = perf.get("baseline", {}).get("accuracy")
            ca = perf.get("current", {}).get("accuracy")
            drop = perf.get("drops", {}).get("accuracy", 0)
            drop_color = "#d62728" if drop > 0.05 else "#ff7f0e" if drop > 0 else "#2ca02c"
            perf_section = f"""
            <div class="card">
                <h3>Model Performance</h3>
                <div class="perf-grid">
                    <div class="perf-item">
                        <div class="perf-label">Current Accuracy</div>
                        <div class="perf-value">{ca*100:.1f}%</div>
                    </div>
                    <div class="perf-item">
                        <div class="perf-label">Baseline Accuracy</div>
                        <div class="perf-value">{ba*100:.1f}%</div>
                    </div>
                    <div class="perf-item">
                        <div class="perf-label">Accuracy Drop</div>
                        <div class="perf-value" style="color:{drop_color}">{drop*100:.1f}%</div>
                    </div>
                </div>
            </div>"""

        trend_json = json.dumps(trend, default=str)

        return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Drift Monitoring Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; background:#f5f7fa; color:#333; padding:20px; }}
.header {{ background:linear-gradient(135deg,#667eea,#764ba2); color:#fff; padding:25px 30px; border-radius:10px; margin-bottom:24px; }}
.header h1 {{ font-size:24px; margin-bottom:6px; }}
.header p {{ opacity:.85; font-size:13px; }}
.grid {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; margin-bottom:24px; }}
.card {{ background:#fff; padding:24px; border-radius:10px; box-shadow:0 2px 4px rgba(0,0,0,.05); }}
.card h3 {{ font-size:16px; margin-bottom:16px; padding-bottom:8px; border-bottom:2px solid #eee; }}
table {{ width:100%; border-collapse:collapse; font-size:13px; }}
th,td {{ padding:10px 12px; text-align:left; border-bottom:1px solid #eee; }}
th {{ background:#f8f9fa; font-weight:600; color:#555; }}
.chart-wrap {{ position:relative; height:350px; }}
.perf-grid {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:16px; }}
.perf-item {{ text-align:center; padding:16px; background:#f8f9fa; border-radius:8px; }}
.perf-label {{ font-size:12px; color:#666; margin-bottom:4px; }}
.perf-value {{ font-size:24px; font-weight:bold; color:#1f77b4; }}
.footer {{ text-align:center; color:#999; margin-top:30px; font-size:12px; }}
</style>
</head>
<body>
<div class="header">
    <h1>🔍 Drift Monitoring Dashboard</h1>
    <p>Last detection: {latest.get('generated_at', 'No data yet')} &nbsp;|&nbsp;
       <a href="/metrics" style="color:#ddd;">Prometheus /metrics</a> &nbsp;|&nbsp;
       <a href="/api/latest" style="color:#ddd;">Latest JSON</a></p>
</div>
{perf_section}
<div class="grid">
    <div class="card">
        <h3>Feature Drift Summary</h3>
        <table>
            <thead><tr><th>Feature</th><th>PSI</th><th>Level</th><th>Test</th></tr></thead>
            <tbody>{feature_rows or '<tr><td colspan="4" style="text-align:center;color:#999;">No data</td></tr>'}</tbody>
        </table>
    </div>
    <div class="card">
        <h3>PSI Trend (Last 30 Days)</h3>
        <div class="chart-wrap"><canvas id="trendChart"></canvas></div>
    </div>
</div>
<div class="footer">Drift Monitoring Dashboard &mdash; Auto-refreshes every 60s</div>
<script>
const trendData = {trend_json};
if (trendData.length > 0) {{
    const labels = trendData.map(d => d.timestamp ? d.timestamp.slice(0,16) : '');
    const psiKeys = Object.keys(trendData[0]).filter(k => k.startsWith('psi_'));
    const colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f'];
    const datasets = psiKeys.map((k, i) => ({{
        label: k.replace('psi_',''),
        data: trendData.map(d => d[k]),
        borderColor: colors[i % colors.length],
        tension: 0.3,
        fill: false,
        pointRadius: 3,
    }}));
    datasets.push({{
        label: 'PSI Critical (0.2)',
        data: labels.map(() => 0.2),
        borderColor: '#d62728',
        borderDash: [6,4],
        pointRadius: 0,
        fill: false,
    }});
    datasets.push({{
        label: 'PSI Warning (0.1)',
        data: labels.map(() => 0.1),
        borderColor: '#ff7f0e',
        borderDash: [6,4],
        pointRadius: 0,
        fill: false,
    }});
    new Chart(document.getElementById('trendChart'), {{
        type: 'line',
        data: {{ labels, datasets }},
        options: {{
            responsive: true,
            maintainAspectRatio: false,
            scales: {{ y: {{ beginAtZero: true, title: {{ display: true, text: 'PSI' }} }} }},
        }},
    }});
}}
setTimeout(() => location.reload(), 60000);
</script>
</body>
</html>"""
