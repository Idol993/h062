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
                path = self.path.split("?")[0]
                if path == "/metrics":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(generate_latest(exporter_ref.registry))
                elif path == "/" or path == "/dashboard":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.end_headers()
                    html = exporter_ref._render_dashboard()
                    self.wfile.write(html.encode("utf-8"))
                elif path == "/api/latest":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.end_headers()
                    data = exporter_ref._latest_json or {}
                    self.wfile.write(json.dumps(data, ensure_ascii=False, default=str).encode("utf-8"))
                elif path == "/api/trend":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(json.dumps(exporter_ref._trend_data, ensure_ascii=False, default=str).encode("utf-8"))
                elif path == "/api/models":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.end_headers()
                    models = exporter_ref._list_models()
                    self.wfile.write(json.dumps(models, ensure_ascii=False, default=str).encode("utf-8"))
                elif path == "/api/history":
                    from urllib.parse import parse_qs, urlparse
                    parsed = urlparse(self.path)
                    params = parse_qs(parsed.query)
                    model = params.get("model", [None])[0]
                    env = params.get("env", [None])[0]
                    limit = int(params.get("limit", [100])[0])
                    history = exporter_ref._list_history(model_filter=model, env_filter=env, limit=limit)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(json.dumps(history, ensure_ascii=False, default=str).encode("utf-8"))
                elif path.startswith("/api/history/"):
                    ts_token = path[len("/api/history/"):]
                    record = exporter_ref._get_history_record(ts_token)
                    if record is None:
                        self.send_response(404)
                        self.send_header("Content-Type", "application/json; charset=utf-8")
                        self.end_headers()
                        self.wfile.write(json.dumps({"error": "not found"}).encode("utf-8"))
                    else:
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json; charset=utf-8")
                        self.end_headers()
                        self.wfile.write(json.dumps(record, ensure_ascii=False, default=str).encode("utf-8"))
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
        print(f"  Model list:   http://localhost:{server_port}/api/models")
        print(f"  History:      http://localhost:{server_port}/api/history")

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
                data = json.load(f)
            if "alert_summary" not in data or data.get("alert_summary") is None:
                alerts = data.get("alerts", []) or []
                data["alert_summary"] = {
                    "total": len(alerts),
                    "critical": sum(1 for a in alerts if a.get("level") == "critical"),
                    "warning": sum(1 for a in alerts if a.get("level") == "warning"),
                    "info": sum(1 for a in alerts if a.get("level") == "info"),
                }
            self._latest_json = data
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

    def _get_sorted_json_files(self) -> List[str]:
        if not os.path.isdir(self.data_dir):
            return []
        return sorted([
            os.path.join(self.data_dir, f)
            for f in os.listdir(self.data_dir)
            if f.startswith("drift_metrics_") and f.endswith(".json")
        ])

    def _build_light_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        meta = data.get("meta") or {}
        alerts_arr = data.get("alerts", []) or []
        drift_results = data.get("drift_results", {}) or {}
        highest_feature = None
        highest_psi = 0.0
        severe = 0
        slight = 0
        for feature, results in drift_results.items():
            psi_info = results.get("psi", {}) or {}
            psi_val = float(psi_info.get("psi", 0.0) or 0.0)
            level = psi_info.get("level", "no_drift")
            if psi_val > highest_psi:
                highest_psi = psi_val
                highest_feature = feature
            if level == "severe_drift":
                severe += 1
            elif level == "slight_drift":
                slight += 1
        perf = data.get("performance") or {}
        drops = perf.get("drops", {}) or {}
        acc_drop = drops.get("accuracy")
        perf_degraded = False
        if acc_drop is not None:
            try:
                perf_degraded = float(acc_drop) > 0.05
            except (TypeError, ValueError):
                perf_degraded = False
        summary = {
            "timestamp": data.get("generated_at", ""),
            "model": meta.get("model_name", "default_model"),
            "env": meta.get("env", "default_env"),
            "run_tag": meta.get("run_tag"),
            "total_alerts": len(alerts_arr),
            "critical_alerts": sum(1 for a in alerts_arr if a.get("level") == "critical"),
            "warning_alerts": sum(1 for a in alerts_arr if a.get("level") == "warning"),
            "severe_drift": severe,
            "slight_drift": slight,
            "no_drift": max(0, len(drift_results) - severe - slight),
            "highest_psi_feature": highest_feature,
            "highest_psi": round(highest_psi, 4),
            "accuracy_drop": round(float(acc_drop), 4) if acc_drop is not None else None,
            "perf_degraded": perf_degraded,
            "production_samples": data.get("production_samples"),
            "window_days": meta.get("window_days"),
        }
        summary["_ts_token"] = summary["timestamp"].replace(":", "-").replace("T", "_")
        return summary

    def _list_models(self) -> List[Dict[str, Any]]:
        files = self._get_sorted_json_files()
        model_set = set()
        out = []
        for jf in files[::-1]:
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue
            meta = data.get("meta") or {}
            key = (meta.get("model_name", "default_model"), meta.get("env", "default_env"))
            if key in model_set:
                continue
            model_set.add(key)
            out.append({
                "model": key[0],
                "env": key[1],
                "last_timestamp": data.get("generated_at", ""),
                "file": os.path.basename(jf),
            })
        return out

    def _list_history(self, model_filter: Optional[str] = None, env_filter: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        files = self._get_sorted_json_files()
        out = []
        for jf in files[::-1]:
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue
            meta = data.get("meta") or {}
            if model_filter and meta.get("model_name", "default_model") != model_filter:
                continue
            if env_filter and meta.get("env", "default_env") != env_filter:
                continue
            summary = self._build_light_summary(data)
            summary["_file"] = os.path.basename(jf)
            out.append(summary)
            if len(out) >= limit:
                break
        return out

    def _get_history_record(self, ts_token: str) -> Optional[Dict[str, Any]]:
        files = self._get_sorted_json_files()
        for jf in files[::-1]:
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue
            summary = self._build_light_summary(data)
            if summary.get("_ts_token") == ts_token:
                meta = data.get("meta") or {}
                alerts_arr = data.get("alerts", []) or []
                drift_results = data.get("drift_results", {}) or {}
                features_out = []
                for feature, results in drift_results.items():
                    psi_info = results.get("psi", {}) or {}
                    ks_info = results.get("ks", {}) or None
                    chi2_info = results.get("chi2", {}) or None
                    entry = {
                        "feature": feature,
                        "type": results.get("type", "unknown"),
                        "psi": {
                            "value": psi_info.get("psi", 0.0),
                            "level": psi_info.get("level", "no_drift"),
                        },
                    }
                    if ks_info:
                        entry["ks"] = {"statistic": ks_info.get("statistic"), "p_value": ks_info.get("p_value")}
                    if chi2_info:
                        entry["chi2"] = {"statistic": chi2_info.get("statistic"), "p_value": chi2_info.get("p_value")}
                    features_out.append(entry)
                perf = data.get("performance")
                return {
                    "summary": {k: v for k, v in summary.items() if not k.startswith("_")},
                    "meta": meta,
                    "features": features_out,
                    "alerts": alerts_arr[:50],
                    "performance": perf,
                    "generated_at": data.get("generated_at"),
                    "file": os.path.basename(jf),
                }
        return None

    def _render_dashboard(self) -> str:
        latest = self._latest_json or {}
        meta = latest.get("meta") or {}
        summary = latest.get("alert_summary") or {}

        return """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Drift Monitoring Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; background:#f5f7fa; color:#333; padding:20px; }
.header { background:linear-gradient(135deg,#667eea,#764ba2); color:#fff; padding:25px 30px; border-radius:10px; margin-bottom:24px; }
.header h1 { font-size:24px; margin-bottom:6px; }
.meta-bar { display:flex; flex-wrap:wrap; gap:16px; margin-top:14px; font-size:13px; align-items:center; }
.meta-bar .tag { background:rgba(255,255,255,.15); padding:4px 10px; border-radius:20px; }
.meta-bar a { color:#eee; }
.layout { display:grid; grid-template-columns: 340px 1fr; gap:20px; }
@media (max-width: 1000px) { .layout { grid-template-columns: 1fr; } }
.sidebar .card { background:#fff; padding:20px; border-radius:10px; box-shadow:0 2px 4px rgba(0,0,0,.05); margin-bottom:20px; }
.sidebar h3 { font-size:15px; margin-bottom:14px; padding-bottom:8px; border-bottom:2px solid #eee; }
.filter-row { display:flex; flex-direction:column; gap:6px; margin-bottom:12px; }
.filter-row label { font-size:12px; color:#666; font-weight:600; }
.filter-row select, .filter-row input { padding:7px 10px; border:1px solid #ddd; border-radius:6px; font-size:13px; }
.history-list { max-height: 520px; overflow-y:auto; border:1px solid #eee; border-radius:8px; }
.history-item { padding:12px 14px; border-bottom:1px solid #f0f0f0; cursor:pointer; font-size:12px; transition:background .15s; }
.history-item:hover { background:#f0f4ff; }
.history-item.active { background:#eef2ff; border-left:3px solid #667eea; padding-left:11px; }
.history-item .ts { color:#333; font-weight:600; margin-bottom:4px; }
.history-item .sub { color:#888; margin:2px 0; }
.history-item .badge { display:inline-block; padding:1px 7px; border-radius:10px; font-size:10px; font-weight:600; margin-right:4px; }
.badge-sev { background:#ffe0e0; color:#c0392b; }
.badge-sl { background:#fff4e0; color:#e67e22; }
.badge-perf { background:#fadbd8; color:#c0392b; }
.grid-2 { display:grid; grid-template-columns:1fr 1fr; gap:20px; margin-bottom:24px; }
@media (max-width: 700px) { .grid-2 { grid-template-columns:1fr; } }
.card { background:#fff; padding:24px; border-radius:10px; box-shadow:0 2px 4px rgba(0,0,0,.05); }
.card h3 { font-size:16px; margin-bottom:16px; padding-bottom:8px; border-bottom:2px solid #eee; }
table { width:100%; border-collapse:collapse; font-size:13px; }
th,td { padding:10px 12px; text-align:left; border-bottom:1px solid #eee; }
th { background:#f8f9fa; font-weight:600; color:#555; }
.chart-wrap { position:relative; height:350px; }
.perf-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; }
@media (max-width: 700px) { .perf-grid { grid-template-columns:repeat(2,1fr); } }
.perf-item { text-align:center; padding:14px; background:#f8f9fa; border-radius:8px; }
.perf-label { font-size:11px; color:#666; margin-bottom:3px; }
.perf-value { font-size:20px; font-weight:bold; color:#1f77b4; }
.footer { text-align:center; color:#999; margin-top:30px; font-size:12px; }

.modal-mask { position:fixed; inset:0; background:rgba(0,0,0,.5); display:none; align-items:center; justify-content:center; z-index:100; padding:20px; }
.modal-mask.show { display:flex; }
.modal { background:#fff; width:100%; max-width:800px; max-height:85vh; overflow-y:auto; border-radius:12px; padding:28px; box-shadow:0 20px 60px rgba(0,0,0,.3); }
.modal h2 { font-size:20px; margin-bottom:6px; }
.modal .sub-meta { color:#666; font-size:12px; margin-bottom:18px; }
.modal-close { float:right; background:#eee; border:0; padding:6px 14px; border-radius:6px; cursor:pointer; font-weight:600; }
.modal-close:hover { background:#ddd; }
.kv-row { display:grid; grid-template-columns:repeat(3,1fr); gap:10px; margin-bottom:18px; }
@media (max-width: 600px) { .kv-row { grid-template-columns:1fr 1fr; } }
.kv-item { background:#f8f9fa; padding:10px 12px; border-radius:8px; }
.kv-item .k { font-size:11px; color:#888; }
.kv-item .v { font-size:15px; font-weight:bold; color:#333; margin-top:2px; }
.section-title { font-size:14px; font-weight:700; margin:18px 0 10px; padding-bottom:5px; border-bottom:1px solid #eee; }
.dot { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; }
</style>
</head>
<body>
<div class="header">
    <h1>🔍 Drift Monitoring Dashboard</h1>
    <div class="meta-bar" id="metaBar">
        <span class="tag">Loading latest detection...</span>
        <a href="/metrics">/metrics</a>
        <a href="/api/latest">/api/latest</a>
    </div>
</div>

<div class="layout">
    <aside class="sidebar">
        <div class="card">
            <h3>🔎 Filter &amp; History</h3>
            <div class="filter-row">
                <label>Model</label>
                <select id="modelSelect"><option value="">(all models)</option></select>
            </div>
            <div class="filter-row">
                <label>Environment</label>
                <select id="envSelect"><option value="">(all envs)</option></select>
            </div>
            <div class="history-list" id="historyList">
                <div style="padding:30px; text-align:center; color:#aaa;">Loading history...</div>
            </div>
        </div>
    </aside>

    <main>
        <div id="perfSection" class="card" style="margin-bottom:24px;">
            <h3>📈 Model Performance</h3>
            <div id="perfGrid" class="perf-grid">
                <div class="perf-item"><div class="perf-label">Accuracy</div><div class="perf-value">—</div></div>
                <div class="perf-item"><div class="perf-label">F1 Score</div><div class="perf-value">—</div></div>
                <div class="perf-item"><div class="perf-label">Precision</div><div class="perf-value">—</div></div>
                <div class="perf-item"><div class="perf-label">Recall</div><div class="perf-value">—</div></div>
            </div>
        </div>

        <div class="grid-2">
            <div class="card">
                <h3>Feature Drift Summary</h3>
                <table>
                    <thead><tr><th>Feature</th><th>PSI</th><th>Level</th><th>Test</th></tr></thead>
                    <tbody id="featureTbody"><tr><td colspan="4" style="text-align:center;color:#999;">No data</td></tr></tbody>
                </table>
            </div>
            <div class="card">
                <h3>PSI Trend (Last 30 Days)</h3>
                <div class="chart-wrap"><canvas id="trendChart"></canvas></div>
            </div>
        </div>
    </main>
</div>

<div class="footer">Drift Monitoring Dashboard &mdash; Auto-refreshes every 60s</div>

<!-- Detail Modal -->
<div class="modal-mask" id="modalMask" onclick="if(event.target===this)closeModal()">
    <div class="modal">
        <button class="modal-close" onclick="closeModal()">Close &times;</button>
        <h2 id="modalTitle">Run Detail</h2>
        <div class="sub-meta" id="modalSubMeta"></div>
        <div class="kv-row" id="modalKv"></div>
        <div class="section-title">Feature Drift</div>
        <table id="modalFeatureTable">
            <thead><tr><th>Feature</th><th>PSI</th><th>Level</th><th>Test</th></tr></thead>
            <tbody></tbody>
        </table>
        <div class="section-title" id="modalAlertsTitle" style="display:none;">Recent Alerts</div>
        <div id="modalAlerts"></div>
        <div class="section-title" id="modalPerfTitle" style="display:none;">Performance</div>
        <div id="modalPerf"></div>
    </div>
</div>

<script>
let chartInstance = null;
let allHistory = [];
let currentToken = null;

const $ = id => document.getElementById(id);

function psiColor(level) {
    return level === 'severe_drift' ? '#d62728' : level === 'slight_drift' ? '#ff7f0e' : '#2ca02c';
}
function fmt(v, digits) {
    if (digits === undefined) digits = 4;
    if (v === null || v === undefined || Number.isNaN(Number(v))) return '\u2014';
    return Number(v).toFixed(digits);
}
function pct(v) {
    if (v === null || v === undefined || Number.isNaN(Number(v))) return '\u2014';
    return (Number(v) * 100).toFixed(1) + '%';
}

async function loadLatest() {
    const resp = await fetch('/api/latest');
    const data = await resp.json();
    if (!data || !data.drift_results) return;
    currentToken = (data.generated_at || '').replace(/:/g, '-').replace('T', '_');

    const meta = data.meta || {};
    const parts = [];
    if (meta.model_name) parts.push('<span class="tag">Model: <b>' + meta.model_name + '</b></span>');
    if (meta.env) parts.push('<span class="tag">Env: <b>' + meta.env + '</b></span>');
    if (meta.run_tag) parts.push('<span class="tag">Run: <b>' + meta.run_tag + '</b></span>');
    const alert = data.alert_summary || {};
    parts.push('<span class="tag">Alerts: <b>' + (alert.total||0) + '</b> (C:' + (alert.critical||0) + ' / W:' + (alert.warning||0) + ')</span>');
    parts.push('<span class="tag">' + (data.generated_at || '\u2014') + '</span>');
    parts.push('<a href="/metrics">/metrics</a>');
    parts.push('<a href="/api/latest">/api/latest</a>');
    $('metaBar').innerHTML = parts.join(' ');

    const dr = data.drift_results || {};
    const rows = Object.entries(dr).map(([f, r]) => {
        const psi = r.psi || {};
        const psiv = psi.psi || 0;
        const lvl = psi.level || 'unknown';
        const col = psiColor(lvl);
        let test = '';
        if (r.ks && r.ks.statistic !== undefined && r.ks.statistic !== null) {
            test = 'KS=' + fmt(r.ks.statistic) + ', p=' + fmt(r.ks.p_value, 6);
        } else if (r.chi2 && r.chi2.statistic !== undefined && r.chi2.statistic !== null) {
            test = 'Chi2=' + fmt(r.chi2.statistic) + ', p=' + fmt(r.chi2.p_value, 6);
        }
        return '<tr><td><span class="dot" style="background:' + col + '"></span>' + f +
            '</td><td style="color:' + col + ';font-weight:bold;">' + fmt(psiv) +
            '</td><td>' + String(lvl).replace(/_/g, ' ') +
            '</td><td style="font-size:12px;color:#666;">' + test + '</td></tr>';
    }).join('') || '<tr><td colspan="4" style="text-align:center;color:#999;">No data</td></tr>';
    $('featureTbody').innerHTML = rows;

    const perf = data.performance;
    const perfDiv = $('perfGrid');
    if (perf) {
        const cur = perf.current || {};
        const base = perf.baseline || {};
        const drops = perf.drops || {};
        const makeItem = function(lbl, curV, baseV, dropV) {
            let dropHtml = '';
            if (dropV !== null && dropV !== undefined && !Number.isNaN(Number(dropV))) {
                const dc = dropV > 0.05 ? '#c0392b' : dropV > 0 ? '#e67e22' : '#27ae60';
                dropHtml = '<div style="font-size:11px;color:' + dc + ';margin-top:2px;">Δ ' + (dropV*100).toFixed(1) + '%</div>';
            }
            return '<div class="perf-item"><div class="perf-label">' + lbl + '</div><div class="perf-value">' + pct(curV) +
                '</div><div style="font-size:11px;color:#888;margin-top:2px;">Base: ' + pct(baseV) +
                '</div>' + dropHtml + '</div>';
        };
        perfDiv.innerHTML =
            makeItem('Accuracy', cur.accuracy, base.accuracy, drops.accuracy) +
            makeItem('F1 Score', cur.f1, base.f1, drops.f1) +
            makeItem('Precision', cur.precision, base.precision, drops.precision) +
            makeItem('Recall', cur.recall, base.recall, drops.recall);
    } else {
        perfDiv.innerHTML = '<div style="padding:20px;color:#888;text-align:center;grid-column:span 4;">No performance data provided for this run</div>';
    }

    const trendResp = await fetch('/api/trend');
    const trendData = await trendResp.json();
    if (trendData && trendData.length) {
        const labels = trendData.map(function(d){ return (d.timestamp || '').slice(0,16); });
        const psiKeys = Object.keys(trendData[0]).filter(function(k){ return k.startsWith('psi_'); });
        const colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f'];
        const datasets = psiKeys.map(function(k,i){
            return { label: k.replace('psi_',''), data: trendData.map(function(d){ return d[k]; }),
                borderColor: colors[i % colors.length], tension:0.3, fill:false, pointRadius:3 };
        });
        datasets.push({ label:'PSI 0.2 (Critical)', data: labels.map(function(){ return 0.2; }),
            borderColor:'#d62728', borderDash:[6,4], pointRadius:0, fill:false });
        datasets.push({ label:'PSI 0.1 (Warning)', data: labels.map(function(){ return 0.1; }),
            borderColor:'#ff7f0e', borderDash:[6,4], pointRadius:0, fill:false });
        if (chartInstance) chartInstance.destroy();
        chartInstance = new Chart($('trendChart'), {
            type:'line',
            data:{ labels: labels, datasets: datasets },
            options:{
                responsive:true, maintainAspectRatio:false,
                scales:{ y:{ beginAtZero:true, title:{display:true, text:'PSI'} } }
            }
        });
    }
}

async function loadFilters() {
    const resp = await fetch('/api/models');
    const models = await resp.json();
    const modelSet = new Set();
    const envSet = new Set();
    models.forEach(function(m){ modelSet.add(m.model); envSet.add(m.env); });
    const ms = $('modelSelect');
    ms.innerHTML = '<option value="">(all models)</option>' + [...modelSet].map(function(m){ return '<option value="' + m + '">' + m + '</option>'; }).join('');
    const es = $('envSelect');
    es.innerHTML = '<option value="">(all envs)</option>' + [...envSet].map(function(e){ return '<option value="' + e + '">' + e + '</option>'; }).join('');
    ms.onchange = loadHistory;
    es.onchange = loadHistory;
}

async function loadHistory() {
    const params = new URLSearchParams();
    const m = $('modelSelect').value;
    const e = $('envSelect').value;
    if (m) params.set('model', m);
    if (e) params.set('env', e);
    params.set('limit', 100);
    const resp = await fetch('/api/history?' + params.toString());
    allHistory = await resp.json();
    const list = $('historyList');
    if (!allHistory.length) { list.innerHTML = '<div style="padding:30px;text-align:center;color:#aaa;">No history</div>'; return; }
    list.innerHTML = allHistory.map(function(h){
        const token = h._ts_token;
        const active = token === currentToken ? 'active' : '';
        const badges = [];
        if (h.severe_drift) badges.push('<span class="badge badge-sev">' + h.severe_drift + ' S</span>');
        if (h.slight_drift) badges.push('<span class="badge badge-sl">' + h.slight_drift + ' Sl</span>');
        if (h.perf_degraded) badges.push('<span class="badge badge-perf">Perf &darr;</span>');
        const extra = [];
        if (h.highest_psi_feature) extra.push('Top PSI: <b>' + h.highest_psi_feature + '</b> ' + fmt(h.highest_psi));
        if (h.accuracy_drop !== null && h.accuracy_drop !== undefined) extra.push('Acc\u0394 ' + pct(h.accuracy_drop));
        return '<div class="history-item ' + active + '" data-token="' + token + '">' +
            '<div class="ts">' + (h.timestamp||'').slice(0,19).replace('T',' ') + '</div>' +
            '<div class="sub"><b>' + h.model + '</b> &middot; ' + h.env + (h.run_tag?' &middot; ' + h.run_tag:'') + '</div>' +
            '<div style="margin-top:4px;">' + badges.join(' ') + (badges.length?' &middot; ':'') + extra.join(' &middot; ') + '</div>' +
            '</div>';
    }).join('');
    list.querySelectorAll('.history-item').forEach(function(el){
        el.onclick = function(){ openDetail(el.dataset.token); };
    });
}

async function openDetail(token) {
    const resp = await fetch('/api/history/' + encodeURIComponent(token));
    const data = await resp.json();
    if (!data || data.error) { alert('Record not found'); return; }
    currentToken = token;
    document.querySelectorAll('.history-item').forEach(function(el){
        el.classList.toggle('active', el.dataset.token === token);
    });

    const s = data.summary || {};
    $('modalTitle').textContent = s.model + ' \u00b7 ' + s.env + (s.run_tag?' \u00b7 ' + s.run_tag:'');
    $('modalSubMeta').textContent = s.timestamp + '   \u00b7   ' + (data.file || '');
    const kv = [
        ['Total Alerts', s.total_alerts],
        ['Critical Alerts', s.critical_alerts],
        ['Warning Alerts', s.warning_alerts],
        ['Severe Drift', s.severe_drift],
        ['Slight Drift', s.slight_drift],
        ['No Drift', s.no_drift],
        ['Top PSI Feature', s.highest_psi_feature || '\u2014'],
        ['Top PSI Value', fmt(s.highest_psi)],
        ['Accuracy Drop', s.accuracy_drop === null || s.accuracy_drop === undefined ? '\u2014' : pct(s.accuracy_drop)],
        ['Perf Degraded', s.perf_degraded ? 'Yes (>5%)' : 'No'],
        ['Production Samples', s.production_samples || '\u2014'],
        ['Window (days)', s.window_days || '\u2014'],
    ];
    $('modalKv').innerHTML = kv.map(function(x){
        return '<div class="kv-item"><div class="k">' + x[0] + '</div><div class="v">' + x[1] + '</div></div>';
    }).join('');

    const fRows = (data.features || []).map(function(f){
        const col = psiColor(f.psi.level);
        let test = '';
        if (f.ks) test = 'KS=' + fmt(f.ks.statistic) + ', p=' + fmt(f.ks.p_value,6);
        else if (f.chi2) test = 'Chi2=' + fmt(f.chi2.statistic) + ', p=' + fmt(f.chi2.p_value,6);
        return '<tr><td><span class="dot" style="background:' + col + '"></span>' + f.feature +
            ' <small style="color:#888;">(' + f.type + ')</small></td>' +
            '<td style="color:' + col + ';font-weight:bold;">' + fmt(f.psi.value) + '</td>' +
            '<td>' + String(f.psi.level).replace(/_/g,' ') + '</td>' +
            '<td style="font-size:12px;color:#666;">' + test + '</td></tr>';
    }).join('');
    $('modalFeatureTable').querySelector('tbody').innerHTML = fRows ||
        '<tr><td colspan="4" style="text-align:center;color:#999;">No features</td></tr>';

    const alerts = data.alerts || [];
    if (alerts.length) {
        $('modalAlertsTitle').style.display = '';
        $('modalAlerts').innerHTML = '<table><thead><tr><th>Feature</th><th>Metric</th><th>Value</th><th>Level</th><th>Message</th></tr></thead><tbody>' +
            alerts.map(function(a){
                const col = a.level === 'critical' ? '#c0392b' : a.level === 'warning' ? '#e67e22' : '#27ae60';
                return '<tr><td>' + a.feature + '</td><td>' + a.metric +
                    '</td><td>' + fmt(a.value, 4) + '</td>' +
                    '<td style="color:' + col + ';font-weight:bold;">' + a.level +
                    '</td><td style="font-size:12px;">' + a.message + '</td></tr>';
            }).join('') + '</tbody></table>';
    } else {
        $('modalAlertsTitle').style.display = 'none';
    }

    const perf = data.performance;
    if (perf) {
        const cur = perf.current || {}, base = perf.baseline || {}, drops = perf.drops || {};
        $('modalPerfTitle').style.display = '';
        const rows = [
            ['Accuracy', cur.accuracy, base.accuracy, drops.accuracy],
            ['F1 Score', cur.f1, base.f1, drops.f1],
            ['Precision', cur.precision, base.precision, drops.precision],
            ['Recall', cur.recall, base.recall, drops.recall],
        ];
        $('modalPerf').innerHTML = '<table><thead><tr><th>Metric</th><th>Current</th><th>Baseline</th><th>Drop</th></tr></thead><tbody>' +
            rows.map(function(r){
                const dc = (r[3] !== null && r[3] !== undefined && !Number.isNaN(Number(r[3])))
                    ? (r[3] > 0.05 ? '#c0392b' : r[3] > 0 ? '#e67e22' : '#27ae60') : '#999';
                const dropStr = (r[3] === null || r[3] === undefined || Number.isNaN(Number(r[3]))) ? '\u2014' : pct(r[3]);
                return '<tr><td><b>' + r[0] + '</b></td><td>' + pct(r[1]) + '</td><td>' + pct(r[2]) +
                    '</td><td style="color:' + dc + ';font-weight:bold;">' + dropStr + '</td></tr>';
            }).join('') + '</tbody></table>';
    } else {
        $('modalPerfTitle').style.display = 'none';
    }

    $('modalMask').classList.add('show');
}
function closeModal() { $('modalMask').classList.remove('show'); }

Promise.all([loadLatest(), loadFilters()]).then(loadHistory);
setTimeout(function(){ location.reload(); }, 60000);
</script>
</body>
</html>"""
