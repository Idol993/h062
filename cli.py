#!/usr/bin/env python3
"""
Model Drift Detection CLI Tool
Detects feature distribution drift and model performance decay.
"""

import os
import sys
import time
import json
from typing import Optional, List, Dict, Any
from datetime import datetime

import click
import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from drift_detector.utils.config_parser import ConfigParser
from drift_detector.utils.time_utils import TimeUtils
from drift_detector.data_loader.baseline_loader import BaselineLoader
from drift_detector.data_loader.production_loader import ProductionLoader
from drift_detector.data_loader.schema_validator import SchemaValidator
from drift_detector.drift_metrics.psi_calculator import PSICalculator
from drift_detector.drift_metrics.ks_tester import KSTester
from drift_detector.drift_metrics.chi2_tester import Chi2Tester
from drift_detector.drift_metrics.performance_tracker import PerformanceTracker
from drift_detector.alerting.threshold_checker import ThresholdChecker, AlertLevel
from drift_detector.alerting.notifier import Notifier
from drift_detector.alerting.report_scheduler import ReportScheduler
from drift_detector.visualization.dist_plotter import DistributionPlotter
from drift_detector.visualization.drift_chart import DriftChart
from drift_detector.visualization.html_report import HTMLReport
from drift_detector.monitoring.metrics_exporter import MetricsExporter
from drift_detector.monitoring.log_collector import LogCollector

console = Console()


class DriftDetector:
    def __init__(
        self,
        config_path: Optional[str] = None,
        features: Optional[List[str]] = None,
        ignore_features: Optional[List[str]] = None,
    ):
        self.config_parser = ConfigParser(config_path)
        self.config = self.config_parser.load()

        self.features = features
        self.ignore_features = ignore_features or []

        self.baseline_loader = BaselineLoader()
        self.production_loader = ProductionLoader()
        self.schema_validator = SchemaValidator()

        self.psi_calculator = PSICalculator(n_bins=10)
        self.ks_tester = KSTester(alpha=self.config.thresholds.p_value)
        self.chi2_tester = Chi2Tester(alpha=self.config.thresholds.p_value)
        self.performance_tracker = PerformanceTracker(
            threshold=self.config.thresholds.performance_drop
        )

        self.threshold_checker = ThresholdChecker(
            psi_warning=self.config.thresholds.psi["warning"],
            psi_critical=self.config.thresholds.psi["critical"],
            p_value_threshold=self.config.thresholds.p_value,
            performance_drop_threshold=self.config.thresholds.performance_drop,
        )

        notif_config = self.config.notifications
        self.notifier = Notifier(
            enable_terminal=notif_config.terminal.get("enabled", True),
            webhook_url=notif_config.webhook.get("url") if notif_config.webhook.get("enabled") else None,
            log_file_path=notif_config.log_file.get("path") if notif_config.log_file.get("enabled") else None,
            log_max_bytes=notif_config.log_file.get("max_bytes", 10 * 1024 * 1024),
            log_backup_count=notif_config.log_file.get("backup_count", 5),
        )

        viz_config = self.config.visualization
        self.dist_plotter = DistributionPlotter(
            output_dir=viz_config.output_dir,
            dpi=viz_config.dpi,
            figsize=tuple(viz_config.figsize),
        )
        self.drift_chart = DriftChart(
            output_dir=viz_config.output_dir,
            dpi=viz_config.dpi,
            figsize=tuple(viz_config.figsize),
        )
        self.html_report = HTMLReport(output_dir=".")

        self.metrics_exporter = MetricsExporter(
            port=self.config.monitoring.prometheus_port,
            data_dir=".",
        )

        self.scheduler = ReportScheduler()

    def _get_features_to_check(
        self,
        baseline_data: pd.DataFrame,
    ) -> List[str]:
        all_features = list(baseline_data.columns)

        if self.features:
            features_to_check = [f for f in self.features if f in all_features]
        else:
            features_to_check = all_features.copy()

        features_to_check = [
            f for f in features_to_check if f not in self.ignore_features
        ]

        return features_to_check

    def _is_numeric_feature(
        self,
        series: pd.Series,
    ) -> bool:
        return pd.api.types.is_numeric_dtype(series)

    def detect(
        self,
        baseline_path: str,
        production_path: str,
        label_file: Optional[str] = None,
        prediction_file: Optional[str] = None,
        baseline_predictions: Optional[str] = None,
        baseline_labels: Optional[str] = None,
        baseline_accuracy: Optional[float] = None,
        baseline_f1: Optional[float] = None,
        baseline_precision: Optional[float] = None,
        baseline_recall: Optional[float] = None,
        date_column: Optional[str] = None,
        window_days: int = 7,
        generate_plots: bool = True,
        generate_report: bool = True,
        api_url: Optional[str] = None,
        api_window_days: int = 7,
    ) -> Dict[str, Any]:
        start_time = time.time()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:

            load_task = progress.add_task("Loading data...", total=100)

            console.print("\n[cyan]📥 Loading baseline data...[/cyan]")
            baseline_data = self.baseline_loader.load(baseline_path)
            console.print(f"[green]✓ Loaded {len(baseline_data)} baseline samples[/green]")
            progress.update(load_task, advance=25)

            console.print("\n[cyan]📥 Loading production data...[/cyan]")
            self.production_loader.date_column = date_column
            self.production_loader.window_days = window_days
            production_data = self.production_loader.load(production_path)
            console.print(f"[green]✓ Loaded {len(production_data)} production samples[/green]")
            progress.update(load_task, advance=20)

            if api_url:
                console.print("\n[cyan]🌐 Pulling prediction logs from API...[/cyan]")
                try:
                    log_collector = LogCollector(
                        api_url=api_url,
                        api_window_days=api_window_days,
                        date_column=date_column,
                    )
                    api_data = log_collector.collect_from_api()
                    if len(api_data) > 0:
                        console.print(f"[green]✓ Pulled {len(api_data)} records from API[/green]")
                        shared_cols = list(set(production_data.columns) & set(api_data.columns))
                        if shared_cols:
                            production_data = pd.concat(
                                [production_data[shared_cols], api_data[shared_cols]],
                                ignore_index=True,
                            )
                            console.print(f"[green]✓ Merged production data: {len(production_data)} total samples[/green]")
                        else:
                            console.print("[yellow]⚠️ API data has no matching columns with production data, skipping merge[/yellow]")
                    else:
                        console.print("[yellow]⚠️ API returned 0 records[/yellow]")
                except RuntimeError as e:
                    console.print(f"[red]❌ {str(e)}[/red]")
                except Exception as e:
                    console.print(f"[red]❌ API log pull error: {str(e)}[/red]")
                progress.update(load_task, advance=5)
            else:
                progress.update(load_task, advance=5)

            console.print("\n[cyan]🔍 Validating schema...[/cyan]")
            validation_result = self.schema_validator.validate_against_baseline(
                production_data, baseline_data
            )
            if not validation_result.valid:
                console.print("[yellow]⚠️ Schema validation warnings found:[/yellow]")
                for warning in validation_result.warnings:
                    console.print(f"  - {warning}")
                if validation_result.errors:
                    console.print("[red]❌ Schema validation errors found:[/red]")
                    for error in validation_result.errors:
                        console.print(f"  - {error}")
            else:
                console.print("[green]✓ Schema validation passed[/green]")
            progress.update(load_task, advance=10)

            features_to_check = self._get_features_to_check(baseline_data)
            console.print(f"\n[cyan]🔬 Analyzing {len(features_to_check)} features...[/cyan]")

            drift_results: Dict[str, Any] = {}
            plots: Dict[str, Any] = {}

            feat_task = progress.add_task(
                "Detecting drift...",
                total=len(features_to_check),
            )

            for i, feature in enumerate(features_to_check):
                progress.update(
                    feat_task,
                    advance=1,
                    description=f"Analyzing feature: {feature} ({i+1}/{len(features_to_check)})",
                )

                baseline_series = baseline_data[feature]
                production_series = production_data.get(feature, pd.Series())

                if len(production_series) == 0:
                    console.print(f"[yellow]⚠️ Feature '{feature}' not found in production data, skipping...[/yellow]")
                    continue

                baseline_clean = baseline_series.dropna().values
                production_clean = production_series.dropna().values

                if len(baseline_clean) == 0 or len(production_clean) == 0:
                    console.print(f"[yellow]⚠️ Insufficient data for feature '{feature}', skipping...[/yellow]")
                    continue

                is_numeric = self._is_numeric_feature(baseline_series)
                drift_results[feature] = {"type": "numerical" if is_numeric else "categorical"}

                if is_numeric:
                    psi_result = self.psi_calculator.calculate(
                        baseline_clean, production_clean
                    )
                    drift_results[feature]["psi"] = psi_result

                    ks_result = self.ks_tester.calculate(
                        baseline_clean, production_clean
                    )
                    drift_results[feature]["ks"] = ks_result

                    if generate_plots:
                        plot_result = self.dist_plotter.plot_numerical(
                            feature,
                            baseline_clean,
                            production_clean,
                            psi_result,
                        )
                        plots[feature] = plot_result

                    level = psi_result["level"]
                    color = (
                        "[red]"
                        if level == "severe_drift"
                        else "[yellow]"
                        if level == "slight_drift"
                        else "[green]"
                    )
                    console.print(
                        f"  {color}• {feature}: PSI = {psi_result['psi']:.4f} "
                        f"({level.replace('_', ' ')}) | KS p-value = {ks_result['p_value']:.6f}[/]"
                    )
                else:
                    psi_result = self.psi_calculator.calculate_categorical(
                        baseline_clean, production_clean
                    )
                    drift_results[feature]["psi"] = psi_result

                    chi2_result = self.chi2_tester.calculate(
                        baseline_clean, production_clean
                    )
                    drift_results[feature]["chi2"] = chi2_result

                    if generate_plots:
                        plot_result = self.dist_plotter.plot_categorical(
                            feature,
                            baseline_clean,
                            production_clean,
                            chi2_result,
                        )
                        plots[feature] = plot_result

                    level = psi_result["level"]
                    color = (
                        "[red]"
                        if level == "severe_drift"
                        else "[yellow]"
                        if level == "slight_drift"
                        else "[green]"
                    )
                    console.print(
                        f"  {color}• {feature}: PSI = {psi_result['psi']:.4f} "
                        f"({level.replace('_', ' ')}) | Chi2 p-value = {chi2_result['p_value']:.6f}[/]"
                    )

            progress.update(load_task, advance=20)

            performance_result = None
            if label_file and prediction_file and os.path.exists(label_file):
                console.print("\n[cyan]📊 Evaluating model performance...[/cyan]")
                try:
                    label_df = pd.read_csv(label_file)
                    pred_df = pd.read_csv(prediction_file)

                    baseline_set = False
                    if baseline_accuracy is not None:
                        self.performance_tracker.set_baseline_metrics(
                            accuracy=baseline_accuracy,
                            f1=baseline_f1,
                            precision=baseline_precision,
                            recall=baseline_recall,
                        )
                        baseline_set = True
                        console.print(
                            f"[green]✓ Baseline set from CLI parameters: "
                            f"accuracy={baseline_accuracy}"
                            + (f", f1={baseline_f1}" if baseline_f1 is not None else "")
                            + (f", precision={baseline_precision}" if baseline_precision is not None else "")
                            + (f", recall={baseline_recall}" if baseline_recall is not None else "")
                            + "[/green]"
                        )
                    elif baseline_predictions and os.path.exists(baseline_predictions):
                        base_pred_df = pd.read_csv(baseline_predictions)
                        base_pred = base_pred_df.iloc[:, -1].values
                        if baseline_labels and os.path.exists(baseline_labels):
                            base_true_df = pd.read_csv(baseline_labels)
                            base_true = base_true_df.iloc[:, -1].values
                            base_true = base_true[: len(base_pred)]
                            base_pred = base_pred[: len(base_true)]
                            self.performance_tracker.set_baseline(base_true, base_pred)
                            baseline_set = True
                            console.print("[green]✓ Baseline set from baseline labels + predictions[/green]")
                        else:
                            console.print("[yellow]⚠️ --baseline-labels not provided, cannot compute baseline from prediction file alone[/yellow]")

                    if not baseline_set and self.performance_tracker.baseline_accuracy is None:
                        console.print("[yellow]⚠️ No baseline performance specified. Use --baseline-accuracy or --baseline-labels + --baseline-predictions. Performance evaluation skipped.[/yellow]")

                    y_true = label_df.iloc[:, -1].values
                    y_pred = pred_df.iloc[:, -1].values

                    if baseline_set and len(y_true) == len(y_pred):
                        performance_result = self.performance_tracker.evaluate(
                            y_true, y_pred, timestamp=TimeUtils.get_iso_timestamp()
                        )
                        console.print(
                            f"[green]✓ Current accuracy: {performance_result['current']['accuracy']:.4f} "
                            f"(drop: {performance_result['drops']['accuracy']:.4f})[/green]"
                        )
                    elif len(y_true) != len(y_pred):
                        console.print(
                            "[yellow]⚠️ Label and prediction file lengths don't match[/yellow]"
                        )
                except Exception as e:
                    console.print(f"[red]❌ Error evaluating performance: {str(e)}[/red]")

            progress.update(load_task, advance=10)

            console.print("\n[cyan]⚠️ Checking alerts...[/cyan]")
            alerts = self.threshold_checker.check_all(drift_results, performance_result)

            report_files = None
            if generate_report:
                console.print("\n[cyan]📄 Generating reports...[/cyan]")

                if generate_plots:
                    overview_plot = self.drift_chart.plot_overall_dashboard(drift_results)
                    plots["overview"] = overview_plot

                report_files = self.html_report.generate(
                    drift_results=drift_results,
                    alerts=alerts,
                    performance_result=performance_result,
                    validation_result={
                        "valid": validation_result.valid,
                        "errors": validation_result.errors,
                        "warnings": validation_result.warnings,
                        "missing_columns": validation_result.missing_columns,
                    },
                    plots=plots,
                    baseline_samples=len(baseline_data),
                    production_samples=len(production_data),
                    date_range=self.production_loader.get_date_range(),
                )
                console.print(f"[green]✓ HTML report: {report_files['html_path']}[/green]")
                console.print(f"[green]✓ JSON metrics: {report_files['json_path']}[/green]")

            progress.update(load_task, advance=10)

        duration = time.time() - start_time

        self.notifier.notify(alerts)

        alert_summary = self.threshold_checker.get_alert_summary(alerts)

        self.metrics_exporter.increment_detection_runs()
        self.metrics_exporter.observe_duration(duration)
        self.metrics_exporter.update_drift_results(drift_results)
        self.metrics_exporter.update_alerts(alerts)
        if performance_result:
            self.metrics_exporter.update_performance(
                accuracy=performance_result["current"]["accuracy"],
                accuracy_drop=performance_result["drops"]["accuracy"],
                f1=performance_result["current"]["f1"],
            )

        latest_summary = {
            "generated_at": TimeUtils.get_iso_timestamp(),
            "drift_results": drift_results,
            "performance": performance_result,
            "alert_summary": alert_summary,
        }
        self.metrics_exporter.update_latest_result(latest_summary)

        console.print("\n" + "=" * 60)
        console.print("[bold]📊 Detection Summary[/bold]")
        console.print(f"  Features analyzed: {len(drift_results)}")
        console.print(f"  Total alerts: {alert_summary['total']}")
        if alert_summary["critical"] > 0:
            console.print(f"  [red]Critical alerts: {alert_summary['critical']}[/red]")
        if alert_summary["warning"] > 0:
            console.print(f"  [yellow]Warning alerts: {alert_summary['warning']}[/yellow]")
        console.print(f"  Duration: {duration:.2f}s")
        console.print("=" * 60 + "\n")

        return {
            "drift_results": drift_results,
            "alerts": [a.__dict__ for a in alerts],
            "alert_summary": alert_summary,
            "performance": performance_result,
            "validation": {
                "valid": validation_result.valid,
                "errors": validation_result.errors,
                "warnings": validation_result.warnings,
            },
            "plots": {k: {"path": v["path"]} for k, v in plots.items()},
            "report_files": report_files,
            "duration": duration,
        }

    def serve(self, port: int = 8080) -> None:
        console.print(f"[cyan]🚀 Starting drift monitoring server on port {port}...[/cyan]")
        self.metrics_exporter.start_server(port=port)

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            console.print("\n[yellow]👋 Server stopped[/yellow]")

    def schedule(
        self,
        baseline_path: str,
        production_path: str,
        interval: str = "daily",
        time_str: str = "08:00",
        run_now: bool = False,
        **kwargs,
    ) -> None:
        def run_detection():
            try:
                self.detect(baseline_path, production_path, **kwargs)
            except Exception as e:
                console.print(f"[red]❌ Scheduled detection failed: {str(e)}[/red]")

        self.scheduler.schedule_report(
            job_id="drift_detection",
            task=run_detection,
            interval=interval,
            time_str=time_str,
        )

        jobs = self.scheduler.get_jobs()
        for job_id, info in jobs.items():
            console.print(
                f"[green]✓ Scheduled '{job_id}' - next run: {info['next_run']}[/green]"
            )

        self.scheduler.start(run_once_now=run_now)

        console.print(f"\n[cyan]⏰ Scheduler running. Press Ctrl+C to stop.[/cyan]")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.scheduler.stop()
            console.print("\n[yellow]👋 Scheduler stopped[/yellow]")


@click.group()
@click.version_option(version="1.0.0")
@click.option(
    "--config",
    type=click.Path(exists=False),
    help="Path to configuration file",
    default="config.yaml",
)
@click.option(
    "--features",
    type=str,
    help="Comma-separated list of features to check (whitelist)",
    default=None,
)
@click.option(
    "--ignore-features",
    type=str,
    help="Comma-separated list of features to ignore (blacklist)",
    default=None,
)
@click.pass_context
def cli(ctx, config, features, ignore_features):
    """
    📊 Model Drift Detection CLI - Monitor ML model performance and data drift.
    """
    feature_list = features.split(",") if features else None
    ignore_list = ignore_features.split(",") if ignore_features else None

    ctx.obj = DriftDetector(
        config_path=config,
        features=feature_list,
        ignore_features=ignore_list,
    )


@cli.command()
@click.option(
    "--baseline",
    "-b",
    type=click.Path(exists=True),
    required=True,
    help="Path to baseline dataset (CSV/Parquet/JSON)",
)
@click.option(
    "--production",
    "-p",
    type=click.Path(exists=True),
    required=True,
    help="Path to production dataset (CSV/Parquet/JSON)",
)
@click.option(
    "--labels",
    "-l",
    type=click.Path(exists=False),
    help="Path to labeled samples for performance evaluation",
)
@click.option(
    "--predictions",
    "-pred",
    type=click.Path(exists=False),
    help="Path to model predictions on labeled samples",
)
@click.option(
    "--baseline-predictions",
    "-bp",
    type=click.Path(exists=False),
    help="Path to baseline predictions for comparison",
)
@click.option(
    "--baseline-labels",
    "-bl",
    type=click.Path(exists=False),
    help="Path to baseline ground-truth labels (used with --baseline-predictions)",
)
@click.option(
    "--baseline-accuracy",
    type=float,
    default=None,
    help="Baseline model accuracy (e.g. 0.92). Alternative to --baseline-predictions.",
)
@click.option(
    "--baseline-f1",
    type=float,
    default=None,
    help="Baseline model F1 score (e.g. 0.90).",
)
@click.option(
    "--baseline-precision",
    type=float,
    default=None,
    help="Baseline model precision.",
)
@click.option(
    "--baseline-recall",
    type=float,
    default=None,
    help="Baseline model recall.",
)
@click.option(
    "--date-column",
    type=str,
    help="Name of date column for time window filtering",
    default=None,
)
@click.option(
    "--window-days",
    type=int,
    help="Number of days to include in sliding window",
    default=7,
)
@click.option(
    "--no-plots",
    is_flag=True,
    help="Disable plot generation",
)
@click.option(
    "--no-report",
    is_flag=True,
    help="Disable report generation",
)
@click.option(
    "--api-url",
    type=str,
    default=None,
    help="API endpoint URL to pull prediction logs (e.g. http://api.example.com/predictions)",
)
@click.option(
    "--api-window-days",
    type=int,
    default=7,
    help="Time window (days) for API log pull",
)
@click.pass_obj
def detect(
    detector,
    baseline,
    production,
    labels,
    predictions,
    baseline_predictions,
    baseline_labels,
    baseline_accuracy,
    baseline_f1,
    baseline_precision,
    baseline_recall,
    date_column,
    window_days,
    no_plots,
    no_report,
    api_url,
    api_window_days,
):
    """
    🔍 Detect feature drift and model performance decay.
    """
    try:
        result = detector.detect(
            baseline_path=baseline,
            production_path=production,
            label_file=labels,
            prediction_file=predictions,
            baseline_predictions=baseline_predictions,
            baseline_labels=baseline_labels,
            baseline_accuracy=baseline_accuracy,
            baseline_f1=baseline_f1,
            baseline_precision=baseline_precision,
            baseline_recall=baseline_recall,
            date_column=date_column,
            window_days=window_days,
            generate_plots=not no_plots,
            generate_report=not no_report,
            api_url=api_url,
            api_window_days=api_window_days,
        )
        return result
    except Exception as e:
        console.print(f"[red]❌ Detection failed: {str(e)}[/red]")
        console.print_exception(show_locals=False)
        sys.exit(1)


@cli.command()
@click.option(
    "--baseline",
    "-b",
    type=click.Path(exists=True),
    required=True,
    help="Path to baseline dataset",
)
@click.option(
    "--production",
    "-p",
    type=click.Path(exists=True),
    required=True,
    help="Path to production dataset",
)
@click.option(
    "--interval",
    type=click.Choice(["daily", "weekly", "hourly"]),
    default="daily",
    help="Report generation interval",
)
@click.option(
    "--time",
    "time_str",
    type=str,
    default="08:00",
    help="Time to run daily/weekly reports (HH:MM format)",
)
@click.option(
    "--run-now",
    is_flag=True,
    help="Run detection immediately after scheduling",
)
@click.option(
    "--labels",
    "-l",
    type=click.Path(exists=False),
    help="Path to labeled samples",
)
@click.option(
    "--predictions",
    "-pred",
    type=click.Path(exists=False),
    help="Path to model predictions",
)
@click.pass_obj
def schedule(
    detector,
    baseline,
    production,
    interval,
    time_str,
    run_now,
    labels,
    predictions,
):
    """
    ⏰ Schedule periodic drift detection reports.
    """
    try:
        detector.schedule(
            baseline_path=baseline,
            production_path=production,
            interval=interval,
            time_str=time_str,
            run_now=run_now,
            label_file=labels,
            prediction_file=predictions,
        )
    except Exception as e:
        console.print(f"[red]❌ Scheduling failed: {str(e)}[/red]")
        sys.exit(1)


@cli.command()
@click.option(
    "--port",
    type=int,
    default=8080,
    help="Port for the monitoring server",
)
@click.pass_obj
def serve(detector, port):
    """
    🚀 Start monitoring server with web dashboard and Prometheus metrics.

    \b
    Endpoints:
      /            Web dashboard (latest metrics + PSI trend chart)
      /metrics     Prometheus metrics endpoint
      /api/latest  Latest detection result as JSON
      /api/trend   PSI trend data (last 30 days) as JSON
    """
    try:
        detector.serve(port=port)
    except Exception as e:
        console.print(f"[red]❌ Server failed: {str(e)}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    cli()
