#!/usr/bin/env python3
"""
Model Drift Detection CLI Tool
Detects feature distribution drift and model performance decay.
"""

import os
import sys
import time
import json
import glob
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
        api_window_days: Optional[int] = None,
        api_retries: int = 0,
        model_name: Optional[str] = None,
        env: Optional[str] = None,
        run_tag: Optional[str] = None,
    ) -> Dict[str, Any]:
        start_time = time.time()

        run_meta = {
            "model_name": model_name or "default_model",
            "env": env or "default_env",
            "run_tag": run_tag,
            "window_days": window_days,
            "baseline_file": os.path.basename(baseline_path),
            "production_file": os.path.basename(production_path),
        }

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
                effective_api_window = api_window_days if api_window_days is not None else window_days
                try:
                    log_collector = LogCollector(
                        api_url=api_url,
                        api_window_days=effective_api_window,
                        date_column=date_column,
                        window_days=effective_api_window,
                    )
                    api_data = log_collector.collect_from_api(max_retries=api_retries)
                    if len(api_data) > 0:
                        console.print(f"[green]✓ Pulled {len(api_data)} records from API (window: {effective_api_window} days)[/green]")
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
                    if self.notifier.logger:
                        self.notifier.logger.error(str(e))
                except Exception as e:
                    msg = f"API log pull error: {str(e)}"
                    console.print(f"[red]❌ {msg}[/red]")
                    if self.notifier.logger:
                        self.notifier.logger.error(msg)
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
                    run_meta=run_meta,
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
            "meta": run_meta,
            "drift_results": drift_results,
            "performance": performance_result,
            "alert_summary": alert_summary,
        }
        self.metrics_exporter.update_latest_result(latest_summary)

        console.print("\n" + "=" * 60)
        console.print("[bold]📊 Detection Summary[/bold]")
        console.print(f"  Model: [cyan]{run_meta['model_name']}[/cyan]  |  Env: [cyan]{run_meta['env']}[/cyan]")
        if run_meta.get("run_tag"):
            console.print(f"  Run Tag: [cyan]{run_meta['run_tag']}[/cyan]")
        console.print(f"  Features analyzed: {len(drift_results)}")
        console.print(f"  Total alerts: {alert_summary['total']}")
        if alert_summary["critical"] > 0:
            console.print(f"  [red]Critical alerts: {alert_summary['critical']}[/red]")
        if alert_summary["warning"] > 0:
            console.print(f"  [yellow]Warning alerts: {alert_summary['warning']}[/yellow]")
        console.print(f"  Duration: {duration:.2f}s")
        console.print("=" * 60 + "\n")

        return {
            "meta": run_meta,
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
    default=None,
    help="Time window (days) for API log pull (defaults to --window-days)",
)
@click.option(
    "--api-retries",
    type=int,
    default=0,
    help="Number of retries for API pull on failure (default: 0)",
)
@click.option(
    "--model-name",
    type=str,
    default=None,
    help="Model name to tag this detection run (e.g. credit_risk_v2)",
)
@click.option(
    "--env",
    type=str,
    default=None,
    help="Environment name to tag this detection run (e.g. prod/staging/canary)",
)
@click.option(
    "--run-tag",
    type=str,
    default=None,
    help="Optional tag for this run (e.g. weekly_check, experiment_A)",
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
    api_retries,
    model_name,
    env,
    run_tag,
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
            api_retries=api_retries,
            model_name=model_name,
            env=env,
            run_tag=run_tag,
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


def _build_history_summary(
    json_files: List[str],
    model_filter: Optional[str] = None,
    env_filter: Optional[str] = None,
    run_tag_filter: Optional[str] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
) -> List[Dict[str, Any]]:
    from datetime import datetime

    def _parse_ts(ts: str) -> Optional[datetime]:
        for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(ts, fmt)
            except (ValueError, TypeError):
                continue
        return None

    def _parse_since(ts: str) -> Optional[datetime]:
        dt = _parse_ts(ts)
        if dt is None:
            return None
        if len(ts.strip()) == 10:
            return dt
        return dt

    def _parse_until(ts: str) -> Optional[datetime]:
        dt = _parse_ts(ts)
        if dt is None:
            return None
        if len(ts.strip()) == 10:
            return dt.replace(hour=23, minute=59, second=59, microsecond=999999)
        return dt

    since_dt = _parse_since(since) if since else None
    until_dt = _parse_until(until) if until else None

    summaries = []
    for jf in sorted(json_files):
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        meta = data.get("meta") or {}
        model_name = meta.get("model_name", "default_model")
        env_name = meta.get("env", "default_env")
        run_tag = meta.get("run_tag")
        if model_filter and model_name != model_filter:
            continue
        if env_filter and env_name != env_filter:
            continue
        if run_tag_filter and run_tag != run_tag_filter:
            continue

        ts = data.get("generated_at", "")
        ts_dt = _parse_ts(ts) if ts else None
        if since_dt and (not ts_dt or ts_dt < since_dt):
            continue
        if until_dt and (not ts_dt or ts_dt > until_dt):
            continue

        alerts_arr = data.get("alerts", []) or []
        total_alerts = len(alerts_arr)
        critical_alerts = sum(1 for a in alerts_arr if a.get("level") == "critical")
        warning_alerts = sum(1 for a in alerts_arr if a.get("level") == "warning")

        drift_results = data.get("drift_results", {}) or {}
        highest_psi_feature = None
        highest_psi_value = 0.0
        severe_drift_count = 0
        slight_drift_count = 0
        for feature, results in drift_results.items():
            psi_res = results.get("psi", {}) or {}
            psi_val = psi_res.get("psi", 0.0)
            level = psi_res.get("level", "no_drift")
            if psi_val > highest_psi_value:
                highest_psi_value = psi_val
                highest_psi_feature = feature
            if level == "severe_drift":
                severe_drift_count += 1
            elif level == "slight_drift":
                slight_drift_count += 1

        perf = data.get("performance") or {}
        drops = perf.get("drops", {}) or {}
        acc_drop = drops.get("accuracy")

        perf_degraded = perf.get("is_degraded")
        perf_threshold = perf.get("threshold")
        if perf_degraded is None:
            if acc_drop is not None:
                try:
                    perf_degraded = float(acc_drop) > 0.05
                except (TypeError, ValueError):
                    perf_degraded = False
            else:
                perf_degraded = False
        if perf_threshold is None:
            perf_threshold = 0.05

        current_acc = (perf.get("current") or {}).get("accuracy")
        baseline_acc = (perf.get("baseline") or {}).get("accuracy")

        summaries.append({
            "timestamp": data.get("generated_at", ""),
            "file": os.path.basename(jf),
            "model": model_name,
            "env": env_name,
            "run_tag": run_tag,
            "total_alerts": total_alerts,
            "critical_alerts": critical_alerts,
            "warning_alerts": warning_alerts,
            "severe_drift": severe_drift_count,
            "slight_drift": slight_drift_count,
            "no_drift": max(0, len(drift_results) - severe_drift_count - slight_drift_count),
            "highest_psi_feature": highest_psi_feature,
            "highest_psi": round(highest_psi_value, 4),
            "accuracy_drop": round(float(acc_drop), 4) if acc_drop is not None else None,
            "accuracy": round(float(current_acc), 4) if current_acc is not None else None,
            "baseline_accuracy": round(float(baseline_acc), 4) if baseline_acc is not None else None,
            "perf_degraded": bool(perf_degraded),
            "perf_threshold": float(perf_threshold) if perf_threshold is not None else 0.05,
            "baseline_samples": data.get("baseline_samples"),
            "production_samples": data.get("production_samples"),
            "window_days": meta.get("window_days"),
        })
    return summaries


def _aggregate_stats(summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    from collections import Counter

    total_runs = len(summaries)
    total_alerts = sum(s["total_alerts"] for s in summaries)
    total_critical = sum(s["critical_alerts"] for s in summaries)
    total_warning = sum(s["warning_alerts"] for s in summaries)
    total_severe_drift = sum(s["severe_drift"] for s in summaries)
    total_slight_drift = sum(s["slight_drift"] for s in summaries)

    feature_counter = Counter()
    high_risk_features = set()
    for s in summaries:
        if s["highest_psi_feature"]:
            feature_counter[s["highest_psi_feature"]] += 1
        if s.get("severe_drift", 0) > 0:
            high_risk_features.add(s["highest_psi_feature"])

    top_drift_features = []
    for feat, cnt in feature_counter.most_common(5):
        feats = [s for s in summaries if s["highest_psi_feature"] == feat]
        avg_psi = sum(s["highest_psi"] for s in feats) / len(feats) if feats else 0.0
        top_drift_features.append({
            "feature": feat,
            "count": cnt,
            "avg_highest_psi": round(avg_psi, 4),
        })

    perf_degraded_count = sum(1 for s in summaries if s["perf_degraded"])
    acc_drops = [s["accuracy_drop"] for s in summaries if s["accuracy_drop"] is not None]
    avg_acc_drop = round(sum(acc_drops) / len(acc_drops), 4) if acc_drops else None
    max_acc_drop = round(max(acc_drops), 4) if acc_drops else None

    return {
        "total_runs": total_runs,
        "total_alerts": total_alerts,
        "total_critical_alerts": total_critical,
        "total_warning_alerts": total_warning,
        "total_severe_drift_occurrences": total_severe_drift,
        "total_slight_drift_occurrences": total_slight_drift,
        "top_drift_features": top_drift_features,
        "high_risk_features": sorted(high_risk_features),
        "perf_degraded_count": perf_degraded_count,
        "perf_degraded_ratio": round(perf_degraded_count / total_runs, 4) if total_runs else 0.0,
        "avg_accuracy_drop": avg_acc_drop,
        "max_accuracy_drop": max_acc_drop,
    }


@cli.command(name="history")
@click.option(
    "--limit",
    "-n",
    type=int,
    default=10,
    help="Number of recent runs to show (default: 10)",
)
@click.option(
    "--model",
    type=str,
    default=None,
    help="Filter by model name",
)
@click.option(
    "--env",
    type=str,
    default=None,
    help="Filter by environment name",
)
@click.option(
    "--run-tag",
    type=str,
    default=None,
    help="Filter by run tag",
)
@click.option(
    "--since",
    type=str,
    default=None,
    help="Include records from this timestamp (e.g. '2026-06-01' or '2026-06-01T00:00:00')",
)
@click.option(
    "--until",
    type=str,
    default=None,
    help="Include records up to this timestamp",
)
@click.option(
    "--last-days",
    type=int,
    default=None,
    help="Include records from the last N days (overrides --since)",
)
@click.option(
    "--export-csv",
    type=click.Path(),
    default=None,
    help="Export all history (after filtering) to a CSV file",
)
@click.option(
    "--export-json",
    type=click.Path(),
    default=None,
    help="Export all history (after filtering) to a JSON file",
)
@click.option(
    "--data-dir",
    type=click.Path(exists=True, file_okay=False),
    default=".",
    help="Directory containing drift_metrics_*.json files (default: current dir)",
)
@click.option(
    "--summary-only",
    is_flag=True,
    default=False,
    help="Show only the aggregate stats summary, skip the per-run table",
)
@click.pass_context
def history_cmd(
    ctx, limit: int, model: Optional[str], env: Optional[str], run_tag: Optional[str],
    since: Optional[str], until: Optional[str], last_days: Optional[int],
    export_csv: Optional[str], export_json: Optional[str], data_dir: str,
    summary_only: bool,
):
    """
    📜 List detection history with summaries, advanced filters, and CSV/JSON export.

    \b
    Summary columns:
      - Timestamp / Model / Env / Run Tag
      - Total alerts (critical / warning)
      - PSI drift counts (severe / slight / no)
      - Highest-PSI feature and value
      - Accuracy drop and whether model is degraded (per detection threshold)

    \b
    Time filter examples:
      history --last-days 7
      history --since 2026-06-01 --until 2026-06-15
      history --model credit_risk_v2 --env prod --last-days 30
    """
    try:
        from datetime import datetime, timedelta

        if last_days is not None:
            since_dt = datetime.now() - timedelta(days=last_days)
            since = since_dt.strftime("%Y-%m-%dT%H:%M:%S")

        json_files = glob.glob(os.path.join(data_dir, "drift_metrics_*.json"))
        if not json_files:
            console.print("[yellow]⚠️ No drift_metrics_*.json files found in " + data_dir + "[/yellow]")
            return

        all_summaries = _build_history_summary(
            json_files, model_filter=model, env_filter=env,
            run_tag_filter=run_tag, since=since, until=until,
        )
        if not all_summaries:
            console.print("[yellow]⚠️ No history matched the given filters[/yellow]")
            return

        all_summaries.sort(key=lambda r: r["timestamp"], reverse=True)
        display = all_summaries[:limit]
        stats = _aggregate_stats(all_summaries)

        from rich.table import Table
        from rich.panel import Panel
        from rich.columns import Columns
        from rich.console import Group

        stat_cards = []
        stat_cards.append(Panel(
            f"[cyan]Total Runs\n[bold white]{stats['total_runs']}[/bold white]",
            title="Runs", border_style="cyan", width=18,
        ))
        stat_cards.append(Panel(
            f"[red]Critical Alerts\n[bold white]{stats['total_critical_alerts']}[/bold white]\n"
            f"[yellow]Warning\n[bold white]{stats['total_warning_alerts']}[/bold white]",
            title="Total Alerts", border_style="red", width=22,
        ))
        stat_cards.append(Panel(
            f"[red]Severe Drift\n[bold white]{stats['total_severe_drift_occurrences']}[/bold white]\n"
            f"[yellow]Slight\n[bold white]{stats['total_slight_drift_occurrences']}[/bold white]",
            title="Drift Occurrences", border_style="yellow", width=22,
        ))
        perf_txt = f"[red]YES {stats['perf_degraded_count']}[/red] / {stats['total_runs']}\n"
        if stats['avg_accuracy_drop'] is not None:
            perf_txt += f"avg drop [yellow]{stats['avg_accuracy_drop']*100:.2f}%[/yellow]\n"
        if stats['max_accuracy_drop'] is not None:
            perf_txt += f"max drop [red]{stats['max_accuracy_drop']*100:.2f}%[/red]"
        stat_cards.append(Panel(perf_txt, title="Perf Degraded", border_style="magenta", width=22))

        if stats['top_drift_features']:
            top_feat_lines = []
            for i, feat in enumerate(stats['top_drift_features'], 1):
                top_feat_lines.append(
                    f"{i}. [bold]{feat['feature']}[/bold]  "
                    f"[cyan]{feat['count']}x[/cyan]  "
                    f"avg PSI [yellow]{feat['avg_highest_psi']:.4f}[/yellow]"
                )
            stat_cards.append(Panel("\n".join(top_feat_lines), title="Top Drift Features", border_style="green", width=36))

        console.print(Panel(Columns(stat_cards, equal=False, expand=True),
                            title=f"📊 Aggregate Summary  ({len(all_summaries)} runs)",
                            border_style="bold cyan"))

        if not summary_only:
            table = Table(title=f"Recent Runs (showing {len(display)} of {len(all_summaries)})", show_lines=True)
            table.add_column("#", justify="right", style="cyan", no_wrap=True)
            table.add_column("Timestamp", style="white", no_wrap=True)
            table.add_column("Model", style="magenta")
            table.add_column("Env", style="yellow")
            table.add_column("Run Tag", style="blue")
            table.add_column("Alerts", justify="center")
            table.add_column("Drift (S/Sl/No)", justify="center")
            table.add_column("Top PSI Feature", style="bold")
            table.add_column("Top PSI", justify="right")
            table.add_column("Acc Drop", justify="right")
            table.add_column("Perf Degr", justify="center")

            for idx, row in enumerate(display, 1):
                alerts_str = f"[red]{row['critical_alerts']}[/red]C / [yellow]{row['warning_alerts']}[/yellow]W / {row['total_alerts']}"
                drift_str = f"[red]{row['severe_drift']}[/red] / [yellow]{row['slight_drift']}[/yellow] / [green]{row['no_drift']}[/green]"
                psi_color = "red" if row['highest_psi'] > 0.2 else "yellow" if row['highest_psi'] > 0.1 else "green"
                top_psi_str = f"[{psi_color}]{row['highest_psi']:.4f}[/{psi_color}]"
                if row['accuracy_drop'] is None:
                    acc_drop_str = "—"
                else:
                    ad = row['accuracy_drop']
                    ad_color = "red" if ad > row['perf_threshold'] else "yellow" if ad > 0 else "green"
                    acc_drop_str = f"[{ad_color}]{ad*100:.1f}%[/{ad_color}]"
                degraded_str = "[red]YES[/red]" if row['perf_degraded'] else "[green]NO[/green]"
                table.add_row(
                    str(idx),
                    row['timestamp'][:19].replace("T", " "),
                    row['model'],
                    row['env'],
                    row['run_tag'] or "—",
                    alerts_str,
                    drift_str,
                    row['highest_psi_feature'] or "—",
                    top_psi_str,
                    acc_drop_str,
                    degraded_str,
                )

            console.print(table)

        if export_csv:
            import csv
            fieldnames = [
                "timestamp", "file", "model", "env", "run_tag",
                "total_alerts", "critical_alerts", "warning_alerts",
                "severe_drift", "slight_drift", "no_drift",
                "highest_psi_feature", "highest_psi",
                "accuracy_drop", "accuracy", "baseline_accuracy",
                "perf_degraded", "perf_threshold",
                "baseline_samples", "production_samples", "window_days",
            ]
            with open(export_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in all_summaries:
                    writer.writerow(row)
            console.print(f"[green]✓ Exported {len(all_summaries)} records to {export_csv}[/green]")

        if export_json:
            payload = {
                "summary": stats,
                "filters": {
                    "model": model, "env": env, "run_tag": run_tag,
                    "since": since, "until": until, "last_days": last_days,
                },
                "records": all_summaries,
            }
            with open(export_json, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            console.print(f"[green]✓ Exported {len(all_summaries)} records to {export_json}[/green]")

    except Exception as e:
        console.print(f"[red]❌ History command failed: {str(e)}[/red]")
        console.print_exception(show_locals=False)
        sys.exit(1)


@cli.command(name="compare")
@click.option("--model", type=str, default=None, help="Filter by model name")
@click.option("--env", type=str, default=None, help="Filter by environment name")
@click.option(
    "--a-since", type=str, required=True,
    help="Window A start date (e.g. '2026-06-01')",
)
@click.option(
    "--a-until", type=str, default=None,
    help="Window A end date (default: midpoint between a-since and b-since)",
)
@click.option(
    "--b-since", type=str, required=True,
    help="Window B start date (e.g. '2026-06-08')",
)
@click.option(
    "--b-until", type=str, default=None,
    help="Window B end date (default: now)",
)
@click.option("--export-csv", type=click.Path(), default=None, help="Export comparison to CSV")
@click.option("--export-json", type=click.Path(), default=None, help="Export comparison to JSON")
@click.option(
    "--data-dir", type=click.Path(exists=True, file_okay=False), default=".",
    help="Directory containing drift_metrics_*.json files (default: current dir)",
)
@click.pass_context
def compare_cmd(
    ctx, model: Optional[str], env: Optional[str],
    a_since: str, a_until: Optional[str], b_since: str, b_until: Optional[str],
    export_csv: Optional[str], export_json: Optional[str], data_dir: str,
):
    """
    📊 Compare two time windows side by side (ring-ratio analysis).

    \b
    Compare metrics between Window A and Window B:
      - Total runs, alerts, drift occurrences
      - High-risk feature changes (new / disappeared)
      - Performance degradation count and accuracy drop
      - Ring-ratio change for each metric

    \b
    Example:
      compare --model credit_risk_v2 --a-since 2026-06-01 --a-until 2026-06-07 --b-since 2026-06-08 --b-until 2026-06-14
    """
    try:
        from datetime import datetime

        json_files = glob.glob(os.path.join(data_dir, "drift_metrics_*.json"))
        if not json_files:
            console.print("[yellow]⚠️ No drift_metrics_*.json files found in " + data_dir + "[/yellow]")
            return

        a_summaries = _build_history_summary(
            json_files, model_filter=model, env_filter=env,
            since=a_since, until=a_until or b_since,
        )
        b_summaries = _build_history_summary(
            json_files, model_filter=model, env_filter=env,
            since=b_since, until=b_until,
        )

        a_stats = _aggregate_stats(a_summaries) if a_summaries else _aggregate_stats([])
        b_stats = _aggregate_stats(b_summaries) if b_summaries else _aggregate_stats([])

        from rich.table import Table
        from rich.panel import Panel

        def _ring_ratio(a_val, b_val):
            if a_val is None or b_val is None or a_val == 0:
                return "—"
            ratio = (b_val - a_val) / abs(a_val)
            sign = "+" if ratio >= 0 else ""
            color = "red" if ratio > 0 else "green" if ratio < 0 else "white"
            return f"[{color}]{sign}{ratio*100:.1f}%[/{color}]"

        def _diff_str(a_val, b_val):
            if a_val is None or b_val is None:
                return "—"
            diff = b_val - a_val
            sign = "+" if diff >= 0 else ""
            color = "red" if diff > 0 else "green" if diff < 0 else "white"
            return f"[{color}]{sign}{diff}[/{color}]"

        a_label = f"{a_since} ~ {a_until or b_since}"
        b_label = f"{b_since} ~ {b_until or 'now'}"

        metric_rows = [
            ("Total Runs", a_stats["total_runs"], b_stats["total_runs"], True),
            ("Total Alerts", a_stats["total_alerts"], b_stats["total_alerts"], True),
            ("Critical Alerts", a_stats["total_critical_alerts"], b_stats["total_critical_alerts"], True),
            ("Warning Alerts", a_stats["total_warning_alerts"], b_stats["total_warning_alerts"], True),
            ("Severe Drift Occ.", a_stats["total_severe_drift_occurrences"], b_stats["total_severe_drift_occurrences"], True),
            ("Slight Drift Occ.", a_stats["total_slight_drift_occurrences"], b_stats["total_slight_drift_occurrences"], True),
            ("Perf Degraded", a_stats["perf_degraded_count"], b_stats["perf_degraded_count"], True),
        ]

        if a_stats["avg_accuracy_drop"] is not None and b_stats["avg_accuracy_drop"] is not None:
            metric_rows.append(("Avg Acc Drop", a_stats["avg_accuracy_drop"], b_stats["avg_accuracy_drop"], False))
        if a_stats["max_accuracy_drop"] is not None and b_stats["max_accuracy_drop"] is not None:
            metric_rows.append(("Max Acc Drop", a_stats["max_accuracy_drop"], b_stats["max_accuracy_drop"], False))

        table = Table(title=f"Window Comparison: A vs B", show_lines=True)
        table.add_column("Metric", style="bold white")
        table.add_column(f"Window A\n{a_label}", justify="right", style="cyan")
        table.add_column(f"Window B\n{b_label}", justify="right", style="cyan")
        table.add_column("Diff", justify="center")
        table.add_column("Ring Ratio", justify="center")

        for label, a_v, b_v, is_int in metric_rows:
            a_str = str(a_v) if is_int else f"{a_v:.4f}"
            b_str = str(b_v) if is_int else f"{b_v:.4f}"
            diff = _diff_str(a_v, b_v)
            ratio = _ring_ratio(a_v, b_v)
            table.add_row(label, a_str, b_str, diff, ratio)

        console.print(table)

        a_risk = set(a_stats.get("high_risk_features", []))
        b_risk = set(b_stats.get("high_risk_features", []))
        new_features = sorted(b_risk - a_risk)
        disappeared_features = sorted(a_risk - b_risk)
        stable_features = sorted(a_risk & b_risk)

        if new_features or disappeared_features:
            feat_lines = []
            if new_features:
                feat_lines.append("[red]🔴 New high-risk:[/red] " + ", ".join(f"[bold]{f}[/bold]" for f in new_features))
            if disappeared_features:
                feat_lines.append("[green]🟢 Disappeared high-risk:[/green] " + ", ".join(f"[bold]{f}[/bold]" for f in disappeared_features))
            if stable_features:
                feat_lines.append("[yellow]🟡 Stable high-risk:[/yellow] " + ", ".join(f"[bold]{f}[/bold]" for f in stable_features))
            console.print(Panel("\n".join(feat_lines), title="High-Risk Feature Changes", border_style="bold red" if new_features else "yellow"))

        if export_csv:
            import csv
            csv_rows = []
            for label, a_v, b_v, is_int in metric_rows:
                diff_val = (b_v - a_v) if a_v is not None and b_v is not None else None
                ratio_val = ((b_v - a_v) / abs(a_v)) if a_v and a_v != 0 and b_v is not None else None
                csv_rows.append({
                    "metric": label,
                    "window_a": a_v, "window_b": b_v,
                    "diff": diff_val,
                    "ring_ratio": round(ratio_val, 4) if ratio_val is not None else None,
                })
            csv_rows.append({"metric": "new_high_risk_features", "window_a": "", "window_b": ", ".join(new_features), "diff": "", "ring_ratio": ""})
            csv_rows.append({"metric": "disappeared_high_risk_features", "window_a": ", ".join(disappeared_features), "window_b": "", "diff": "", "ring_ratio": ""})
            with open(export_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["metric", "window_a", "window_b", "diff", "ring_ratio"])
                writer.writeheader()
                for row in csv_rows:
                    writer.writerow(row)
            console.print(f"[green]✓ Comparison exported to {export_csv}[/green]")

        if export_json:
            payload = {
                "window_a": {"label": a_label, "since": a_since, "until": a_until, "stats": a_stats, "run_count": len(a_summaries)},
                "window_b": {"label": b_label, "since": b_since, "until": b_until, "stats": b_stats, "run_count": len(b_summaries)},
                "feature_changes": {
                    "new_high_risk": new_features,
                    "disappeared_high_risk": disappeared_features,
                    "stable_high_risk": stable_features,
                },
            }
            with open(export_json, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
            console.print(f"[green]✓ Comparison exported to {export_json}[/green]")

    except Exception as e:
        console.print(f"[red]❌ Compare command failed: {str(e)}[/red]")
        console.print_exception(show_locals=False)
        sys.exit(1)


if __name__ == "__main__":
    cli()
