from typing import List, Dict, Any, Optional
import json
import logging
from logging.handlers import RotatingFileHandler
import requests
from datetime import datetime

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from .threshold_checker import Alert, AlertLevel
from ..utils.time_utils import TimeUtils


class Notifier:
    def __init__(
        self,
        enable_terminal: bool = True,
        webhook_url: Optional[str] = None,
        log_file_path: Optional[str] = None,
        log_max_bytes: int = 10 * 1024 * 1024,
        log_backup_count: int = 5,
    ):
        self.enable_terminal = enable_terminal
        self.webhook_url = webhook_url
        self.log_file_path = log_file_path
        self.console = Console()
        self.logger = self._setup_logger(log_file_path, log_max_bytes, log_backup_count)

    def _setup_logger(
        self,
        log_file_path: Optional[str],
        max_bytes: int,
        backup_count: int,
    ) -> Optional[logging.Logger]:
        if not log_file_path:
            return None

        logger = logging.getLogger("drift_alerts")
        logger.setLevel(logging.INFO)

        if logger.handlers:
            return logger

        handler = RotatingFileHandler(
            log_file_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger

    def _get_level_color(self, level: AlertLevel) -> str:
        color_map = {
            AlertLevel.CRITICAL: "#ff0000",
            AlertLevel.WARNING: "#ffff00",
            AlertLevel.INFO: "#00ff00",
        }
        return color_map.get(level, "#ffffff")

    def _get_level_style(self, level: AlertLevel) -> str:
        style_map = {
            AlertLevel.CRITICAL: "bold red",
            AlertLevel.WARNING: "bold yellow",
            AlertLevel.INFO: "bold green",
        }
        return style_map.get(level, "white")

    def notify(self, alerts: List[Alert]) -> None:
        timestamp = TimeUtils.get_iso_timestamp()
        for alert in alerts:
            alert.timestamp = timestamp

        if self.enable_terminal:
            self._notify_terminal(alerts)

        if self.logger:
            self._notify_log(alerts)

        if self.webhook_url:
            self._notify_webhook(alerts)

    def _notify_terminal(self, alerts: List[Alert]) -> None:
        if not alerts:
            self.console.print(
                Panel.fit(
                    "[green]✅ No drift detected - all features are stable[/green]",
                    title="Drift Detection Results",
                    border_style="green",
                )
            )
            return

        table = Table(
            title="🚨 Drift Detection Alerts",
            show_header=True,
            header_style="bold white",
            border_style="dim",
        )
        table.add_column("Level", style="bold", width=10)
        table.add_column("Feature", style="cyan", width=25)
        table.add_column("Metric", style="magenta", width=15)
        table.add_column("Value", justify="right", width=12)
        table.add_column("Threshold", justify="right", width=12)
        table.add_column("Message", width=60, overflow="fold")

        critical_count = sum(1 for a in alerts if a.level == AlertLevel.CRITICAL)
        warning_count = sum(1 for a in alerts if a.level == AlertLevel.WARNING)

        for alert in alerts:
            style = self._get_level_style(alert.level)
            value_str = (
                f"{alert.value:.4f}" if isinstance(alert.value, float) else str(alert.value)
            )
            threshold_str = (
                f"{alert.threshold:.4f}"
                if isinstance(alert.threshold, float)
                else str(alert.threshold)
            )

            table.add_row(
                Text(alert.level.value.upper(), style=style),
                alert.feature,
                alert.metric,
                value_str,
                threshold_str,
                alert.message,
            )

        self.console.print(table)

        if critical_count > 0 or warning_count > 0:
            summary = Text()
            summary.append(f"Total alerts: {len(alerts)}", style="bold")
            if critical_count > 0:
                summary.append(f" | CRITICAL: {critical_count}", style="bold red")
            if warning_count > 0:
                summary.append(f" | WARNING: {warning_count}", style="bold yellow")

            self.console.print(Panel(summary, border_style="red" if critical_count > 0 else "yellow"))

    def _notify_log(self, alerts: List[Alert]) -> None:
        if not self.logger:
            return

        for alert in alerts:
            log_data = {
                "timestamp": alert.timestamp,
                "level": alert.level.value,
                "feature": alert.feature,
                "metric": alert.metric,
                "value": alert.value,
                "threshold": alert.threshold,
                "message": alert.message,
                "details": alert.details,
            }

            if alert.level == AlertLevel.CRITICAL:
                self.logger.critical(json.dumps(log_data, ensure_ascii=False))
            elif alert.level == AlertLevel.WARNING:
                self.logger.warning(json.dumps(log_data, ensure_ascii=False))
            else:
                self.logger.info(json.dumps(log_data, ensure_ascii=False))

    def _notify_webhook(self, alerts: List[Alert]) -> None:
        if not self.webhook_url:
            return

        for alert in alerts:
            payload = {
                "alert": alert.message,
                "feature": alert.feature,
                "metric": alert.metric,
                "value": alert.value,
                "threshold": alert.threshold,
                "level": alert.level.value,
                "timestamp": alert.timestamp,
                "details": alert.details,
            }

            try:
                response = requests.post(
                    self.webhook_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=10,
                )
                response.raise_for_status()
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to send webhook: {str(e)}")
                if self.enable_terminal:
                    self.console.print(f"[red]⚠️ Webhook failed: {str(e)}[/red]")

    def send_webhook_async(self, alert: Alert) -> None:
        import threading

        def send():
            self._notify_webhook([alert])

        threading.Thread(target=send, daemon=True).start()
