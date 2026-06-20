from typing import Callable, Optional, Dict, Any
import time
import schedule
from datetime import datetime
import threading

from ..utils.time_utils import TimeUtils


class ReportScheduler:
    def __init__(self):
        self.jobs: Dict[str, schedule.Job] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def schedule_report(
        self,
        job_id: str,
        task: Callable,
        interval: str = "daily",
        time_str: str = "08:00",
        *args,
        **kwargs,
    ) -> None:
        if job_id in self.jobs:
            self.cancel_job(job_id)

        if interval == "daily":
            job = schedule.every().day.at(time_str).do(self._wrap_task(task, *args, **kwargs))
        elif interval == "weekly":
            job = schedule.every().week.at(time_str).do(self._wrap_task(task, *args, **kwargs))
        elif interval == "hourly":
            job = schedule.every().hour.do(self._wrap_task(task, *args, **kwargs))
        elif interval == "minutes":
            job = schedule.every(int(time_str)).minutes.do(self._wrap_task(task, *args, **kwargs))
        else:
            raise ValueError(f"Unsupported interval: {interval}")

        self.jobs[job_id] = job

    def _wrap_task(self, task: Callable, *args, **kwargs) -> Callable:
        def wrapper():
            try:
                timestamp = TimeUtils.get_iso_timestamp()
                print(f"[{timestamp}] Running scheduled task...")
                result = task(*args, **kwargs)
                print(f"[{TimeUtils.get_iso_timestamp()}] Scheduled task completed")
                return result
            except Exception as e:
                print(f"[{TimeUtils.get_iso_timestamp()}] Scheduled task failed: {str(e)}")
                raise

        return wrapper

    def cancel_job(self, job_id: str) -> None:
        if job_id in self.jobs:
            schedule.cancel_job(self.jobs[job_id])
            del self.jobs[job_id]

    def start(self, run_once_now: bool = False) -> None:
        if self._running:
            return

        if run_once_now:
            for job_id in list(self.jobs.keys()):
                self.run_job(job_id)

        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _run_loop(self) -> None:
        while self._running:
            try:
                schedule.run_pending()
                time.sleep(1)
            except Exception as e:
                print(f"[{TimeUtils.get_iso_timestamp()}] Scheduler error: {str(e)}")

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def run_job(self, job_id: str) -> Any:
        if job_id not in self.jobs:
            raise ValueError(f"Job '{job_id}' not found")
        return self.jobs[job_id].run()

    def get_jobs(self) -> Dict[str, Dict[str, Any]]:
        job_info = {}
        for job_id, job in self.jobs.items():
            job_info[job_id] = {
                "next_run": job.next_run.isoformat() if job.next_run else None,
                "interval": str(job),
            }
        return job_info

    def is_running(self) -> bool:
        return self._running
