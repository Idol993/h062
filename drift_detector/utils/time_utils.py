from datetime import datetime, timedelta
from typing import Optional, Tuple, List
from dateutil.relativedelta import relativedelta


class TimeUtils:
    @staticmethod
    def get_sliding_window(
        window_days: int = 7,
        end_date: Optional[datetime] = None,
    ) -> Tuple[datetime, datetime]:
        if end_date is None:
            end_date = datetime.now()
        start_date = end_date - timedelta(days=window_days)
        return start_date, end_date

    @staticmethod
    def get_date_range(
        start_date: datetime,
        end_date: datetime,
        freq: str = "D",
    ) -> List[datetime]:
        dates = []
        current = start_date

        if freq == "D":
            delta = timedelta(days=1)
        elif freq == "W":
            delta = timedelta(weeks=1)
        elif freq == "M":
            while current <= end_date:
                dates.append(current)
                current = current + relativedelta(months=1)
            return dates
        else:
            delta = timedelta(days=1)

        while current <= end_date:
            dates.append(current)
            current += delta

        return dates

    @staticmethod
    def parse_date(date_str: str, fmt: str = "%Y-%m-%d") -> datetime:
        return datetime.strptime(date_str, fmt)

    @staticmethod
    def format_date(date: datetime, fmt: str = "%Y-%m-%d") -> str:
        return date.strftime(fmt)

    @staticmethod
    def get_timestamp() -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    @staticmethod
    def get_iso_timestamp() -> str:
        return datetime.now().isoformat()

    @staticmethod
    def is_within_window(
        target_date: datetime,
        start_date: datetime,
        end_date: datetime,
    ) -> bool:
        return start_date <= target_date <= end_date

    @staticmethod
    def get_window_label(window_days: int) -> str:
        if window_days == 1:
            return "last 24 hours"
        elif window_days == 7:
            return "last 7 days"
        elif window_days == 30:
            return "last 30 days"
        elif window_days == 90:
            return "last 90 days"
        else:
            return f"last {window_days} days"

    @staticmethod
    def generate_time_points(
        start_date: datetime,
        end_date: datetime,
        n_points: int = 30,
    ) -> List[datetime]:
        total_seconds = (end_date - start_date).total_seconds()
        interval = total_seconds / max(1, n_points - 1)

        points = []
        for i in range(n_points):
            point = start_date + timedelta(seconds=i * interval)
            points.append(point)

        return points
