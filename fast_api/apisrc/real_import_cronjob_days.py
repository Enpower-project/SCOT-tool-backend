import sys
from real_utils import collect_data, calculate_true_consumption, import_dataframes_to_db
from datetime import datetime, timedelta, timezone


def get_time_range_days(days_back: int = 1, hours_back: int = 0) -> tuple[str, str]:
    """
    Get time range for last N days and hours

    Args:
        days_back: Number of days back from now
        hours_back: Additional hours back from now

    Returns:
        Tuple of (start_time, end_time) in RFC3339 format

    Examples:
        # Last 7 days
        start, end = get_time_range_days(7)

        # Last 2 days and 6 hours
        start, end = get_time_range_days(2, 6)

        # Last 12 hours
        start, end = get_time_range_days(0, 12)
    """
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=days_back, hours=hours_back)

    def format_time(dt):
        return dt.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'

    return format_time(start_time), format_time(end_time)


def main():
    try:
        # Get last 60 minutes of data (with overlap for upsert)
        start_time, end_time = get_time_range_days(1)

        # Collect data
        pv_prod = collect_data("pv_prod", start_time, end_time)
        tc = calculate_true_consumption(start_time, end_time)

        # Import to database
        prod_rows, cons_rows = import_dataframes_to_db(pv_prod, tc)
        print(
            f"✅ Import completed: {prod_rows} production, {cons_rows} consumption rows")

    except Exception as e:
        print(f"❌ Cronjob failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
