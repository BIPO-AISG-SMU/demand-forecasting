from typing import List, Dict
from datetime import datetime, timedelta

def generate_default_daily_date(num_dates: int = 7) -> List:
    """Function to generate default dates for inference.

    Args:
        num_dates (int, optional): Number of days. Defaults to 7 days.

    Returns:
        num_days_list (List): list of dates for inference
    """
    # generate inference list
    today_date = datetime.today()
    # Create a list to store the dates for the next 7 days
    num_days_list = []
    # Loop to generate dates for the next 7 days
    days = 2  # inference period is after 2 days
    for day in range(num_dates):
        day = days + day
        next_day = today_date + timedelta(days=day)
        num_days_list.append(next_day.strftime("%Y-%m-%d"))
    return num_days_list


def generate_default_lag_date(num_dates: int = 14) -> List:
    """Function to generate default lag dates for inference.

    Args:
        num_dates (int, optional): Number of days. Defaults to 14.

    Returns:
        previous_days_list (List): list of lag dates for inference
    """
    # generate inference list
    today_date = datetime.today()
    # Create a list to store the dates for the next 7 days
    previous_days_list = []
    # Loop to generate dates for the next 7 days
    for day in range(num_dates):
        day = day + 1
        next_day = today_date - timedelta(days=day)
        previous_days_list.append(next_day.strftime("%Y-%m-%d"))
    previous_days_list.sort(reverse=False)
    return previous_days_list
