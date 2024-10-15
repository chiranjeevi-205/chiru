import datetime
import time


class time_operations:

    @staticmethod
    def get_now_time():
        return datetime.datetime.now()

    @staticmethod
    def get_current_time():
        """
        Get the current UTC time.

        Returns:
            datetime.datetime: Current UTC time.
        """
        return datetime.datetime.utcnow()

    @staticmethod
    def string_to_datetime(current_time):
        """
        Convert a string representation of time to a datetime object.

        Args:
            current_time (str): String representing time.

        Returns:
            datetime.datetime: Datetime object.
        """
        format_string = "%Y-%m-%d %H:%M:%S.%f%z"
        current_time = datetime.datetime.strptime(current_time, format_string)
        return current_time

    @staticmethod
    def get_previous_time(days=0, hours=0, minutes=0, seconds=0):
        """
        Get the datetime object representing a time in the past.

        Args:
            days (int): Number of days in the past.
            hours (int): Number of hours in the past.
            minutes (int): Number of minutes in the past.
            seconds (int): Number of seconds in the past.

        Returns:
            datetime.datetime: Datetime object representing the past time.
        """
        return time_operations.get_current_time() - datetime.timedelta(
            seconds=seconds, hours=hours, minutes=minutes, days=days
        )

    @staticmethod
    def wait(seconds):
        """
        Pause the execution for the specified number of seconds.

        Args:
            seconds (int): Number of seconds to wait.

        Returns:
            None
        """
        time.sleep(seconds)
