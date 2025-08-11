# utils.py
import datetime
import logging

# Set up logging configuration
logging.basicConfig(
    filename="trading_bot.log",
    level=logging.DEBUG,  # You can adjust this level to INFO, WARNING, etc.
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def log_message(message, level="INFO"):
    """
    Log messages with a specified log level.
    Levels: INFO, WARNING, ERROR, DEBUG
    Parameters:
    - message (str): The message to log.
    - level (str): The level of the log message. Defaults to INFO.
    """
    log_levels = {
        "INFO": logging.info,
        "WARNING": logging.warning,
        "ERROR": logging.error,
        "DEBUG": logging.debug
    }

    # Log the message using the appropriate log level
    log_function = log_levels.get(level.upper(), logging.info)
    log_function(message)

def convert_timestamp(timestamp):
    """
    Convert a UNIX timestamp to a human-readable format (YYYY-MM-DD HH:MM:SS).
    
    Parameters:
    - timestamp (int): The UNIX timestamp to convert.
    
    Returns:
    - str: A human-readable date-time string or None if the timestamp is invalid.
    """
    try:
        dt_object = datetime.datetime.fromtimestamp(int(timestamp))
        return dt_object.strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        log_message(f"Error converting timestamp: {e}", level="ERROR")
        return None  # Return None instead of a string to indicate failure
