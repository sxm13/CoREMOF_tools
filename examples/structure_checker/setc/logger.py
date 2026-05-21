import logging

# handle older python versions
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from enum import Enum


# logging level for type hinting
class LogLevel(Enum):
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


LogLevelType = Literal[
    LogLevel.DEBUG, LogLevel.INFO, LogLevel.WARNING, LogLevel.ERROR, LogLevel.CRITICAL
]


# initialize logger function
def logger_setup(name, log_file, log_level: LogLevelType) -> logging.Logger:
    """
    Set up logging in the main process.

    Parameters:
        name (str): logger name.
        log_file (str): file to writing logging messages.
        log_level (logging.LOGLEVEL): desired level of logging information.

    Returns:
        logger (logging.Logger): main process logging.
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # remove duplicate handlers
    if not logger.handlers:
        s_handler = logging.StreamHandler()
        s_handler.setFormatter(get_formatter())
        logger.addHandler(s_handler)

        f_handler = logging.FileHandler(filename=log_file, mode="a")
        f_handler.setFormatter(get_formatter())
        logger.addHandler(f_handler)

    return logger


# set log message format
def get_formatter() -> logging.Formatter:
    return logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s (%(funcName)s) | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# get logger for other files
def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
