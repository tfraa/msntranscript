"""
Shared utilities and logging setup
"""
import logging
import sys

class _PipelineFormatter(logging.Formatter):
    """
    Log formatter for the msnpip pipeline.

    Format:
        [HH:MM:SS]  LEVEL   message
    Level labels are padded so output columns align.
    """
    LEVEL_LABELS = {
        logging.DEBUG:    "DEBUG  ",
        logging.INFO:     "INFO   ",
        logging.WARNING:  "WARNING",
        logging.ERROR:    "ERROR  ",
        logging.CRITICAL: "CRITICAL",
    }

    def format(self, record):
        time_str = self.formatTime(record, datefmt="%H:%M:%S")
        level = self.LEVEL_LABELS.get(record.levelno, record.levelname)
        return f"[{time_str}]  {level}  {record.getMessage()}"


def setup_logging(verbose: bool = False):
    """
    Configure logging for the msnpip pipeline.

    Sets up a readable format and suppresses noisy third-party
    libraries (imaging_transcriptomics, nilearn, neuromaps, matplotlib, …).

    Args:
        verbose: If True, set log level to DEBUG, otherwise INFO
    """
    level = logging.DEBUG if verbose else logging.INFO

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_PipelineFormatter())

    # Configure the root logger
    root = logging.getLogger()
    root.setLevel(level)
    # Remove any handlers already attached
    root.handlers.clear()
    root.addHandler(handler)

    # Suppress noisy third-party libraries
    for noisy_logger in (
        "imaging_transcriptomics",
        "neuromaps",
        "nilearn",
        "matplotlib",
        "PIL",
        "nibabel",
        "gseapy",
    ):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
