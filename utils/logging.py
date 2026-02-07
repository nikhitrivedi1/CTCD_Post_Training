"""
Shared logging utilities for training and analysis.

Provides TeeWriter for capturing stdout/stderr to file while handling
carriage returns from tqdm progress bars cleanly.
"""

import logging
import sys
from pathlib import Path


class TeeWriter:
    """Write to both a file and the original stream.

    Handles carriage returns so tqdm progress bars don't spam the log file.
    Terminal gets full interactive experience; file gets clean final states.
    """

    def __init__(self, file_path: Path, original_stream):
        self.file = open(file_path, "a")
        self.original = original_stream
        self.encoding = getattr(original_stream, "encoding", "utf-8")
        self._line_buffer = ""

    def write(self, data):
        self.original.write(data)
        for char in data:
            if char == "\r":
                self._line_buffer = ""
            elif char == "\n":
                self.file.write(self._line_buffer + "\n")
                self.file.flush()
                self._line_buffer = ""
            else:
                self._line_buffer += char

    def flush(self):
        self.original.flush()

    def close(self):
        if self._line_buffer:
            self.file.write(self._line_buffer + "\n")
        self.file.close()

    def isatty(self):
        return self.original.isatty()

    def fileno(self):
        return self.original.fileno()


def setup_file_logging(log_path: Path, update_transformers: bool = False):
    """
    Set up file logging with TeeWriter for clean progress bar handling.

    Captures stdout/stderr to log file while handling \\r so tqdm progress
    bars don't spam the log. Terminal gets interactive experience, file
    gets clean final states.

    Args:
        log_path: Path to the log file
        update_transformers: If True, also update transformers logging handlers

    Returns:
        The log_path for confirmation
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Capture stdout/stderr with TeeWriter
    sys.stdout = TeeWriter(log_path, sys.stdout)
    sys.stderr = TeeWriter(log_path, sys.stderr)

    # Update existing logging handlers to use TeeWriter
    for handler in logging.root.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.stream = sys.stdout

    if update_transformers:
        try:
            import transformers.utils.logging as tf_logging
            for handler in tf_logging.get_logger().handlers:
                if isinstance(handler, logging.StreamHandler):
                    handler.stream = sys.stdout
        except ImportError:
            pass

    return log_path
