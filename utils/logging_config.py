"""
Centralized logging configuration.

Library modules should call `logging.getLogger(__name__)` and use it
normally -- they should NOT call `logging.basicConfig()` themselves (that's
an application-level decision, not a library one, and calling it in a
library causes surprising behavior for anyone who imports it). Entry points
(training scripts, the API server, notebooks) should call
`configure_logging()` once at startup.
"""
from __future__ import annotations

import logging
import sys


def configure_logging(level: str = "INFO", json_format: bool = False) -> None:
    """
    Call once, from an application entry point (train.py, api/main.py, a
    notebook's first cell) -- not from library modules.

    Args:
        level: "DEBUG", "INFO", "WARNING", "ERROR".
        json_format: structured (one-line-JSON-ish) output, easier to parse
            in log aggregation systems (e.g. CloudWatch, Datadog) than the
            default human-readable format. Off by default for local/dev use.
    """
    handler = logging.StreamHandler(sys.stdout)
    if json_format:
        fmt = (
            '{"time": "%(asctime)s", "level": "%(levelname)s", '
            '"logger": "%(name)s", "message": "%(message)s"}'
        )
    else:
        fmt = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
    handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%dT%H:%M:%S"))

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers = [handler]  # replace, don't stack, in case of re-configuration
