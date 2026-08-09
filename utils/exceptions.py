"""
Custom exception hierarchy for CardioFusion-AI.

Using specific exception types (rather than bare ValueError/RuntimeError
everywhere) lets calling code -- an API layer, an ingestion pipeline, a
notebook -- catch and handle failure modes differently: a malformed input
signal is recoverable (skip this window, log it, continue), a missing model
checkpoint is not (fail startup loudly).
"""


class CardioFusionError(Exception):
    """Base class for all CardioFusion-AI-specific errors."""


class InvalidSignalError(CardioFusionError):
    """Raised when an input signal is empty, wrong-shaped, all-NaN, or otherwise unusable."""


class ConfigurationError(CardioFusionError):
    """Raised for invalid or missing configuration (bad YAML, out-of-range parameters)."""


class ModelNotLoadedError(CardioFusionError):
    """Raised when inference is requested but no model checkpoint has been loaded."""


class InsufficientDataError(CardioFusionError):
    """Raised when a signal is technically valid but too short/sparse for a given operation
    (e.g. fewer than 2 peaks, needed for any interval-based feature)."""
