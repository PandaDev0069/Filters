"""Independent solar-filament filters extracted from the preprocessing notebook.

Use get_config() for fresh presets, import an individual algorithm, or run
``python -m Filters --help``. Imports never run demos or write output files.
"""

from .config import get_config

__all__ = ["get_config"]
