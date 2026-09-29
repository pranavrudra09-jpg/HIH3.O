"""Stable ASGI entrypoint for the existing DealBrief AI application."""

import importlib.util
import sys
from pathlib import Path

_implementation_path = Path(__file__).resolve().parents[1] / "app.py"
_implementation_name = "_dealbrief_existing_application"
_spec = importlib.util.spec_from_file_location(_implementation_name, _implementation_path)
if _spec is None or _spec.loader is None:
    raise RuntimeError("Could not load the existing DealBrief AI application module.")

_implementation = importlib.util.module_from_spec(_spec)
sys.modules[_implementation_name] = _implementation
try:
    _spec.loader.exec_module(_implementation)
except Exception:
    sys.modules.pop(_implementation_name, None)
    raise

app = _implementation.app