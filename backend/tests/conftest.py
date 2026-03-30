"""
Pytest configuration for the backend test suite.

`modal` is not available in the CI test environment, so we register a
MagicMock for it before any test module (or the app itself) is imported.
This lets us import and test the pure-math helpers in app.py without
requiring a real Modal token.
"""
import sys
from unittest.mock import MagicMock

# Must happen before any import that transitively pulls in app.py
sys.modules.setdefault("modal", MagicMock())
