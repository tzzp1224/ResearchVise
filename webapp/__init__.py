"""Phase 5 web interface package."""

from .app import app
from .v2_app import app as v2_app

__all__ = ["app", "v2_app"]
