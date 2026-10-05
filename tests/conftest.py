"""Run the suite against another package, e.g. ``LABS_PACKAGE=labs_solutions uv run pytest``."""

import importlib
import os
import pkgutil
import sys

if package := os.environ.get("LABS_PACKAGE"):
    for module in pkgutil.iter_modules(importlib.import_module(package).__path__):
        sys.modules[f"labs.{module.name}"] = importlib.import_module(f"{package}.{module.name}")
