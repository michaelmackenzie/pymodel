"""Backend registry.  Backends are imported lazily so that, e.g., hfmodel works without zfit."""

import importlib

BACKEND_NAMES = ("hfmodel", "zmodel", "roomodel")


def get_backend(name: str):
    if name not in BACKEND_NAMES:
        raise KeyError(f"Unknown backend '{name}'; choose from {', '.join(BACKEND_NAMES)}")
    return importlib.import_module(f"stat_backends.{name}").BACKEND
