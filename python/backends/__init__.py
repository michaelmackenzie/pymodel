"""Backend registry for pymodel."""

from typing import Dict

from backends.hfmodel.implementation import create_backend as create_hf_backend
from backends.zmodel.implementation import create_backend as create_z_backend
from backends.base import BaseBackend


def get_backends() -> Dict[str, BaseBackend]:
    """Return supported backends keyed by CLI name."""
    hf_backend = create_hf_backend()
    z_backend = create_z_backend()
    return {
        hf_backend.name: hf_backend,
        z_backend.name: z_backend,
    }
