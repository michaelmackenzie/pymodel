"""Backend contract.

A backend is a package ``stat_backends.<name>`` exposing a ``Backend`` subclass instance named
``BACKEND``.  A backend only has to turn a ``ModelIR`` into an ``inference.model.Likelihood``;
the CLI, fitting, toys and every statistical method are shared.

Rules for backends
------------------
* ``supported_features`` lists the ``ModelIR.features()`` strings the backend implements.
  Likelihoods are created through ``build_likelihood``, which refuses a model that uses
  anything else and names the unsupported features.  Never approximate or drop silently: if a
  feature can only be approximated, it needs an explicit option and an entry in
  ``Likelihood.notes``.
* ``nll_main`` follows the convention in ``inference/model.py`` so that every backend gives the
  same absolute NLL for the same model, parameters and data.
* Backend-specific options are added with ``add_arguments`` and arrive in ``options``.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple


class UnsupportedByBackend(RuntimeError):
    pass


class Backend(ABC):
    name = ""
    description = ""
    supported_features: frozenset = frozenset()

    def add_arguments(self, parser):
        """Add backend-specific CLI options (called for every command)."""

    @abstractmethod
    def runtime_versions(self) -> List[Tuple[str, str]]:
        """(package, version) pairs for the run header and the result file."""

    @abstractmethod
    def create(self, model, options):
        """Return an ``inference.model.Likelihood`` for ``model``."""

    def export(self, model, path: str, options):
        """Write the model in the backend's native format (pyhf JSON, RooWorkspace, ...)."""
        raise UnsupportedByBackend(f"{self.name} has no native export")

    def build_likelihood(self, model, options):
        missing = sorted(model.features() - set(self.supported_features))
        if missing:
            raise UnsupportedByBackend(
                f"backend '{self.name}' does not support these model features: {', '.join(missing)}")
        return self.create(model, options)
