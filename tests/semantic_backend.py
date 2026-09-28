"""The numpy oracle (inference.semantic_likelihood) wrapped as a Backend, for the tests.

It is not a user-facing backend.  It lives in its own importable module so that parallel
workers (multiprocessing "spawn", see inference.parallel) can unpickle it.
"""

import numpy as np

from inference.semantic_likelihood import SemanticLikelihood
from stat_backends.base import Backend, UnsupportedByBackend


class SemanticBackend(Backend):
    """The numpy oracle wrapped as a backend (not a user-facing backend)."""

    name = "semantic"

    @property
    def supported_features(self):
        # read at call time: SemanticLikelihood.supported may grow
        return frozenset(SemanticLikelihood.supported | {"shape:parametric"})

    def runtime_versions(self):
        return [("numpy", np.__version__)]

    def create(self, model, options):
        try:
            return SemanticLikelihood(model)
        except NotImplementedError as exc:
            raise UnsupportedByBackend(f"semantic oracle: {exc}") from exc
