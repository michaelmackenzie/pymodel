"""zmodel: zfit / TensorFlow backend.

Counting, binned and unbinned (optionally weighted) channels; histogram templates with
Combine's vertical-morph shape systematics; parametric RooFit pdfs translated into zfit pdfs
(class map in ``roofit.py``); lnN, asymmetric lnN, lnU, gmN, rateParam, rateParam formulas
and ``<pdf>_norm`` functions built from RooFormulaVar/RooProduct/RooRealVar.  Every
translated pdf is compared with RooFit when the likelihood is created (``likelihood.py``).
"""

import os

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from stat_backends.base import Backend  # noqa: E402


class ZBackend(Backend):
    name = "zmodel"
    description = "zfit pdfs + TensorFlow graph of the full parameter vector"
    supported_features = frozenset({
        "data:count", "data:binned", "data:unbinned", "data:weighted",
        "shape:counting", "shape:template", "shape:parametric", "shape:parametric-histogram",
        "syst:shape",
        "norm:lnN", "norm:asym_lnN", "norm:lnU", "norm:gmN", "norm:rate_param", "norm:formula", "norm:ws_norm",
        "constraint:gauss", "constraint:bifurgauss", "constraint:poisson", "constraint:flat",
    })

    def add_arguments(self, parser):
        g = parser.add_argument_group("zmodel")
        g.add_argument("--zmodel-no-xla", action="store_true",
                       help="evaluate the NLL graph without XLA compilation (XLA is ~1.5x faster)")

    def runtime_versions(self):
        import tensorflow as tf
        import zfit

        return [("zfit", zfit.__version__), ("tensorflow", tf.__version__), ("numpy", np.__version__)]

    def create(self, model, options):
        from stat_backends.zmodel.likelihood import ZLikelihood

        return ZLikelihood(model, xla=not getattr(options, "zmodel_no_xla", False))


BACKEND = ZBackend()
