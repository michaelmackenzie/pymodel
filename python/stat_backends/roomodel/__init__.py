"""roomodel: RooFit backend (the most general one).

The ModelIR becomes one RooWorkspace (workspace.py): RooFormulaVar/RooProduct yields with
Combine's lnN / asymPow / gmN / rateParam / formula terms, imported workspace pdfs and
``<pdf>_norm`` functions (their variables are the IR parameters), RooFormulaVar vertical
template morphing (``shape`` and ``shapeN``) and RooMultiPdf envelopes (Combine library).
The NLL of the inference/model.py convention is summed in C++ from RooFit evaluations
(evaluator.py, likelihood.py).  ``export`` writes a RooWorkspace with a ModelConfig
(export.py).  Envelopes need PYMODEL_ROOT_LIBS=libHiggsAnalysisCombinedLimit.so.
"""

import os

from stat_backends.base import Backend


class RooBackend(Backend):
    name = "roomodel"
    description = "RooFit workspace model (C++ evaluation of RooFit objects)"
    supported_features = frozenset({
        "data:count", "data:binned", "data:unbinned", "data:weighted",
        "shape:counting", "shape:template", "shape:parametric", "shape:parametric-histogram",
        "shape:envelope", "discrete",
        "syst:shape", "syst:shapeN",
        "norm:lnN", "norm:asym_lnN", "norm:lnU", "norm:gmN", "norm:rate_param", "norm:formula", "norm:ws_norm",
        "constraint:gauss", "constraint:bifurgauss", "constraint:poisson", "constraint:flat",
    })

    def runtime_versions(self):
        from modelspec import rootinput as R

        ROOT = R.root()
        out = [("ROOT", ROOT.gROOT.GetVersion())]
        libs = os.environ.get("PYMODEL_ROOT_LIBS", "")
        if libs:
            out.append(("PYMODEL_ROOT_LIBS", libs))
        return out

    def create(self, model, options):
        from stat_backends.roomodel.likelihood import RooLikelihood

        return RooLikelihood(model)

    def export(self, model, path, options):
        """RooWorkspace ``w`` with model_s, ModelConfig(_bonly), constraints and data_obs."""
        from stat_backends.roomodel.export import export_workspace

        missing = sorted(model.features() - set(self.supported_features))
        if missing:
            from stat_backends.base import UnsupportedByBackend

            raise UnsupportedByBackend(f"roomodel cannot export: {', '.join(missing)}")
        return export_workspace(model, path)


BACKEND = RooBackend()
