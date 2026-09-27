"""hfmodel: pyhf (HistFactory, numpy tensor backend) backend.

Supports counting and binned channels with histogram templates or fixed parametric
histograms, lnN (symmetric and asymmetric), rateParam, gmN (one process, single-bin
channel), template shape systematics, unit Gaussian and gmN Poisson constraints.
See ``builder.py`` for the IR -> pyhf mapping and the notes it attaches.
"""

import json
import os

import numpy as np

from stat_backends.base import Backend, UnsupportedByBackend
from stat_backends.hfmodel.builder import MODIFIER_SETTINGS, build_spec
from stat_backends.hfmodel.likelihood import HFLikelihood, import_pyhf


class HFBackend(Backend):
    name = "hfmodel"
    description = "pyhf HistFactory model (numpy tensor backend)"
    supported_features = frozenset({
        "data:count", "data:binned",
        "shape:counting", "shape:template", "shape:parametric", "shape:parametric-histogram",
        "syst:shape",
        "norm:lnN", "norm:asym_lnN", "norm:rate_param", "norm:gmN",
        "constraint:gauss", "constraint:poisson",
    })

    def runtime_versions(self):
        pyhf = import_pyhf()
        return [("pyhf", pyhf.__version__), ("numpy", np.__version__)]

    def create(self, model, options):
        # "shape:parametric" is listed only because every parametric histogram also carries it;
        # parametric shapes with floating parameters are rejected by the builder.
        return HFLikelihood(model)

    def export(self, model, path, options):
        """Write a pyhf JSON workspace plus ``<stem>_settings.json`` (interpolation codes,
        parameter map and notes, which the pyhf workspace format cannot hold)."""
        pyhf = import_pyhf()
        lik = HFLikelihood(model)
        if model.poi not in lik.spec.links:
            raise UnsupportedByBackend("hfmodel export: the model has no signal process, so no POI modifier")
        ws = lik.spec.workspace(model.poi)
        check = pyhf.Workspace(ws).model(modifier_settings=MODIFIER_SETTINGS)
        pars = lik.pyhf_pars(lik.nominal_values())
        if check.config.par_order != lik.pdf.config.par_order or not np.allclose(
                check.main_model.expected_data(pars), lik.pdf.main_model.expected_data(pars), rtol=1e-12):
            raise RuntimeError("hfmodel export: the re-read workspace differs from the model")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(ws, handle, indent=1)
        stem = os.path.splitext(path)[0]
        settings = {
            "load_with": {"modifier_settings": MODIFIER_SETTINGS},
            "parameter_map": {n: {"ir_name": l.ir_name, "kind": l.kind, "pyhf_equals_ir_times": l.factor}
                              for n, l in lik.spec.links.items()},
            "notes": lik.notes,
        }
        with open(stem + "_settings.json", "w", encoding="utf-8") as handle:
            json.dump(settings, handle, indent=1)
        print(f"  pyhf workspace uses normsys {MODIFIER_SETTINGS['normsys']['interpcode']} and histosys "
              f"{MODIFIER_SETTINGS['histosys']['interpcode']}: load it with "
              f"pyhf.Workspace(spec).model(modifier_settings=...) as in {stem}_settings.json")
        for note in lik.notes:
            print(f"  note: {note}")


BACKEND = HFBackend()
