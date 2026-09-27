"""Write the model as a RooWorkspace usable by RooStats.

Contents of workspace ``w``:
* per channel ``pdf_<ch>_nuis`` = extended RooAddPdf of the process pdfs with coefficients
  ``n_exp_<ch>_<proc>``; process pdfs: counting -> RooUniform on a one-bin observable,
  template -> RooParametricStepFunction with heights f_i/width_i, parametric -> the
  imported pdf, envelope -> the imported RooMultiPdf;
* constraint pdfs ``<param>_Pdf`` with global observables ``<param>_In``:
  gauss RooGaussian, bifurgauss RooBifurGauss(theta; mean=g, sigma_lo, sigma_hi),
  poisson RooPoisson(g; theta) (no rounding); flat -> the parameter range only;
* ``pdf_<ch>`` = RooProdPdf(main, constraints), ``model_s`` = RooSimultaneous over the
  category ``CMS_channel``;
* ``data_obs``: weighted RooDataSet (binned channels at bin centres with the counts);
* ``ModelConfig`` (and ``ModelConfig_bonly`` with the snapshot r = 0) and the sets POI,
  nuisances, globalObservables, observables, discreteParams;
* TNamed ``pymodel_ir`` with the IR as JSON.

A RooFit NLL of this workspace differs from ``nll_main`` only by data-only constants.
"""

import json

import numpy as np

from inference.model import observed_dataset
from modelspec import ir as I
from modelspec import rootinput as R
from stat_backends.roomodel.workspace import ModelWorkspace


def export_workspace(model: I.ModelIR, path: str):
    ROOT = R.root()
    mw = ModelWorkspace(model, name="w", export=True)
    ws = mw.ws
    imp = mw._imp
    cat = ROOT.RooCategory("CMS_channel", "CMS_channel")
    for cb in mw.channels:
        cat.defineType(cb.name)
    cat = imp(cat)

    # constraints and global observables
    constraints, globs = [], ROOT.RooArgSet()
    for p in model.parameters.values():
        c = p.constraint
        if c is None or p.role == I.ROLE_CONSTANT or c.kind == I.CONSTRAINT_FLAT:
            continue
        theta = mw.vars[p.name]
        g = ROOT.RooRealVar(f"{p.name}_In", f"{p.name}_In", c.center)
        g.setConstant(True)
        g = imp(g)
        globs.add(g)
        if c.kind == I.CONSTRAINT_GAUSS:
            pdf = ROOT.RooGaussian(f"{p.name}_Pdf", "", theta, g, mw._const(f"{p.name}_sigma", c.sigma_hi))
        elif c.kind == I.CONSTRAINT_BIFURGAUSS:
            pdf = ROOT.RooBifurGauss(f"{p.name}_Pdf", "", theta, g, mw._const(f"{p.name}_sigmaL", c.sigma_lo),
                                     mw._const(f"{p.name}_sigmaR", c.sigma_hi))
        elif c.kind == I.CONSTRAINT_POISSON:
            pdf = ROOT.RooPoisson(f"{p.name}_Pdf", "", g, theta, True)
        else:
            raise ValueError(c.kind)
        constraints.append(imp(pdf))

    sim = ROOT.RooSimultaneous("model_s", "model_s", cat)
    obs_set = ROOT.RooArgSet(cat)
    for cb in mw.channels:
        obs_set.add(cb.obs)
        pdfs, coefs = ROOT.RooArgList(), ROOT.RooArgList()
        for pb in cb.procs:
            name = f"shape_{cb.name}_{pb.name}"
            if pb.kind == "counting" or len(cb.edges) == 2 and pb.kind == "template":
                pdf = imp(ROOT.RooUniform(f"{name}_pdf", "", ROOT.RooArgSet(cb.obs)))
            elif pb.kind == "template":
                st = pb.states[0]
                widths = np.diff(cb.edges)
                heights = ROOT.RooArgList()
                for i in range(len(widths) - 1):
                    if st.cfrac is not None:
                        heights.add(mw._const(f"{name}_h{i}", st.cfrac[i] / widths[i]))
                    else:
                        heights.add(mw._formula(f"{name}_h{i}", f"@0/{widths[i]!r}", [st.ffrac[i]]))
                limits = ROOT.TArrayD(len(cb.edges), np.asarray(cb.edges, dtype=float))
                pdf = imp(ROOT.RooParametricStepFunction(f"{name}_pdf", "", cb.obs, heights, limits,
                                                         len(widths)))
            elif pb.kind == "envelope":
                pdf = pb.multipdf
            else:
                pdf = pb.states[0].pdf
            pdfs.add(pdf)
            coefs.add(pb.yield_func)
        main = imp(ROOT.RooAddPdf(f"pdf_{cb.name}_nuis", "", pdfs, coefs))
        prod_list = ROOT.RooArgList(main)
        for c in constraints:
            prod_list.add(c)
        chpdf = imp(ROOT.RooProdPdf(f"pdf_{cb.name}", "", prod_list))
        sim.addPdf(chpdf, cb.name)
    sim = imp(sim)

    # data
    wvar = ROOT.RooRealVar("_weight_", "_weight_", 1.0)
    dvars = ROOT.RooArgSet(obs_set)
    dvars.add(wvar)
    data = ROOT.RooDataSet("data_obs", "data_obs", dvars, ROOT.RooFit.WeightVar(wvar))
    observed = observed_dataset(model)
    for cb in mw.channels:
        cat.setLabel(cb.name)
        md = observed.main[cb.name]
        if md.kind == "unbinned":
            xs = md.values
            ws_ = md.weights if md.weights is not None else np.ones(len(xs))
        else:
            xs = 0.5 * (cb.edges[:-1] + cb.edges[1:])
            ws_ = md.counts
        for x, w in zip(xs, ws_):
            cb.obs.setVal(float(x))
            data.add(obs_set, float(w))
    getattr(ws, "import")(data)

    # sets and ModelConfig
    poi = ROOT.RooArgSet(mw.vars[model.poi])
    nuis, discrete = ROOT.RooArgSet(), ROOT.RooArgSet()
    for p in model.parameters.values():
        if p.role == I.ROLE_DISCRETE:
            discrete.add(mw.vars[p.name])
        elif p.floating and p.name != model.poi:
            nuis.add(mw.vars[p.name])
    for name, s in (("POI", poi), ("nuisances", nuis), ("globalObservables", globs), ("observables", obs_set),
                    ("discreteParams", discrete)):
        ws.defineSet(name, s)
    mc = ROOT.RooStats.ModelConfig("ModelConfig", ws)
    mc.SetPdf(sim)
    mc.SetParametersOfInterest(poi)
    mc.SetObservables(obs_set)
    mc.SetNuisanceParameters(nuis)
    mc.SetGlobalObservables(globs)
    getattr(ws, "import")(mc)
    mcb = ROOT.RooStats.ModelConfig("ModelConfig_bonly", ws)
    mcb.SetPdf(sim)
    mcb.SetParametersOfInterest(poi)
    mcb.SetObservables(obs_set)
    mcb.SetNuisanceParameters(nuis)
    mcb.SetGlobalObservables(globs)
    r = mw.vars[model.poi]
    rval = r.getVal()
    r.setVal(0.0)
    mcb.SetSnapshot(poi)
    r.setVal(rval)
    getattr(ws, "import")(mcb)
    ws.writeToFile(path, True)
    f = ROOT.TFile.Open(path, "UPDATE")
    ROOT.TNamed("pymodel_ir", json.dumps(model.to_dict())).Write()
    f.Close()
    return path
