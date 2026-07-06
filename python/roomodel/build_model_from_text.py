import os
import pathlib
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

from backends.path_bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path(__file__)

from backends.builder_common import resolve_shape_file_for_term
from backends.card_parser import CardSpec, parse_model_card as parse_common_model_card
from roomodel.model_io import save_fit_model_bundle


def _get_root():
    import ROOT

    try:
        if not getattr(ROOT, "_pymodel_roofit_quiet", False):
            ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)
            ROOT._pymodel_roofit_quiet = True
    except Exception:
        pass

    return ROOT


def parse_model_card(card_path: str) -> CardSpec:
    return parse_common_model_card(
        card_path,
        shape_extension=".root",
        shape_description="a ROOT file",
    )


def _common_process_aliases(process: str) -> List[str]:
    aliases = [process]
    lowered = process.lower()
    if lowered == "sig":
        aliases.append("signal")
    if lowered == "signal":
        aliases.append("sig")
    if lowered == "bkg":
        aliases.append("background")
    if lowered == "background":
        aliases.append("bkg")
    return aliases


def _first_workspace_from_file(root_path: str):
    ROOT = _get_root()
    tf = ROOT.TFile.Open(root_path)
    if tf is None or tf.IsZombie():
        raise ValueError(f"Could not open ROOT shapes file '{root_path}'")
    try:
        for key in tf.GetListOfKeys() or []:
            obj = tf.Get(key.GetName())
            if obj is not None and obj.InheritsFrom("RooWorkspace"):
                ws = obj.Clone(obj.GetName()) if hasattr(obj, "Clone") else obj
                ws.SetName(str(obj.GetName()))
                return ws
    finally:
        tf.Close()
    raise ValueError(f"No RooWorkspace found in shapes file '{root_path}'")


def _iter_roo_collection(collection):
    if collection is None:
        return
    try:
        for obj in collection:
            yield obj
    except TypeError:
        return


def _pdf_by_name_or_aliases(workspace, process_name: str):
    names = _common_process_aliases(process_name)
    for name in list(names):
        names.append(name.lower())
        names.append(name.upper())
    seen = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        pdf = workspace.pdf(name)
        if pdf is not None:
            return pdf

    # Fallback: exact lowercase compare against all PDFs.
    target = process_name.lower()
    for obj in _iter_roo_collection(workspace.allPdfs()):
        if str(obj.GetName()).lower() == target:
            return obj
    return None


def _data_obs_by_channel(workspace, channel: str):
    candidates = [
        f"{channel}__data_obs",
        f"data_obs__{channel}",
        "data_obs",
    ]
    for name in candidates:
        data = workspace.data(name)
        # cppyy wraps ROOT null pointers as non-None Python objects; use
        # bool() to trigger the underlying pointer validity check.
        if data is not None and bool(data):
            return data
    return None


def _make_obs_var(name: str, lower: float = 0.0, upper: float = 1.0):
    ROOT = _get_root()
    return ROOT.RooRealVar(name, name, lower, upper)


def _make_signal_strength_var(ws, signal_processes: List[str], mu_min: float = 0.0, mu_max: float = 10.0):
    """Create the primary signal-strength POI if a signal process exists."""
    ROOT = _get_root()
    if not signal_processes:
        return None
    # Use bare "mu" for a single signal process; for multiple signal processes
    # use the per-process name so each has a distinct parameter.
    if len(signal_processes) == 1:
        poi_name = "mu"
    else:
        poi_name = f"mu_{signal_processes[0]}"
    poi = ROOT.RooRealVar(poi_name, poi_name, 1.0, float(mu_min), float(mu_max))
    getattr(ws, "import")(poi)
    return poi


def _physical_mu_min_from_card(card: CardSpec, signal_processes: List[str]) -> float:
    lower_bounds = []
    for channel in card.channels:
        sig_rate, bkg_rate = _channel_rate_split(card, channel, signal_processes)
        if sig_rate > 0.0:
            lower_bounds.append(-float(bkg_rate) / float(sig_rate))
    if not lower_bounds:
        return 0.0
    # Keep total expected yield slightly positive in every channel.
    return float(max(lower_bounds) + 1.0e-6)


def _channel_rate_split(card: CardSpec, channel: str, signal_processes: List[str]) -> Tuple[float, float]:
    sig_rate = 0.0
    bkg_rate = 0.0
    signal_set = set(signal_processes)
    for process, bin_name, rate in zip(card.process_names, card.bin_names, card.rates):
        if bin_name != channel:
            continue
        val = float(rate or 0.0)
        if process in signal_set:
            sig_rate += val
        else:
            bkg_rate += val
    return sig_rate, bkg_rate


def _build_lnn_constraints(ws, card, process_names, bin_names, yield_var_names):
    """Build lnN/gs nuisance parameters, constraint PDFs, and scale factors.

    For each active uncertainty row in the card:
      - Create a floating nuisance parameter ``theta_<name>`` (range -7..7, init 0).
      - Create a ``RooGaussian`` constraint PDF ``constraint_<name>`` (mean=0, sigma=1).
      - For each (process, channel) pair with a non-"-" value, replace the existing
        yield variable in *yield_var_names* with a ``RooFormulaVar`` that scales the
        nominal yield by the appropriate factor.

    Parameters
    ----------
    ws : RooWorkspace
        The workspace being built (modified in-place).
    card : CardSpec
        Parsed model card.
    process_names : list[str]
        Process names in card column order.
    bin_names : list[str]
        Channel names in card column order (parallel to process_names).
    yield_var_names : dict[tuple[str,str], str]
        Mapping (process, channel) -> name of the nominal yield object in *ws*.
        This dict is updated in-place: constrained yields get new names.

    Returns
    -------
    list[str]
        Names of the constraint PDFs imported into *ws*.
    """
    ROOT = _get_root()
    from backends.builder_common import kind_token

    constraint_pdf_names = []

    for unc in (card.uncertainties or []):
        kind = kind_token(unc.kind)
        if kind not in ("lnN", "gs"):
            # shape uncertainties not supported in counting/shape workspaces yet
            continue

        # Check if at least one process/channel is affected
        active_indices = [
            i for i, v in enumerate(unc.values)
            if v not in (None, "-", "")
        ]
        if not active_indices:
            continue

        # 1. Create floating nuisance parameter theta ~ Gauss(0,1)
        theta_name = f"theta_{unc.name}"
        theta = ROOT.RooRealVar(theta_name, theta_name, 0.0, -7.0, 7.0)
        theta.setVal(0.0)
        getattr(ws, "import")(theta)
        theta_ws = ws.var(theta_name)

        # 2. Create Gaussian constraint PDF for theta: Gauss(theta | 0, 1)
        mean_name = f"constraint_mean_{unc.name}"
        sigma_name = f"constraint_sigma_{unc.name}"
        mean_var = ROOT.RooRealVar(mean_name, mean_name, 0.0)
        mean_var.setConstant(True)
        # Give sigma a finite positive range to suppress RooFit range warnings.
        sigma_var = ROOT.RooRealVar(sigma_name, sigma_name, 1.0, 1e-6, 100.0)
        sigma_var.setConstant(True)
        getattr(ws, "import")(mean_var)
        getattr(ws, "import")(sigma_var)
        constraint_name = f"constraint_{unc.name}"
        constraint_pdf = ROOT.RooGaussian(
            constraint_name, constraint_name,
            theta_ws,
            ws.var(mean_name),
            ws.var(sigma_name),
        )
        getattr(ws, "import")(constraint_pdf)
        constraint_pdf_names.append(constraint_name)

        # 3. Scale each affected yield by the constraint factor
        for i in active_indices:
            process = process_names[i]
            channel = bin_names[i]
            raw_value = unc.values[i]
            try:
                kappa = float(raw_value)
            except (TypeError, ValueError):
                continue

            key = (process, channel)
            nominal_yield_name = yield_var_names.get(key)
            if nominal_yield_name is None:
                continue

            nominal_obj = ws.function(nominal_yield_name) or ws.var(nominal_yield_name)
            if nominal_obj is None:
                continue

            scaled_name = f"yield_scaled_{unc.name}_{process}__{channel}"

            if kind == "lnN":
                # Scaled yield = nominal * kappa^theta
                log_kappa = float(np.log(kappa)) if kappa > 0 else 0.0
                log_kappa_var_name = f"log_kappa_{unc.name}_{process}__{channel}"
                log_kappa_var = ROOT.RooRealVar(log_kappa_var_name, log_kappa_var_name, log_kappa)
                log_kappa_var.setConstant(True)
                getattr(ws, "import")(log_kappa_var)
                # formula: nominal * exp(log_kappa * theta)
                # @0 = nominal_yield, @1 = log_kappa, @2 = theta
                scaled = ROOT.RooFormulaVar(
                    scaled_name, scaled_name,
                    "@0*exp(@1*@2)",
                    ROOT.RooArgList(nominal_obj, ws.var(log_kappa_var_name), theta_ws),
                )
            else:
                # gs: scaled yield = nominal * max(0, 1 + sigma*theta)
                # sigma = kappa - 1 (for kappa >= 1), or kappa (for kappa < 1)
                sigma_val = kappa - 1.0 if kappa >= 1.0 else kappa
                sigma_const_name = f"gs_sigma_{unc.name}_{process}__{channel}"
                sigma_const = ROOT.RooRealVar(sigma_const_name, sigma_const_name, sigma_val)
                sigma_const.setConstant(True)
                getattr(ws, "import")(sigma_const)
                # formula: nominal * max(0, 1 + sigma*theta)
                scaled = ROOT.RooFormulaVar(
                    scaled_name, scaled_name,
                    "@0*max(0.0, 1.0+@1*@2)",
                    ROOT.RooArgList(nominal_obj, ws.var(sigma_const_name), theta_ws),
                )

            getattr(ws, "import")(scaled, ROOT.RooFit.RecycleConflictNodes())
            # Update the mapping so downstream code picks up the constrained yield
            yield_var_names[key] = scaled_name

    return constraint_pdf_names


def _wrap_with_constraints(ws, sim_pdf_name, constraint_pdf_names):
    """Replace simPdf in ws with simPdf * prod(constraints) as 'constrainedPdf'.

    Returns the name of the final PDF to use as the model.
    """
    ROOT = _get_root()
    if not constraint_pdf_names:
        return sim_pdf_name

    sim_pdf = ws.pdf(sim_pdf_name)
    if sim_pdf is None:
        raise ValueError(f"Missing '{sim_pdf_name}' in workspace")

    pdf_list = ROOT.RooArgList()
    pdf_list.add(sim_pdf)
    for cname in constraint_pdf_names:
        cpdf = ws.pdf(cname)
        if cpdf is None:
            raise ValueError(f"Missing constraint PDF '{cname}'")
        pdf_list.add(cpdf)

    constrained = ROOT.RooProdPdf("constrainedPdf", "constrainedPdf", pdf_list)
    getattr(ws, "import")(constrained, ROOT.RooFit.RecycleConflictNodes())
    return "constrainedPdf"


def _build_counting_workspace(card: CardSpec):
    ROOT = _get_root()
    ws = ROOT.RooWorkspace("workspace")
    channel_model_names = []

    observed_counts = {}
    signal_processes = list(dict.fromkeys(
        process
        for process, pid in zip(card.process_names, card.process_ids)
        if int(pid) <= 0
    ))
    poi = _make_signal_strength_var(ws, signal_processes, mu_min=_physical_mu_min_from_card(card, signal_processes))

    # nominal yield names keyed by (process, channel) – needed for constraint wiring
    yield_var_names: Dict[Tuple[str, str], str] = {}

    for channel in card.channels:
        # Counting observable is an event count; keep it in a wide non-negative range.
        obs = _make_obs_var(f"count_obs_{channel}", 0.0, 1.0e6)
        getattr(ws, "import")(obs)

        sig_rate, bkg_rate = _channel_rate_split(card, channel, signal_processes)
        total_rate = sig_rate + bkg_rate

        # Build per-process yield variables so constraints can scale them individually
        for process, pid, bin_name, rate in zip(
            card.process_names, card.process_ids, card.bin_names, card.rates
        ):
            if bin_name != channel:
                continue
            rate_val = float(rate or 0.0)
            is_signal = int(pid) <= 0
            yield_name = f"yield_{process}__{channel}"
            if is_signal and poi is not None:
                nom_rate_name = f"nom_rate_{process}__{channel}"
                nom_rate = ROOT.RooRealVar(nom_rate_name, nom_rate_name, rate_val)
                nom_rate.setConstant(True)
                getattr(ws, "import")(nom_rate)
                term_yield = ROOT.RooFormulaVar(
                    yield_name, yield_name, "@0*@1",
                    ROOT.RooArgList(poi, ws.var(nom_rate_name)),
                )
            else:
                term_yield = ROOT.RooRealVar(yield_name, yield_name, rate_val, -1.0e12, 1.0e12)
                term_yield.setConstant(True)
            getattr(ws, "import")(term_yield, ROOT.RooFit.RecycleConflictNodes())
            yield_var_names[(process, channel)] = yield_name

        obs_val = float(card.observations.get(channel, card.observation_count or 0.0))
        observed_counts[channel] = obs_val
        ws_obs = ws.var(f"count_obs_{channel}")
        if ws_obs is not None and bool(ws_obs):
            ws_obs.setVal(obs_val)

    # Apply lnN/gs constraints (updates yield_var_names in-place)
    constraint_pdf_names = _build_lnn_constraints(
        ws, card, card.process_names, card.bin_names, yield_var_names
    )

    # Build per-channel total-yield formula and Poisson PDF using (possibly scaled) yields
    for channel in card.channels:
        yields_in_channel = []
        for process, bin_name in zip(card.process_names, card.bin_names):
            if bin_name != channel:
                continue
            yname = yield_var_names.get((process, channel))
            if yname is None:
                continue
            yobj = ws.function(yname) or ws.var(yname)
            if yobj is not None:
                yields_in_channel.append(yobj)

        if not yields_in_channel:
            raise ValueError(f"No yields resolved for counting channel '{channel}'")

        total_yield_list = ROOT.RooArgList()
        for y in yields_in_channel:
            total_yield_list.add(y)
        expected = ROOT.RooAddition(
            f"yield_total__{channel}", f"yield_total__{channel}", total_yield_list
        )
        getattr(ws, "import")(expected, ROOT.RooFit.RecycleConflictNodes())

        expected_obj = ws.function(f"yield_total__{channel}") or ws.var(f"yield_total__{channel}")

        # Build an extended shape model that mimics test_card.C's approach:
        #   RooUniform over a dummy x ∈ [0,1] with one bin, extended by yield_total.
        # This makes the model extended so createNLL includes the -lambda Poisson
        # normalization term.  Without it, NLL = -n*log(lambda) is monotonically
        # decreasing in mu when mu_hat < 0, making q_mu = 0 everywhere and the
        # CLs scan meaningless.
        dummy_x_name = f"dummy_x_{channel}"
        dummy_x = ROOT.RooRealVar(dummy_x_name, dummy_x_name, 0.5, 0.0, 1.0)
        dummy_x.setBins(1)
        getattr(ws, "import")(dummy_x)

        flat_name = f"flat_pdf_{channel}"
        # Build RooArgSet from the local variable (workspace pointer may not dereference)
        dummy_x_set = ROOT.RooArgSet(dummy_x)
        flat_pdf = ROOT.RooUniform(flat_name, flat_name, dummy_x_set)
        getattr(ws, "import")(flat_pdf, ROOT.RooFit.RecycleConflictNodes())

        ext_name = f"ext_pdf_total__{channel}"
        ext_pdf = ROOT.RooExtendPdf(ext_name, ext_name,
                                    ws.pdf(flat_name), expected_obj)
        getattr(ws, "import")(ext_pdf, ROOT.RooFit.RecycleConflictNodes())
        channel_model_names.append(ext_name)

    # Use RooSimultaneous over a channel-index category so the model is extended
    # (each channel PDF is a RooExtendPdf) and fitTo/createNLL with Extended(True)
    # handles the Poisson normalization correctly across all channels.
    channel_cat = ROOT.RooCategory("channelCat", "Channel index")
    for ch in card.channels:
        channel_cat.defineType(ch)
    getattr(ws, "import")(channel_cat)

    sim_pdf = ROOT.RooSimultaneous("simPdf", "simPdf", ws.cat("channelCat"))
    for ch, model_name in zip(card.channels, channel_model_names):
        ch_pdf = ws.pdf(model_name)
        if ch_pdf is None:
            raise ValueError(f"Missing extended channel model '{model_name}' in counting workspace")
        sim_pdf.addPdf(ch_pdf, ch)
    getattr(ws, "import")(sim_pdf, ROOT.RooFit.RecycleConflictNodes())

    final_pdf_name = _wrap_with_constraints(ws, "simPdf", constraint_pdf_names)
    return ws, final_pdf_name, None, observed_counts, signal_processes


def _resolve_shape_terms(card: CardSpec, card_dir: str) -> List[Tuple[str, str, str]]:
    terms = []
    for process, channel in zip(card.process_names, card.bin_names):
        rel_path = resolve_shape_file_for_term(card, process, channel)
        full = rel_path if os.path.isabs(rel_path) else os.path.join(card_dir, rel_path)
        terms.append((process, channel, os.path.abspath(full)))
    return terms


def _extract_obs_from_pdf(pdf, workspace):
    obs_set = pdf.getObservables(workspace.allVars())
    for obj in obs_set:
        if obj.InheritsFrom("RooRealVar"):
            return obj
    return None


def _build_shape_workspace(card: CardSpec, card_dir: str):
    ROOT = _get_root()
    ws = ROOT.RooWorkspace("workspace")

    signal_processes = list(dict.fromkeys(
        process
        for process, pid in zip(card.process_names, card.process_ids)
        if int(pid) <= 0
    ))
    signal_set = set(signal_processes)
    poi = _make_signal_strength_var(ws, signal_processes, mu_min=_physical_mu_min_from_card(card, signal_processes))

    by_channel = {ch: {"obs_name": None, "min": None, "max": None, "terms": []} for ch in card.channels}
    observed_counts = {ch: 0.0 for ch in card.channels}

    shapes_cache: Dict[str, object] = {}

    for process, pid, channel, rate, (_, _, root_path) in zip(
        card.process_names,
        card.process_ids,
        card.bin_names,
        card.rates,
        _resolve_shape_terms(card, card_dir),
    ):
        if root_path not in shapes_cache:
            shapes_cache[root_path] = _first_workspace_from_file(root_path)
        src_ws = shapes_cache[root_path]

        src_pdf = _pdf_by_name_or_aliases(src_ws, process)
        if src_pdf is None:
            raise ValueError(
                f"Could not resolve PDF for process '{process}' in shapes file '{root_path}'"
            )

        obs = _extract_obs_from_pdf(src_pdf, src_ws)
        if obs is None:
            raise ValueError(f"PDF '{src_pdf.GetName()}' has no RooRealVar observable")

        ch_info = by_channel[channel]
        obs_name = str(obs.GetName())
        if ch_info["obs_name"] is None:
            ch_info["obs_name"] = obs_name
            ch_info["min"] = float(obs.getMin())
            ch_info["max"] = float(obs.getMax())
        elif ch_info["obs_name"] != obs_name:
            raise ValueError(
                f"Channel '{channel}' mixes observables '{ch_info['obs_name']}' and '{obs_name}'"
            )

        ch_info["terms"].append(
            {
                "process": str(process),
                "rate": float(rate if rate is not None else 1.0),
                "root_path": root_path,
                "pdf_name": str(src_pdf.GetName()),
            }
        )

        src_data = _data_obs_by_channel(src_ws, channel)
        # cppyy wraps ROOT null pointers as non-None Python objects; use
        # bool() to trigger the underlying validity check.
        if src_data is not None and bool(src_data):
            try:
                observed_counts[channel] = float(src_data.sumEntries())
            except Exception:
                pass

    # --- Pass 1: import all shape PDFs and build nominal yield variables ---
    # Track the PDF name and nominal yield name per (process, channel).
    # yield_var_names will be updated by _build_lnn_constraints to point at
    # the scaled yield objects for any constrained processes.
    yield_var_names: Dict[Tuple[str, str], str] = {}
    shape_pdf_names: Dict[Tuple[str, str], str] = {}  # (process,channel) -> imported pdf name

    for channel in card.channels:
        channel_info = by_channel[channel]
        if channel_info["min"] is None or channel_info["max"] is None or not channel_info["terms"]:
            raise ValueError(f"No terms for channel '{channel}'")

        sig_rate_total = 0.0
        bkg_rate_total = 0.0
        for term in channel_info["terms"]:
            process = term["process"]
            root_path = term["root_path"]
            src_pdf_name = term["pdf_name"]
            rate_val = float(term["rate"])

            src_ws = shapes_cache[root_path]
            src_pdf = src_ws.pdf(src_pdf_name)
            if src_pdf is None:
                raise ValueError(
                    f"Could not reload PDF '{src_pdf_name}' for process '{process}' in '{root_path}'"
                )

            term_suffix = str(channel)
            imported_pdf_name = f"{src_pdf_name}_{term_suffix}"
            term_pdf = ws.pdf(imported_pdf_name)
            if term_pdf is None or not bool(term_pdf):
                getattr(ws, "import")(
                    src_pdf,
                    ROOT.RooFit.RenameAllNodes(term_suffix),
                    ROOT.RooFit.RenameAllVariables(term_suffix),
                )
                term_pdf = ws.pdf(imported_pdf_name)
            if term_pdf is None or not bool(term_pdf):
                raise ValueError(f"Failed to import shape PDF '{src_pdf_name}' for channel '{channel}'")

            shape_pdf_names[(process, channel)] = imported_pdf_name

            yield_name = f"yield_{process}__{channel}"
            if process in signal_set and poi is not None:
                rate_const = ROOT.RooRealVar(
                    f"rate_{process}__{channel}",
                    f"rate_{process}__{channel}",
                    rate_val,
                    -1.0e12,
                    1.0e12,
                )
                rate_const.setConstant(True)
                getattr(ws, "import")(rate_const)
                term_yield = ROOT.RooFormulaVar(
                    yield_name,
                    "@0*@1",
                    ROOT.RooArgList(poi, rate_const),
                )
                getattr(ws, "import")(term_yield)
                sig_rate_total += rate_val
            else:
                term_yield = ROOT.RooRealVar(
                    yield_name,
                    yield_name,
                    rate_val,
                    -1.0e12,
                    1.0e12,
                )
                term_yield.setConstant(True)
                getattr(ws, "import")(term_yield)
                bkg_rate_total += rate_val

            yield_var_names[(process, channel)] = yield_name

        sig_rate_const = ROOT.RooRealVar(f"sig_rate__{channel}", f"sig_rate__{channel}", sig_rate_total, -1.0e12, 1.0e12)
        bkg_rate_const = ROOT.RooRealVar(f"bkg_rate__{channel}", f"bkg_rate__{channel}", bkg_rate_total, -1.0e12, 1.0e12)
        sig_rate_const.setConstant(True)
        bkg_rate_const.setConstant(True)
        getattr(ws, "import")(sig_rate_const)
        getattr(ws, "import")(bkg_rate_const)

    # --- Pass 2: apply lnN/gs constraints (updates yield_var_names in-place) ---
    constraint_pdf_names = _build_lnn_constraints(
        ws, card, card.process_names, card.bin_names, yield_var_names
    )

    # --- Pass 3: build per-channel RooAddPdf using (possibly scaled) yields ---
    channel_model_names = []
    for channel in card.channels:
        channel_info = by_channel[channel]
        ch_pdf_list = ROOT.RooArgList()
        ch_yield_total_objs = []

        for term in channel_info["terms"]:
            process = term["process"]
            imported_pdf_name = shape_pdf_names[(process, channel)]
            term_pdf = ws.pdf(imported_pdf_name)
            if term_pdf is None or not bool(term_pdf):
                raise ValueError(f"Cannot find imported PDF '{imported_pdf_name}'")

            yname = yield_var_names.get((process, channel))
            if yname is None:
                raise ValueError(f"No yield resolved for process '{process}' channel '{channel}'")
            yobj = ws.function(yname) or ws.var(yname)
            if yobj is None:
                raise ValueError(f"Failed to resolve yield object '{yname}'")

            ch_pdf_list.add(term_pdf)
            ch_yield_total_objs.append(yobj)

        if not ch_yield_total_objs:
            raise ValueError(f"No yields resolved for channel '{channel}'")

        total_yield_list = ROOT.RooArgList()
        for y in ch_yield_total_objs:
            total_yield_list.add(y)
        ROOT.RooAddition(
            f"yield_total__{channel}",
            f"yield_total__{channel}",
            total_yield_list,
        )

        coeff_list = ROOT.RooArgList()
        for coeff in ch_yield_total_objs:
            coeff_list.add(coeff)
        ch_model_pdf = ROOT.RooAddPdf(
            f"pdf_total__{channel}",
            f"pdf_total__{channel}",
            ch_pdf_list,
            coeff_list,
            False,
        )
        getattr(ws, "import")(ch_model_pdf, ROOT.RooFit.RecycleConflictNodes())
        channel_model_names.append(f"pdf_total__{channel}")

    if len(channel_model_names) == 1:
        only_pdf = ws.pdf(channel_model_names[0])
        if only_pdf is None:
            raise ValueError(f"Missing channel model '{channel_model_names[0]}' in shape workspace")
        sim_list = ROOT.RooArgList()
        sim_list.add(only_pdf)
        sim_pdf = ROOT.RooProdPdf("simPdf", "simPdf", sim_list)
        getattr(ws, "import")(sim_pdf, ROOT.RooFit.RecycleConflictNodes())
    else:
        channel_cat = ROOT.RooCategory("channel", "channel")
        for channel in card.channels:
            channel_cat.defineType(channel)
        sim_pdf = ROOT.RooSimultaneous("simPdf", "simPdf", channel_cat)
        for channel in card.channels:
            model_name = f"pdf_total__{channel}"
            model_pdf = ws.pdf(model_name)
            if model_pdf is None:
                raise ValueError(f"Missing channel model '{model_name}' in shape workspace")
            sim_pdf.addPdf(model_pdf, channel)
        getattr(ws, "import")(sim_pdf, ROOT.RooFit.RecycleConflictNodes())

    # Import data_obs from the shape file(s) into the output workspace so
    # that analyze can load it as observed data instead of generating a toy.
    # The PDFs were imported with RenameAllVariables(channel), so the observable
    # inside the output workspace is named mass_{channel}, not mass.  Build a
    # new RooDataHist using the renamed observable and the bin contents from
    # the source data_obs.
    data_name = None
    for channel in card.channels:
        ch_info = by_channel[channel]
        obs_name_orig = ch_info.get("obs_name")
        if obs_name_orig is None:
            continue
        obs_name_renamed = f"{obs_name_orig}_{channel}"
        renamed_obs = ws.var(obs_name_renamed)
        if renamed_obs is None or not bool(renamed_obs):
            continue
        for root_path in shapes_cache:
            src_ws = shapes_cache[root_path]
            src_data = _data_obs_by_channel(src_ws, channel)
            if src_data is None or not bool(src_data):
                src_data = src_ws.data("data_obs")
            if src_data is None or not bool(src_data):
                continue
            # Use the source observable (same range/binning) to create the TH1,
            # then construct a new RooDataHist bound to the renamed observable.
            src_obs = src_ws.var(obs_name_orig)
            if src_obs is None or not bool(src_obs):
                continue
            th1 = src_data.createHistogram("_data_obs_tmp_th1", src_obs)
            if th1 is None or not bool(th1):
                continue
            arg_list = ROOT.RooArgList(renamed_obs)
            rebound = ROOT.RooDataHist("data_obs", "Observed data", arg_list, th1)
            getattr(ws, "import")(rebound)
            data_name = "data_obs"
            break
        if data_name is not None:
            break

    final_pdf_name = _wrap_with_constraints(ws, "simPdf", constraint_pdf_names)
    return ws, final_pdf_name, data_name, observed_counts, signal_processes


def build_model_from_card(card: CardSpec, card_dir: str):
    if card.is_counting:
        ws, model_name, data_name, observed_counts, signal_processes = _build_counting_workspace(card)
    else:
        ws, model_name, data_name, observed_counts, signal_processes = _build_shape_workspace(card, card_dir)

    # Use bare "mu" for a single signal process; for multiple signal processes
    # use the per-process name consistent with _make_signal_strength_var.
    if len(signal_processes) == 1:
        poi_name = "mu"
    elif signal_processes:
        poi_name = f"mu_{signal_processes[0]}"
    else:
        poi_name = "mu"

    # Count constraints and floating (nuisance) parameters from the card.
    # Each UncertaintySpec row that has at least one non-"-" value contributes
    # one constraint (a Gaussian penalty term) and one floating nuisance
    # parameter.  The POI itself is intentionally excluded from this count.
    n_constraints = 0
    for unc in (card.uncertainties or []):
        active = any(v not in (None, "-", "") for v in unc.values)
        if active:
            n_constraints += 1
    n_floating = n_constraints

    metadata = {
        "format": "fit_model_bundle_v1_roomodel",
        "workspace_name": str(ws.GetName()),
        "model_name": model_name,
        "data_name": data_name,
        "channels": list(card.channels),
        "process_names": list(card.process_names),
        "process_ids": list(card.process_ids),
        "signal_processes": list(signal_processes),
        "observed_counts_by_channel": dict(observed_counts),
        "poi_name": poi_name,
        "n_constraints": n_constraints,
        "n_floating": n_floating,
    }

    return {
        "workspace": ws,
        "metadata": metadata,
    }


def build_and_save_model_from_card_file(card_path: str, output_file: str) -> str:
    card_path = os.path.abspath(card_path)
    output_file = os.path.abspath(output_file)

    card = parse_model_card(card_path)
    card_dir = os.path.dirname(card_path)
    payload = build_model_from_card(card, card_dir)
    return save_fit_model_bundle(payload, output_file)
