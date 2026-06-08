import os

from roomodel.model_io import load_fit_model, load_workspace_and_metadata


def _iter_roo_collection(collection):
    if collection is None:
        return
    try:
        for obj in collection:
            yield obj
    except TypeError:
        return


def _collect_observable_names(workspace):
    """Return the set of variable names that are true fit observables.

    A fit observable is a non-constant RooRealVar that is a direct server of
    a leaf (non-composite) PDF.  This correctly excludes parameters, which
    are constant in the workspace layout produced by the roomodel builder.
    """
    COMPOSITE_CLASSES = {
        "RooAddPdf", "RooProdPdf", "RooSimultaneous",
        "RooProduct", "RooAddition",
    }
    observable_names = set()
    for pdf in _iter_roo_collection(workspace.allPdfs()):
        if pdf.ClassName() in COMPOSITE_CLASSES:
            continue
        try:
            for server in pdf.servers():
                if not server.InheritsFrom("RooRealVar"):
                    continue
                try:
                    if not bool(server.isConstant()):
                        observable_names.add(str(server.GetName()))
                except Exception:
                    pass
        except Exception:
            pass
    return observable_names


def _count_floating_params(workspace, observable_names):
    """Count non-constant workspace variables that are not fit observables."""
    n_float = 0
    for var in _iter_roo_collection(workspace.allVars()):
        try:
            is_constant = bool(var.isConstant())
        except Exception:
            is_constant = False
        if not is_constant and str(var.GetName()) not in observable_names:
            n_float += 1
    return n_float


def load_and_summarize_model(model_file: str, verbose: int = 0):
    model_path = os.path.abspath(model_file)
    fit_model = load_fit_model(model_path)
    workspace, metadata = load_workspace_and_metadata(model_path)

    # Count floating parameters and constraints directly from the workspace so
    # the reported values always reflect what is actually in the model, rather
    # than what the card claimed.  Constraints are the number of non-constant
    # non-observable parameters (same as floating params, since each nuisance
    # parameter has a corresponding constraint term).
    observable_names = _collect_observable_names(workspace)
    n_floating = _count_floating_params(workspace, observable_names)
    # Subtract the POI from the nuisance/constraint count (the POI floats but
    # is not a constrained nuisance parameter).
    poi_name = fit_model.poi_name or metadata.get("poi_name")
    n_constraints = max(0, n_floating - (1 if poi_name else 0))

    summary = {
        "model_path": model_path,
        "model_name": fit_model.model_name,
        "obs_range": metadata.get("obs_range", "workspace-defined"),
        "channels": list(fit_model.channels),
        "processes": list(fit_model.process_names),
        "signal_process": ", ".join(fit_model.signal_processes) if fit_model.signal_processes else "none",
        "poi_name": poi_name,
        "constraints": n_constraints,
        "floating_params": n_floating,
        "observed_count": sum(float(v) for v in fit_model.observed_counts_by_channel.values()) if fit_model.observed_counts_by_channel else None,
        "pdf_lines": [],
    }

    if verbose:
        lines = []
        lines.append(f"Workspace: {workspace.GetName()}")
        lines.append("PDFs:")
        for pdf in _iter_roo_collection(workspace.allPdfs()):
            lines.append(f"  - {pdf.GetName()} ({pdf.ClassName()})")

        # observable_names is already computed above; reuse it here so the
        # verbose variable listing is consistent with the summary counts.
        lines.append("Variables:")
        for var in _iter_roo_collection(workspace.allVars()):
            var_name = str(var.GetName())
            is_observable = var_name in observable_names
            is_constant = False
            try:
                is_constant = bool(var.isConstant())
            except Exception:
                pass
            label = " (observable)" if is_observable else (" (fixed)" if is_constant else "")
            try:
                lines.append(f"  - {var_name} = {float(var.getVal()):.6g} [{float(var.getMin()):.6g}, {float(var.getMax()):.6g}]{label}")
            except Exception:
                lines.append(f"  - {var_name}{label}")

        summary["pdf_lines"] = lines

    return summary
