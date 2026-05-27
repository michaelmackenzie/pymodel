import os

from roomodel.model_io import load_fit_model, load_workspace_and_metadata


def _iter_roo_collection(collection):
    if collection is None:
        return
    create_iterator = getattr(collection, "createIterator", None)
    if callable(create_iterator):
        iterator = create_iterator()
        while True:
            obj = iterator.Next()
            if not obj:
                break
            yield obj
        return
    try:
        for obj in collection:
            yield obj
    except TypeError:
        return


def load_and_summarize_model(model_file: str, verbose: int = 0):
    model_path = os.path.abspath(model_file)
    fit_model = load_fit_model(model_path)
    workspace, metadata = load_workspace_and_metadata(model_path)

    summary = {
        "model_path": model_path,
        "model_name": fit_model.model_name,
        "obs_range": metadata.get("obs_range", "workspace-defined"),
        "channels": list(fit_model.channels),
        "processes": list(fit_model.process_names),
        "signal_process": ", ".join(fit_model.signal_processes) if fit_model.signal_processes else "none",
        "poi_name": fit_model.poi_name,
        "constraints": int(metadata.get("n_constraints", 0)),
        "floating_params": int(metadata.get("n_floating", 0)),
        "observed_count": sum(float(v) for v in fit_model.observed_counts_by_channel.values()) if fit_model.observed_counts_by_channel else None,
        "pdf_lines": [],
    }

    if verbose:
        lines = []
        lines.append(f"Workspace: {workspace.GetName()}")
        lines.append("PDFs:")
        for pdf in _iter_roo_collection(workspace.allPdfs()):
            lines.append(f"  - {pdf.GetName()} ({pdf.ClassName()})")
        lines.append("Variables:")
        n_float = 0
        for var in _iter_roo_collection(workspace.allVars()):
            is_constant = False
            try:
                is_constant = bool(var.isConstant())
            except Exception:
                pass
            if not is_constant:
                n_float += 1
            try:
                lines.append(f"  - {var.GetName()} = {float(var.getVal()):.6g} [{float(var.getMin()):.6g}, {float(var.getMax()):.6g}]" + (" (fixed)" if is_constant else ""))
            except Exception:
                lines.append(f"  - {var.GetName()}" + (" (fixed)" if is_constant else ""))

        summary["floating_params"] = n_float
        summary["pdf_lines"] = lines

    return summary
