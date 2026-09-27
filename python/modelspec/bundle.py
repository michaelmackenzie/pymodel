"""Save/load a built model (ModelIR) as a self-contained bundle.

A bundle is a JSON file holding the IR.  When the IR references RooFit objects, they are
copied (with everything they depend on) into ``<stem>_objects.root`` next to the JSON and
the references are rewritten to point there, so a bundle never depends on the original
workspaces.  Loading a bundle reproduces the model exactly: every backend rebuilds its
likelihood from the IR.
"""

import copy
import json
import os
from typing import Dict, Tuple

from modelspec import ir as I

BUNDLE_FORMAT = "pymodel-model"
OBJECT_WORKSPACE = "pymodel_objects"


def _iter_refs(model: I.ModelIR):
    for ch in model.channels:
        if ch.obs_ref is not None:
            yield ch, "obs_ref", ch.obs_ref
        for proc in ch.processes:
            if proc.shape.ref is not None:
                yield proc.shape, "ref", proc.shape.ref
            for key, refs in proc.shape.syst_refs.items():
                for i, ref in enumerate(refs):
                    yield refs, i, ref
            for term in proc.norm_terms:
                if term.ref is not None:
                    yield term, "ref", term.ref


def _copy_objects(model: I.ModelIR, root_path: str):
    from modelspec import rootinput as R

    ROOT = R.root()
    out_ws = ROOT.RooWorkspace(OBJECT_WORKSPACE)
    moved: Dict[Tuple[str, str, str], str] = {}
    for holder, key, ref in list(_iter_refs(model)):
        src = (ref.file, ref.workspace, ref.name)
        if src not in moved:
            ws = R.get_workspace(ref.file, ref.workspace)
            obj = ws.arg(ref.name)
            if not obj:
                raise KeyError(f"Object {ref.name} not found in {ref.file}:{ref.workspace}")
            if not out_ws.arg(ref.name):
                if getattr(out_ws, "import")(obj, ROOT.RooFit.RecycleConflictNodes(), ROOT.RooFit.Silence()):
                    raise RuntimeError(f"Could not copy {ref.name} into the bundle workspace")
            moved[src] = ref.name
        new_ref = I.RooRef(file=os.path.basename(root_path), workspace=OBJECT_WORKSPACE, name=ref.name,
                           class_name=ref.class_name)
        if isinstance(holder, list):
            holder[key] = new_ref
        else:
            setattr(holder, key, new_ref)
    if moved:
        out_ws.writeToFile(root_path, True)
    return bool(moved)


def save_bundle(model: I.ModelIR, path: str) -> str:
    path = os.path.abspath(path)
    model = copy.deepcopy(model)
    stem = os.path.splitext(path)[0]
    has_refs = any(True for _ in _iter_refs(model))
    if has_refs:
        _copy_objects(model, stem + "_objects.root")
    payload = {"format": BUNDLE_FORMAT, "model": model.to_dict()}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1)
    return path


def load_bundle(path: str) -> I.ModelIR:
    path = os.path.abspath(path)
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("format") != BUNDLE_FORMAT:
        raise ValueError(f"'{path}' is not a pymodel model bundle")
    model = I.ir_from_dict(payload["model"])
    base = os.path.dirname(path)
    for holder, key, ref in list(_iter_refs(model)):
        if not os.path.isabs(ref.file):
            ref.file = os.path.join(base, ref.file)
    return model


def load_model(path: str, **build_options) -> I.ModelIR:
    """Load a model from a datacard (.txt / any non-.json) or a bundle (.json)."""
    if path.endswith(".json"):
        return load_bundle(path)
    from modelspec.datacard import build_ir

    return build_ir(path, **build_options)
