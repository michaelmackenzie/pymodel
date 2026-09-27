"""Result files: every command writes one JSON document with a common header.

Schema (``format: pymodel-result``, version 1)::

    {
      "format": "pymodel-result", "version": 1,
      "command": "limit", "backend": "hfmodel", "argv": [...], "seed": 123456,
      "input": "/abs/path/card.txt", "created": "2026-09-27T12:00:00",
      "versions": {"pymodel": "...", "pyhf": "..."},
      "model_notes": [...],            # informational notes from building the model
      "flags": [...],                   # anything that makes the result suspect (see below)
      "result": {...}                   # command-specific payload
    }

``flags`` is the single place to look before trusting a number: unbracketed limits, failed or
invalid fits, parameters at bounds, excluded toys and unsupported approximations all end up
there.  An empty list means no problem was detected.
"""

import datetime
import json
import math
import sys

import numpy as np

RESULT_FORMAT = "pymodel-result"
RESULT_VERSION = 1
PYMODEL_VERSION = "2.0.0.dev0"


def _clean(obj):
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if math.isfinite(v) else (None if math.isnan(v) else ("inf" if v > 0 else "-inf"))
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def write_result(path, *, command, backend, versions, seed, input_path, model_notes, flags, result):
    doc = {
        "format": RESULT_FORMAT, "version": RESULT_VERSION,
        "command": command, "backend": backend, "argv": sys.argv, "seed": seed,
        "input": input_path, "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "versions": dict(versions, pymodel=PYMODEL_VERSION),
        "model_notes": list(model_notes), "flags": list(dict.fromkeys(flags)), "result": result,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_clean(doc), handle, indent=1)
    return path
