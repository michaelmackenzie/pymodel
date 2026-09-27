#!/usr/bin/env bash
# Install the pure-python packages pymodel needs on top of rootana (Mu2e "pyenv rootana 2.5.0").
#
# rootana 2.5.0 has numpy, scipy, iminuit, zfit, click, tqdm, jsonschema, jsonpointer and pyyaml,
# but not pyhf (hfmodel backend) or jsonpatch (a pyhf dependency).  They are installed with
# --no-deps into $PYMODEL_REPO/.pydeps, which setup_env.sh appends to PYTHONPATH, so rootana's
# own numpy/scipy are never shadowed.
#
# Usage:  source setup_env.sh && scripts/install_python_deps.sh
set -euo pipefail

if [[ -z "${PYMODEL_REPO:-}" ]]; then
    echo "PYMODEL_REPO is not set: source setup_env.sh first" >&2
    exit 1
fi

PYHF_VERSION="${PYHF_VERSION:-0.7.6}"
TARGET="$PYMODEL_REPO/.pydeps"
PACKAGES=("pyhf==${PYHF_VERSION}" "jsonpatch>=1.15")

mkdir -p "$TARGET"
# 'python3 -m pip' avoids the Mu2e 'pip' shell function
python3 -m pip install --upgrade --no-deps --target "$TARGET" "${PACKAGES[@]}"

# Check that every pyhf requirement is importable with rootana + .pydeps and that numpy/scipy
# come from rootana, not from .pydeps.
PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$TARGET" python3 - <<'EOF'
import importlib, sys
missing = []
for mod in ("pyhf", "jsonpatch", "jsonpointer", "jsonschema", "click", "tqdm", "yaml", "scipy", "numpy"):
    try:
        importlib.import_module(mod)
    except ImportError as err:
        missing.append(f"{mod} ({err})")
if missing:
    sys.exit("missing after install: " + ", ".join(missing))
import numpy, scipy, pyhf
for mod in (numpy, scipy):
    if ".pydeps" in mod.__file__:
        sys.exit(f"{mod.__name__} is shadowed by .pydeps: {mod.__file__}")
print(f"pyhf {pyhf.__version__} OK (numpy {numpy.__version__}, scipy {scipy.__version__} from rootana)")
EOF
