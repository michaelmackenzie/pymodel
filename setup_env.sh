#! /bin/bash

source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh
pyenv rootana 2.5.0
export PYMODEL_REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
export PATH="$PATH:$PYMODEL_REPO/bin"
export PYTHONPATH="$PYMODEL_REPO/python:${PYTHONPATH:+$PYTHONPATH:}$PYMODEL_REPO"
# pure-python extras (pyhf, jsonpatch) installed by scripts/install_python_deps.sh; appended so
# that rootana's numpy/scipy always win
if [[ -d "$PYMODEL_REPO/.pydeps" ]]; then
    export PYTHONPATH="$PYTHONPATH:$PYMODEL_REPO/.pydeps"
fi
if ! python3 -c "import pyhf" &> /dev/null; then
    echo "setup_env.sh: warning: pyhf is not importable (hfmodel backend unavailable); run scripts/install_python_deps.sh" >&2
fi
