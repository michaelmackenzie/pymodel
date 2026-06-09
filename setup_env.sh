#! /bin/bash

source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh
pyenv rootana 2.5.0
export PYMODEL_REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
export PATH="$PATH:$PYMODEL_REPO/bin"
export PYTHONPATH="$PYTHONPATH:$PYMODEL_REPO"
