#!/bin/bash
# Produce tests/fixtures/*.json from Combine runs on copies of the example cards and of the
# mumep_ana reference cards.  Combine is run ONLY inside WORKDIR (a scratch directory).
#
#   tests/make_combine_fixtures.sh WORKDIR [--only name1,name2] [--jobs N]
#
# Needs combine + text2workspace.py on PATH (e.g. source the Combine environment:
# rootana 2.5.0 + env_standalone_mu2e.sh of HiggsAnalysis/CombinedLimit), or set
# COMBINE_SETUP to a script that sets it up.  Takes ~10-20 minutes with 16 jobs.
set -euo pipefail
if [[ $# -lt 1 ]]; then
    sed -n '2,10p' "$0"; exit 1
fi
if ! command -v combine &> /dev/null; then
    if [[ -n "${COMBINE_SETUP:-}" ]]; then
        # shellcheck disable=SC1090
        source "$COMBINE_SETUP"
    else
        echo "combine is not on PATH; source the Combine environment or set COMBINE_SETUP" >&2; exit 1
    fi
fi
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
exec python3 "$HERE/combine_fixtures.py" "$@"
