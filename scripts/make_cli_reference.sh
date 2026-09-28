#!/bin/bash
# Regenerate docs/cli-reference.md from the CLI --help output (run after changing options).
set -euo pipefail
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &> /dev/null && pwd)
OUT="$REPO/docs/cli-reference.md"
{
    echo "# CLI reference"
    echo
    echo 'Generated from `--help` for roomodel; the other backends differ only in their'
    echo 'backend-specific options. Regenerate with `scripts/make_cli_reference.sh`.'
    for c in build inspect nll fit scan impacts limit fc significance generate merge export; do
        echo
        echo "## \`$c\`"
        echo
        echo '```'
        COLUMNS=100 python3 "$REPO/python/pymodel" roomodel "$c" --help 2>&1 | sed '1{/^pymodel/d}'
        echo '```'
    done
} > "$OUT"
echo "Wrote $OUT"
