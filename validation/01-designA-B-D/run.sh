#!/usr/bin/env bash
#
# Run the context-strategy comparison and print the metrics report.
#
# Sets up the venv on first use, then forwards every argument to the harness:
#
#   ./run.sh                          # all five configurations, full script
#   ./run.sh --smoke                  # cheapest run that still exercises all three
#   ./run.sh --configs baseline all   # just two
#   ./run.sh --turns 3 --verbose      # short run with the plugins' own logging
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$HERE/.venv"
PY="$VENV/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "creating venv at $VENV"
  uv venv --python 3.12 "$VENV"
  uv pip install --python "$PY" -r "$HERE/requirements.txt"
  "$PY" -m playwright install chromium
fi

# No credential is exported here and none is written to disk: the run resolves them through the
# standard AWS chain, or through the profile named in VALIDATION_AWS_PROFILE. Missing credentials are
# caught by the preflight check in src/run.py, which prints the fix.
cd "$HERE"
exec "$PY" -m src.run "$@"
