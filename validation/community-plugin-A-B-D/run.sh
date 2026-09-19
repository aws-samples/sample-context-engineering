#!/usr/bin/env bash
#
# Run the community-plugin comparison and print the metrics report.
#
# Sets up the venv on first use, then forwards every argument to the harness:
#
#   ./run.sh                                      # all five configurations, full script
#   ./run.sh --smoke                              # cheapest run that still wires all three
#   ./run.sh --configs baseline all                # just two
#   ./run.sh --total-turns 60 --tag cm60           # the 60-turn comparison
#   ./run.sh --turns 3 --verbose                   # short run with the plugins' own logging
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$HERE/.venv"
PY="$VENV/bin/python"

# cd first: requirements.txt installs the three community packages by RELATIVE path, so pip has to
# resolve them from this directory rather than from wherever the caller happened to be.
cd "$HERE"

if [[ ! -x "$PY" ]]; then
  echo "creating venv at $VENV"
  uv venv --python 3.12 "$VENV"
  uv pip install --python "$PY" -r requirements.txt
  "$PY" -m playwright install chromium
fi

# No credential is exported here and none is written to disk: the run resolves them through the
# standard AWS chain, or through the profile named in VALIDATION_AWS_PROFILE. Missing credentials
# are caught by the preflight check in src/run.py, which prints the fix.
exec "$PY" -m src.run "$@"
