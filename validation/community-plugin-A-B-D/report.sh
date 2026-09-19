#!/usr/bin/env bash
#
# Build the full report for a run that already exists.
#
#   ./report.sh cm60              # everything from results/run-cm60.json
#   ./report.sh cm60 --verify     # also check the tokens against Bedrock's logs
#
# Reads results/run-<tag>.json and writes, beside it:
#
#   comparison-<tag>.md   the comparison table, pasteable into a PR
#   curve-<tag>.csv       the per-turn series
#   curve-<tag>.html      the deliverable: result table plus charts, self-contained
#
# It never calls Bedrock unless --verify is passed, so re-rendering is free. To change a price and
# re-render, edit PRICING in src/config.py and run this again -- do not re-run the measurement.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$HERE/.venv/bin/python"
RESULTS="$HERE/results"

if [[ $# -lt 1 ]]; then
  echo "usage: $(basename "$0") <tag> [--verify]" >&2
  echo "" >&2
  echo "available runs:" >&2
  ls -1t "$RESULTS"/run-*.json 2>/dev/null | sed 's|.*/run-|  |; s|\.json$||' >&2 || echo "  none" >&2
  exit 2
fi

TAG="$1"
shift
VERIFY=0
[[ "${1:-}" == "--verify" ]] && VERIFY=1

RUN_JSON="$RESULTS/run-$TAG.json"
if [[ ! -f "$RUN_JSON" ]]; then
  echo "no such run: $RUN_JSON" >&2
  echo "run it first with: ./run.sh --total-turns 60 --tag $TAG" >&2
  exit 1
fi

if [[ ! -x "$PY" ]]; then
  echo "venv missing at $HERE/.venv -- run ./run.sh once to create it" >&2
  exit 1
fi

cd "$HERE"

echo "==> comparison table and per-turn series"
"$PY" -m src.compare "$RUN_JSON" --curve

CURVE_CSV="$RESULTS/curve-$TAG.csv"
if [[ ! -f "$CURVE_CSV" ]]; then
  echo "compare did not write $CURVE_CSV -- cannot chart" >&2
  exit 1
fi

echo "==> HTML report"
# --run is implicit: chart.py derives run-<tag>.json from the CSV name. Passed explicitly so a
# missing file fails here rather than silently rendering a page with no result table.
"$PY" -m src.chart "$CURVE_CSV" --run "$RUN_JSON"

if [[ "$VERIFY" == "1" ]]; then
  echo "==> verifying tokens against Bedrock invocation logs"
  # Needs credentials and only reads CloudWatch. Non-fatal: a log-group lag should not invalidate a
  # report that is already written.
  "$PY" -m src.verify_logs "$RUN_JSON" || echo "note: verification did not complete"
fi

echo
echo "written:"
for f in "$RESULTS/report-$TAG.md" "$RESULTS/comparison-$TAG.md" "$CURVE_CSV" "$RESULTS/curve-$TAG.html"; do
  [[ -f "$f" ]] && printf "  %s\n" "$f"
done
