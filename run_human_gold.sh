#!/usr/bin/env bash
# Human gold pass — a real hand-annotated set, so the LLM-vs-LLM numbers can be
# checked against something that isn't an LLM.
#
# 100 comments, stratified exactly like the rest of the paper (15/42.5/42.5
# random/topic/emotive, topic slice spread over all 14 cause categories), drawn
# from the 300 that claude-sonnet-5 and gemini-3.5-flash already labeled — so
# every human label has two model labels to compare against.
#
# Single annotator, writing to its own sqlite so the existing 300-comment store
# is untouched.
#
# NOTE: the RELEASED benchmark (data/labels/gold/human_gold_100.offsets.jsonl)
# was produced with --assist — hand-curated from pooled machine candidates, not
# blind. Every kept pair carries `_accepted_from`, so this is verifiable from the
# file itself. Running without --assist gives the blind pass instead, which is a
# different (stricter) benchmark, not a reproduction of the released one.
#
#   ./run_human_gold.sh            # start API + UI (localhost)
#   ./run_human_gold.sh --assist   # show model labels with click-to-accept
#   ./run_human_gold.sh --api      # API only (UI already running elsewhere)
set -uo pipefail
cd "$(dirname "$0")"

# --assist: show every model's labels with click-to-accept. Faster, but the
# labels are then model-anchored: each accepted pair is stamped `_accepted_from`
# with its source, and human_gold_eval reports the assisted share and downgrades
# the shared-bias rate to a floor (it counts only readings the annotator rejected
# outright, so it understates by construction). Without the flag the pass is
# blind — the server withholds model and consensus predictions entirely.
ASSIST=0
for a in "$@"; do [[ "$a" == "--assist" ]] && ASSIST=1; done
if [[ "$ASSIST" == "1" ]]; then
  set -- "${@/--assist/}"
fi

# Serve on localhost. To reach the UI from another machine, put it behind your
# own tunnel/reverse proxy and set VITE_API_BASE to where the API is reachable.
HOST_ARG="${GOLD_HOST_ARG:-}"

export GOLD_SAMPLE="$PWD/data/samples/human_gold_100_seed7.jsonl"
export GOLD_DB="$PWD/data/labels/gold/human_gold_labels.sqlite"
export GOLD_ANNOTATORS="${GOLD_ANNOTATORS:-A-human}"
export GOLD_BLIND=$(( 1 - ASSIST ))

if [[ ! -f "$GOLD_SAMPLE" ]]; then
  echo "[human-gold] missing $GOLD_SAMPLE — draw it with:"
  echo "  python -m pipeline.sample --gold --n 100 --seed 7 \\"
  echo "    --src data/samples/gold_300_seed7.jsonl --out $GOLD_SAMPLE"
  exit 1
fi

echo "[human-gold] sample    : $GOLD_SAMPLE ($(wc -l < "$GOLD_SAMPLE") comments)"
echo "[human-gold] store     : $GOLD_DB"
if [[ "$GOLD_BLIND" == "1" ]]; then
  echo "[human-gold] annotator : $GOLD_ANNOTATORS (blind — model panels hidden)"
else
  echo "[human-gold] annotator : $GOLD_ANNOTATORS (ASSISTED — all model labels shown)"
fi
[[ -n "${VITE_API_BASE:-}" ]] && echo "[human-gold] api base   : $VITE_API_BASE"

node gold-tool/server/index.js &
API_PID=$!
trap 'kill $API_PID 2>/dev/null' EXIT

if [[ "${1:-}" == "--api" ]]; then
  wait $API_PID
else
  npm --prefix gold-tool/client run dev -- $HOST_ARG
fi
