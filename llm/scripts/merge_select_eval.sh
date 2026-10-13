#!/bin/bash
# Coefficient-based merging (task_arithmetic | ties | tsvm):
#   1. merge every coefficient in the paper's grid
#   2. evaluate each on the validation partition
#   3. select the coefficient with the highest validation average
#   4. evaluate the selected model on the test partition
#
#   bash llm/scripts/merge_select_eval.sh llama-3.1-8b-instruct ties ri
# EXPERTS: vanilla (original MergeBench experts), ri (released RI experts),
#          or 5 comma-separated local paths.
set -euo pipefail
FAMILY=$1; METHOD=$2; EXPERTS=$3; TAG=${4:-$EXPERTS}
MERGED=merged/$FAMILY/${METHOD}_$TAG
VAL=results/$FAMILY/val/${METHOD}_$TAG
python -m llm.merging.static --family "$FAMILY" --experts "$EXPERTS" --method "$METHOD" --sweep --output_dir "$MERGED"
for M in "$MERGED"/*/; do
  C=$(basename "$M")
  python -m llm.eval.run --model "$M" --family "$FAMILY" --split val --output_dir "$VAL/$C"
done
python -m llm.eval.score select --sweep "$VAL" | tee "$VAL/selection.txt"
BEST=$(awk '/selected coefficient/{print $3}' "$VAL/selection.txt")
python -m llm.eval.run --model "$MERGED/$BEST" --family "$FAMILY" --split test \
    --output_dir "results/$FAMILY/test/${METHOD}_$TAG"
