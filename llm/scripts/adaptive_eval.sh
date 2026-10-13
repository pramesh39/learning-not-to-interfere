#!/bin/bash
# Adaptive merging (adamerging | adarank) followed by test evaluation.
# Needs several GPUs on one node (the model is spread with device_map="balanced").
#   bash llm/scripts/adaptive_eval.sh llama-3.1-8b-instruct adamerging ri
set -euo pipefail
FAMILY=$1; METHOD=$2; EXPERTS=$3; TAG=${4:-$EXPERTS}
OUT=merged/$FAMILY/${METHOD}_$TAG
python -m llm.merging.adaptive --family "$FAMILY" --experts "$EXPERTS" --method "$METHOD" --output_dir "$OUT"
python -m llm.eval.run --model "$OUT" --family "$FAMILY" --split test --output_dir "results/$FAMILY/test/${METHOD}_$TAG"
