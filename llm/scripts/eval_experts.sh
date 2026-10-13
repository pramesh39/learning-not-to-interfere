#!/bin/bash
# Evaluate each original expert on its own benchmark (the "Individual" row,
# also the reference for Norm. Avg.).
#   bash llm/scripts/eval_experts.sh llama-3.1-8b-instruct
set -euo pipefail
FAMILY=$1
for DOMAIN in instruction math multilingual coding safety; do
  MODEL=$(python -c "from llm.config import family_cfg; print(family_cfg('$FAMILY')['experts']['$DOMAIN'])")
  python -m llm.eval.run --model "$MODEL" --family "$FAMILY" --split test --benchmarks "$DOMAIN" \
      --output_dir "results/$FAMILY/test/experts/$DOMAIN"
done
