#!/bin/bash
# Adapt all five experts of a family with RI (each expert is independent and
# can run in parallel on separate GPUs).
#   bash llm/scripts/train_ri.sh llama-3.1-8b-instruct runs/ri
set -euo pipefail
FAMILY=$1; OUT=${2:-runs/ri}
for DOMAIN in instruction math multilingual coding safety; do
  python -m llm.ri.train --family "$FAMILY" --domain "$DOMAIN" --output_dir "$OUT/$FAMILY"
done
