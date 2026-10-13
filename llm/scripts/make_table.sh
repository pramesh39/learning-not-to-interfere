#!/bin/bash
# Results table for one family, from the outputs written by the scripts above.
#   bash llm/scripts/make_table.sh llama-3.1-8b-instruct
set -euo pipefail
FAMILY=$1; R=results/$FAMILY/test
python -m llm.eval.score table --experts "$R/experts" --models \
  "TA=$R/task_arithmetic_vanilla" "TIES=$R/ties_vanilla" "TSV-M=$R/tsvm_vanilla" \
  "AdaMerging=$R/adamerging_vanilla" "AdaRank=$R/adarank_vanilla" \
  "TA + RI=$R/task_arithmetic_ri" "TIES + RI=$R/ties_ri" "TSV-M + RI=$R/tsvm_ri" \
  "AdaMerging + RI=$R/adamerging_ri" "AdaRank + RI=$R/adarank_ri"
