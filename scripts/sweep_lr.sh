#!/usr/bin/env bash
# Learning-rate search on the dev split only (seed 1), with the same epoch cap and
# early stopping as the final runs. The lr with the best mean dev nDCG@10 over the two
# swept arms is then used for every full fine-tune in scripts/run_all.sh (LR=...).
# Results: results/sweep/*.json, table in results/sweep/summary.md.
set -euo pipefail

PY=${PY:-python}
[[ -f artifacts/negatives_dense_margin95_seed1.jsonl ]] ||
  $PY scripts/mine.py --method dense --margin 0.95 --name dense_margin95
for arm in inbatch dense_margin95; do
  for lr in 5e-06 1e-05 2e-05 5e-05 0.0001; do  # spelled like Python formats them (:g)
    out="results/sweep/${arm}_seed1_lr${lr}.json"
    if [[ -f $out ]]; then echo "skip $out"; continue; fi
    $PY scripts/train.py --arm "$arm" --seed 1 --lr "$lr" --dev-only
  done
done
