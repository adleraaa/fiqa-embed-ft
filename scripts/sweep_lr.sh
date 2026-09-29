#!/usr/bin/env bash
# Learning-rate search on the dev split only (seed 1). The chosen lr is then used for
# every arm in scripts/run_all.sh. Results: results/sweep/*.json
set -euo pipefail

PY=${PY:-python}
for arm in inbatch dense_margin95; do
  [[ $arm == inbatch ]] || [[ -f artifacts/negatives_$arm.jsonl ]] ||
    $PY scripts/mine.py --method dense --margin 0.95 --name "$arm"
  for lr in 5e-06 1e-05 2e-05; do
    out="results/sweep/${arm}_seed1_lr${lr}.json"
    if [[ -f $out ]]; then echo "skip $out"; continue; fi
    $PY scripts/train.py --arm "$arm" --seed 1 --lr "$lr" --dev-only
  done
done
