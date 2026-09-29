#!/usr/bin/env bash
# Reproduce every number in results/. Needs a CUDA GPU (about 2.5 h on an RTX 5070 Laptop).
# Each step skips work whose result file already exists, so the script can be resumed.
set -euo pipefail

PY=${PY:-python}
run() { # run <result file> <command...>
  local out=$1; shift
  if [[ -f $out ]]; then echo "skip $out"; else echo "== $*"; "$@"; fi
}

run results/published_mteb.json $PY scripts/fetch_published.py
run results/runs/baseline_bge-small.json $PY scripts/evaluate.py --name baseline_bge-small
run results/runs/baseline_bge-small_noinstr.json \
  $PY scripts/evaluate.py --name baseline_bge-small_noinstr --no-instruction
run results/runs/baseline_bge-base.json \
  $PY scripts/evaluate.py --name baseline_bge-base --model BAAI/bge-base-en-v1.5
run results/runs/dev_baseline_bge-small.json \
  $PY scripts/evaluate.py --name dev_baseline_bge-small --split dev
run results/runs/bm25.json $PY scripts/evaluate.py --bm25 --name bm25

# Mined negatives live in artifacts/ (gitignored); always regenerate if missing.
[[ -f artifacts/negatives_bm25.jsonl ]] || $PY scripts/mine.py --method bm25 --name bm25
[[ -f artifacts/negatives_dense_nofilter.jsonl ]] ||
  $PY scripts/mine.py --method dense --name dense_nofilter
[[ -f artifacts/negatives_dense_margin95.jsonl ]] ||
  $PY scripts/mine.py --method dense --margin 0.95 --name dense_margin95

for seed in 1 2 3; do
  for arm in inbatch bm25 dense_nofilter dense_margin95; do
    run "results/runs/${arm}_seed${seed}.json" $PY scripts/train.py --arm "$arm" --seed "$seed"
  done
done

for arm in inbatch dense_margin95; do
  run "results/runs/${arm}_lora_seed1.json" $PY scripts/train.py --arm "$arm" --seed 1 --lora
done

# Batch-size ablation: 128 examples per step via GradCache (same peak memory as 32).
for arm in inbatch dense_margin95; do
  run "results/runs/${arm}_full_bs128_seed1.json" \
    $PY scripts/train.py --arm "$arm" --seed 1 --batch-size 128 --gradcache-mini-batch 32
done

$PY scripts/summarize.py
