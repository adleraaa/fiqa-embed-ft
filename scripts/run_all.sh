#!/usr/bin/env bash
# Reproduce every number in results/. Needs a CUDA GPU (about 3 h on an RTX 5070 Laptop,
# plus about 1 h for scripts/sweep_lr.sh, which chose LR).
# Each step skips work whose result file already exists, so the script can be resumed.
set -euo pipefail

PY=${PY:-python}
LR=${LR:-5e-05} # best mean dev nDCG@10 in results/sweep/summary.md
run() { # run <result file> <command...>
  local out=$1; shift
  if [[ -f $out ]]; then echo "skip $out"; else echo "== $*"; "$@"; fi
}

run results/published_mteb.json $PY scripts/fetch_published.py
run results/eval_conditions.json $PY scripts/eval_conditions.py
run results/runs/baseline_bge-small.json $PY scripts/evaluate.py --name baseline_bge-small
run results/runs/baseline_bge-small_noinstr.json \
  $PY scripts/evaluate.py --name baseline_bge-small_noinstr --no-instruction
run results/runs/baseline_bge-base.json \
  $PY scripts/evaluate.py --name baseline_bge-base --model BAAI/bge-base-en-v1.5
run results/runs/dev_baseline_bge-small.json \
  $PY scripts/evaluate.py --name dev_baseline_bge-small --split dev
run results/runs/bm25.json $PY scripts/evaluate.py --bm25 --name bm25

# Mined negatives live in artifacts/ (gitignored), one file per training seed.
[[ -f artifacts/negatives_bm25_seed3.jsonl ]] || $PY scripts/mine.py --method bm25 --name bm25
[[ -f artifacts/negatives_dense_nofilter_seed3.jsonl ]] ||
  $PY scripts/mine.py --method dense --name dense_nofilter
[[ -f artifacts/negatives_dense_margin95_seed3.jsonl ]] ||
  $PY scripts/mine.py --method dense --margin 0.95 --name dense_margin95

for seed in 1 2 3; do
  for arm in inbatch bm25 dense_nofilter dense_margin95; do
    run "results/runs/${arm}_seed${seed}.json" \
      $PY scripts/train.py --arm "$arm" --seed "$seed" --lr "$LR"
  done
done

# LoRA ablation (untuned lr 2e-4, the usual LoRA default; one seed).
for arm in inbatch dense_margin95; do
  run "results/runs/${arm}_lora_seed1.json" $PY scripts/train.py --arm "$arm" --seed 1 --lora
done

# Batch-size ablation: 128 examples per step via GradCache in mini-batches of 32.
for arm in inbatch dense_margin95; do
  run "results/runs/${arm}_full_bs128_seed1.json" \
    $PY scripts/train.py --arm "$arm" --seed 1 --lr "$LR" --batch-size 128 --gradcache-mini-batch 32
done

$PY scripts/summarize.py
