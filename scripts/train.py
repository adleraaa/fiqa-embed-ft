"""Fine-tune bge-small-en-v1.5 on FiQA train with MultipleNegativesRankingLoss.

Arms:
  inbatch            (query, positive) pairs; the other positives in the batch are negatives
  <negatives name>   (query, positive, hard negative) triplets from
                     artifacts/negatives_<name>.jsonl (see scripts/mine.py)

Model selection: dev nDCG@10 after every epoch, keep the best, stop after one epoch
without improvement. The test split is scored exactly once, with the selected weights.

Example:
  python scripts/train.py --arm inbatch --seed 1
  python scripts/train.py --arm dense_margin95 --seed 1
  python scripts/train.py --arm inbatch --seed 1 --lora
  python scripts/train.py --arm inbatch --seed 1 --batch-size 128 --gradcache-mini-batch 32
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import torch
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.base.sampler import BatchSamplers
from sentence_transformers.sentence_transformer.losses import (
    CachedMultipleNegativesRankingLoss,
    MultipleNegativesRankingLoss,
)

from fiqa_ft.data import QUERY_INSTRUCTION, load_fiqa, train_pairs
from fiqa_ft.evaluation import evaluate_dense, write_json
from fiqa_ft.training import DevRetrievalEvaluator, EarlyStopOnDev

BASE_MODEL = "BAAI/bge-small-en-v1.5"


def load_triplets(name: str) -> list[tuple[str, str, str]]:
    path = Path(f"artifacts/negatives_{name}.jsonl")
    if not path.exists():
        raise SystemExit(f"{path} not found; run scripts/mine.py --name {name} first")
    with path.open(encoding="utf-8") as f:
        return [tuple(json.loads(line)) for line in f]


def build_dataset(fiqa, arm: str) -> Dataset:
    if arm == "inbatch":
        rows = train_pairs(fiqa.qrels["train"])
        return Dataset.from_dict(
            {
                "anchor": [fiqa.queries[q] for q, _ in rows],
                "positive": [fiqa.corpus[d] for _, d in rows],
            }
        )
    rows = load_triplets(arm)
    return Dataset.from_dict(
        {
            "anchor": [fiqa.queries[q] for q, _, _ in rows],
            "positive": [fiqa.corpus[p] for _, p, _ in rows],
            "negative": [fiqa.corpus[n] for _, _, n in rows],
        }
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True, help="'inbatch' or a mined negatives name")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=None, help="default 2e-5 full, 2e-4 LoRA")
    p.add_argument("--max-seq-length", type=int, default=256)
    p.add_argument("--lora", action="store_true")
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument(
        "--gradcache-mini-batch",
        type=int,
        default=None,
        help="use GradCache (CachedMNRL) so --batch-size can exceed what fits in memory",
    )
    p.add_argument(
        "--dev-only",
        action="store_true",
        help="hyperparameter search: report dev scores only, never touch test",
    )
    args = p.parse_args()
    lr = args.lr or (2e-4 if args.lora else 2e-5)
    method = ("lora" if args.lora else "full") + (
        f"_bs{args.batch_size}" if args.batch_size != 32 else ""
    )
    run_name = f"{args.arm}{'' if method == 'full' else '_' + method}_seed{args.seed}"

    fiqa = load_fiqa()
    train_ds = build_dataset(fiqa, args.arm)

    model = SentenceTransformer(BASE_MODEL, device="cuda")
    # Training truncates at 256 tokens (~80% of FiQA answers fit) so a batch of 32
    # triplets stays at ~4 GB; batch 64 overflowed the 8 GB card. Evaluation always
    # uses the model's full 512.
    eval_max_len = model.max_seq_length
    model.max_seq_length = args.max_seq_length
    if args.lora:
        from peft import LoraConfig, TaskType

        model.add_adapter(
            LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=args.lora_rank,
                lora_alpha=2 * args.lora_rank,
                lora_dropout=0.1,
                target_modules=["query", "key", "value", "dense"],
            )
        )
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    evaluator = DevRetrievalEvaluator(fiqa.queries, fiqa.corpus, fiqa.qrels["dev"])
    out_dir = Path("checkpoints") / run_name
    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=lr,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        bf16=True,
        # Same query or document twice in a batch would be a false in-batch negative;
        # FiQA queries have ~2.6 positives each, so this matters.
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        prompts={"anchor": QUERY_INSTRUCTION},
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
    )
    if args.gradcache_mini_batch:
        # GradCache computes the same loss as one big batch, but back-propagates in
        # mini-batches, so memory is set by the mini-batch size.
        loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=args.gradcache_mini_batch)
    else:
        loss = MultipleNegativesRankingLoss(model)
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        loss=loss,
        evaluator=evaluator,
        callbacks=[EarlyStopOnDev(evaluator, patience=1)],
    )

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    trainer.train()
    wall = time.perf_counter() - t0
    peak_vram_gb = torch.cuda.max_memory_allocated() / 2**30

    shutil.rmtree(out_dir, ignore_errors=True)
    record = {
        "name": run_name,
        "arm": args.arm,
        "method": method,
        "seed": args.seed,
        "config": {
            "base_model": BASE_MODEL,
            "epochs_max": args.epochs,
            "batch_size": args.batch_size,
            "gradcache_mini_batch": args.gradcache_mini_batch,
            "lr": lr,
            "train_max_seq_length": args.max_seq_length,
            "lora_rank": args.lora_rank if args.lora else None,
            "train_examples": len(train_ds),
            "trainable_params": n_trainable,
            "gpu": torch.cuda.get_device_name(0),
        },
        "best_dev_epoch": evaluator.best_epoch,
        "best_dev_ndcg@10": evaluator.best_score,
        "dev_history": evaluator.history,
        "train_seconds": wall - evaluator.seconds,  # excludes per-epoch dev evaluation
        "dev_eval_seconds": evaluator.seconds,
        "global_steps": trainer.state.global_step,
        "peak_vram_gb": peak_vram_gb,
    }
    if args.dev_only:
        write_json(f"results/sweep/{run_name}_lr{lr:g}.json", record)
        print(run_name, f"lr={lr:g}", "best dev nDCG@10", round(evaluator.best_score, 4))
        return

    evaluator.restore_best(model)
    model.max_seq_length = eval_max_len
    test = evaluate_dense(model, fiqa.queries, fiqa.corpus, fiqa.qrels["test"])
    record.update(split="test", mean=test["mean"], per_query=test["per_query"])
    write_json(f"results/runs/{run_name}.json", record)
    print(
        run_name,
        {k: round(v, 4) for k, v in test["mean"].items()},
        f"train {record['train_seconds']:.0f}s, peak {peak_vram_gb:.2f} GB",
    )


if __name__ == "__main__":
    main()
