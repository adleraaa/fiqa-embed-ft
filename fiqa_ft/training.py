"""Training-time helpers: training data, dev-set model selection and early stopping.

We keep the best weights in CPU memory instead of relying on the Trainer's
checkpoint-based load_best_model_at_end. bge-small is ~130 MB, so a copy is cheap,
nothing touches the disk, and the same code works for full fine-tuning and LoRA.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from datasets import Dataset
from sentence_transformers.sentence_transformer.evaluation import BaseEvaluator
from transformers import TrainerCallback

from fiqa_ft.data import FiQA, train_pairs
from fiqa_ft.evaluation import evaluate_dense


def negatives_path(name: str, seed: int, root: str | Path = "artifacts") -> Path:
    return Path(root) / f"negatives_{name}_seed{seed}.jsonl"


def load_triplets(path: Path) -> list[tuple[str, str, str]]:
    if not path.exists():
        raise SystemExit(f"{path} not found; run scripts/mine.py first")
    with path.open(encoding="utf-8") as f:
        return [tuple(json.loads(line)) for line in f]


def build_dataset(fiqa: FiQA, arm: str, seed: int, root: str | Path = "artifacts") -> Dataset:
    """Training rows for one arm: (anchor, positive) pairs or (anchor, positive, negative).

    Arm "inbatch" uses every labelled train pair. Any other arm name refers to mined
    negatives, one file per seed, so the random choice among eligible candidates
    varies with the training seed too.
    """
    if arm == "inbatch":
        rows = train_pairs(fiqa.qrels["train"])
        return Dataset.from_dict(
            {
                "anchor": [fiqa.queries[q] for q, _ in rows],
                "positive": [fiqa.corpus[d] for _, d in rows],
            }
        )
    triplets = load_triplets(negatives_path(arm, seed, root))
    return Dataset.from_dict(
        {
            "anchor": [fiqa.queries[q] for q, _, _ in triplets],
            "positive": [fiqa.corpus[p] for _, p, _ in triplets],
            "negative": [fiqa.corpus[n] for _, _, n in triplets],
        }
    )


class DevRetrievalEvaluator(BaseEvaluator):
    """Full-corpus dev retrieval with best-epoch tracking.

    Call it once with epoch=0 before training to score the base model. That score
    counts for early stopping and tells us whether dev would prefer no fine-tuning
    at all (`base_won`), but only fine-tuned epochs (epoch > 0) are kept as weights:
    the base model's test scores are already in the zero-shot baseline file.
    """

    def __init__(
        self, queries, corpus, qrels, name: str = "dev", max_seq_length: int | None = None
    ):
        super().__init__()
        self.queries, self.corpus, self.qrels, self.name = queries, corpus, qrels, name
        # Training truncates documents to save memory; dev must be scored at the same
        # length as test, or model selection optimises a different task.
        self.max_seq_length = max_seq_length
        self.primary_metric = "ndcg@10"
        self.base_score: float | None = None
        self.best_score = float("-inf")  # best fine-tuned epoch
        self.best_state: dict[str, torch.Tensor] | None = None
        self.best_epoch: float | None = None
        self.best_overall = float("-inf")  # includes the base model
        self.evals_since_best = 0
        self.seconds = 0.0  # time spent here, subtracted from training wall-clock
        self.history: list[dict] = []

    def __call__(self, model, output_path=None, epoch: float = -1, steps: int = -1):
        t0 = time.perf_counter()
        train_len = model.max_seq_length
        if self.max_seq_length is not None:
            model.max_seq_length = self.max_seq_length
        try:
            # fp16 autocast makes dev evaluation ~3x faster. It applies to every dev
            # score, including the base model's epoch-0 score, so selection stays
            # like-for-like; results/eval_conditions.json has the fp16-vs-fp32 check.
            with torch.autocast("cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                scores = evaluate_dense(model, self.queries, self.corpus, self.qrels)["mean"]
        finally:
            model.max_seq_length = train_len

        score = scores["ndcg@10"]
        if score > self.best_overall:
            self.best_overall, self.evals_since_best = score, 0
        else:
            self.evals_since_best += 1
        if epoch == 0:
            self.base_score = score
        elif score > self.best_score:
            self.best_score, self.best_epoch = score, epoch
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        self.history.append({"epoch": epoch, "steps": steps, **scores})
        self.seconds += time.perf_counter() - t0
        return self.prefix_name_to_metrics(scores, self.name)

    @property
    def base_won(self) -> bool:
        """True if no fine-tuned epoch beat the untouched base model on dev."""
        return self.base_score is not None and self.base_score >= self.best_score

    def restore_best(self, model) -> None:
        """Load the best fine-tuned weights (not the base model; see class docstring)."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


class EarlyStopOnDev(TrainerCallback):
    """Stop once dev nDCG@10 has not improved for `patience` consecutive evaluations."""

    def __init__(self, evaluator: DevRetrievalEvaluator, patience: int = 2):
        self.evaluator, self.patience = evaluator, patience

    def on_evaluate(self, args, state, control, **kwargs):
        if self.evaluator.evals_since_best >= self.patience:
            control.should_training_stop = True
        return control


def require_cuda(bf16: bool = False) -> None:
    """Fail early with a readable message instead of a CUDA error deep inside torch."""
    if not torch.cuda.is_available():
        raise SystemExit("This script needs a CUDA GPU (see README, Quickstart).")
    if bf16 and not torch.cuda.is_bf16_supported():
        raise SystemExit("Training uses bf16 mixed precision, which this GPU does not support.")


def library_versions() -> dict[str, str]:
    import sentence_transformers
    import transformers

    return {
        "torch": torch.__version__,
        "cuda": str(torch.version.cuda),
        "sentence_transformers": sentence_transformers.__version__,
        "transformers": transformers.__version__,
    }
