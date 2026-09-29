"""Training-time helpers: dev-set model selection and early stopping.

We keep the best weights in CPU memory instead of relying on the Trainer's
checkpoint-based load_best_model_at_end. bge-small is ~130 MB, so a copy is cheap,
nothing touches the disk, and the same code works for full fine-tuning and LoRA.
"""

from __future__ import annotations

import time

import torch
from sentence_transformers.sentence_transformer.evaluation import BaseEvaluator
from transformers import TrainerCallback

from fiqa_ft.evaluation import evaluate_dense


class DevRetrievalEvaluator(BaseEvaluator):
    """Full-corpus dev retrieval; remembers the best weights by dev nDCG@10."""

    def __init__(self, queries, corpus, qrels, name: str = "dev"):
        super().__init__()
        self.queries, self.corpus, self.qrels, self.name = queries, corpus, qrels, name
        self.primary_metric = "ndcg@10"
        self.best_score = float("-inf")
        self.best_state: dict[str, torch.Tensor] | None = None
        self.best_epoch: float | None = None
        self.evals_since_best = 0
        self.seconds = 0.0  # time spent here, subtracted from training wall-clock
        self.history: list[dict] = []

    def __call__(self, model, output_path=None, epoch: float = -1, steps: int = -1):
        t0 = time.perf_counter()
        # fp16 autocast halves dev-eval time; on the zero-shot model it moves test
        # nDCG@10 by 0.0003, far below the differences we select on.
        with torch.autocast("cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
            result = evaluate_dense(model, self.queries, self.corpus, self.qrels)
        scores = result["mean"]
        if scores["ndcg@10"] > self.best_score:
            self.best_score = scores["ndcg@10"]
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            self.best_epoch = epoch
            self.evals_since_best = 0
        else:
            self.evals_since_best += 1
        self.history.append({"epoch": epoch, "steps": steps, **scores})
        self.seconds += time.perf_counter() - t0
        return self.prefix_name_to_metrics(scores, self.name)

    def restore_best(self, model) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


class EarlyStopOnDev(TrainerCallback):
    """Stop once dev nDCG@10 has not improved for `patience` consecutive evaluations."""

    def __init__(self, evaluator: DevRetrievalEvaluator, patience: int = 1):
        self.evaluator, self.patience = evaluator, patience

    def on_evaluate(self, args, state, control, **kwargs):
        if self.evaluator.evals_since_best >= self.patience:
            control.should_training_stop = True
        return control
