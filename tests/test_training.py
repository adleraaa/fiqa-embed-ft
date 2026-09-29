"""Model selection logic, exercised with a tiny fake encoder (no downloads, CPU only)."""

import pytest
import torch

pytest.importorskip("sentence_transformers")

from transformers import TrainerControl, TrainerState, TrainingArguments  # noqa: E402

from fiqa_ft.training import DevRetrievalEvaluator, EarlyStopOnDev  # noqa: E402

CORPUS = {"d1": "tax", "d2": "fund", "d3": "loan"}
QUERIES = {"q1": "tax", "q2": "fund"}
QRELS = {"q1": {"d1": 1}, "q2": {"d2": 1}}
VOCAB = ["tax", "fund", "loan"]


class FakeEncoder(torch.nn.Module):
    """One-hot text embeddings rotated by a learnable permutation weight.

    With `good=True` queries match their own document; otherwise every text maps to
    the same vector, so ranking is arbitrary and nDCG drops.
    """

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.eye(3))

    def encode(self, texts, prompt=None, **_):
        onehot = torch.stack([torch.eye(3)[VOCAB.index(t)] for t in texts])
        return torch.nn.functional.normalize(onehot @ self.w + 1e-6, dim=1)


def make_bad(model):
    with torch.no_grad():
        model.w.copy_(torch.ones(3, 3))


def test_evaluator_keeps_and_restores_best_weights():
    model = FakeEncoder()
    ev = DevRetrievalEvaluator(QUERIES, CORPUS, QRELS)
    metrics = ev(model, epoch=1)
    assert metrics["dev_ndcg@10"] == pytest.approx(1.0)
    assert ev.primary_metric == "dev_ndcg@10"

    make_bad(model)
    ev(model, epoch=2)
    assert ev.evals_since_best == 1 and ev.best_epoch == 1

    ev.restore_best(model)
    assert torch.equal(model.w, torch.eye(3))
    assert [h["epoch"] for h in ev.history] == [1, 2]


def test_early_stop_after_patience_evaluations_without_gain():
    model = FakeEncoder()
    ev = DevRetrievalEvaluator(QUERIES, CORPUS, QRELS)
    cb = EarlyStopOnDev(ev, patience=1)
    args, state = TrainingArguments(output_dir="unused", report_to="none"), TrainerState()

    ev(model, epoch=1)
    assert not cb.on_evaluate(args, state, TrainerControl()).should_training_stop
    make_bad(model)
    ev(model, epoch=2)
    assert cb.on_evaluate(args, state, TrainerControl()).should_training_stop
