"""Model selection logic, exercised with a tiny fake encoder (no downloads, CPU only)."""

import json

import pytest
import torch

pytest.importorskip("sentence_transformers")

from transformers import TrainerControl, TrainerState, TrainingArguments  # noqa: E402

from fiqa_ft.data import FiQA  # noqa: E402
from fiqa_ft.training import (  # noqa: E402
    DevRetrievalEvaluator,
    EarlyStopOnDev,
    build_dataset,
    negatives_path,
)

CORPUS = {"d1": "tax", "d2": "fund", "d3": "loan"}
QUERIES = {"q1": "tax", "q2": "fund"}
QRELS = {"q1": {"d1": 1}, "q2": {"d2": 1}}
VOCAB = ["tax", "fund", "loan"]


class FakeEncoder(torch.nn.Module):
    """One-hot text embeddings multiplied by a learnable weight.

    With the identity weight every query matches its own document (nDCG = 1); after
    make_bad() every text maps to the same vector and nDCG drops. It also records the
    max_seq_length in force at each encode call, like SentenceTransformer truncation.
    """

    def __init__(self, max_seq_length: int = 256):
        super().__init__()
        self.w = torch.nn.Parameter(torch.eye(3))
        self.max_seq_length = max_seq_length
        self.seen_lengths: list[int] = []

    def encode(self, texts, prompt=None, **_):
        self.seen_lengths.append(self.max_seq_length)
        onehot = torch.stack([torch.eye(3)[VOCAB.index(t)] for t in texts])
        return torch.nn.functional.normalize(onehot @ self.w + 1e-6, dim=1)


def make_bad(model):
    with torch.no_grad():
        model.w.copy_(torch.ones(3, 3))


def make_good(model):
    with torch.no_grad():
        model.w.copy_(torch.eye(3))


def test_dev_evaluation_uses_eval_length_and_restores_training_length():
    model = FakeEncoder(max_seq_length=256)
    ev = DevRetrievalEvaluator(QUERIES, CORPUS, QRELS, max_seq_length=512)
    ev(model, epoch=1)
    assert model.seen_lengths and set(model.seen_lengths) == {512}
    assert model.max_seq_length == 256  # training continues with its own length


def test_evaluator_keeps_and_restores_best_finetuned_weights():
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


def test_base_model_scored_but_never_kept_as_fine_tuned_weights():
    model = FakeEncoder()
    ev = DevRetrievalEvaluator(QUERIES, CORPUS, QRELS)
    ev(model, epoch=0)  # the base model is the best there is here
    make_bad(model)
    ev(model, epoch=1)
    assert ev.base_score == pytest.approx(1.0)
    assert ev.best_epoch == 1 and ev.base_won
    assert ev.evals_since_best == 1  # the base score counts for early stopping
    ev.restore_best(model)
    assert torch.equal(model.w, torch.ones(3, 3))  # epoch 1 weights, not the base

    make_good(model)
    ev(model, epoch=2)
    assert ev.best_epoch == 2
    assert ev.base_won  # a tie goes to the base model: fine-tuning bought nothing


def test_early_stop_after_patience_evaluations_without_gain():
    model = FakeEncoder()
    ev = DevRetrievalEvaluator(QUERIES, CORPUS, QRELS)
    cb = EarlyStopOnDev(ev, patience=2)
    args, state = TrainingArguments(output_dir="unused", report_to="none"), TrainerState()

    def should_stop():
        return cb.on_evaluate(args, state, TrainerControl()).should_training_stop

    ev(model, epoch=0)
    make_bad(model)
    ev(model, epoch=1)
    assert not should_stop()  # one epoch without gain is tolerated
    ev(model, epoch=2)
    assert should_stop()


def test_build_dataset_pairs_and_per_seed_triplets(tmp_path):
    fiqa = FiQA(
        corpus={"d1": "tax text", "d2": "fund text", "d3": "loan text"},
        queries={"10": "tax?", "9": "fund?"},
        qrels={"train": {"10": {"d1": 1}, "9": {"d2": 1}}},
    )
    pairs = build_dataset(fiqa, "inbatch", seed=1, root=tmp_path)
    assert pairs.column_names == ["anchor", "positive"]
    assert pairs["anchor"] == ["fund?", "tax?"]  # numeric id order: 9 before 10
    assert pairs["positive"] == ["fund text", "tax text"]

    path = negatives_path("hard", seed=2, root=tmp_path)
    path.write_text(json.dumps(["10", "d1", "d3"]) + "\n", encoding="utf-8")
    triplets = build_dataset(fiqa, "hard", seed=2, root=tmp_path)
    assert triplets[0] == {"anchor": "tax?", "positive": "tax text", "negative": "loan text"}

    with pytest.raises(SystemExit, match="mine.py"):
        build_dataset(fiqa, "hard", seed=3, root=tmp_path)  # that seed was never mined
