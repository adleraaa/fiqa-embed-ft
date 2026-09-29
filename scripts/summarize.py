"""Aggregate results/runs/*.json into results/summary.{json,md} and results/ndcg10.png.

For a system trained with several seeds, per-query scores are first averaged over
seeds, then a query-level bootstrap gives the CI and a paired sign-flip test compares
it with the zero-shot baseline and with the in-batch-only arm.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # never open a window
import matplotlib.pyplot as plt  # noqa: E402

from fiqa_ft.evaluation import read_json, write_json  # noqa: E402
from fiqa_ft.metrics import DEFAULT_METRICS  # noqa: E402
from fiqa_ft.stats import (  # noqa: E402
    bootstrap_ci,
    paired_bootstrap_delta,
    paired_randomization_test,
    seed_average,
)

RUNS = Path("results/runs")
BASELINE = "baseline_bge-small"
LABELS = {
    "bm25": "BM25 (bm25s)",
    "baseline_bge-small_noinstr": "bge-small zero-shot, no query instruction",
    BASELINE: "bge-small zero-shot",
    "baseline_bge-base": "bge-base zero-shot (reference, 110M)",
    "inbatch/full": "(a) MNRL, in-batch negatives",
    "bm25/full": "(b) + BM25 hard negatives",
    "dense_nofilter/full": "(c0) + dense hard negatives, no filter",
    "dense_margin95/full": "(c) + dense hard negatives, 95% margin filter",
    "inbatch/lora": "(a) with LoRA r=16",
    "dense_margin95/lora": "(c) with LoRA r=16",
}


def load_systems() -> dict[str, list[dict]]:
    systems: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(RUNS.glob("*.json")):
        run = read_json(path)
        if run.get("split", "test") != "test":
            continue
        key = f"{run['arm']}/{run['method']}" if "arm" in run else run["name"]
        systems[key].append(run)
    return systems


def summarize_system(key: str, runs: list[dict], base_pq: dict, ref_pq: dict | None) -> dict:
    ndcg_pq = seed_average([r["per_query"]["ndcg@10"] for r in runs])
    qids = sorted(ndcg_pq)
    scores = [ndcg_pq[q] for q in qids]
    mean, lo, hi = bootstrap_ci(scores)
    row: dict = {
        "key": key,
        "label": LABELS.get(key, key),
        "seeds": [r.get("seed") for r in runs],
        "ndcg@10_ci": [lo, hi],
    }
    for m in DEFAULT_METRICS:
        vals = [r["mean"][m] for r in runs]
        row[m] = statistics.fmean(vals)
        row[f"{m}_sd"] = statistics.stdev(vals) if len(vals) > 1 else None
    for name, other in [("baseline", base_pq), ("inbatch", ref_pq)]:
        if other is None or key in (BASELINE, "inbatch/full"):
            continue
        a = [other[q] for q in qids]
        delta, dlo, dhi = paired_bootstrap_delta(a, scores)
        row[f"delta_vs_{name}"] = {
            "delta": delta,
            "ci": [dlo, dhi],
            "p": paired_randomization_test(a, scores),
        }
    if "train_seconds" in runs[0]:
        row["train_seconds"] = statistics.fmean(r["train_seconds"] for r in runs)
        row["peak_vram_gb"] = max(r["peak_vram_gb"] for r in runs)
        row["trainable_params"] = runs[0]["config"]["trainable_params"]
        row["best_dev_epochs"] = [r["best_dev_epoch"] for r in runs]
    return row


def fmt(x: float | None, nd: int = 3) -> str:
    return "" if x is None else f"{x:.{nd}f}"


def to_markdown(rows: list[dict]) -> str:
    head = (
        "| System | Seeds | nDCG@10 | 95% CI | Δ vs zero-shot [95% CI], p | Δ vs (a), p "
        "| R@10 | R@100 | MRR@10 | Train s | Peak VRAM GB |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    lines = []
    for r in rows:
        ndcg = fmt(r["ndcg@10"]) + (f" ± {fmt(r['ndcg@10_sd'])}" if r["ndcg@10_sd"] else "")
        ci = f"[{fmt(r['ndcg@10_ci'][0])}, {fmt(r['ndcg@10_ci'][1])}]"
        d = r.get("delta_vs_baseline")
        dtxt = (
            f"{d['delta']:+.3f} [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}], p={d['p']:.4f}" if d else ""
        )
        di = r.get("delta_vs_inbatch")
        ditxt = f"{di['delta']:+.3f}, p={di['p']:.4f}" if di else ""
        seeds = str(len(r["seeds"])) if r["seeds"][0] is not None else "-"
        lines.append(
            f"| {r['label']} | {seeds} | {ndcg} | {ci} | {dtxt} | {ditxt} | {fmt(r['recall@10'])} "
            f"| {fmt(r['recall@100'])} | {fmt(r['mrr@10'])} | {fmt(r.get('train_seconds'), 0)} "
            f"| {fmt(r.get('peak_vram_gb'), 2)} |"
        )
    return head + "\n".join(lines) + "\n"


def plot(rows: list[dict], path: str) -> None:
    rows = [r for r in rows if r["key"] != "bm25"]
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(rows) + 1.2))
    y = range(len(rows))
    means = [r["ndcg@10"] for r in rows]
    err = [
        [m - r["ndcg@10_ci"][0] for m, r in zip(means, rows, strict=True)],
        [r["ndcg@10_ci"][1] - m for m, r in zip(means, rows, strict=True)],
    ]
    ax.errorbar(means, list(y), xerr=err, fmt="o", color="#2b5d8a", capsize=3)
    ax.set_yticks(list(y), [r["label"] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("FiQA-2018 test nDCG@10 (95% bootstrap CI over 648 queries)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main() -> None:
    systems = load_systems()
    base_pq = systems[BASELINE][0]["per_query"]["ndcg@10"]
    ref_runs = systems.get("inbatch/full")
    ref_pq = seed_average([r["per_query"]["ndcg@10"] for r in ref_runs]) if ref_runs else None

    order = [k for k in LABELS if k in systems] + sorted(set(systems) - set(LABELS))
    rows = [summarize_system(k, systems[k], base_pq, ref_pq) for k in order]
    write_json("results/summary.json", rows)
    Path("results/summary.md").write_text(to_markdown(rows), encoding="utf-8")
    plot(rows, "results/ndcg10.png")
    print(to_markdown(rows))


if __name__ == "__main__":
    main()
