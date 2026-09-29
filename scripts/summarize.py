"""Aggregate results/runs/*.json into results/summary.{json,md} and results/deltas.png.

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
    "inbatch/full_bs128": "(a) with batch 128 (GradCache)",
    "dense_margin95/full_bs128": "(c) with batch 128 (GradCache)",
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


TESTED = ("ndcg@10", "recall@100")  # metrics that get paired CIs and p-values


def per_query(runs: list[dict], metric: str) -> dict[str, float]:
    return seed_average([r["per_query"][metric] for r in runs])


def paired(a_pq: dict[str, float], b_pq: dict[str, float]) -> dict:
    """Paired comparison of system b against system a on the same queries."""
    qids = sorted(b_pq)
    a, b = [a_pq[q] for q in qids], [b_pq[q] for q in qids]
    delta, lo, hi = paired_bootstrap_delta(a, b)
    return {"delta": delta, "ci": [lo, hi], "p": paired_randomization_test(a, b)}


def summarize_system(key: str, runs: list[dict], base: list[dict], ref: list[dict] | None) -> dict:
    trained = "train_seconds" in runs[0]
    _, lo, hi = bootstrap_ci(list(per_query(runs, "ndcg@10").values()))
    row: dict = {
        "key": key,
        "label": LABELS.get(key, key),
        "seeds": [r["seed"] for r in runs] if trained else [],
        "ndcg@10_ci": [lo, hi],
    }
    for m in DEFAULT_METRICS:
        vals = [r["mean"][m] for r in runs]
        row[m] = statistics.fmean(vals)
        row[f"{m}_sd"] = statistics.stdev(vals) if len(vals) > 1 else None
    for m in TESTED:
        if key != BASELINE:
            row[f"{m}_vs_zero_shot"] = paired(per_query(base, m), per_query(runs, m))
        if trained and ref is not None and key != "inbatch/full":
            row[f"{m}_vs_inbatch"] = paired(per_query(ref, m), per_query(runs, m))
    if trained:
        secs = [r["train_seconds"] for r in runs]
        row["train_seconds"] = statistics.fmean(secs)
        row["train_seconds_range"] = [min(secs), max(secs)]
        row["peak_vram_gb"] = max(r["peak_vram_gb"] for r in runs)
        row["trainable_params"] = runs[0]["config"]["trainable_params"]
        row["best_dev_epochs"] = [r["best_dev_epoch"] for r in runs]
    return row


def fmt(x: float | None, nd: int = 3) -> str:
    return "" if x is None else f"{x:.{nd}f}"


def fmt_delta(d: dict | None, with_ci: bool = True) -> str:
    if not d:
        return ""
    ci = f" [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}]" if with_ci else ""
    # 10k resamples cannot resolve p below ~1e-4, so do not print false precision
    p = "p<0.001" if d["p"] < 0.001 else f"p={d['p']:.3f}"
    return f"{d['delta']:+.3f}{ci}, {p}"


def to_markdown(rows: list[dict]) -> str:
    head = (
        "| System | Seeds | nDCG@10 | 95% CI | Δ nDCG@10 vs zero-shot [95% CI], p "
        "| Δ nDCG@10 vs (a), p | R@100 | Δ R@100 vs zero-shot [95% CI], p | R@10 | MRR@10 "
        "| Train s | Peak VRAM GB |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    lines = []
    for r in rows:
        ndcg = fmt(r["ndcg@10"], 4) + (f" ± {fmt(r['ndcg@10_sd'], 4)}" if r["ndcg@10_sd"] else "")
        ci = f"[{fmt(r['ndcg@10_ci'][0])}, {fmt(r['ndcg@10_ci'][1])}]"
        lines.append(
            f"| {r['label']} | {len(r['seeds']) or '-'} | {ndcg} | {ci} "
            f"| {fmt_delta(r.get('ndcg@10_vs_zero_shot'))} "
            f"| {fmt_delta(r.get('ndcg@10_vs_inbatch'), with_ci=False)} "
            f"| {fmt(r['recall@100'])} | {fmt_delta(r.get('recall@100_vs_zero_shot'))} "
            f"| {fmt(r['recall@10'])} | {fmt(r['mrr@10'])} "
            f"| {fmt(r.get('train_seconds'), 0)} | {fmt(r.get('peak_vram_gb'), 2)} |"
        )
    return head + "\n".join(lines) + "\n"


def plot(rows: list[dict], path: str) -> None:
    """Paired differences vs the zero-shot model, one panel per tested metric.

    Paired CIs are much narrower than the CIs of the absolute scores, because most
    of the query-to-query variance is shared by both systems.
    """
    rows = [r for r in rows if "ndcg@10_vs_zero_shot" in r and r["key"] != "bm25"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.42 * len(rows) + 1.4), sharey=True)
    y = list(range(len(rows)))
    for ax, metric in zip(axes, TESTED, strict=True):
        d = [r[f"{metric}_vs_zero_shot"] for r in rows]
        mid = [x["delta"] for x in d]
        err = [[x["delta"] - x["ci"][0] for x in d], [x["ci"][1] - x["delta"] for x in d]]
        ax.errorbar(mid, y, xerr=err, fmt="o", color="#2b5d8a", capsize=3)
        ax.axvline(0, color="#888888", lw=1)
        ax.set_xlabel(f"Δ {metric} vs zero-shot bge-small (95% paired CI)")
        ax.grid(axis="x", alpha=0.3)
    axes[0].set_yticks(y, [r["label"] for r in rows])
    axes[0].invert_yaxis()
    fig.suptitle("FiQA-2018 test, 648 queries", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def sweep_markdown() -> str:
    """Dev-only learning-rate search table (results/sweep/*.json)."""
    dev_base = read_json(RUNS / "dev_baseline_bge-small.json")["mean"]["ndcg@10"]
    lines = [
        f"Zero-shot dev nDCG@10: {dev_base:.4f}\n",
        "| Arm | lr | dev nDCG@10 epoch 1 | epoch 2 | best |",
        "|---|---|---|---|---|",
    ]
    runs = [read_json(p) for p in Path("results/sweep").glob("*.json")]
    for r in sorted(runs, key=lambda r: (r["arm"], r["config"]["lr"])):
        per_epoch = [f"{h['ndcg@10']:.4f}" for h in r["dev_history"]]
        per_epoch += [""] * (2 - len(per_epoch))
        lines.append(
            f"| {r['arm']} | {r['config']['lr']:g} | {' | '.join(per_epoch[:2])} "
            f"| {r['best_dev_ndcg@10']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    Path("results/sweep/summary.md").write_text(sweep_markdown(), encoding="utf-8", newline="\n")
    systems = load_systems()

    order = [k for k in LABELS if k in systems] + sorted(set(systems) - set(LABELS))
    rows = [
        summarize_system(k, systems[k], systems[BASELINE], systems.get("inbatch/full"))
        for k in order
    ]
    write_json("results/summary.json", rows)
    Path("results/summary.md").write_text(to_markdown(rows), encoding="utf-8", newline="\n")
    plot(rows, "results/deltas.png")
    print(to_markdown(rows))


if __name__ == "__main__":
    main()
