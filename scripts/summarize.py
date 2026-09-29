"""Aggregate results/runs/*.json into results/summary.{json,md} and results/deltas.png.

For a system trained with several seeds, per-query scores are first averaged over
seeds. A query-level bootstrap gives each CI, a paired sign-flip test gives each
p-value, and Holm's method adjusts all p-values in the table together, so "survives
correction" means it holds for the whole family of comparisons we report.
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
    holm_adjust,
    paired_bootstrap_delta,
    paired_randomization_test,
    seed_average,
)

RUNS = Path("results/runs")
BASELINE = "baseline_bge-small"
INBATCH = "inbatch/full"
LABELS = {
    "bm25": "BM25 (bm25s)",
    "baseline_bge-small_noinstr": "bge-small zero-shot, no query instruction",
    BASELINE: "bge-small zero-shot",
    "baseline_bge-base": "bge-base zero-shot (reference, 110M)",
    INBATCH: "(a) MNRL, in-batch negatives",
    "bm25/full": "(b) + BM25 hard negatives",
    "dense_nofilter/full": "(c0) + dense hard negatives, no filter",
    "dense_margin95/full": "(c) + dense hard negatives, 95% margin filter",
    "inbatch/full_bs128": "(a) with batch 128 (GradCache)",
    "dense_margin95/full_bs128": "(c) with batch 128 (GradCache)",
    "inbatch/lora": "(a) with LoRA r=16",
    "dense_margin95/lora": "(c) with LoRA r=16",
}
SHORT = {BASELINE: "zero-shot", INBATCH: "(a)", "dense_nofilter/full": "(c0)"}
TESTED = ("ndcg@10", "recall@100")  # metrics that get paired CIs and p-values
# Extra comparison beyond "vs zero-shot" and "vs (a)": does the margin filter itself
# matter? Same dense candidates, with and without the filter.
EXTRA = [("dense_margin95/full", "dense_nofilter/full")]


def load_systems() -> dict[str, list[dict]]:
    systems: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(RUNS.glob("*.json")):
        run = read_json(path)
        if run.get("split", "test") != "test":
            continue
        key = f"{run['arm']}/{run['method']}" if "arm" in run else run["name"]
        systems[key].append(run)
    return systems


def per_query(runs: list[dict], metric: str) -> dict[str, float]:
    return seed_average([r["per_query"][metric] for r in runs])


def paired(a_pq: dict[str, float], b_pq: dict[str, float]) -> dict:
    """Paired comparison of system b against system a on the same queries."""
    if set(a_pq) != set(b_pq):
        raise ValueError("paired systems were scored on different queries")
    qids = sorted(b_pq)
    a, b = [a_pq[q] for q in qids], [b_pq[q] for q in qids]
    delta, lo, hi = paired_bootstrap_delta(a, b)
    return {"delta": delta, "ci": [lo, hi], "p": paired_randomization_test(a, b)}


def comparison_pairs(systems: dict[str, list[dict]]) -> list[tuple[str, str]]:
    """(system, reference) pairs, in table order."""
    pairs = []
    for key in systems:
        if key != BASELINE:
            pairs.append((key, BASELINE))
        if "train_seconds" in systems[key][0] and key != INBATCH and INBATCH in systems:
            pairs.append((key, INBATCH))
    pairs += [(s, r) for s, r in EXTRA if s in systems and r in systems]
    return pairs


def compare_all(systems: dict[str, list[dict]]) -> list[dict]:
    rows = []
    for key, ref in comparison_pairs(systems):
        for m in TESTED:
            d = paired(per_query(systems[ref], m), per_query(systems[key], m))
            rows.append({"system": key, "reference": ref, "metric": m, **d})
    for row, p_holm in zip(rows, holm_adjust([r["p"] for r in rows]), strict=True):
        row["p_holm"] = p_holm
    return rows


def summarize_system(key: str, runs: list[dict]) -> dict:
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
    if trained:
        secs = [r["train_seconds"] for r in runs]
        row["train_seconds"] = statistics.fmean(secs)
        row["train_seconds_range"] = [min(secs), max(secs)]
        n_evals = [len(r["dev_history"]) - 1 for r in runs]  # minus the epoch-0 base eval
        row["dev_eval_seconds_per_epoch"] = statistics.fmean(
            r["dev_eval_seconds"] / n for r, n in zip(runs, n_evals, strict=True)
        )
        row["peak_vram_gb"] = max(r["peak_vram_gb"] for r in runs)
        row["trainable_params"] = runs[0]["config"]["trainable_params"]
        row["lr"] = runs[0]["config"]["lr"]
        row["epochs_run"] = [r["epochs_run"] for r in runs]
        row["best_dev_epochs"] = [r["best_dev_epoch"] for r in runs]
        row["best_dev_ndcg@10_mean"] = statistics.fmean(r["best_dev_ndcg@10"] for r in runs)
        row["base_dev_ndcg@10"] = runs[0]["base_dev_ndcg@10"]
        row["base_won_on_dev"] = sum(r["base_won_on_dev"] for r in runs)
    return row


def fmt(x: float | None, nd: int = 3) -> str:
    return "" if x is None else f"{x:.{nd}f}"


def fmt_p(p: float) -> str:
    # 10k resamples cannot resolve p below ~1e-4, so do not print false precision
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def systems_markdown(rows: list[dict]) -> str:
    lines = [
        "| System | Seeds | nDCG@10 | 95% CI | R@10 | R@100 | MRR@10 | lr | Epochs run "
        "| Best dev epoch | Best dev nDCG@10 (mean) | Base won on dev | Train s (range) "
        "| Dev eval s/epoch | Peak VRAM GB |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        ndcg = fmt(r["ndcg@10"], 4) + (f" ± {fmt(r['ndcg@10_sd'], 4)}" if r["ndcg@10_sd"] else "")
        ci = f"[{fmt(r['ndcg@10_ci'][0])}, {fmt(r['ndcg@10_ci'][1])}]"
        trained = "train_seconds" in r
        cells = [
            r["label"],
            str(len(r["seeds"]) or "-"),
            ndcg,
            ci,
            fmt(r["recall@10"]),
            fmt(r["recall@100"]),
            fmt(r["mrr@10"]),
        ]
        if trained:
            lo, hi = r["train_seconds_range"]
            cells += [
                f"{r['lr']:g}",
                ", ".join(f"{e:g}" for e in r["epochs_run"]),
                ", ".join(f"{e:g}" for e in r["best_dev_epochs"]),
                fmt(r["best_dev_ndcg@10_mean"], 4),
                f"{r['base_won_on_dev']}/{len(r['seeds'])}",
                f"{r['train_seconds']:.0f} ({lo:.0f}-{hi:.0f})",
                fmt(r["dev_eval_seconds_per_epoch"], 0),
                fmt(r["peak_vram_gb"], 2),
            ]
        else:
            cells += [""] * 8
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def comparisons_markdown(comps: list[dict]) -> str:
    lines = [
        "| System | vs | Metric | Δ | 95% CI | p | Holm-adjusted p |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in comps:
        ref = SHORT.get(c["reference"], c["reference"])
        lines.append(
            f"| {LABELS.get(c['system'], c['system'])} | {ref} | {c['metric']} "
            f"| {c['delta']:+.4f} | [{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}] "
            f"| {fmt_p(c['p'])} | {fmt_p(c['p_holm'])} |"
        )
    return "\n".join(lines) + "\n"


def summary_markdown(rows: list[dict], comps: list[dict], n_queries: int) -> str:
    return (
        f"# FiQA-2018 test results ({n_queries} queries)\n\n"
        "Generated by `scripts/summarize.py`. Fine-tuned rows with several seeds show the "
        "seed mean ± SD. Dev columns: nDCG@10 on the dev split at 512 tokens (fp16), "
        "epoch 0 is the untouched base model; 'Base won on dev' counts seeds where no "
        "fine-tuned epoch beat it. Train seconds exclude dev evaluation.\n\n"
        + systems_markdown(rows)
        + "\n## Paired comparisons\n\n"
        "Per-query scores are averaged over seeds before pairing. CIs: 10,000 bootstrap "
        "resamples of queries. p: two-sided sign-flip test, 10,000 resamples. Holm "
        f"adjustment over all {len(comps)} tests in this table.\n\n" + comparisons_markdown(comps)
    )


def plot(rows: list[dict], comps: list[dict], path: str) -> None:
    """Paired differences vs the zero-shot model, one panel per tested metric.

    Paired CIs are much narrower than the CIs of the absolute scores, because most
    of the query-to-query variance is shared by both systems.
    """
    vs_zero = {
        (c["system"], c["metric"]): c
        for c in comps
        if c["reference"] == BASELINE and c["system"] != "bm25"
    }
    rows = [r for r in rows if (r["key"], "ndcg@10") in vs_zero]
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.42 * len(rows) + 1.4), sharey=True)
    y = list(range(len(rows)))
    for ax, metric in zip(axes, TESTED, strict=True):
        d = [vs_zero[(r["key"], metric)] for r in rows]
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
    runs = [read_json(p) for p in Path("results/sweep").glob("*.json")]
    n_cols = max(len(r["dev_history"]) for r in runs)
    head = " | ".join(f"epoch {i}" for i in range(n_cols))
    lines = [
        "# Learning-rate search (dev split, seed 1)\n",
        "Generated by `scripts/summarize.py` from runs of `scripts/sweep_lr.sh`. Dev "
        "nDCG@10 at 512 tokens under fp16 autocast; epoch 0 is the base model before "
        f"training. For reference, zero-shot dev nDCG@10 in fp32 is {dev_base:.4f} "
        "(`results/runs/dev_baseline_bge-small.json`). A blank cell means early "
        "stopping ended the run.\n",
        f"| Arm | lr | {head} | Best fine-tuned | Base won |",
        "|---|---|" + "---|" * n_cols + "---|---|",
    ]
    by_lr: dict[float, list[float]] = defaultdict(list)
    for r in sorted(runs, key=lambda r: (r["arm"], r["config"]["lr"])):
        per_epoch = [f"{h['ndcg@10']:.4f}" for h in r["dev_history"]]
        per_epoch += [""] * (n_cols - len(per_epoch))
        lines.append(
            f"| {r['arm']} | {r['config']['lr']:g} | {' | '.join(per_epoch)} "
            f"| {r['best_dev_ndcg@10']:.4f} (epoch {r['best_dev_epoch']:g}) "
            f"| {'yes' if r['base_won_on_dev'] else 'no'} |"
        )
        by_lr[r["config"]["lr"]].append(r["best_dev_ndcg@10"])
    lines += ["", "Mean best dev nDCG@10 over the swept arms:", ""]
    lines += [f"- lr {lr:g}: {statistics.fmean(v):.4f}" for lr, v in sorted(by_lr.items())]
    return "\n".join(lines) + "\n"


def main() -> None:
    Path("results/sweep/summary.md").write_text(sweep_markdown(), encoding="utf-8", newline="\n")
    systems = load_systems()
    order = [k for k in LABELS if k in systems] + sorted(set(systems) - set(LABELS))
    systems = {k: systems[k] for k in order}

    rows = [summarize_system(k, runs) for k, runs in systems.items()]
    comps = compare_all(systems)
    n_queries = len(systems[BASELINE][0]["per_query"]["ndcg@10"])
    write_json("results/summary.json", {"systems": rows, "comparisons": comps})
    md = summary_markdown(rows, comps, n_queries)
    Path("results/summary.md").write_text(md, encoding="utf-8", newline="\n")
    plot(rows, comps, "results/deltas.png")
    print(md)


if __name__ == "__main__":
    main()
