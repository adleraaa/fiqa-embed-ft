"""Download the published MTEB FiQA2018 scores for the reference models.

Source: https://github.com/embeddings-benchmark/results (the MTEB leaderboard's
results repository), pinned to one commit so the reference numbers cannot move.
Files under "external/" are results submitted by the model authors, not runs by
the MTEB maintainers. Writes results/published_mteb.json.
"""

from __future__ import annotations

import json
import urllib.request

from fiqa_ft.evaluation import write_json

RESULTS_COMMIT = "6dc609bacf737ea0ddb3799008116f775df12eda"
URL = (
    f"https://raw.githubusercontent.com/embeddings-benchmark/results/{RESULTS_COMMIT}/"
    "results/{model}/external/FiQA2018.json"
)
MODELS = {
    "BAAI/bge-small-en-v1.5": "BAAI__bge-small-en-v1.5",
    "BAAI/bge-base-en-v1.5": "BAAI__bge-base-en-v1.5",
}
KEYS = {
    "ndcg@10": "ndcg_at_10",
    "recall@10": "recall_at_10",
    "recall@100": "recall_at_100",
    "mrr@10": "mrr_at_10",
}


def main() -> None:
    out = {}
    for model, folder in MODELS.items():
        url = URL.format(model=folder)
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.load(resp)
        test = data["scores"]["test"][0]
        out[model] = {
            "source": url,
            "kind": "author-reported (MTEB external results)",
            **{ours: test[theirs] for ours, theirs in KEYS.items()},
        }
    write_json("results/published_mteb.json", out)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
