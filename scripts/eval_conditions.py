"""Check the two shortcuts the training script takes, on the zero-shot model and dev split.

1. Training truncates documents to 256 tokens. How many FiQA documents are longer?
2. Dev evaluation during training runs under fp16 autocast. How much does that move
   dev nDCG@10 compared with fp32? (Scored at 256 and 512 tokens, which also shows
   how much the truncation alone changes the dev score.)

Writes results/eval_conditions.json. Needs a CUDA GPU; about 5 minutes.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from fiqa_ft.data import BASE_MODEL, MODEL_REVISIONS, load_fiqa
from fiqa_ft.evaluation import evaluate_dense, write_json
from fiqa_ft.training import require_cuda


def main() -> None:
    require_cuda()
    fiqa = load_fiqa()
    model = SentenceTransformer(BASE_MODEL, device="cuda", revision=MODEL_REVISIONS[BASE_MODEL])

    # Token counts include [CLS] and [SEP], matching what max_seq_length truncates.
    lengths = np.array(
        [len(ids) for ids in model.tokenizer(list(fiqa.corpus.values()))["input_ids"]]
    )
    doc_lengths = {
        "documents": int(lengths.size),
        "percentiles": {str(p): float(np.percentile(lengths, p)) for p in (50, 75, 90, 95, 99)},
        "share_over_256": float((lengths > 256).mean()),
        "share_over_512": float((lengths > 512).mean()),
    }
    print(doc_lengths)

    dev = []
    for max_len in (512, 256):
        model.max_seq_length = max_len
        for fp16 in (False, True):
            t0 = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
                mean = evaluate_dense(model, fiqa.queries, fiqa.corpus, fiqa.qrels["dev"])["mean"]
            row = {
                "max_seq_length": max_len,
                "precision": "fp16 autocast" if fp16 else "fp32",
                "seconds": time.perf_counter() - t0,
                **mean,
            }
            print(row)
            dev.append(row)

    write_json(
        "results/eval_conditions.json",
        {"model": BASE_MODEL, "doc_lengths": doc_lengths, "dev_zero_shot": dev},
    )


if __name__ == "__main__":
    main()
