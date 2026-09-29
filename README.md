# fiqa-embed-ft

How much does in-domain fine-tuning help a small (33M-parameter) embedding model on
financial question answering retrieval, which kind of hard negatives is worth mining,
and what does it cost on one 8 GB laptop GPU? This repo fine-tunes
`BAAI/bge-small-en-v1.5` on the FiQA-2018 training split (BEIR version) with
MultipleNegativesRankingLoss and compares in-batch negatives, BM25-mined hard
negatives and dense-mined hard negatives with and without false-negative filtering.
Model selection uses the dev split only; the test split (648 queries, 57,638
documents) is scored once per trained model, and every comparison carries a
query-level bootstrap CI and a paired permutation test.

**Short answer:** the zero-shot model is already strong on FiQA. No arm improves
test nDCG@10 by a statistically clear margin; the best arm (dense negatives + 95%
margin filter) gains +0.006 nDCG@10 (95% CI -0.006 to +0.019). Every fine-tuned arm
except the BM25 one significantly improves Recall@100 (up to +0.037, p<0.001). Among the fine-tuning arms, the
filter matters: unfiltered dense negatives and BM25 negatives are no better, or
worse, than in-batch negatives alone.

## Results

FiQA-2018 test split, full-corpus exact search, 512-token documents, fp32 inference.
Fine-tuned rows with 3 seeds report the seed mean (± standard deviation across
seeds); their CIs and p-values use per-query scores averaged over the seeds. Δ values
are paired differences on the same 648 queries (10,000 bootstrap resamples for CIs,
10,000 sign flips for p-values). Full table: [results/summary.md](results/summary.md);
per-query scores: `results/runs/*.json`.

| System | Seeds | nDCG@10 | Δ nDCG@10 vs zero-shot [95% CI], p | R@100 | Δ R@100 vs zero-shot, p | MRR@10 | Train s | Peak VRAM GB |
|---|---|---|---|---|---|---|---|---|
| BM25 (bm25s, stemmed) | - | 0.2514 | -0.152 [-0.176, -0.129], p<0.001 | 0.559 | -0.137, p<0.001 | 0.307 | | |
| bge-small zero-shot, no query instruction | - | 0.3848 | -0.019 [-0.027, -0.010], p<0.001 | 0.687 | -0.010, p=0.206 | 0.465 | | |
| **bge-small zero-shot** | - | **0.4035** | | 0.696 | | 0.488 | | |
| bge-base zero-shot (110M, reference) | - | 0.4062 | +0.003 [-0.012, +0.018], p=0.740 | 0.741 | +0.045, p<0.001 | 0.485 | | |
| (a) MNRL, in-batch negatives | 3 | 0.3967 ± 0.0004 | -0.007 [-0.020, +0.006], p=0.314 | 0.724 | +0.027, p=0.002 | 0.474 | 132 | 2.41 |
| (b) + BM25 hard negatives | 3 | 0.3819 ± 0.0004 | -0.022 [-0.036, -0.007], p=0.004 | 0.716 | +0.019, p=0.054 | 0.459 | 208 | 4.00 |
| (c0) + dense hard negatives, no filter | 3 | 0.3934 ± 0.0016 | -0.010 [-0.024, +0.004], p=0.170 | 0.723 | +0.026, p=0.012 | 0.467 | 203 | 4.00 |
| (c) + dense hard negatives, 95% margin filter | 3 | **0.4099** ± 0.0006 | +0.006 [-0.006, +0.019], p=0.326 | **0.734** | +0.037, p<0.001 | 0.490 | 199 | 4.00 |
| (a) at batch 128 via GradCache | 1 | 0.4040 | +0.001 [-0.012, +0.013], p=0.939 | 0.722 | +0.026, p=0.006 | 0.483 | 148 | 2.20 |
| (c) at batch 128 via GradCache | 1 | 0.4074 | +0.004 [-0.008, +0.016], p=0.550 | 0.731 | +0.035, p<0.001 | 0.484 | 242 | 2.20 |
| (a) with LoRA r=16 (1.3M trainable params) | 1 | 0.3941 | -0.009 [-0.022, +0.004], p=0.168 | 0.725 | +0.029, p=0.003 | 0.467 | 161 | 2.70 |
| (c) with LoRA r=16 (1.3M trainable params) | 1 | 0.4064 | +0.003 [-0.010, +0.016], p=0.664 | 0.731 | +0.034, p<0.001 | 0.487 | 296 | 4.77 |

![Paired differences vs zero-shot](results/deltas.png)

What the numbers say:

- **Zero-shot baseline matches the published number.** Our bge-small nDCG@10 is
  0.4035 against 0.40343 in the MTEB results repository, and bge-base is 0.4062
  against 0.40646 ([results/published_mteb.json](results/published_mteb.json)).
  R@100 differs by less than 0.002. There is no gap to explain, so the evaluation
  code (metrics, instruction prefix, 512-token documents, exact search) matches MTEB's.
- **Hard-negative filtering is the only thing that separates the arms.** Compared
  with in-batch negatives (a), the margin-filtered dense arm (c) gains +0.013 nDCG@10
  (p<0.001) and +0.010 R@100 (p=0.010). The same dense negatives without the filter
  (c0) are no better than (a) (-0.003, p=0.380), and BM25 negatives are worse
  (-0.015, p<0.001).
- **Fine-tuning mostly helps recall, not the top of the ranking.** All fine-tuned arms
  raise R@100 by 0.019 to 0.037 over zero-shot (all with p<0.02 except BM25 negatives,
  p=0.054), which would matter in front of a re-ranker. nDCG@10 moves by at most
  0.022, and only the BM25 arm's change (a drop) is significant.
- **Batch size explains part of the in-batch result.** With 32 pairs per batch, (a)
  loses 0.007 nDCG@10. With 128 pairs per batch (GradCache, same peak memory) the loss
  is gone (+0.001), but one seed only.
- **Cost is small.** A full fine-tune is 886 optimizer steps (2 epochs) and takes 1.5
  to 4 minutes on the laptop GPU, plus 30 to 40 s of dev evaluation per epoch (fp16)
  and about 1.5 minutes to mine dense negatives. LoRA does not save memory at this model
  size: activations, not the optimizer state of 33M parameters, dominate peak VRAM.

Dev-split learning-rate search (seed 1, in-batch and filtered-dense arms,
[results/sweep/summary.md](results/sweep/summary.md)): 2e-5 was best for both arms and
is used for every full fine-tune. Zero-shot dev nDCG@10 is 0.4094 (fp32); the best
dev score of any fine-tuned model was 0.4175 (filtered dense arm, seed 3, fp16).

Mining statistics ([results/mining/](results/mining/)): with the 95% margin rule, 3,601
of 14,166 training pairs (25.4%) had no candidate in the top 1,000 below 0.95 x
cos(query, positive), so they received a random negative instead. These are pairs
whose labelled positive the base model ranks very low.

Hardware and software: RTX 5070 Laptop GPU (8 GB), i9-14900HX, Windows 11, PyTorch
2.9.0+cu129, sentence-transformers 6.1.0, transformers 5.17.0. Train seconds exclude
dev evaluation and model loading. They are noisy (in-batch runs took 98 to 195 s)
because other CPU-heavy jobs shared the machine and tokenization runs on the CPU.
Peak VRAM is `torch.cuda.max_memory_allocated` during training, which includes the
per-epoch dev evaluation.

## Architecture

```
BeIR/fiqa + BeIR/fiqa-qrels (HF Hub)
        |
   fiqa_ft/data.py ---- corpus, queries, qrels{train,dev,test}
        |
        +--> scripts/mine.py  --(BM25 | base-model dense top-1000)--> fiqa_ft/mining.py
        |        filters: drop labelled positives, skip_top, margin x cos(q, pos)
        |        -> artifacts/negatives_<name>.jsonl   (1 negative per pair)
        |
        +--> scripts/train.py  (sentence-transformers Trainer, MNRL or GradCache MNRL,
        |        bf16, no-duplicate batches, 2 epochs, lr 2e-5, batch 32)
        |        fiqa_ft/training.py: dev nDCG@10 each epoch -> keep best weights
        |        -> test retrieval once -> results/runs/<run>.json (per-query scores)
        |
        +--> scripts/evaluate.py (zero-shot models, BM25)
        |
   scripts/summarize.py -- fiqa_ft/stats.py (bootstrap, sign-flip test)
        -> results/summary.{md,json}, results/deltas.png
```

`fiqa_ft/retrieval.py` does exact cosine search on the GPU (chunked `q @ D.T` +
`topk`) and BM25 through `bm25s`; `fiqa_ft/metrics.py` implements nDCG/Recall/MRR with
trec_eval definitions.

## Quickstart / Reproduce

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu129
pip install -e ".[train,dev]"

pytest -q && ruff check .                     # CPU-only tests, no downloads

python scripts/evaluate.py --name baseline_bge-small   # zero-shot, about 2 min on GPU
bash scripts/sweep_lr.sh                               # dev-only lr search
bash scripts/run_all.sh                                # everything in results/, about 2.5 h
```

`run_all.sh` skips any result file that already exists, so delete the files under
`results/runs/` that you want to recompute. Mined negatives are regenerated into
`artifacts/` (gitignored). No model weights are saved or uploaded.

## Design decisions

- **Exact search over approximate search.** 57,638 x 384 embeddings fit easily in
  GPU memory, and exact top-k removes ANN recall loss as a confounder.
- **The margin filter is relative to each positive's own score**
  (TopK-PercPos from NV-Retriever, [arXiv:2407.15831](https://arxiv.org/abs/2407.15831)).
  FiQA has about 2.6 labelled answers per query and many unlabelled correct ones, so
  the candidates that look most like the positive are often unlabelled positives.
  A fixed rank cutoff (`--skip-top`) is also implemented and tested, but it was not
  one of the evaluated arms.
- **No-duplicate batches.** A query with several positives would otherwise appear
  twice in a batch, and each copy's positive would become a false in-batch negative
  for the other.
- **Best weights kept in memory, not in checkpoints.** The evaluator copies the state
  dict (about 130 MB) when dev nDCG@10 improves. Nothing is written to disk, and the
  same code serves full fine-tuning and LoRA.
- **Queries, not seeds, as the unit of resampling.** Seed-to-seed standard deviation
  is below 0.002 nDCG@10, while the CI from the 648-query sample is about ±0.028. The
  query sample is the larger source of uncertainty, so the paired tests resample
  queries.
- **Batch 32 with 256-token training inputs.** A batch of 64 triplets did not fit in
  8 GB. The batch-128 GradCache runs show what a larger batch buys without extra memory.

## Limitations

- The margin arm is partly a mixture: 25.4% of its pairs got random, easy negatives
  because no candidate passed the filter. Part of its advantage over (c0) may come
  from that mixture rather than from removing false negatives; this was not separated.
- Only one hyperparameter setting per arm was used for the final runs (lr chosen on
  dev for arms a and c, reused for b and c0). The LoRA runs use an untuned lr of 2e-4
  and one seed; the GradCache runs use one seed.
- The first in-batch run (lr 2e-5, seed 1) printed its test score before the dev-only
  sweep mode existed. The final seed-1 run for that arm uses the same settings and
  reproduced the same score (0.3963); no setting was chosen based on it.
- Training was capped at 2 epochs for time. The filtered dense arm was still improving
  on dev at epoch 2, so longer training might help it.
- One dataset, one base model. We do not know whether FiQA-like data was part of the
  bge-v1.5 training mix, which would explain why fine-tuning adds so little.

## Data and model licenses

- FiQA-2018 via BEIR: [BeIR/fiqa](https://huggingface.co/datasets/BeIR/fiqa) and
  [BeIR/fiqa-qrels](https://huggingface.co/datasets/BeIR/fiqa-qrels), listed as
  CC BY-SA 4.0 on the Hugging Face Hub. Original task: FiQA 2018 open challenge
  (Maia et al., WWW'18 companion). BEIR: Thakur et al., NeurIPS 2021 Datasets and
  Benchmarks. The data is downloaded at runtime and not stored in this repo.
- [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) and
  bge-base-en-v1.5: MIT.
- Published reference scores: [embeddings-benchmark/results](https://github.com/embeddings-benchmark/results).

## License

MIT, copyright 2026 Yunlong Lu. See [LICENSE](LICENSE).
