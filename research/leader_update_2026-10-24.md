# Leaderboard update 2026-10-24

## Leaderboard (top)
Hod Carriers 0.700 · Bootstrapping 0.688 · Le Viet 0.674 · Vishak K Bhat 0.673 · Søren Ravn Andersen 0.673 · T.SHIGA 0.666 · Adisak Sukul 0.666.
Rank 20 is at 0.650. Ours: v6a 0.556.

## Public notebooks (best first)
| Score | Notebook | What it adds |
|---|---|---|
| 0.639 | kirilltym/0-639-teach-more-keep-the-signature | 0.636 base + fixed 10% helper: Fourier / PCA-Fourier / prompt-residual / Laplace / kNN / RBF readouts on LoRA24 features, synthetic 4404 |
| 0.639 | maverickss26/kaggle-name-the-llm-v2 | restructured, fail-safe copy of the 0.639 |
| 0.637 | kirilltym/0-637-remove-the-prompt-keep-the-signature | same helper, without the synthetic set |
| 0.636 | fleongg/ntl-part-two-complement | Nishant part two + 15% complementary stack + 10% ATLAS on uncertain rows + Sinkhorn |
| 0.636 | lightningv08/0-636-lora-llama-qwen3-8-27b | reference ensemble + 10% full-text Qwen3.8-27B (NF4, LoRA layers 56-63, 60.5k external rows) on the half with the smallest margin |
| 0.627 | mrpicklerick / adityaakbar | 0.624 + small encoders (UltraBERT 380M, ELECTRA-large) |
| 0.620 | nishantdahal/two-stage-lora-part-two | SVC + Qwen3-14B LoRA (27/40 L) + Qwen2.5-14B LoRA (32/48 L) + DeBERTa-v3-large, all two-stage; CV 0.642 |

## The lever: data
Nishant part two: the same Qwen3-14B recipe moved from 0.569 to **0.611 OOF** only by switching the public pool from 35k to 84k passages
(26 public datasets, filtered). After stage 1 alone it scores 0.572 on host rows (0.493 with the old pool).
The new pool is public: `nishantdahal/ntl-handwriting-blend/pool_v4.parquet` (35 MB). It ships with ready adapters
(Qwen3-14B, Qwen2.5-14B, DeBERTa-v3-large).

## Consequences for us
- The `nb/v7` fork of 0.607 is outdated. Fork the 0.639 instead (maverickss26 v2 is the readable copy).
- The Colab OLMo-2-7B run used pool_v1 (35k); it is obsolete. Any own two-stage run must use pool_v4.
- Frozen-feature backbone screen (E047, partial): OLMo-2-7B +0.0048 on strong (3/5 folds), Qwen3-8B -0.0007. Both fail the
  acceptance rule; the eval crashed on a truncated npz for the remaining models. Frozen features are not worth more effort.

## Update 2026-10-27: new public data
- **crtproduction/name-that-llm-open-pool** (CC BY-SA 4.0, 2026-10-27): 246,978 de-duplicated, provenance-tracked passages
  from 56 public datasets, filtered to look like competition passages (README, sources.csv, filter_log.csv,
  excluded_sources.csv). About 3x pool_v4. The biggest lever so far was pool size (35k -> 84k: +0.04 member OOF), so the
  next stage-1 training should use it (or a large subsample; full size is ~3x the stage-1 GPU time).
- **alessandrolair/ntl-kaggle-benchmarks-generated-passages** (CC BY 4.0): 3,866 passages from the 12 Benchmarks-served
  labels on 335 invented prompts, same protocol as ours (fresh chat, "single paragraph of about N words"); not in the
  open pool (published later the same day).
- Data is no longer an edge by itself: everyone has the open pool. Our own-prompt Benchmarks set is the unique part.
