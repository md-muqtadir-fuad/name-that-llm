# Next steps (updated 2026-11-02)

## Where we stand
- Public LB: v8 0.639 (copy of a public notebook), v9 0.637, v10b 0.635, v11a timeout (CPU docker image).
  **v12 submitted 2026-11-02 (ref 57882460)** = public 0.664 + our Gemma-4-12B two-stage row (plain 2, centred twin 2.8).
  Best public notebook 0.664; LB top 0.737. One public reading moves by up to 0.009 on identical code.
- No paper unless an own method beats the base (user, 2026-11-02). Everything goes to the score.
- Decisions now come from honest OOF (seeded split of the public fold adapters, test-like 2-3 answer groups, infer.py
  estimated clusters): `experiments/honest/` (E065-E067b). Members we can score honestly: q14v5 0.635, q8v5 0.615,
  ll8v5 0.593, svc4 0.504, tfidf 0.421, distort 0.322; our Gemma two-stage plain 0.578 / centred 0.563 (folds 0-2).
- Settled: keep the 0.664 hand weights (NNLS -0.004); Sinkhorn 0-0.45 equal, never 1.0; Gemma centred head +0.011.

## Queue (one at a time, each logged as an E-row before the next starts)
1. **E068 - our Benchmarks passages as SVC/TF-IDF training data** (CPU, Kaggle). ~3,400 passages (b002-b004, o001-o003,
   drift_202611; 12 labels) added to the open pool for the 0.664 CPU members; honest OOF with vs without, 5 folds, plus
   the honest blend. Risk to watch: data for only 12 labels can pull predictions away from the 6 unserved labels.
   If it gains, v13 = v12 + our data in the SVC pool (data must be public by the deadline).
2. **Daily Benchmarks** own-prompt batch: `python tools/kbench/daily.py --own --n 40` (credit is separate).
3. **v12 result** (user reports it): if >= 0.665 keep Gemma and test centred weight 2.8 vs 4 on honest OOF;
   if < 0.655 drop Gemma (stop rule).
4. **Open-pool stage-1 retrain** of one LoRA member (15-18 GPU-h; user decision on quota). Candidate: Llama-3.1-8B
   two-stage (Llama-family diversity is worth +0.005-0.008 on the LB, public A/Bs), stage 1 on the 198k open pool.
5. Head/tail pooling: the other agent tests it (user, 2026-11-02); read its result when reported.

## Rules learned this week
- GPU kernels: GPU byod image `gcr.io/kaggle-private-byod/python@sha256:2757e0c7...`, `assert torch.cuda.is_available()`,
  watch `kaggle kernels logs -f` for the first minutes; `tools/check_meta.py` before every push (now fails on CPU images).
- Do not poll submission scores.
- Publish any external data a submission uses before 2027-01-13 (darragh run-1 is already public, CC0).

## Untested ideas (lower priority)
SYNTHESIS M3 soft kernel centring, M5 sibling-aware decoding, M7 distribution-matching transduction; TEST_PLAN #9
SupCon, #12e self-distillation, #14 fold soup, #18 test-time augmentation, #20 distillation.
