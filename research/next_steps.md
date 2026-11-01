# Next steps (updated 2026-10-26)

## Where we stand
- Public LB: **v8 = 0.639** (private copy of the public 0.639 stack; best). v9 (v8 + word-BiLSTM 10%) = 0.637, rejected.
  Public top notebook 0.641 (= 0.639 + Qwen3.8-27B 10%, about +0.001 on its own); leaderboard top 0.700.
- The public stack's LLMs are all Qwen-family (plus DeBERTa, GLM-9B in the complement, a Llama in the 0.641 version).
  E049: unique signal comes from other families, so we train non-Qwen two-stage LoRAs.
- Public-LB steps of +-0.002 are inside noise; trust prompt-grouped CV and changes of meaningful size.

## Running
1. **Gemma-4-12B two-stage** (`nb/gemma_twostage`): s1 done (stage1a/b, pool_v4 + darragh = 77,264 rows);
   s2 `ntl-gemma-ts-s2` (stage1c, folds 0-2, final) running since 2026-10-26 08:50 UTC (~6.5 h).
2. **Daily Benchmarks** on the own prompt set: `python tools/kbench/daily.py --own --n 40` (n 32 on days with an Opus
   retry). Done: o001, o001r, o002 (72 prompts x 10 models). Credit is separate from GPU quota.

## Next (in order)
1. Gemma s2 done -> OOF eval on folds 0-2 against the strong bundle (`experiments/gemma_oof.py`-style) -> E054.
2. v10 = v8 + Gemma at 10-15% (single change) -> 3-row test run -> **runtime check #19** (v8 + Gemma must stay < 9 h;
   needs ~3 GPU-h after the weekly reset) -> ask the user -> submit.
3. **Llama-3.1-8B two-stage** after the weekly GPU reset (cancelled 2026-10-26 for quota):
   `python nb/gemma_twostage/build.py jobs:stage1a,stage1b ntl-llama-ts-s1 "" llama31-8b`, push; then s2 with
   `jobs:stage1c,fold0,fold1,fold2,final ntl-llama-ts-s2 ntl-llama-ts-s1 llama31-8b` (~10.5 GPU-h total).
   Add **length-crop augmentation (test plan #10)** to the next new LoRA as its one recipe change.
4. **LSTM probe of our own Benchmarks data** at ~2,000 own passages (CPU): does it raise CV? If yes, include our passages
   in the next stage-1 pool (and publish them before the deadline).

## Paper obligations (paper/PROTOCOL.md)
- **Drift panel**: month 0 = drift_202611 (2026-11-01, 720/720, frozen `tools/kbench/ntl_gen_task_drift.py`,
  g0003-g0062). Next: `drift_202612` on **Dec 1-3** via `tools/kbench/drift.py drift_202612 <12 slugs>`, then monthly.
- **Final freeze (late Dec)**: 5-fold OOF for every final member, leave-one-component-out ablation, dry sensitivity grid;
  class-mix part done (E053).
- **Final 2 submissions**: best stack with full Sinkhorn + same stack with Sinkhorn strength 0.5 (E053 hedge).
- **After 2027-01-13**: late submissions of the ablated notebooks (private-test ablation).
- Publish any external data a submission uses before the deadline (darragh run-1 is already public, CC0).

## Untested ideas (lower priority)
SYNTHESIS M3 soft kernel centring, M4 hierarchical / contrastive objective, M5 sibling-aware decoding, M6 stacking with
per-member temperature, M7 distribution-matching transduction; TEST_PLAN #9 SupCon, #12e self-distillation, #14 fold
soup, #18 test-time augmentation, #20 distillation.
