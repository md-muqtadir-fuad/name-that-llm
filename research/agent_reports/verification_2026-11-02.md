# Verification of the two pasted agent reports (2026-11-02)

Status per claim: **verified true**, **verified false**, **partly true** (measured, with the measured size), or **not
verified** (no check run; no judgement). Each verified status names its evidence. Corrections to my own earlier verdicts are
listed at the end.

## Report 1 ("Breakthrough Blueprint")

| Claim | Status | Evidence |
|---|---|---|
| Blend 0.6506 on 1,620 rows (our folds 0-2); recalls glm-5.3 0.244, gemini-3.7 0.378, fable 0.411, ... | verified true | re-ran its `experiments/scratch_diag.py` (plain heads q14v5 3 / q8v5 2 / ll8v5 1, svc4 12, tfidf 0.25, distort 0.10, Gemma plain 2; no twins/Gaussian): identical numbers |
| "Over 88% of errors are in 4 clusters" (pie 32/35/21/12) | verified false | same predictions: 566 errors; inside gemini 3.6/3.7/3.8 = 99 (17%), glm-5.3/glm-5.3-flash/fable = 99 (17%), gpt-5.6 sol/terra/luna = 61 (11%); total 259 = 46%. Its tier lists omit gpt-5.5 (0.656) and claude-opus-4-8 (0.667) |
| "+0.101 by raising 6 classes from 44.6% to 75%" | arithmetic only | no method shown; paired n-gram probes on 1,600-2,500 pool prompts reach 0.53 (gemini-3.7/3.8) and 0.55 (glm pair) (E055) |
| Lab classifier ">92.5%", Anthropic vs ZAI ">95%" | verified false (for this blend) | lab-level accuracy of the same blend 0.873; fable -> glm* 29 rows, glm-5.3 -> fable 14 rows |
| Layer table (layers 5-12 best 0.35-0.37, 13-20 0.34-0.35, 21-28 0.30-0.31) | verified false (middle rows) | `experiments/backbones/out/layer_scores.csv` (E047, frozen Qwen3-8B, 18-way): L4 0.368, L7 0.356, L11 0.315, L16 0.338, L22 0.359, L29 0.307, L36 0.301 - not monotonic. Its two cited end points are correct |
| Monroe log-odds: siblings statistically indistinguishable (max z 2.42 / 2.36) | partly true | not re-run. On larger paired data a weak real signal exists (E055: 0.530 and 0.550 vs shuffled-label null max 0.509/0.508) |
| v10b failed because Gemma was "uncalibrated" | verified false (as the cause) | v10b used only the plain head; honest sim: plain Gemma alone +0.003, the gain is the centred head (E067b) |
| "E067b = 0.6757 -> v12 LB target 0.675-0.680" | verified false (as an LB target) | 0.6757 is the honest subset (no n14b, q25_14, DeBERTa, ATLAS, comp stack, typography) on 90 prompts |
| "Keep Sinkhorn at 0.35" | verified false | v12 run command: `'--sinkhorn', '0.45'` |
| "ATLAS margin < 0.20 -> under 6.5 h, zero timeout risk" | verified false | v12 infer.py: ATLAS loops until `ATLAS_BUDGET = int(8.3 * 3600 - COMP_SEC - 600)`; the margin changes coverage, not the end time |
| Mean pooling is "85% content words" | not verified | |
| High-IDF content masking before pooling | not verified (transformer) | n-gram analogue: topic-masked TF-IDF 0.322 vs 0.421 unmasked (E065) - evidence against for n-grams only |
| Head/body/tail pooling | not verified | the agent tests it (user, 2026-11-02) |

## Report 2 ("Comprehensive Technical Audit")

| Claim | Status | Evidence |
|---|---|---|
| Per-member prediction bias on hard pairs (svc4 3.8 208 vs 3.7 86; ll8v5 211/104; tfidf 249/119; tfidf fable 212 vs glm-5.3 44; distort 164/63; q14v5 156/138; q8v5 152/137) | verified true | honest OOF, all 2,699 rows: identical counts. Plain-head blend 3.8 182 vs 3.7 101 (true 150/150) |
| "Correcting the bias raises bacc" (Strategy 1) | verified false (no net gain) | E069: learned per-class offset (class-balanced CE, 4 folds -> 5th) and prior correction; uniform mix -0.0004/+0.0004, Dirichlet(3) +0.0020 (se 0.0013) / +0.0006; with Gemma -0.0000/-0.0012. Mechanism: offset moves gemini-3.7 recall 0.467 -> 0.533 but 3.8 0.647 -> 0.580; fable 0.423 -> 0.483, glm-5.3 0.327 -> 0.293 (errors move between near-chance siblings) |
| Gemma layer 24 alone beats all 8 layers (3.7/3.8 AUC 0.6117 vs 0.5925; sol/terra BA 0.7111 vs 0.6722) | partly true (smaller) | my re-run, 10 random prompt splits, LR C=0.01, 180 rows/pair: 3.7/3.8 L24 0.584 +- 0.013 vs all-8 0.571 +- 0.018; sol/terra L24 0.745 +- 0.013 vs all-8 0.730 +- 0.013; L24 is the best single layer for both pairs. A frozen 180-row probe does not test "train the head on layer 24 only" |
| "Head has 552,960 weights fit on 1,080 samples" | verified false | the head is trained in stage 1 on 77,264 pool rows and in stage 2 on host folds (twostage_train.py) |
| Centring raises sibling AUC by +0.035 to +0.072 | method verified: oracle | `experiments/test_centred_sibling_pairs.py` line 36-37 centres on the TRUE full prompt (18 answers); numbers not re-run |
| Gemma's twin is robust because it was trained on 2-3 chunks | first half verified true; causal part not verified | `twostage_train.py` line 276: chunk size `rng.integers(2, 4)`; the public v5 adapters' training code is not in our pulls, so whether their centred heads were trained on full prompts is unknown |
| v11a CPU image `17b2...` "Docker registry logs confirm no CUDA 12.4" | not verified | our evidence is behavioural (same family of `kaggle-images` digests gave `torch.cuda.is_available() == False`) |
| "MARGIN_THRESHOLD = 0.3 and a 7,200 s watchdog at line 29 of v12" | verified false | `grep` in nb/v12/ntl-v12.ipynb: no MARGIN_THRESHOLD, no 7200; margin is `--atlas_margin 0.3`, budget formula above |
| "0.664 default Sinkhorn is 0.0" | verified false | run command `--sinkhorn 0.45 --sinkhorn_target_mix 0.35` |
| "Stage 1 needs 6-8 GPU-h on A100" | not applicable | we train on T4; our estimate for 198k rows is ~15-20 T4 GPU-h (not measured) |
| "gemini-3.7-flash and glm-5.3 are unreleased/internal checkpoints" | not verified | we only know they are not served by Kaggle Benchmarks, and the Gemini API serves 3.7 as 3.8 |
| "88% error wall", "lab ID 99.7% Google / 94.7% OpenAI" | 88% verified false (46%, above); per-lab numbers not verified | |

## Corrections to my earlier verdicts (same day)

- I called the report-1 layer-24 gain "within ~0.04 SE" without running it. Re-run: direction confirmed, about +0.013-0.015 AUC.
- I said report 2's Docker claim was false ("it had no such logs") without checking what it accessed: changed to not verified.
- I called "unreleased internal checkpoints" unsupported: changed to not verified.
