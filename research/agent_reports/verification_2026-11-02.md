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
| Gemma's twin is robust because it was trained on 2-3 chunks | chunk training verified to help (E070: +0.026 on 2-3 groups, frozen Gemma features); causal claim not supported (E070b: public twins lose the same 0.07-0.08 from whole prompt to 2-3 groups as our chunk-trained Gemma twin, and Gemma's twin is weaker standalone) | `twostage_train.py` line 276: chunk size `rng.integers(2, 4)`; the public v5 adapters' training code is not in our pulls, so whether their centred heads were trained on full prompts is unknown |
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

## Report 3 ("Gemma status & strategic recommendation", pasted 2026-11-02)

| Claim | Status | Evidence |
|---|---|---|
| Plain bacc q14v5 0.6290, q8v5 0.6068, ll8v5 0.5790, Gemma 0.5778 (same 1,620 rows) | verified true | honest OOF re-computed: identical |
| Agreement with q14v5: q8v5 71.9%, ll8v5 68.9%, Gemma 64.3% | verified true | re-computed: 0.719 / 0.689 / 0.643 |
| "Plain + Centred (Twin)" column (0.669 / 0.657 / 0.644 / 0.637) | not verified | combination method not stated |
| E067b's +0.0114 is on an incomplete subset (no n14b, q25_14, deb, comp, ATLAS) | verified true | stated in E066/E067 descriptions |
| v12 used one Gemma adapter (final), no fold averaging | verified true | v12 MODELS row `'adapters': [GEMMA_FINAL]`; folds 3-4 never trained |
| "Gemma ~2.2 s/row on 2x T4, ~2.4 h of the budget, starved ATLAS" | verified false | ntl-gemma-parity log: 129 s for 540 rows including load -> <= ~16 min for 4,000 rows (same loader, same hardware) |
| "infer.py `_tk` did not prepend BOS; the position-0 mask removed the first real word" | verified false | Gemma tokenizer saved with the checkpoint: `tok(text)` (infer.py style, add_special_tokens default) -> ids start with 2 = bos_token_id; parity: argmax agreement 1.0 and identical accuracy (0.5741) on 216 rows vs training-time OOF |
| "ATLAS is worth +0.010 to +0.012 on the LB" | not verified | our community record has +0.008 (ATLAS-off 0.643 vs 0.651, *claim*) |
| "Early layers (2, 4, 8) are pure noise (AUC 0.50)" | verified false | re-run (10 prompt splits): gemini-3.7/3.8 L2 0.535, L4 0.549, L8 0.527; sol/terra L2 0.657, L4 0.650 |
| "552,960 head parameters on 1,080 samples" | verified false | head trained on 77,264 stage-1 rows + host folds |
| Qwen3.5-27B NF4 "too slow, caused timeouts" | verified false (timeout part) | community record: it scored 0.638 (a completed run); no reason stated |
| ModernBERT "overfitted on local prompts" | not verified | no reason in our record |
| OLMo2-7B 0.597 -> 0.594, 0.377 standalone | verified true (two sources) | community: OLMo2 member at log-prob weight 0.03 -> 0.594; 0.377 = our E047 frozen top-3 probe |
| "New families have a 100% failure rate" | partly true | every public new-member A/B lost except Llama (+0.005 to +0.008 when removed); Llama is in the stack |
| "Calibrating the sibling prior repairs 3.7-flash recall at zero cost" | verified false (as a gain) | E069: offset raises 3.7 recall 0.467 -> 0.533 but 3.8 drops 0.647 -> 0.580; bacc ~0 |
| "Data pool expansion for Qwen/Llama is the real lever" (Nishant 35k -> 84k: 0.569 -> 0.611 member OOF) | verified true (as a reported result) | community record (*claim*); our option 4 (open-pool stage-1 retrain) |
| "Dropping Gemma restores 0.664" | not verified | 0.664 is the base author's reading, not ours |

## Report 4 ("why v12 scored 0.658", pasted 2026-11-02)

| Claim | Status | Evidence |
|---|---|---|
| Accepts BOS and runtime corrections (report 3) | consistent | see report 3 rows |
| Grid on 1,620 rows: base 0.6698; Gemma plain 0 + twin 2.8 = 0.6907 (+0.0210); plain 2 + twin 2.8 = 0.6833 (+0.0136) | numbers verified true; setting is oracle-like | re-ran `experiments/test_gemma_twin_on_clusters.py`: identical numbers. It clusters all 1,620 rows (90 prompts x 18 answers) at once: 193 clusters, median cluster size of a row 18, 81% of rows in clusters >= 10, 3% singletons -> near whole-prompt centring, not the test's 2-3 answers per prompt. Test-like version (E067b, one 2-3 group per prompt, same clustering, 40 draws): plain 0 + centred 2.8 +0.0098, plain 2 + centred 2.8 +0.0114 |
| "Gemma's plain head (w 2) drags lone passages toward a 0.578 model -> LB drop" | verified false (on honest OOF) | all 1,620 rows as singletons (no twin, no Gaussian, as infer.py does for size-1 clusters): base 0.6407 -> + Gemma plain w2 0.6506 without Sinkhorn; 0.6426 -> 0.6457 with Sinkhorn 0.45 (w 0.5: 0.6457/0.6469, w 1: 0.6469/0.6457) |
| Single final adapter vs 6-adapter averages | verified true (fact) / effect not verified | v12 row `adapters: [GEMMA_FINAL]` |
| Recommendations (Gemma plain 0, folds 3-4, layer-24 head) | not pursued | Gemma dropped by the user (2026-11-02) |
| "Do not switch to outside architectures" | partly true | public A/Bs cited are real (*claim*); Llama is the exception; E072 tests Phi-4 / OLMo-2 at layer level on honest OOF (user request) |

Why v12 read 0.658 stays unexplained by measurement: one public reading (+-0.009 on identical code), members we cannot score
honestly (n14b, q25_14, DeBERTa, comp stack, ATLAS), ATLAS coverage reduced by Gemma's ~16 min, unequal public class mix.

## Re-experiments (what settles each open point)

| Open point | Needed experiment | Status |
|---|---|---|
| Report 1 layer-24 claim | 10-split repeated probe on Gemma z | done (partly true, +0.013-0.015 AUC) |
| Report 2 class-bias correction | E069 offset on 4 folds -> 5th, Sinkhorn 0.45, uniform + Dirichlet(3) | done (~0; errors move between siblings) |
| Report 2 chunk-trained twins as the cause | E070 (frozen Gemma features) + E070b (twin drop per member) | done (chunk training helps; public twins show no deficit) |
| Report 3 BOS bug / runtime | tokenizer check + ntl-gemma-parity log | done (both false) |
| Report 4 plain-head drag on lone passages | all-singleton honest OOF blend +- Gemma plain | done (false: +0.0099 / +0.0031) |
| Report 4 grid regime | cluster-size check of its script | done (oracle-like: median cluster 18) |
| Were the public v5 twins trained on whole prompts or chunks? | pull dhruvtibarewal's training notebook (if public) and read its centring code | open (cheap, no GPU) |
| How much is ATLAS worth (+0.008 claim vs +0.010-0.012) | needs an honest ATLAS bundle per fold (GPU); otherwise only LB A/Bs | open (not measurable honestly today) |
| Why v12 read 0.658 | only separable with paired LB readings (control run of the base) - user declined Gemma work | closed by user decision |
| E068 with class-balanced weighting of our data | rerun ntl-honest-cpu-own with per-class weights | open (low priority: +0.0010 unweighted) |
| Phi-4-14B / OLMo-2-13B value per layer | E072 (running) | in progress |

## Report 5 (E072 failure analysis, pasted 2026-11-02)

| Claim | Status | Evidence |
|---|---|---|
| `WGRID = (0.5, 1, 2, 3)` has no 0, so every layer is forced to a harmful weight | verified true | code line 23; fixed-weight check: weight 0.5 hurts every model's layer 4 (-0.0024 to -0.0033), 1.0 -0.011 to -0.015, 2.0 -0.033 |
| Small weights help a little (OLMo +0.0015, Phi-4 +0.0030, Qwen3-8B +0.0037 at w 0.2; base 0.6494) | direction true; values not reproduced | our check (base 0.6687, Sinkhorn 0.45 per fold slice, 6 draws): OLMo-2 +0.0012 (w 0.1, se 0.0008), Phi-4 +0.0019 (w 0.2, se 0.0008, 5/6), Qwen3-8B +0.0005 (w 0.05); its base differs, setup not stated |
| ~17,850 Sinkhorn calls at ~72 ms -> > 17 min | plausible | measured 23 ms per call on a 540-row fold slice |
| "Phi-4-14B (fp16)" | verified false | nb/layer_screen/run.py loads phi4-14b in 8-bit |
| 15 Sinkhorn iterations give identical argmax | not verified | |
| Frozen probes do not move the baseline; pursue only as trained two-stage LoRA | consistent with data | best fixed-weight gains < +0.002; full per-layer rerun with the fixed grid running |
