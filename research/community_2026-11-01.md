# Community notebooks, 2026-11-01

16 notebooks pulled with `kaggle kernels pull -m` into `research/kernels_2026-11-01/<owner>__<slug>/` (git-ignored).
None of the pulled .ipynb files contain cell outputs, so every number below is taken from the authors' markdown or read
from the code. Labels used here:
- **claim**: a public-LB score or offline number the author reports. We cannot check it.
- **code**: a value I read from the notebook's code, such as a flag, weight or budget formula.
- **est**: my own estimate.
Our references: the 0.651 base is `research/kernels_2026-10-26/avazaku__name-that-llm-0-651-verified-reproduction/`, our v8 (0.639) is `nb/v8/`, and our failed copy is `nb/v11a/`.

Lineage, as each author states it (claims):
nlm-v27 (dhruvtibarewal, 0.651) -> Roman Tamrazov *Who am I?* (0.654, typography likelihood) -> mercurydust **V31** (0.660)
-> omerfarukmerey **V31 + faster run + ATLAS 20%** (0.663) -> alexchilton (0.662, public Llama upload) -> judith007 v2
(0.664, Sinkhorn 0.30) and zbigniewkirchner (0.664, Open Pool SVC) -> vasily61 (a fork of judith v2 that has not been scored).

Code identity, checked by md5 of the `%%writefile` cells:
- `infer.py` is byte-identical in omerfarukmerey, alexchilton, vasily61 and zbigniewkirchner.
- judith007's current version differs from that by one constant (typography scale 1.5 -> 1.2).
- `comp_stack.py` is identical in all six V31-family notebooks, and it differs from the 0.651 base's (68 diff lines: V31's family gates).
- `gauss.py` is identical everywhere, including the 0.651 base.
- So the whole 0.660-0.664 family is **one code base plus 1-2 run flags**.

## 1. Recipe table (top notebooks)

| Notebook | Public LB (claim) | Base | What it changes (code-verified) | Runtime |
|---|---|---|---|---|
| omerfarukmerey/0-663-0-660-v31-atlas-20 | **0.663** (another account re-ran the same code: 0.662) | mercurydust V31 (0.660) | 1. `--atlas_w 0.20` (V31 used 0.10).<br>2. **Faster run**: Llama-3.1-8B is copied to /tmp before loading; the two 8B LoRA members (q8v5, ll8v5, flag `"dp"`) keep one full copy per T4 and alternate batches in two threads, with hidden-state means computed on GPU.<br>3. ATLAS keeps scoring (shortest passages first) until a **hard deadline** and keeps the rows it finished. V31's stop on a projected overrun is removed.<br>Other flags are V31's: `--sinkhorn 0.45 --sinkhorn_target_mix 0.35 --tfidf_w 0.25 --distort_w 0.10 --svc_w 1.5 --extra_w 0.5 --adapt_thr 0.15 --adapt_w 1.0 --post 0 --atlas_margin 0.3 --atlas_bs 4 --atlas_chunk 3 --comp_w 0.15`. | **code**: `ATLAS_BUDGET = int(8.3*3600 - COMP_SEC - 600)`, measured from the start of infer.py. The ATLAS loop stops at `budget - 60`.<br>**claim**: commit ~5 min (placeholder on 3 rows); hidden rerun "several hours".<br>**claim**: Llama read from the mount took ~2,050-2,140 s; the copy takes 88-135 s and the local load ~60 s. The faster run reaches ATLAS "over an hour earlier". |
| alexchilton/name-that-llm-0-662-all-public-what-failed | **0.662** (re-run of the 0.663 code) | omerfarukmerey 0.663 | Same infer.py and run cell. The setup cell loads Llama from `alexchilton/llama-3-1-8b-base` (omer says the sha256 of all shards matches the avazaku cache). `FAST_COMMIT` also requires `KAGGLE_IS_COMPETITION_RERUN` to be unset, so the scoring rerun never takes the placeholder path. | Same as omer. **claim**: commit ~10 GPU min. |
| judith007/0-664-hyperparameter-tuned-who-are-u | **0.664** belongs to **version 2** (scriptVersionId 363834272). vasily61 and lightningv08 say that version is the 0.663 code with `--sinkhorn 0.30`. | alexchilton 0.662 | **The current pulled version is not the scored one**: `--sinkhorn 0.60` and typography evidence scale `1.2` (instead of 1.5). It has no score of its own. | Same structure as omer. |
| vasily61/name-that-llm-0-664-reproduction-notes | none ("this fork has not yet been evaluated") | judith v2 | `--sinkhorn 0.30`; everything else is the 0.663 code (identical infer.py). Notes list only. | "Runtime and scores must be established by our own run." |
| zbigniewkirchner/0-664-one-swap-198k-clean-pool-fully-public | **0.664** | alexchilton 0.662 | 1. SVC `--pool` swapped from `nlm_pool_v5` (120k) to **crtproduction/name-that-llm-open-pool** (`ntl_open_pool.parquet`, ~198k cleaned passages, rows with `flag_train_prompt_regen` dropped).<br>2. `--svc_w 12` (was 1.5).<br>**claim**: the SVC alone in prompt-grouped CV went from 0.4805 to 0.5008 (5 folds, pool weight 0.35, measured on the 193k version).<br>svc_w ladder on the LB: 1.5 -> 0.664, 6 -> 0.659, 12 -> 0.664, 20 -> 0.664. The author says the higher weight "is not a proven gain". | **claim**: commit ~10 GPU min; **hidden run "about 8 hours"**.<br>**est**: the SVC fit on ~200k rows takes longer, but that time comes out of ATLAS's budget, so ATLAS covers fewer rows. Total time does not grow. |
| lightningv08/0-662-qwen3-8-27b-lora-weighted-ensemble | Title says 0.662, but the text says "**This candidate has no confirmed leaderboard score**" | Re-implementation of mercurydust V31 v2 (script 363008892), with prefit caches | 1. Its own **GSQ3 Qwen3.8-27B LoRA** (trained on 119,899 public rows + 2 official epochs, full text, batch size 1, one worker per T4), mixed at **7%** after ATLAS.<br>2. Core and complementary stack mixed 85/15.<br>3. ATLAS at **10%** on rows with top-2 margin < 0.3.<br>4. **Sinkhorn strength 0** (marginal correction off). 37-marker typography unchanged (1.5).<br>5. Fitted states come from a 12-part numeric cache: **prefit SVC4**, Gaussian states, style heads.<br>6. Llama is a 24-layer truncated copy (`llama31-prefix24`) inside its own dataset.<br>7. Qwen3.5 token-attention off; DeBERTa token-attention on. | **code**: **no time budget at all.** The ATLAS worker scores every selected row, and GSQ-27B runs at batch size 1. Runtime is not stated. **Not safe.** |
| leoprovorov/adaptive-centered-stylegram | none. It is "V31 with analysis", and the author writes that the "Public LB 0.654 … belongs to the previous V30" | mercurydust V31 | V31 code with reliability fixes:<br>- comp_stack is launched with `check=True` and stale output is deleted first.<br>- `--save_lp` saves scores after typography.<br>- CPU EDA and prompt-grouped TF-IDF baselines (no outputs saved). | **claim** (from V31's own 3-row log): comp_stack **~2,614 s** (2,629 s with startup); infer.py **~8,426 s on 3 rows**, i.e. fixed overhead; SVC4 fit on **123,828 rows**; **24,443** external texts after de-duplication. Turning Qwen3.5 token-attention back on "may add roughly 1.7-2.7 hours". |
| leoprovorov/he-who-must-not-be-named | **0.552** (title "NTL v11 Candidate, Sinkhorn alpha=0.25"; no score given for alpha 0.25) | Lucas Boesen *LoRA x Adaptive Centering* (0.549) | Previous-generation stack:<br>- an extra Qwen3-8B LoRA with a 5-way Claude head;<br>- Claude gates (stylegram LinearSVC + LoRA head).<br>**claim**: on outer0 rows, LoRA plain 0.4559 -> 0.4838, and 0.4932 with the gate. Writes submissions for alpha 0/0.1/0.25/0.5. | **claim**: base run ~1.5 h. Safe, but off the frontier. |
| ozertuu/fresh-top-30 | n/a | n/a | **Not a Name That LLM notebook.** It is a Kaggriculture game agent; the metadata only lists the competition. Ignore. | n/a |

### Runtime verdicts

**0.663/0.664 family** (omer, alexchilton, judith, vasily, zbig) (**est**):
- ATLAS is built to fill the time up to a fixed deadline. The deadline sits at `8.3 h - COMP_SEC - 660 s` after infer.py starts.
- Total ≈ setup (~2-5 min including the Llama copy) + COMP_SEC + (8.3 h - COMP_SEC - 11 min) + post-processing ≈ **8.1-8.3 h**. That leaves **~40-50 min** under 9 h, independent of test size.
- zbig's "about 8 hours" agrees.
- These are the only top notebooks with a **hard** guard. Verdict: **safe-ish**. The margin is fixed, but there is little extra room.

**lightningv08** (code): no budget on ATLAS or on the 27B member. **Unsafe / unknown.**

**0.651 base and our v11a** (code). This is where our timeout most likely came from. Each point below was checked in `nb/v11a` and `research/kernels_2026-10-27/v11a_out/infer.py`:
- `atlas_scores` stops only by **raising** when a projection exceeds the budget. The projection is token-proportional ×1.1, checked every 100 rows after 50 rows. There is no per-batch deadline.
- Llama is loaded straight from the mount (`BL = mdir('/llama-3.1/','/8b/')`, no copy), which costs ~35 min (claim, omer).
- Both 8B members are split across the GPUs, so they run at about half the 0.663 speed (claim).
- Hypothesis (**est**): ATLAS sorts rows shortest-first, so a token-linear projection made early under-estimates the long-row tail. The budget leaves only ~40 min for everything outside infer.py, so a small mis-projection pushes the run past 9 h.
- Fix: adopt the 0.663 `atlas_scores` (hard deadline that keeps completed rows), the Llama copy and the `"dp"` 8B path.
- infer.py also has `--fake_test N` (score N pool passages as a stand-in test) and `--atlas_probe N`, made for timing runs.

## 2. What failed, and reproduction pitfalls

### Single-change public-LB results that did not help

Each row is the author's claim. The baseline score is in brackets.

| Base | Change | LB | Source |
|---|---|---|---|
| nlm-v27 (0.651) | blend in own five-backbone model | 0.648 | alexchilton |
| nlm-v27 (0.651) | swap q8v5 for Qwen3.5-27B NF4 member (0.642 alone) | 0.638 | alexchilton |
| nlm-v27 (0.651) | Qwen3.5-9B token-attention back on | 0.640 | alexchilton |
| own 5-backbone two-stage blend (nested CV **0.6865**) | plain heads on the board | 0.559-0.560 | alexchilton |
| same | + prompt clusters + centred heads / twin rule / twin + Hungarian | 0.567 / 0.588 / 0.579 | alexchilton |
| lightningv08 LoRA+AC (0.541), saddamaditya (0.542) | Hungarian inside test clusters | 0.524 / 0.525 | alexchilton |
| zbig v6 (0.552), stylegram (0.559) | within-Gemini gate; 4 intra-family gates | 0.552; 0.559 (no gain) | alexchilton |
| stylegram (0.556) | extra seed-11 LoRA24 | 0.553 | alexchilton |
| nishantdahal (0.592) | third two-stage Qwen2.5-7B member | 0.592 | alexchilton |
| own 0.597 | OLMo2 native24 member at log-prob weight 0.03; in-style seed-11 swap | 0.594; 0.597 | alexchilton |
| 0.651 recipe | Dhruv's ModernBERT-large member (offline +0.008) | 0.645 | omer |
| nlm-v27 | without Llama | **0.643** (Llama worth +0.008) | omer, alekseyosipenko |
| 0.651 recipe | SVM with quality-tiered pool weights (member +0.028 offline) | +0.001 / -0.001 / -0.003 | omer |
| 0.651 recipe | own Ministral-3-8B two-stage LoRA24 (OOF 0.605 alone) at w2 / w3 (offline +0.004/+0.006) | 0.641 / 0.635 | omer |
| 0.651 recipe | second ATLAS (Qwen3-4B bundle) at 10% after ATLAS-8B (offline +0.007) | 0.650 | omer |
| V31 + faster run | typography margin gate only (offline +0.0019) | 0.660; 0.661 with ATLAS 20% (0.663 without) | omer |
| V31 + faster run | ATLAS 30% | 0.663 (= 20%) | omer |
| zbig 0.664 | svc_w 6 | 0.659 | zbig |
| V31 + ATLAS 20% (0.663) | **Llama removed** | **0.658** | alekseyosipenko |
| V31 without Llama | + prvsiyan two-stage Qwen3-8B adapters (OOF 0.592) at Llama's weight 1.0 | 0.655 | alekseyosipenko |
| V31 without Llama | cluster threshold 0.15 -> 0.08 | 0.652 (-0.006) | alekseyosipenko |
| nlm-v27 without Llama | + own Qwen3-8B two-stage LoRA at 10% | 0.639 | alekseyosipenko |
| V31 family | a Claude specialist replacing family predictions (350-row external sample: **4 wrong->right, 68 right->wrong**) | not submitted | vasily61 |

The only small gains alexchilton reports are 0.552 -> 0.556 (a 4th LoRA member) and 0.592 -> 0.597 (prvsiyan's blend as one log-prob member). An independent run, *0.659: 0.660 Without Llama*, also lands at 0.659. So without Llama this family sits at about **0.658-0.659** (claim).

### Noise on identical code (alexchilton's own resubmissions, claim)

| Pipeline | Scores on identical code |
|---|---|
| nlm-v27 | 0.651, 0.651, 0.643, 0.651 |
| *Who am I?* | 0.645, 0.654 |
| V31 | 0.660, 0.660 |
| omer 0.663 | 0.662 |
| LoRA x AC | 0.549 vs page 0.547 |
| lora-instyle | 0.561 vs 0.563 |
| Teach + ATLAS | 0.606 vs 0.607 |

- The spread is **up to 0.009** on identical code. alekseyosipenko puts one passage at ~0.0006 and the noise floor at ~0.005.
- The whole 0.662 -> 0.664 ladder (Sinkhorn 0.30, Open Pool, svc_w 12) is **inside noise**.
- Forum, via alexchilton (GoPain, comment on discussion 753177): out-of-fold scoring on **2 answers per prompt** tracked the board with correlation **0.91**, against **0.13** when scoring all 18 per prompt (15 submissions).

### Reproduction pitfalls

1. **avazaku's Llama asset cache is no longer readable** (since 2026-10-29). The original 0.663 code (`assert 'name-that-llm-verified-llama-asset-cache' in BL`) fails. Fixes in use:
   - alexchilton's public `alexchilton/llama-3-1-8b-base`: omer says all 4 shard hashes plus config, tokenizer and index match;
   - Meta's `metaresearch/llama-3.1` (Transformers, 8b, v2) after accepting the licence, which needs only the assert change;
   - our own kernel-output weights.
   The 0.664 notebooks also assert the shard sizes (`_SHARDS`).
2. **Pull the scored version, not the latest.** judith's latest is Sinkhorn 0.60 + typography 1.2 and has no score; the 0.664 is version 2 (Sinkhorn 0.30). The version ids lightningv08 cites are 363008892 (V31 v2) and 363834272 (judith v2).
3. **Gated or region-blocked Llama.** alekseyosipenko: "not available in every region". Without it, expect -0.005 to -0.008.
4. **The Qwen3.5 token-attention head** is silently skipped when its backbone is not attached (`NTL_DROP_Q35='0'` changes nothing there). Turning it on costs 1.7-2.7 h and scored 0.640 vs 0.651.
5. **`!python comp_stack.py | grep ...` hides comp_stack failures**, because the exit code comes from grep (leoprovorov). The stack then logs `COMP SKIPPED` and the run continues without the 15% member.
6. **Dataset changes**:
   - the Open Pool has a 193k and a 198k version;
   - the OOF bundle we used earlier is gone (our commit 77131e7);
   - lightningv08's numeric cache is version-pinned (`artifact_hub_version 17`, base version 15).

## 3. EDA insights

### Error map (alekseyosipenko)

Setup: 4-view n-gram LinearSVC on train + pool_v1, StratifiedGroupKFold(5) by prompt, leakage guard (claim).
- OOF balanced accuracy is **0.435**.
- Errors split **about half inside a family** (right family, wrong model) and **half wrong family**.
- Every class already has **1,100-2,700 pool passages**, and recall is not correlated with that count (r ≈ 0).
- The **6 classes that Kaggle Benchmarks cannot generate** have mean recall **~0.28**, against **~0.51** for the 12 it can. The six are claude-fable-5, gemini-3.7-flash, glm-5.3, glm-5.3-flash, kimi-k3 and qwen3.8-max.
- Generating data for only the 12 listed classes pulls predictions towards them. The author suggests keeping the class balance or using such data only inside a family model.

### Sibling separability (fingerprints-inside-a-family, claim)

Setup: one binary char 1-4-gram SVC per pair, prompt-grouped CV on train only.
- Sibling pairs score **0.53-0.74**; cross-family reference pairs score **0.93-0.97**.
- **glm-5.3 vs glm-5.3-flash is barely above chance.** Its top train tokens keep their direction on the pool in **< 50%** of cases, so it has no reliable n-gram fingerprint. This matches our own measurement of 0.56.
- Pairs that do separate differ in **habits, not topic**, and those differences keep their direction on the pool in **68-82%** of cases:
  - gpt-5.6-sol vs terra: quotation marks and first-person narration vs parentheses and hedges;
  - gemini-3.6 vs 3.8-flash: article choice.
- Typography signatures by family:
  - Claude and GLM use em dashes and contractions;
  - Gemini almost never uses em dashes, colons or hyphens;
  - GPT uses curly apostrophes, semicolons and curly quotes.
  This overlap explains errors such as claude-fable-5 <-> GLM.
- The pairs listed as hardest: gemini-3.7 vs 3.8-flash, gpt-5.6-sol vs terra, glm-5.3 vs flash, gpt-5.6-luna vs terra, gemini-3.6 vs 3.8-flash, opus-4-8 vs sonnet-5, opus-4-6 vs opus-4-8, gpt-5.5 vs terra.

### prvsiyan "who sounds like whom" (claim)

- A hashed n-gram MLP scores **0.274** raw and **0.265** after a learned confusion map, on a 30-prompt holdout. The map did not help.
- The strongest OOF confusions are gemini-3.7 <-> 3.8-flash and glm-5.3 <-> glm-5.3-flash.

### Without Llama, what transfers (alekseyosipenko, claim)

- The LB table is in section 2. Offline (CPU, prvsiyan's OOF): threshold 0.15 finds only ~3/4 of same-prompt pairs and leaves "a hundred-odd" singletons; 0.08 finds almost all at nearly the same precision. A hard Hungarian on those groups gains a lot offline.
- On the LB the lower threshold lost 0.006. The likely reason: the recipes use groups softly (twin heads 0.35, Gaussian 0.44-0.70), and those weights were tuned together with 0.15.
- Lesson: an offline check must copy the recipe's own code path. infer.py has `--sim <folds>` and `--sim_k lo-hi`, which hold out train prompts and keep 'lo-hi' passages per prompt.

### Does the Response Corpus help? (olegzholobov, 24,025 answers; grouped CV, claim)

| Variant | 4-view SVC BA | Change |
|---|---|---|
| A: train + pool | 0.4349 | baseline |
| B: + clean paragraphs | 0.4108 | -0.024 |
| C: + flattened corpus | 0.4219 | -0.013 |
| D: C with external weight 0.3 | 0.4241 | -0.011 |

- One standard error is ≈ 0.01. The corpus was written under other system prompts, tasks and decoding settings: chat, lists and code.
- The author suggests it may still help as stage-1 pre-training in a two-stage recipe, but adding it directly to a classifier hurts.

### The prompt-recovery trap (yangkuangou)

**What it is.** In train, 150 prompts each have at most one passage per author; all but one have the full 18. That makes "recover the prompt groups, then give each author at most one passage per group (Hungarian)" very tempting, and prompt-held-out CV rewards it.

**Why it fails.** It does not transfer (the author's own submissions, claim):

| Method | Development OOF BA | Public LB |
|---|---|---|
| SVC, independent | 0.327 | 0.336 |
| SVC + recovered-group assignment | 0.419 | **0.308** |
| relative/style LR + assignment | 0.501 | 0.351 |

The author's inference is that hidden batches break the assumptions behind the assignment. Candidates are repeated authors within a prompt, near-topic prompts merged together, missing authors, or other shift.

**Why it matters.** Five independent sources agree that hard assignment loses on the board:
- alexchilton: -0.009 to -0.017;
- yangkuangou: -0.028;
- alekseyosipenko: lower threshold -0.006;
- the 0.663 family runs with `--post 0`, which disables both the em-dash guard and the dense-cluster Hungarian;
- our own v2 result.

Soft use of groups (twin heads, Gaussian head) at the tuned 0.15 threshold is what survives. CV with all 18 passages per prompt is near-oracle and misleading.

### Test-set structure claims

- **~4,000 hidden rows** (omer, alexchilton).
- The public LB is **~40% (~1,600 passages)** (alekseyosipenko).
- The organisers state that **class counts are unequal and public/private mixes differ**. This is quoted in the V31 code comment and by yangkuangou. Hence V31's damped Sinkhorn towards 0.65 × predicted marginal + 0.35 × train prior.
- **Few passages per prompt in the hidden test** (inferred, not stated). Evidence: every assignment trick loses, and 2-per-prompt OOF correlates 0.91 with the LB.
- No notebook publishes recovered test cluster sizes. infer.py logs them (`test clusters … sizes`) only in the scoring run, whose logs we cannot see.

### Leaderboard snapshot (cosmorin, 2026-10-25, claim)

- #1 Hod Carriers **0.703**; #2 Lian Matsuo 0.689; #3 Bootstrapping 0.688.
- Gold cutoff ≈ 0.66 (16th); the top 13 run from 0.703 to 0.664.
- The rest is speculation. cosmorin wrongly calls the metric "log-loss family"; it is balanced accuracy.

## 4. Opportunities for us

These are cheap ideas that are not yet in the 0.664 stack and have evidence in these notebooks.

1. **Rebase on the 0.663/0.664 code, not the 0.651 base.** This is the runtime fix for v11a/v11b. The 0.663 version adds:
   - the hard ATLAS deadline that keeps completed rows;
   - the Llama copy to /tmp (saves ~33 min);
   - the dual-GPU `"dp"` 8B members (claimed to match upstream log-probs within 7.6e-6);
   - `FAST_COMMIT` + `KAGGLE_IS_COMPETITION_RERUN` (a ~5-10 min commit instead of hours, which saves weekly GPU quota).
   Any new member we add (Gemma, our Llama two-stage) then only shrinks ATLAS coverage and does not cause a timeout.
   Before submitting, measure with `--fake_test 4000 --atlas_probe 200`.
2. **Fit the blend weights; do not hand-set them.** The 0.664 blend uses round hand weights:
   - q14v5 3, q8v5 2, ll8v5 1, n14b 2.5, q25_14 1, deb 1;
   - svc 1.5, tfidf 0.25, distort 0.10;
   - comp 15%, ATLAS 20%.
   zbig shows the SVC weight is effectively uncalibrated: log-softmax of flat SVM margins, and svc_w 1.5-20 gives the same LB. Our E062 NNLS stacking gave +0.013 OOF over hand weights (4/5 folds). That is the largest cheap, untried lever. Caveats:
   - the member OOF must use the public v5 fold adapters' own folds (our fold-split check kernel);
   - evaluate with **1-3 passages per prompt** (`--sim_k`), not full prompts.
3. **Free knobs to test with the 2-per-prompt sim, CPU only from cached log-probs (`--save_lp`).**
   - Sinkhorn strength: 1.0 / 0.45 / 0.30 / 0.0 are all in public use. The LB evidence is noise-level: 0.651 -> soft 0.5 gave 0.653; V31 0.660; judith 0.30 gave 0.664.
   - atlas_w: 0.20 ≈ 0.30.
   - Typography scale: 1.5 vs 1.2.
   - Our E053 hedge (full vs 0.5) still applies. Use one final submission at 0.30-0.45 and one at a different strength.
4. **Pre-fit the SVC4 offline and load it**, as lightningv08 does with `svc4_prefit`, instead of fitting on 124k-200k rows inside the 9 h run. The fit is part of the ~8,426 s fixed overhead seen on a 3-row run.
   - The saved time goes to ATLAS coverage. ATLAS is worth **+0.008** on 0.651 (ATLAS-off 0.643, claim), and the 0.663 notes say ATLAS uses all the time it gets.
   - Combine it with the **Open Pool** swap (SVC CV 0.4805 -> 0.5008, claim; an LB change within noise).
5. **Keep Llama.** It is worth +0.005 to +0.008 (two independent LB A/Bs). Our own Llama-3.1-8B two-stage LoRA would be a second Llama-family view; no notebook has tried it.
6. **Spend specialist effort only where siblings actually separate**: GPT sol/terra/luna and Gemini 3.6/3.8, which differ in habits that hold on the pool 68-82% of the time.
   - Skip glm-5.3 vs flash (chance level).
   - Do not add the Response Corpus directly to classifiers (-0.011 to -0.024).
   - Do not replace family predictions with a specialist (vasily: 4 fixed, 68 broken). V31-style soft gates that keep family mass are the safe form.
7. **Do not spend submissions on**:
   - Hungarian or assignment of any kind, or a lower cluster threshold;
   - a second ATLAS, ModernBERT, Qwen3.5 token-attention or Qwen3.5-27B;
   - adding a strong member at a large weight. Ministral at 0.605 OOF lost 0.010-0.016; prvsiyan's adapters lost 0.003.
   - Single-change gains below ~0.01 cannot be read from one public submission.

## Files

- Pulled notebooks: `research/kernels_2026-11-01/<owner>__<slug>/` (16 folders, .ipynb + kernel-metadata.json).
- Key code to reuse: the omerfarukmerey `infer.py` (`%%writefile` cell 6) has `atlas_scores` with the hard deadline and `l24_predict` with the `"dp"` path. Its setup cell has the Llama copy and `FAST_COMMIT`.
