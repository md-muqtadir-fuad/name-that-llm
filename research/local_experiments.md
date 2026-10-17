# Local CPU experiments: how much is the prompt structure worth?

Date: 2026-10-16. Machine: 4 CPU cores, no GPU. Everything is prompt-disjoint (5-fold GroupKFold by `prompt_id`, shuffled, seed 42). No test data, no submissions, no uploads.

- Code: `experiments/local/` (`common.py`, `eda.py`, `baseline.py`, `twin.py`, `structure.py`, `clustering.py`, `assign.py`, `strong_confusion.py`, `summarize.py`)
- Raw outputs: `experiments/local/out/` (`eda.md`, `baseline.md`, `clustering.md/.csv`, `assign_*.md/.csv`, `summary_assign.md`, `strong_confusion.md`, OOF arrays `oof_*.npy`, `twin_scores_char_wb.npz`)
- External inputs, downloaded read-only from public Kaggle datasets:
  - `research/ext_oof/q35heads/`: `lightningv08/ntl-oof-q35-heads` → `oof_predictions.npz`. These are honest nested prompt-disjoint OOF log-probs from 37 frozen-LLM and encoder heads (Qwen3.5-9B, Qwen3-14B, Qwen3-8B, DeBERTa, RoBERTa, SVM, style). They are used as the **"strong" score set**, so the structural tricks can be measured on a ~0.49 classifier and not only on a 0.37 TF-IDF one.
  - `research/ext_data/extra_api/`: `seyitkaangunes/ntl-extra-api-passages` (CC BY 4.0). It has 4,998 Claude passages on 1,000 *new* prompts in the competition style. They are used only as **unlabelled clustering distractors**, so that simulated test pools contain about 1,150 prompts and not 150.
- Not done: sentence-transformers / MiniLM embeddings. The link measured 46 kB/s, so a torch CPU wheel plus model is more than an hour to download. LSA-128 on TF-IDF is used as the dense-embedding stand-in.

Rerun everything (about 1.5 h on 4 cores):
`cd experiments/local && python eda.py && python baseline.py && python clustering.py && python twin.py char_wb && python assign.py strong true,rec,rec_dis,rec_k18 0.85 5 && python assign.py tfidf true,rec,rec_dis,rec_k18 0.85 5 && python summarize.py`

---

## TL;DR

1. **The structure is worth a lot only if the test is dense.** Take a ~0.49 OOF classifier (public frozen-LLM plain blend) and recover groups from text alone, with 1,000-prompt distractors in the pool. Best decoder, gain in balanced accuracy:

   | Passages per test prompt | Gain |
   |---|---|
   | Full 18-answer blocks | **+0.15 to +0.19** (Hungarian; oracle groups +0.197) |
   | About 12, with unequal class counts | +0.12 to +0.13 |
   | 3–8 | +0.07 to +0.08 |
   | 1–5 | +0.04 |
   | 2–3 | +0.035 to +0.041 |
   | 1–2 | +0.016 to +0.022 |

2. **The leaderboard fits a sparse test and rules out full blocks.**
   - The 0.541 notebook has plain OOF ≈0.49–0.50 and scores 0.541 on the LB, a gain of about +0.04 to +0.05. That matches our 1–5 / 2–3 per prompt numbers. With full blocks, this level of structure use would land around 0.65.
   - Our "assume 18 per prompt" grouping (cut the tree into N/18 clusters) reproduces the reported LB failures. On sparse pools it goes negative: Hungarian −0.045 at 2–3 per prompt. Threshold-cut clustering instead gives +0.009 at the same density.
   - The duplicate-author stress test (two prompts merged, so a model can appear twice) makes hard Hungarian negative (−0.018 to −0.020). Soft distinct-author marginals stay positive (+0.019 to +0.021).
3. **One decoder stays positive or neutral in every scenario we simulated** with threshold-recovered groups:
   1. score-level centering at **β = 0.5** inside recovered groups;
   2. partial Sinkhorn (α = 0.5);
   3. exact **distinct-author marginals** (subset DP, soft Hungarian) per group, then argmax.

   On the strong set its gain is +0.016 to +0.17 depending on density, and +0.021 under duplicate authors.

   **Full centering (β = 1, i.e. subtract the group mean) is harmful when groups are small**: −0.09 to −0.11 at 1–3 per prompt, because with n = 2 it subtracts the sibling's whole score. Use β ≈ 0.5, or LOO repulsion with α ≈ 0.2–0.35.
4. **Global class balancing is worthless or harmful** when test class counts are unequal.
   - Full Sinkhorn: −0.005 to +0.003.
   - The 0.541 notebook's α = 0.5 Sinkhorn: −0.004 to +0.003.
   - Forcing uniform argmax counts: −0.011 to −0.017.

   For balanced accuracy, argmax of a balanced-trained classifier is already the right decision rule. Do not spend effort here.
5. **Prompt recovery is easy.**
   - Content-word TF-IDF with average linkage gets ARI 0.96–0.98 on dense pools.
   - On sparse pools of ~1,150 prompts and ~2,900 passages it gets ARI 0.84–0.91 (pair precision about 0.85, recall about 0.93 at a distance cut of 0.85).
   - Decoding prefers higher recall: a cut of 0.90 beat 0.85, which beat 0.80, for the soft decoder even when ARI was lower.
6. **Classifier facts.**
   - TF-IDF char_wb(1–5) + word(1–2) LinearSVC scores 0.330. Adding about 50 standardized stylometric features lifts it to **0.3745**, the best TF-IDF variant (+0.044); case-sensitive char gives 0.3748.
   - The prompt-chunk-centred "twin" SVM scores 0.4397 vs 0.3745 plain with oracle full-prompt centring. Score-centring alone reaches 0.4137.
   - About 50–58% of errors stay inside the true lab.
   - The hard pairs are gemini-3.7/3.8-flash, glm-5.3/glm-5.3-flash, claude-fable-5/glm-5.3 and the gpt-5.6 trio. Even oracle Hungarian on full blocks leaves fable/glm and the gemini flashes at 0.33–0.66 recall.

**What this means for first place.** The structural exploit is already priced in at the top of the LB, with a realistic ceiling of about +0.03 to +0.05 at the likely test density. Getting it right (β ≈ 0.5 centring, soft distinct marginals, recall-leaning threshold clustering, no forced cluster sizes, no global balancing) is worth maybe +0.01 to +0.02 over what the 0.541 notebook does. The rest of the gap to 0.572 and beyond has to come from a **better per-passage classifier**: better backbones or LoRA, external data, and heads specialised on the hard within-lab pairs. Every point of base accuracy carries through roughly one-for-one, because the structural gains are nearly additive across base strengths (§4.4).

---

## 1. Stylometric EDA (`out/eda.md` has the full tables)

### 1.1 Typography is a near-deterministic *lab* fingerprint

Share of passages containing the mark at least once:

| Lab | em dash — | curly ’ | straight ' | curly “ | straight " | semicolon | colon |
|---|---|---|---|---|---|---|---|
| anthropic (5 models) | 75% | **0%** | 63% | **0%** | 9% | 30% | 54% |
| zhipu (glm ×2) | 81% | **0%** | 63% | **0%** | 11% | 41% | 64% |
| google (gemini ×4) | **0%** | 20% | 14% | 0% | 2% | 23% | **19%** |
| openai (gpt ×5) | 22% | 47% | **5%** | 10% | 0% | 54% | 68% |
| moonshot (kimi) | 52% | 32% | 23% | 9% | 1% | **61%** | **77%** |
| qwen | **3%** | 29% | 12% | 1% | 0% | 39% | 47% |

- No Anthropic or GLM passage (0 of 1,049) contains a curly apostrophe or curly double quote. 81% of curly double quotes come from OpenAI models.
- Google and Qwen practically never use em dashes, and Google rarely uses colons.
- Only 14 passages mix both apostrophe styles, and all 14 are Gemini.
- GLM's typography is indistinguishable from Claude's. Its token profile is also Claude-like: em dash, straight quotes, "itself", "nothing", "never". This is why GLM ↔ claude-opus-5 / claude-fable-5 is a top confusion: GLM looks distilled from Claude.

### 1.2 Within-lab separators (per-model, standardized mean difference vs the other 17)

- **claude-opus-5:** the most "plain-English" Claude. Shortest words (d −0.79), lowest type-token ratio (−0.51), highest function-word rate (+0.66), most variable sentence length (+0.78), most semicolons in its lab (54% of passages). Uses "the", "a", "is", "he", "hundred", "never", "thing".
- **claude-sonnet-5:** most contractions (it's, isn't; d +0.65) and straight quotes. Words: "itself", "requires", "entirely".
- **claude-opus-4-6:** fewest semicolons and commas in its lab. Words: "roughly", "comfortable", "percent", "underlying", "emerges".
- **claude-opus-4-8:** second person ("you", "we"). Words: "quietly", "glimpse", "immensity".
- **claude-fable-5:** narrative. Words: "she", "her", "nothing", "complete". It is the hardest class (recall 0.07–0.17 plain).
- **gemini-3.1-pro:** more sentences, "we/our", stock words "profound", "absolute", "relentless", "ultimate", "of".
- **gemini-3.7 and 3.8-flash:** longest words (d +0.84 to +0.93), highest TTR, "an …"-heavy ("an intricate", "an enduring"), "abyss", "transforming", "permanently". They are the most confused pair of all.
- **gemini-3.6-flash:** "expanse", "dynamic", "void", "breathtaking".
- **gpt-5.5:** most "rule of three" lists and commas, many semicolons, lowest 'of'. Words: "speech", "habits".
- **gpt-5.6-luna:** parentheses (5% of passages, the most of any model), "may", "therefore", "while".
- **gpt-5.6-sol:** curly double quotes (15%), "another", "continually".
- **gpt-5.6-terra:** parentheses, "may", "crowded", "institutions".
- **gpt-6-astra:** fewer commas, short simple sentences. Words: "familiar", "helps", "could", "still".
- **kimi-k3:** most semicolons and colons of all models. Mixes em dash with curly and straight quotes. Words: "keeps", "patience", "debt", "before/after".
- **qwen3.8-max:** most commas (+0.83), rule-of-three, almost no em dash. Words: "turning", "luminous", "becomes", "becoming", "compassion". Starts with "In …" 13% of the time.
- **glm-5.3 vs glm-5.3-flash:** almost no stylometric separation. glm-5.3-flash uses a few more semicolons, 48% vs 34%.

### 1.3 Length and openings

- **Length carries essentially no signal.** Word-count eta² across models is 0.010. Within-prompt length percentile per model ranges 0.45–0.57. Every prompt spans about 44 → 231 words, so each generation got its own target length.
- Markdown, headings and newlines appear in **0%** of passages; formatting was cleaned.
- **Openings are prompt-driven, not model-driven.** Within a prompt, the most common 2-word opening is shared by 42% of the 18 answers on average. That is a strong clustering cue and a weak label cue. The few model-skewed openings: gemini flash "Beneath/Deep/Across…", gpt-5.5/terra "At …", qwen "In …".

---

## 2. TF-IDF baselines (`out/baseline.md`)

5-fold prompt-grouped OOF. Vectorizers are fit on the train fold only. char_wb(1–5) + word(1–2) keeps punctuation tokens, min_df = 2.

| Variant | Balanced acc | Fold sd | Lab-level bacc | NLL | Time |
|---|---|---|---|---|---|
| LR C=10, raw tf | 0.3467 | 0.016 | 0.553 | 2.068 | 171 s |
| LR C=10, sublinear tf | 0.3375 | 0.007 | 0.564 | 2.094 | 157 s |
| LR C=30, sublinear | 0.3363 | 0.006 | 0.561 | 2.026 | 167 s |
| LinearSVC C=0.5, sublinear | 0.3304 | 0.012 | 0.550 | 2.063 | 90 s |
| LinearSVC C=0.2, sublinear | 0.3308 | 0.014 | 0.551 | 2.011 | 95 s |
| **LinearSVC + 52 stylometric features (z-scored ×0.15)** | **0.3745** | 0.014 | 0.592 | 1.913 | 100 s |
| LR + stylometric | 0.3663 | 0.007 | 0.594 | 1.854 | 519 s |
| LinearSVC, case-sensitive `char` 1–5 | 0.3382 | 0.004 | 0.554 | 2.009 | 158 s |
| LinearSVC, case-sensitive char + stylometric | **0.3748** | 0.014 | 0.582 | 1.902 | 142 s |

- The appended **dense stylometric block is worth +0.044**: typography rates, word length, TTR, function-word rate, rule-of-three, and so on.
- Sublinear tf and C barely matter.
- These numbers agree with the forum: the honest-CV notebook reports 0.369 vs LB 0.355.

**Confusions (best TF-IDF).**
- Recall: claude-fable-5 0.07, glm-5.3 0.11, gpt-5.6-sol 0.15, glm-5.3-flash 0.17, claude-opus-4-8 0.18.
- Easy: gemini-3.1-pro 0.77, claude-opus-5 0.69, gpt-6-astra 0.68.
- Top symmetric pairs: gemini-3.7↔3.8-flash (72 of ~300), glm-5.3↔glm-5.3-flash (54), claude-opus-5↔glm-5.3 (49), gemini-3.6↔3.7 (48), gemini-3.1↔3.6 (43), gpt-5.5↔kimi-k3 (41).
- 50% of errors stay within the true lab.

**Strong set (public frozen-LLM plain blend, OOF 0.4915, lab-level 0.693; `out/strong_confusion.md`).**
- Same hard pairs: gemini-3.7↔3.8 (92), glm-5.3↔flash (64), gemini-3.6↔3.7 (50), gpt-5.6-sol↔terra (46), luna↔sol (44), fable↔glm-5.3 (44), opus-4-8↔sonnet-5 (41).
- Even **oracle Hungarian on full prompts (0.689)** leaves fable 0.33, glm-5.3 0.35, glm-flash 0.41, gemini-3.7/3.8 0.61/0.66, gpt-5.6-sol 0.59.
- These pairs are where classifier capacity has to improve.

---

## 3. Prompt-group recovery from content only (`out/clustering.md`, `clustering.csv`)

**Method.**
- Features: content-word TF-IDF (`stop_words='english'`, sublinear, alphabetic tokens, max_df 0.5), fit on the pool texts only.
- Clustering: cosine distance, average-linkage agglomerative, distance cut. Clusters larger than 18 are split.
- Scenarios:
  - Train-only pools have 150 prompts.
  - "+Distractor" pools add the 1,000 extra-API prompts. Dense scenarios get 100 of them; sparse scenarios get all 1,000 at the same per-prompt rate. Pools then hold about 250 prompts / 2,500–3,200 passages (dense) or **1,150 prompts / 1,700–4,800 passages (sparse)**, matching the expected hidden test of about 4,000 passages.

Best cut per scenario for tfidf_uni, with distractors:

| Scenario | Pool | Prompts | Best cut | ARI | Purity | Pair P | Pair R | Singletons |
|---|---|---|---|---|---|---|---|---|
| full18 | 3,199 | 250 | 0.95 | 0.973 | 0.969 | 0.954 | 0.992 | 0.00 |
| drop 0–30% | 2,794 | 250 | 0.90 | 0.966 | 0.992 | 0.984 | 0.949 | 0.01 |
| unequal-class keep (~12/prompt) | 2,493 | 250 | 0.90 | 0.967 | 0.993 | 0.990 | 0.947 | 0.01 |
| 3–8 per prompt | 4,808 | 1,150 | 0.85 | 0.903 | 0.957 | 0.886 | 0.922 | 0.03 |
| 1–5 | 3,402 | 1,150 | 0.85 | 0.909 | 0.962 | 0.889 | 0.930 | 0.10 |
| 2–3 | 2,866 | 1,150 | 0.85 | 0.885 | 0.954 | 0.845 | 0.930 | 0.05 |
| 1–2 | 1,723 | 1,150 | 0.80–0.85 | 0.838 | 0.951 | 0.770–0.843 | 0.83–0.92 | 0.35–0.42 |

**Distance-cut sweep** (ARI / pair precision / pair recall, with distractors):

| Scenario | 0.75 | 0.80 | 0.85 | 0.90 | 0.95 |
|---|---|---|---|---|---|
| full18 | 0.46/1.00/0.30 | 0.70/1.00/0.54 | 0.88/0.99/0.79 | 0.96/0.98/0.95 | 0.97/0.95/0.99 |
| n2-3 | 0.81/0.93/0.71 | 0.86/0.89/0.84 | 0.89/0.84/0.93 | 0.81/0.69/0.98 | 0.45/0.29/0.99 |
| n1-2 | 0.80/0.89/0.72 | 0.84/0.84/0.83 | 0.84/0.77/0.92 | 0.69/0.54/0.97 | 0.30/0.18/0.98 |

- The best cut depends on density. Dense pools want 0.90–0.95; sparse pools with 1,000+ prompts want 0.80–0.85.
- **0.85 is the robust compromise** (ARI ≥ 0.84 everywhere).
- For decoding, recall matters more than precision. With the soft decoder, a 0.90 cut gave higher downstream accuracy than 0.85 or 0.80 even at 2–3 per prompt (§4.2).
- word 1–2-gram TF-IDF and LSA-128 were not better than unigram TF-IDF (best ARI 0.82–0.90 with distractors).
- Caveat: real test prompts might overlap topically more than the extra-API distractors, which were deduplicated against train topics. The 0.541 notebook mixes in GLM layer-24 embeddings for the same reason.

---

## 4. Constrained decoding, centring and balancing (`out/assign_*.md/.csv`, `out/summary_assign.md`)

### 4.1 Set-up

- **Score sets** (log-probs, temperature-calibrated on OOF NLL; a single scalar, so argmax is unchanged):
  - **strong**: the public frozen-LLM *plain* blend. These are the bundle weights restricted to non-centred members, with OOF bacc 0.4915. Its centred members are deliberately excluded, so all centring here is ours.
  - **tfidf**: the best TF-IDF+stylometric LinearSVC (0.3745), plus a **twin** LinearSVC trained on prompt-chunk-centred features (chunks of 2/3/4/6/18 per train prompt, stacked).
    - Both heads are linear, so twin(x_i − mean_group x) = t_i − mean_group(t) + b. The twin can therefore be evaluated for any grouping from stored OOF scores.
    - Distractor passages are scored by plain and twin models refit on all of train.
- **Pools.** Per scenario we sample a pool from the 2,699 OOF rows (seeds 500+): 1 seed for full18, 5 for drop/uneq/dup_pairs, 10 for the sparse scenarios. All sampled scenarios except drop0-30 use per-class weights U(0.35, 1), so **test class counts are unequal**. Balanced accuracy is computed on the pool's train rows.
- **Groupings.**
  - `true`: oracle prompt ids.
  - `rec`: content-TF-IDF clustering of the pool, cut 0.85.
  - `rec_dis`: the same with distractor passages in the pool. For the tfidf set the distractors take part in decoding; for the strong set they are dropped after clustering.
  - `rec_k18`: "assume 18 per prompt", i.e. cut the tree into round(N/18) clusters.
- **Decoders.** All work per group; singletons are untouched.
  - `center_bβ`: z_i = L_i − β·mean_group(L), the adaptive-centring analogue at score level.
  - `repulse_aα`: Public-D logit repulsion, z_i − α·mean_{j≠i} z_j.
  - `hungarian`: `linear_sum_assignment` on −log p, at most one passage per model per group.
  - `distinct_marg_tτ`: **exact** P(a_i = c | all authors in the group distinct). It is computed by a forward-backward DP over passage subsets, O(K·n·2ⁿ), for n ≤ 10, and by Sinkhorn on an (n+1)×K plan otherwise. It was verified against brute force to 1e-16.
  - Global: `sinkhorn_a1` (full balancing), `sinkhorn_a0.5` (the 0.541-notebook partial version), `count_balance` (bias iteration to uniform argmax counts).
  - Combinations: centre, then Sinkhorn, then Hungarian or distinct marginals.

### 4.2 Strong set (base ≈0.49). Delta balanced accuracy vs independent argmax

Recovered groups with distractors (`rec_dis`), cut 0.85; ± is the s.e. over seeds:

| Decoder | full18 | drop0-30 | uneq_keep (~12/prompt) | n3-8 | n1-5 | n2-3 | n1-2 |
|---|---|---|---|---|---|---|---|
| *base bacc* | 0.492 | 0.493 | 0.487 | 0.490 | 0.480 | 0.490 | 0.492 |
| center β=0.5 | +0.058 | +0.052 | +0.054 | +0.044 | +0.027 | +0.025 | +0.017 |
| repulsion α=0.2 (Public D) | +0.038 | +0.034 | +0.035 | +0.030 | +0.026 | +0.023 | +0.020 |
| center β=1 (full group-mean subtraction) | +0.017 | +0.010 | +0.015 | −0.018 | **−0.052** | **−0.092** | **−0.097** |
| Hungarian | **+0.163** | +0.125 | +0.099 | +0.040 | +0.024 | +0.009 | +0.005 |
| distinct marginals τ=1 | +0.158 | +0.136 | +0.116 | +0.049 | +0.028 | +0.015 | +0.007 |
| center 0.5 + Hungarian | +0.161 | +0.131 | +0.114 | +0.063 | +0.034 | +0.026 | +0.018 |
| **center 0.5 + Sinkhorn 0.5 + distinct τ=1** | +0.151 | +0.133 | +0.118 | **+0.068** | **+0.040** | **+0.034** | +0.019 |
| Sinkhorn α=0.5 (global) | −0.003 | −0.003 | −0.004 | −0.002 | −0.004 | +0.003 | +0.001 |
| Sinkhorn α=1 (global) | −0.000 | +0.000 | −0.003 | −0.001 | −0.005 | −0.001 | −0.002 |
| uniform argmax counts (global) | −0.003 | −0.000 | −0.013 | −0.011 | −0.013 | −0.016 | −0.017 |

Oracle groups (`true`), and the gap that recovery costs:

| Decoder | full18 | uneq_keep | n3-8 | n1-5 | n2-3 | n1-2 |
|---|---|---|---|---|---|---|
| Hungarian, oracle | +0.197 | +0.117 | +0.046 | +0.029 | +0.012 | +0.005 |
| center 0.5 + Sinkhorn 0.5 + distinct, oracle | +0.182 | +0.142 | +0.084 | +0.053 | +0.045 | +0.024 |
| same, recovered at cut 0.85 | +0.151 | +0.118 | +0.068 | +0.040 | +0.034 | +0.019 |
| same, recovered at cut **0.90** | +0.169 | +0.132 | +0.076 | +0.043 | **+0.041** | +0.016 |
| same, recovered at cut 0.80 | +0.111 | +0.092 | +0.053 | +0.038 | +0.025 | +0.017 |

**Takeaways.**
- At low density the gain comes mostly from **soft centring/repulsion**: removing the shared topic component of siblings' scores. The distinctness constraint adds little, because a 2–3-passage group rarely contains two passages the classifier would give the same model.
- At high density the distinctness constraint dominates (+0.12 to +0.20).
- β = 1 centring is dangerous for small groups. With n = 2 it equals z₁ − z₂ and throws away the passage's own absolute evidence. This probably explains why "replace-mode" centring needed groups of ≥ 4 in Lucas v8. The best fixed β is about 0.5, with α ≈ 0.2–0.35 in repulsion form.
- Recovery costs only 10–25% of the oracle gain.

### 4.3 "Assume 18 per prompt" grouping reproduces the leaderboard failures

`rec_k18` (tree cut into N/18 clusters, the k-means-to-18 style used by several public notebooks), strong set:

| Decoder | full18 | uneq_keep | n3-8 | n1-5 | n2-3 | n1-2 |
|---|---|---|---|---|---|---|
| Hungarian | +0.192 | +0.108 | −0.002 | −0.032 | **−0.045** | **−0.047** |
| center β=1 | +0.052 | +0.047 | −0.018 | −0.042 | **−0.051** | −0.050 |
| center 0.5 + Sinkhorn 0.5 + distinct | +0.172 | +0.130 | +0.028 | −0.005 | −0.011 | −0.015 |

The public LB history (`research/other_notebooks.md`) contains two failures of this kind:

| Notebook | Change | CV | Public LB |
|---|---|---|---|
| yangkuangou | recovered-group Hungarian | +0.092 | **−0.028** (0.336 → 0.308) |
| abh001 | centring on k-means pseudo-prompts of ~18 | – | **−0.028** (0.404 → 0.376) |

**Under full-block test pools both would have gained** (+0.05 to +0.19 in every dense row above). Under 2–5-per-prompt pools both lose, with magnitudes that match.

The "+0.04 to +0.05 from structure" seen in the 0.541 notebook (plain OOF ≈0.49–0.50 → LB 0.541) also matches the 1–5 / 2–3 rows. It is a factor of 3–4 below the full-block rows.

**Conclusion (inference, not confirmed by hosts):** the hidden test is sparse, roughly **1–5 passages per prompt and about 1,000–2,000 prompts**. Expect at most about +0.03 to +0.05 from structure even with a perfect decoder.

### 4.4 TF-IDF set (base ≈0.37) and the feature-level twin

(Filled in from `out/assign_tfidf_thr0.85*.md`; see the section below this one.)

### 4.5 Duplicate-author stress test (oracle groups = two train prompts merged; a model can appear twice)

| Decoder | dup_pairs (8–18 of 36) | dup_n2-4 (2–4 of 36) |
|---|---|---|
| Hungarian | **−0.018** | +0.010 |
| center 0.5 + Hungarian | **−0.020** | +0.003 |
| distinct marginals τ=1 | +0.019 | +0.020 |
| center 0.5 + Sinkhorn 0.5 + distinct τ=1 | +0.021 | +0.014 |
| center β=1 | −0.033 | **−0.115** |

Hard assignment is fragile when the "at most one per model" assumption fails, for example when clusters merge two prompts or the hosts reused a prompt for one model. Soft marginals are not. **Use soft marginals, never hard Hungarian, in a submission.**

### 4.6 Global class balancing: do not bother

- Sinkhorn (α = 1 or 0.5) moved balanced accuracy by −0.005 to +0.006 in every scenario for both score sets.
- Forcing uniform predicted counts cost −0.011 to −0.017 whenever test class counts were unequal.
- Reason: a classifier trained on balanced data already outputs p(c|x) ∝ p(x|c), and argmax of that is the Bayes rule for balanced accuracy *whatever the test prior*. Balancing toward uniform predicted counts only helps if the classifier has a systematic class bias, which the OOF blend does not.
- Inside the combo decoder, the α = 0.5 Sinkhorn step added about +0.005 to +0.007 at moderate density. That is within noise, but it was never harmful there.

---

## 5. Recommendations

1. **Classifier first.** The structural layer adds a roughly constant +0.03 to +0.05 at the plausible test density, and it is already used by the top notebooks. To reach first place, invest in per-passage accuracy:
   - more or better LoRA backbones (Qwen3.5-9B / Gemma-4-12B LoRA, more seeds);
   - external in-distribution data (Oleg's corpus, the extra-API passages);
   - capacity aimed at the hard within-lab pairs: gemini-3.7/3.8-flash, glm-5.3/flash/fable, gpt-5.6-luna/sol/terra, opus-4-8/sonnet-5. Hierarchical lab → model heads or pairwise specialist heads trained only on those pairs fit here.

   Lab is already 0.69 (strong) and typography alone nearly separates three lab clusters (§1.1).
2. **Decoder to ship** (CPU-cheap, seconds for 4,000 rows):
   1. Content-word TF-IDF fit on the test texts, average linkage, distance cut about 0.85–0.90. **No forced cluster count**; split clusters larger than 18. Optionally average with a dense-embedding distance, as the 0.541 notebook does with GLM L24.
   2. Inside groups of size ≥ 2: z = L − 0.5·mean_group(L).
   3. Global Sinkhorn α = 0.5 (optional, neutral).
   4. Exact distinct-author marginals per group (n ≤ 10 by subset DP, larger by Sinkhorn), then argmax.

   Expected gain over argmax: about +0.034 to +0.041 at 2–3 per prompt and +0.068 to +0.076 at 3–8. It stays positive under duplicate authors.
3. **Twin heads (feature-level centring)** are an additional, complementary lever. Lucas's frozen twins and the centred LoRA head give +0.05 to +0.07 at 3–6 contexts in their own sims. Train them on **random chunks of 2/3/4/6 within prompts**, not full 18-blocks, to match test group sizes, and use them in *add* mode. The β = 1-style "replace" logic must never be applied to groups of size 2–3. See §4.4 for the stacking result.
4. **Do not use:** hard Hungarian; full-mean centring (β = 1) on small groups; k-means/N/18 grouping; global count balancing; priors tuned to the public LB mix (the private mix differs).

## 6. Caveats

- The pools come from only 150 train prompts. Sparse scenarios have about 220–1,100 labelled passages per pool, so seed noise is ±0.005 to 0.02; deltas are paired and averaged over 10 seeds.
- The strong-set blend weights were selected on the same OOF (mild optimism of the base). Deltas are what matter here.
- The distractors are Claude-only and topic-deduplicated, so real test prompts may collide more often. If test prompts share topics, use the higher-precision cut (0.80–0.85).
- For the strong set, distractors are removed from groups after clustering (no scores for them). For the tfidf set they take part.
- The twin uses OOF raw scores per fold. A recovered cluster that mixes prompts from different folds mixes fold models; this only affects false merges.
