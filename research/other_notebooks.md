# Name That LLM: survey of public notebooks and datasets

Snapshot date: 2026-10-16. Sources: 60 public kernels, pulled with `kaggle kernels pull` into `research/kernels2/<slug>/`. Execution logs came from `kaggle kernels logs` and are saved as `research/kernels2/<slug>/log.txt`. Dataset READMEs are in `research/datasets_meta/`.
Every number below was read from a notebook's own markdown, code comments or execution log. "CV" always means prompt-grouped CV (GroupKFold or StratifiedGroupKFold on `prompt_id`) unless stated otherwise. "LB" means public leaderboard, as reported by the author.

Public leaderboard top (from `kaggle competitions leaderboard -s`):

| Rank | Team | Score |
|---|---|---|
| 1 | Adisak Sukul | 0.572 |
| 2 | Søren Ravn Andersen | 0.569 |
| 3 | Lucas Boesen | 0.547 |
| 4 | Komil Parmar | 0.544 |
| 5 | Abdelrehman | 0.543 |
| 6–7 | Richard Wang, Tony Li | 0.542 |
| 8+ | about 15 teams | 0.541 |

The 0.541 teams all submitted the public `lightningv08/0-541-lora-adaptive-centering` notebook, which is a verbatim copy of `lucasboesen/lora-x-adaptive-centering`. Lucas Boesen, its author, now scores 0.547 privately.

---

## 1. Master table: approach, CV and LB

| Notebook (ref) | Method | Backbone(s) | CV (grouped) | LB | Test-time tricks |
|---|---|---|---|---|---|
| **lightningv08/0-541-lora-adaptive-centering** = lucasboesen/lora-x-adaptive-centering (v14) | Log-linear blend of the 4-view TF-IDF LinearSVC ("stylegram"), stylometric LightGBM, frozen-LLM linear probes, frozen encoders, UltraBERT and a LoRA Qwen3-8B classifier | Qwen3-8B-Base L7+22, Qwen2.5-7B L6+20, Qwen3-4B-Base L4+7 (style-token pool), GLM-4-9B-0414 (truncated, L8+3), frozen DeBERTa-v3-large, ELECTRA-large, desklib detector, UltraBERT-380M 5-fold, LoRA Qwen3-8B 24L | v5 base 0.4927. The rest is pooled-sim deltas | **0.541** (v13 0.532, v8 0.499, v5/v4 0.436) | Test-set standardisation of LM features; adaptive prompt-centred "twins" added on test clusters of size 2–18 (clusters from TF-IDF plus GLM L24 distance, cut 0.60); Sinkhorn class-marginal balancing, alpha 0.5 |
| kirilltym/lora-adaptive-centering-content-ablation | 0.541 notebook with the SVM function-word CONTENT channel scaled 0.5 → 0.25 | same | "small uncertain gain" | not reported | same |
| kirilltym/0-525-learned-token-attention | name-that-llm-d plus a learned token-attention head (4096→64 tanh MLP, attention pooling over 512 tokens of Qwen3.5-9B L18) | Gemma-4-12B, Qwen3.5-9B probes, LoRA Qwen3-8B | Ensemble mean of five regimes: 0.480 → 0.498 (+0.017, CI [+0.012, +0.023]); head alone 0.366 | **0.525** (control 0.521) | Group cap 6, topic threshold 0.92, logit repulsion |
| ahmedberatozer/name-that-llm-d ("Public D") | Log-linear blend: 4-view TF-IDF LR, stylometry LR and LGBM, frozen probes, LoRA Qwen3-8B (24 layers, two-head) | Gemma-4-12B (11 layers), Qwen3.5-9B (8 layers), Qwen3-8B LoRA | nested CV 0.4978 | **0.522** | Qwen3-Embedding-0.6B topic grouping (cos > 0.92, cap 6); group-centred heads; test standardisation; logit repulsion (alpha 0.2) |
| ahmedberatozer/name-that-llm-baseline-b | earlier D (LoRA with 10 layers) | same | nested 0.4845 | n/a | same |
| kirilltym "0-520 conditional author decoding" (copy: mtoshidesu/test-conditional-author-decoding) | D plus exact distinct-author marginal decoder (subset DP, g ≤ 6) | same as D | n/a | **0.520** (did not beat D's 0.522) | Exact marginals |
| qasminumber/name-that-llm-v5 (v6) | Calibrated SVM, 4-view TF-IDF LR, frozen probes (plain and chunk-centred), LoRA Qwen3-8B ×2 seeds and LoRA Gemma-4-12B trained in-notebook, frozen DeBERTa-v3-large, exact decoder | Qwen3.5-9B, Gemma-4-12B, Qwen3-14B, Qwen3-8B | sim at 2 / 3 passages per prompt: v1 0.500/0.510 → v6 0.560/0.556 | v1 0.475, v2 0.494, **v3 0.519**, v4 0.517 | Test standardisation; centring inside topic clusters; exact decoder |
| lucasboesen/0-499-llm-heads-adaptive-prompt-centering (v8) | Stylegram SVC, stylometric LGBM, Qwen3-8B and Qwen2.5-7B concatenated layers, Qwen3-4B style pool, GLM-4-9B, UltraBERT | | 0.4927 (v5) | **0.499** | Capped adaptive centring, Sinkhorn |
| rudrabhaskar/0-490-highest-public-notebook-qwen3-fusion (also kiza123123 apex-quintuple and lucifer19 apex-trial) | Geometric blend of 22 heads: Qwen3-8B and 14B prefixes (plain, centre, topic-projection, joint, hierarchy, std), calibrated text SVM, frozen DeBERTa, plus UltraBERT at 0.15 in the Apex fork | Qwen3-8B, Qwen3-14B, DeBERTa-v3-base | selection cross-check 0.5327 | **0.490**; Apex fork 0.505 | Small-group centring, topic projection |
| rudrabhaskar/name-that-llm-original-offline-ensemble | Earlier version of the same fusion | | untouched 30-prompt holdout, 95% CI [0.495, 0.606] | n/a | |
| visheshbanna/frozen-embeddings-prompt-centering-0-483 | 0.36·SVM + 0.27·plain Qwen8B + 0.27·centred Qwen8B + 0.10·frozen DeBERTa-v3-base (T = 2) | Qwen3-8B L4/6/8/22, DeBERTa-v3-base | 5-fold sim: 0.529 (k = 2), 0.528 (k = 3) | v3 0.414, v6 0.475, **v9 0.483** | Clusters at average-linkage cut 0.90, centring at size ≥ 2 |
| hanifnoerrofiq/svm-frozen-qwen | Calibrated LinearSVC (char 1–5, word 1–2, 25 style counts) plus Qwen3-8B per-layer LR (C = 0.002), plain and chunk-centred | Qwen3-8B-Base L4/6/8/22 | svm 0.372, plain 8B 0.444, plain blend 0.469, centred (v4) 0.469 / 0.483 / 0.485 / 0.487 / 0.481 at k = 1/2/3/6/full | n/a (descendant of v1 0.475) | Confidence-gated centring |
| ammunoz/name-that-llm-exp07a-qwen14b-3way | Same three-way recipe with Qwen3-14B | Qwen3-14B | svm 0.372, qwen 0.445, centred 0.461, 3-way 0.517 | n/a | |
| lynnsakurai/llm-decoder-qwen-ver | SVM plus Qwen8B plain and centred, bounded unique decoder | Qwen3-8B | blend 0.510 → 0.513 with decoder | 0.483 (cited) | |
| evgendvorkin/name-that-llm | Word+char TF-IDF, masked syntax and stylometry (LinearSVC), then Qwen3-8B 4-layer LR | Qwen3-8B | text: 0.306 → 0.346 (+masked) → 0.355 (+style) → 0.370 (all); Qwen 0.445 | n/a | |
| souvikdbiswas/ntl-sota-tri-model-fusion | 50% Qwen3-1.7B L17+20, 30% UltraBERT, 20% 4-view; prior shift tau 0.2 | Qwen3-1.7B, UltraBERT | n/a | n/a (cites 0.436) | |
| lucasboesen/qwen-blend (v3) | TF-IDF plus Qwen3-1.7B L17+20 heads | Qwen3-1.7B | 0.431 | **0.414** | |
| abh001/name-that-llm-eda-fingerprints-embedding-clusters | LR blend of Qwen3-1.7B L17/L20, char-SVD, word and char TF-IDF; EmbeddingGemma-2 tested | Qwen3-1.7B | 0.418 | 0.404 without 18-size centring, **0.376 with it** | Centring and Hungarian both turned OFF |
| pjmathematician/ntl-ultrabert-380m-baseline-(train/infer) | Fine-tuned UltraBERT-380M, CLS head, lr 1e-5, 5 epochs | dmis-lab/UltraBERT-380M | 0.408 (folds 0.357–0.451) | n/a | |
| ammunoz/ultrabert-v2-leak-test and exp09 | Re-scored the published UltraBERT folds on train | | **each fold scores 0.86–0.92 on every train fold**, see §5 | | |
| yangkuangou/…-style-aware-char-n-grams-cv-0-384 | 4-view TF-IDF (char 3–5 cased w1.0, word 1–3 w0.7, shape 2–6 w0.5, function-word 1–3 w0.5) + LinearSVC C = 0.5 | none | **0.3845** (baseline char_wb 0.327) | n/a | |
| yangkuangou/…-frozen-qwen3-4b-tf-idf | Qwen3-4B L4+8 LR (C = 0.005) plus the 4-view SVC | Qwen3-4B-Base | LR 0.425, blend 0.458 | n/a | |
| yangkuangou/…-the-prompt-recovery-trap | Topic recovery (ARI 0.99) plus Hungarian assignment | none | baseline 0.327 → assignment 0.419; LR + assignment 0.501 | **0.336 → 0.308 (assignment HURT)**; 0.351 for the 0.501 pipeline | Assignment |
| alekseyosipenko/0-355-…honest-cv | Char 1–5 and word 1–2 TF-IDF plus 25 style features | none | 0.369 (random split 0.217) | **0.355** | Per-class bias did not transfer (−0.005) |
| dariushafshar/…-cv-0-37-tf-idf-svm-baseline | char + word TF-IDF + style SVM | none | 0.3685 | n/a | |
| nulldata/…-fingerprint-eda-tf-idf-baseline | char 1–5 + word 1–2 TF-IDF LR | none | 0.356 (lab 0.711) | n/a | |
| martynaplomecka/starter | TF-IDF baseline | none | 0.3145 | n/a | |
| malikhammadfarooq/…-prompt-clustered-stylometry-cpu | Ridge on cluster-centred char+word TF-IDF, LGBM on style (raw, centred, rank); Hungarian; one pseudo-label round | none | argmax 0.443, assigned 0.484 (oracle 0.486) | n/a | Assignment and pseudo-labelling |
| malikhammadfarooq/…-stylometry-cpu-c-pseudo | Same with assignment OFF, PSEUDO_ROUNDS = 1, PSEUDO_FRAC = 0.6 | none | n/a (CV off) | n/a | Pseudo-labelling, gain never measured |
| sharif485/stylometric-fusion | LinearSVC TF-IDF plus GPU CatBoost on 218 style features + SVD96 | none | SVM 0.336, CatBoost 0.308, blend (w = 0.2) 0.366 | n/a | |
| avikdas567/frontier-llm-attribution-via-stylometry-deberta | LGBM on TF-IDF + 69 style features, LR, deberta-v3-small (fell back to an MLP offline) | | LGBM 0.289, LR 0.328, MLP 0.133, blend 0.332 → 0.361 with per-class thresholds | n/a | Threshold optimisation (in-sample, optimistic) |
| olegzholobov/name-that-llm-curated-tfidf | Word+char TF-IDF SGD or SVC with filtered external corpus at weight 0.1 or 0.3 | none | 0.30 (SGD, ext 0.1) | n/a | External data |
| olegzholobov/name-that-llm-qwen3-14b-style | Qwen3-14B first 25 blocks, mean+std at states 4/6/8/24 (40,960-d), prompt-direction shrinkage, LR, external weight 0.03 | Qwen3-14B | 0.443–0.456 per fold | n/a | External data |
| kingdamon/…-sota-external-corpus | Ridge on word+char TF-IDF + style, 24,025 external rows at weight 0.25, Hungarian | none | not reported | n/a | External data and Hungarian |
| seyitkaangunes/does-extra-data-help-ntl-extra-passages | 4-view SVC with 4,998 new Claude API passages | none | 0.3715 → **0.3882** (extra weight 0.5) | n/a | External data |
| takaito/…-modernbert-training-inference | ModernBERT-large fine-tune | ModernBERT-large | **0.2045** | n/a | |
| takaito/…-gemma2b-training-inference | Gemma-2B sequence classification fine-tune (QLoRA) | Gemma-2B | **0.251** | n/a | |
| prvsiyan/a-fable-of-residual-voices | Sparse TF-IDF, 472 opportunity-normalised style features, Qwen3-Embedding-0.6B L7/14/21 | Qwen3-Emb-0.6B | layers alone 0.345 / 0.349 / 0.294; final embedding 0.17 | n/a | |
| sahangunasekara92/eda-…-pos-patterns | POS n-grams, char and word n-grams, LR | none | macro-F1: char 0.255, POS 0.200, word 0.143, all 0.268 | n/a | |
| verracodeguacas/prism-signatures-eda | Atlas: cached Qwen3-8B L4/6/8/22 + 468 style features; 50/50 blend with D's public LoRA adapter | Qwen3-8B | Atlas OOF reused | n/a | |
| lightningv08/ntl-lora24-fivefold-training | **5-fold LoRA training of Qwen3-8B** (24 layers, r16 / alpha32 on 7 projections, 8 LayerNorm mean pools, plain and centred heads, same-prompt triples, effective batch 18, 3 epochs, cosine) | Qwen3-8B | **plain 0.4816**; centred 0.470 / 0.491 / 0.553 at k = 1/3/6; adding LoRA at 20% to the Gemma baseline: +0.016 | used in 0.541 | |

---

## 2. What the best notebooks have in common (the 0.52–0.54 recipe)

1. **Frozen decoder-LLM probes, not fine-tuned encoders.** Mean-pool the hidden states of early or middle layers, skip the first token (an attention sink), apply StandardScaler, then a heavily regularised LR (C 0.002–0.005). Early layers win: L4/6/8 for Qwen3-8B, L2–6 for Qwen3.5-9B, L17/20 of 28 for Qwen3-1.7B. Single-backbone OOF with plain heads and test standardisation (qasminumber, five folds):
   - Qwen3.5-9B: 0.468
   - Gemma-4-12B: 0.465
   - Qwen3-8B: 0.464 (0.445 without the first-token skip and test standardisation)
   - Qwen3-14B: 0.461
   - Qwen3-4B L4+8: 0.425
   - Qwen3-1.7B: about 0.42

   Frozen bidirectional encoders also help: DeBERTa-v3-large mean L23 plus style pool L24 scores 0.450–0.464, ELECTRA-large 0.434, desklib AI detector 0.428.
2. **LoRA fine-tune of a truncated LLM is the strongest single branch.** Recipe: Qwen3-8B-Base cut to its first 24 layers, LoRA r16 on all linear layers, LayerNorm and mean pooling of 8 layers, a linear head, 3 epochs (5 epochs overfits: 0.456). It scores 0.48–0.49 OOF. Gemma-4-12B with the same recipe scores 0.472 but agrees with the Qwen LoRA on only about 63% of passages, so it is a good ensemble member. Qwen3-14B with 16 layers scored 0.432. Two LoRA seeds plus Gemma moved the simulated score from 0.528 to 0.556 at k = 2.
3. **Text and style branch.** The 4-view TF-IDF LinearSVC/LR reaches 0.384 OOF and gets the largest weight in the 0.541 blend (1.649, on margins). Stylometric LGBM is weak alone (0.29–0.34) but adds a little.
4. **Log-linear (geometric) blending** with weights fitted on OOF (L-BFGS or coordinate ascent).
5. **Transductive test-time steps**, all label-free:
   - *Test-set standardisation* of LM features: the scaler is refit on the test features. Worth +0.5 to +1.0 pt in simulation.
   - *Recovered topic clusters plus centred "twins".* Cluster passages by content TF-IDF, GLM layer-24 or Qwen3-Embedding similarity. Subtract the cluster mean and score with a head trained on prompt-chunk-centred features (random chunks of 2/3/4/6). Add this as extra evidence ("add" mode) rather than replacing the plain head. Gains in the pooled 150-prompt sim: +0.024 (1–2 passages per prompt), +0.042 (1–5), +0.052 (all 18).
   - *Sinkhorn class-marginal balancing* (alpha 0.5): +0.002 to +0.007 in CV, never negative.
   - *Exact distinct-author marginal decoding* in clusters of 2–6 passages: simulated −0.1 / +0.2 / +0.7 / +1.4 pt at 1 / 2 / 3 / 4 passages per prompt. On the LB it did not help: 0.520 vs 0.522.

## 3. Structure of the hidden test

- **Hard assignment over full 18-passage pseudo-prompts hurts on the LB.** Hungarian assignment raised OOF from 0.327 to 0.419 but moved the LB from 0.336 to 0.308 (yangkuangou). Centring on KMeans pseudo-prompts of about 18 passages moved the LB from 0.404 to 0.376 (abh001).
- **The hidden test appears to have about 2–3 passages per unseen prompt.** qasminumber reports this structure "matched every leaderboard result so far", and the best notebooks tune for it. With about 4,000 passages, that implies roughly 1,300–2,000 prompts, so each model does not answer every test prompt.
- **Recovered groups are small and noisy.** That is why the 0.541 notebook allows clusters of 2–18 and the D notebook caps them at 6.
- **Simulate the test this way.** Five grouped folds; inside each held-out fold, sample 1/2/3/6 or mixed passages per prompt, recover clusters from that pool only, then score. svm-frozen-qwen, visheshbanna and lucasboesen all do this.

## 4. Stylometric signals by lab (consistent across EDA notebooks)

| Lab | Signals |
|---|---|
| **Anthropic (Claude ×5)** | Curly apostrophe ’ never appears (0%), only straight '. Em-dash in 63–83% of passages. claude-opus-4-6 leaves `*` markdown asterisks (z +3.1 to +3.4). claude-opus-5 has a high ellipsis rate (z +4.0), low TTR and more "!". claude-sonnet-5 is the "contraction model" (isn't, it's) and uses single quotes. claude-opus-4-8 favours "quietly" and "glimpse". claude-fable-5 has no clear signature (best is em-dash, z +1.3) and is the hardest class. |
| **Zhipu (GLM ×2)** | Same typography as Claude: 0% curly apostrophe and about 80% em-dash (more than 1.6 per passage for glm-5.3). This is why GLM is mislabelled as Claude. glm-5.3 uses more "!"; glm-5.3-flash uses more straight double quotes. |
| **Google (Gemini ×4)** | Almost no em-dashes (0–1%) and few colons (present in only 17–23% of passages) and semicolons. Longest words (5.5–5.9 characters), most "-ly" adverbs, fewest commas. Strong vocabulary: *profound, expanse, breathtaking, vibrant, intricate, fundamentally*. gemini-3.1-pro-preview writes short sentences and has many periods; it is the easiest class (recall 0.73–0.87). gemini-3.8-flash writes shorter passages (z −2.5 on log word count) with higher TTR. gemini-3.7-flash has longer mean word length. Curly ’ appears in 6–31% of passages. |
| **OpenAI (GPT ×5)** | Curly apostrophe ’ in 42–53% of passages; straight ' rarely (≤ 11%). Curly double quotes in 7–15%. Em-dash at a moderate 9–37%. gpt-5.6-luna favours parentheses (z +2.9 to +3.5); gpt-5.6-terra also uses parentheses. gpt-5.6-sol favours curly double quotes. gpt-6-astra often ends **without a final period** and writes short sentences. gpt-5.5 uses more "!". |
| **Moonshot (kimi-k3)** | Semicolons (about 0.95 per passage) and colons (about 0.99), plus en-dashes (z +2.7). Em-dash in 52%, curly quotes in 9%. Vocabulary: *patience, keep, tighten*. |
| **Alibaba (qwen3.8-max)** | Most commas (about 11.7 per 100 words), em-dash in only 3%, few question marks. Vocabulary: *turning, becoming, luminous*. |

The host removed newlines, `#`, backticks and double spaces. There are no whitespace fingerprints.
Character n-grams, case-sensitive with punctuation kept, generalise across prompts better than word n-grams (macro-F1 0.255 vs 0.143), even though word n-grams separate the classes more in descriptive EDA. Lab-level accuracy is 0.57–0.79, against 0.35–0.52 for the exact model.

## 5. Confusion patterns (from logged outputs)

TF-IDF/SVM baseline, top true → predicted pairs (counts out of 150 passages per class), from 0-355-honest-cv and which-models-are-easiest:
- gemini-3.8-flash → gemini-3.7-flash: 47–51
- gemini-3.7-flash → gemini-3.8-flash: 38–44
- glm-5.3 → claude-opus-5: 35; glm-5.3 → glm-5.3-flash: 36
- glm-5.3-flash → claude-opus-5: 31; glm-5.3-flash → glm-5.3: 27–28
- claude-fable-5 → claude-opus-5: 31
- gemini-3.6-flash → gemini-3.1-pro-preview: 33; gemini-3.6-flash → gemini-3.7-flash: 27–30
- kimi-k3 → gpt-5.5: 30 (Qwen and Kimi both leak into gpt-5.5)
- gpt-5.6-sol → gpt-5.6-luna / gpt-5.6-terra: 27–29; gpt-5.6-terra → gpt-5.5: 26 and → gpt-5.6-luna: 27; gpt-5.6-luna → gpt-6-astra: 27

How errors split by lab:
- About 52% of errors stay inside the predicted lab overall.
- Gemini errors stay in-family 85% of the time and GPT errors 58%.
- Claude errors are split evenly between in-family and cross-family.
- GLM errors are mostly cross-family, going to Claude.

Per-class recall:

| Model | TF-IDF baseline | Qwen14B 3-way blend (0.517) |
|---|---|---|
| claude-fable-5 | 0.07–0.13 | 0.22 |
| gpt-5.6-sol | 0.13–0.19 | 0.28 |
| glm-5.3 | 0.14–0.21 | 0.26 |
| glm-5.3-flash | 0.11–0.21 | 0.30 |
| kimi-k3 | 0.19–0.30 | 0.47 |
| claude-opus-4-8 | 0.20–0.23 | 0.56 |
| gemini-3.7-flash | 0.35–0.38 | 0.39 |
| gemini-3.8-flash | 0.35–0.40 | 0.49 |
| gemini-3.1-pro-preview | 0.74–0.83 | 0.87 |
| gpt-6-astra | 0.62–0.66 | 0.81 |
| claude-opus-5 | 0.63–0.70 | 0.81 |
| claude-opus-4-6 | 0.48–0.61 | 0.67 |
| qwen3.8-max | 0.39–0.45 | 0.65 |

The hard core is {claude-fable-5, glm-5.3, glm-5.3-flash, gpt-5.6-sol, gpt-5.6-terra, gpt-5.6-luna, gemini-3.7-flash, gemini-3.8-flash}. prvsiyan names the next targets as pair specialists for Gemini 3.7↔3.8, GPT-5.6 sol/luna/terra and GLM↔Claude.

## 6. Pseudo-labelling and other transductive steps

- **Pseudo-labelling on test.** Only malikhammadfarooq used it: one round, the top 60% most confident test passages after group assignment, then refit. **No CV or LB gain was ever reported**; the pseudo step runs only on the hidden test. No high-scoring notebook uses pseudo-labelling.
- **Transductive steps that did help:**
  - Test-feature standardisation: +0.5 to +1.0 pt in simulation.
  - Centred twins on small clusters: +0.02 to +0.05 in simulation; LB 0.499 → 0.532 together with new encoders.
  - Sinkhorn balancing: +0.002 to +0.007.
- **Transductive steps that hurt on the LB:**
  - Hard Hungarian assignment: −0.028.
  - Centring on clusters of about 18 passages: −0.028.
- **Per-class bias / threshold tuning.** Cross-fitted bias was −0.005 (alekseyosipenko). avikdas reports +0.029, but the thresholds were fitted in-sample, so that number is optimistic.

## 7. Caveats on public assets

- **UltraBERT-380M weights leak** (`pjmathematician/ultrabert-380m-ntl`). Each fold checkpoint scores 0.86–0.92 balanced accuracy on its own "held-out" fold and 0.73–0.83 on the others (ammunoz exp09 and leak-test). The weights are effectively trained on all training data, so they cannot be used for OOF blend fitting. The 0.541 notebook gives UltraBERT a fixed weight of 0.126 that was never OOF-validated. Its claimed OOF was 0.408.
- **The public LoRA adapters** (ahmedberatozer `q8c_L24_s42_full`; lightningv08 lora24 `final`) are full-train fits. Only lightningv08's five fold-specific adapters give honest OOF (0.4816).
- **Blend weights in the 0.541 notebook** come from seed-0 OOF and pooled simulations with many selections on the same 150 prompts, so expect optimism. The private split has a different class mix.

## 8. External and synthetic data

| Dataset | Content | Findings |
|---|---|---|
| **olegzholobov/name-that-llm-response-corpus** (394 MB) | 135,171 unique external responses labelled with the 18 competition model names, from Speechmap, EQ-Bench, Arena-prose, BullshitBench, HF trace datasets, APIFlow and others. `external_short_wide.parquet` has 24,025 passages of 40–250 words. Per-model counts are very uneven: Gemini 3.6/3.7/3.8-flash have only 71–83 short passages; claude-opus-4-6 has 4,911. There is also a quarantined Luna weak-label set of 74,790 rows. | No exact or near-duplicate overlap with train. It is used at low weight (0.03–0.3). Curated TF-IDF with it scored about 0.30, below plain TF-IDF at about 0.356. Qwen14B at external weight 0.03 scored 0.44–0.456, no clear gain over about 0.461 without it. **Domain shift is large**: Lucas Boesen notes that the in-domain model applied to genre-shifted external data collapsed onto one class (BA 0.12, raised to 0.19 by Sinkhorn). |
| **seyitkaangunes/ntl-extra-api-passages** (CC BY 4.0) | 4,998 passages from the official Claude API: 1,000 new prompts × 5 Claude models, generated with the competition-style protocol (single paragraph, lengths sampled from train). Only v1 (Claude) exists; other labs are "planned". | A train-only classifier gets the exact model right on 30–31% and the provider on 65–68%, so the passages are close to but not identical to the competition distribution. Adding them improves 18-class BA by +0.013 to +0.017 with the SVM (0.383 → 0.396; 0.3715 → 0.3882 at weight 0.5). Claude recalls rise: fable 0.10 → 0.20, sonnet 0.43 → 0.63. GLM loses some recall. |
| **pjmathematician/ntl-train-prompts-reconstructed** (CC BY-SA 4.0) | Reconstructed prompts for the 150 train prompts, plus a generation-protocol JSON with a template, domain mix and length mechanism ("(words {min}-{max})", min 36–60, max min+130–220) and example new prompts. | Blueprint for generating matched synthetic data if API access is available. |
| nulldata/ntl-bullshitbench-v2-paragraphs | 40–250-word paragraphs from BullshitBench v2 responses for 16 of the 18 labels (MIT) | No public notebook was found using it. |
| kishor1123/gpf-matched-news-… | 12 older models (GPT-5.4, Claude 4.5, Gemini 2.5…) | Not relevant: wrong models. |

**Pretrained weights and feature caches:**
- Official Kaggle Models: qwen-lm/qwen-3 (1.7b/4b/8b/14b-base), qwen-lm/qwen2.5 7b, qwen-lm/qwen-3-embedding 0.6b, google/gemma-4 gemma-4-12b, google/embeddinggemma-2, answer-ai/modernbert large.
- bbucxi/qwen35-9b-safetensors (Qwen3.5-9B).
- Kernel outputs for DeBERTa-v3-large, ELECTRA-large and desklib-detector: lucasboesen/name-that-llm-extract-encoders2, -electra-large-weights, -desklib-weights. GLM-4-9B truncated to 25 layers: lucasboesen/name-that-llm-glm4-9b-trunc25.
- Public LoRA checkpoints and code: ahmedberatozer/ntl-code (1.9 GB, CC0); lightningv08/ntl-lora24-fivefold-training output (five folds plus final).
- Frozen heads and OOF caches: lightningv08/ntl-oof-q35-heads, ntl-oof-validation-cache, ntl-tail-inference-public, ntl-exact-inference-artifacts; kirilltym/ntl-frozen-heads-and-ce-projection; rudrabhaskar/name-that-llm-qwen-fusion-heads and -original-offline-models.
- Feature caches: abh001/ntl-frozen-lm-features-train; verracodeguacas/name-that-llm-prism-feature-atlas (Qwen3-8B 37 layers × 4096 for train, plus 468 style features).
- UltraBERT: pjmathematician/ultrabert-380m-ntl (leaky, see §7).

## 9. Weak or failed approaches

| Approach | Result |
|---|---|
| Full fine-tune of encoder classifiers on 2.7k rows | ModernBERT-large 0.205, Gemma-2B QLoRA classifier 0.251, deberta-v3-small 0.13, UltraBERT 0.408. All far below frozen-LLM probes at 0.44–0.47. |
| Final-layer sentence embeddings | Qwen3-Embedding final layer 0.17. Use early/middle layers. |
| Gradient boosting on style or TF-IDF | 0.29–0.34 |
| Joint/projection heads, richer style features, token-std | Did not reliably beat simpler heads in sparse 2–3-per-prompt sims (visheshbanna v7/v8). |
| Tighter clustering | Purity is not the objective: cut 0.85 vs 0.90 lowered BA 0.4598 → 0.4550. |

## 10. Implications for reaching first place

Required to reach the 0.54 level:
- A frozen multi-backbone probe ensemble with test standardisation.
- LoRA-tuned truncated LLM classifiers (Qwen3-8B and Gemma-4-12B, several seeds).
- The 4-view SVC.
- Centred twins on small recovered clusters.
- Sinkhorn balancing.

The gap from 0.541 to 0.572 must come from somewhere new. Candidates, ranked by evidence:
1. More and more diverse LoRA classifiers: more seeds, Gemma-4-12B, Qwen3.5-9B, with prompt-triple batches and a centred head. LoRA is the strongest branch and only two or three runs are in public blends.
2. Pair and sibling specialists for the hard core: Gemini 3.7/3.8, GPT-5.6 trio, GLM vs Claude, fable.
3. Token-level attention readouts: +0.017 on top of D.
4. In-domain synthetic data generated with the competition protocol. Claude-only data gave +0.015; all 18 labs would likely help more, but needs API access and has to be published to satisfy the "public" rule.
5. Honest weight refitting without UltraBERT leakage, and simulations matched to about 2–3 passages per prompt.

All training must run on Kaggle T4×2. Locally (4 CPU cores) only TF-IDF/SVM and head fitting on cached features are feasible.
