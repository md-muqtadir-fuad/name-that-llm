# Name That LLM: what the top public notebooks do

Date: 2026-10-16. All sources were pulled read-only with `kaggle kernels pull -m`. Run logs came from `kaggle kernels output <ref> --file-pattern ^zzz$`, which fetches only the log.

- Code: `research/kernels/<slug>/`
- Plain-text dumps of each notebook: `research/kernels_txt/<slug>.txt`
- Run logs: `research/kernel_outputs/<slug>/`
- Small dataset files I downloaded (train args, READMEs, `lora_infer.py`): `research/datasets/`

## 0. Leaderboard ordering and lineage

`kaggle kernels list --competition name-that-llm --sort-by scoreDescending` gives the ranking, but the API returns no numeric scores. The scores below come from notebook titles and text.

| Rank | Kernel | Public score | What it is |
|---|---|---|---|
| 1 | lightningv08/0-541-lora-adaptive-centering | 0.541 | Exact copy (by AST) of lucasboesen/lora-x-adaptive-centering v1 |
| 2 | lucasboesen/lora-x-adaptive-centering | 0.541 (v1) | Original. Lucas's "v14" ensemble |
| 3 | lynnsakurai/lora-opt-ed-bot | (≈0.541) | Same recipe, refactored. Same constants |
| 4 | saddamaditya/name-that-llm-apex-stylometric-centering | ? | Copy of the Public D + Qwen-tail hybrid with the exact decoder |
| 5–6 | tamerlanomralinov/ntl-fast-lora-baseline, ntl-fast-final-decoder | ? | Private reproductions of lightningv08 "0.531 Qwen+LoRA ensemble" v4 |
| 7 | kirilltym/0-525-learned-token-attention | 0.525 | Public D v1 + learned token-attention head |
| 8 | ahmedberatozer/name-that-llm-d ("Public D") | 0.522 (v1) | Frozen Gemma-4-12B / Qwen3.5-9B probes + Qwen3-8B LoRA + TF-IDF + group centering + logit repulsion |
| 9 | kirilltym/lora-adaptive-centering-content-ablation | ? | 0.541 with one change: SVM function-word view scale 0.5→0.25. The notebook is an update of Kirill's former "0.520 conditional author decoding" notebook |
| 10 | qasminumber/name-that-llm-v5 | v1 0.475, v6 unknown | Trains Gemma-4-12B and Qwen3-8B LoRAs inside the submission notebook + exact distinct-author decoding |
| 11 | lightningv08/name-that-llm-qwen-lora-ensemble | 0.531 (v1) | 0.75 × Public D + 0.25 × own Qwen-tail/UltraBERT branch |
| … | lucasboesen/0-499-llm-heads-adaptive-prompt-centering | 0.499 | Lucas v8 |
| … | rudrabhaskar/0-490-highest-public-notebook-qwen3-fusion | 0.490 | Frozen Qwen3-8B/14B prefix heads + SVM + frozen DeBERTa |
| low | kingdamon/*hungarian*, malikhammadfarooq/*prompt-clustered* | ~0.3–0.4 | CPU TF-IDF + hard Hungarian assignment |

Two independent families sit at the top. Everything above 0.52 descends from one of them.

- **Lucas Boesen family:** frozen-feature log-prob blend with "adaptive centering" (§2). It reached 0.541 when lightningv08's LoRA was added.
- **Ahmed Berat Özer "Public D" family (0.522):** frozen probes plus his own two-head Qwen3-8B LoRA, embedding-based grouping, centered heads and logit repulsion (§4). lightningv08's 0.531 hybrid and Kirill's 0.520 and 0.525 notebooks are built on it.

**Accelerators.** Every top notebook has `"machine_shape": "NvidiaTeslaT4"` with GPU on and internet off, and in the UI that is "GPU T4 ×2". The CPU-only notebooks show `"machine_shape": "None"`.

Per the kagglesdk docstring (`kernels_api_service.py`), the machine shapes Kaggle supports are `NvidiaTeslaT4`, `NvidiaTeslaP100` and `Tpu1VmV38`. Your quota is 30 GPU-hours and 20 TPU-hours per week; `kaggle quota` shows 22.6 GPU-hours left this week.

T4 has compute capability 7.5 and no native bf16. So everyone runs the backbone in fp16, keeps the trainable LoRA/head parameters in fp32, uses AMP GradScaler, and adds a "non-finite → redo in bf16" fallback.

---

## 1. The 0.541 notebook: what is in the blend

Kernel lightningv08/0-541-lora-adaptive-centering is the same code as lucasboesen/lora-x-adaptive-centering v1. Settings: T4×2, internet off.

Its interactive run with 3 test rows took about 34 minutes, almost all of it extracting features from the 2,699 training rows (see the log). A full run on 4,000 test rows would add roughly 1.5× of the train-extraction time, so I estimate about 1.5 hours. That is far under the 9-hour limit.

The ensemble is a weighted sum of log-probabilities: `LOGIT += W[key]·log p`. There are 11 members. Weights were fitted by L-BFGS on seed-0 OOF predictions.

| Key | Model / input | Features | Classifier | Weight W | Twin weight |
|---|---|---|---|---|---|
| stylegram | 4-view TF-IDF: char 3–5 (250k), word 1–3 (180k, punctuation tokens kept), "shape" text char 2–6 (letters → X/x), function-word text word 1–3 (non-function words → `CONTENT`). View scales 1.0 / 0.7 / 0.5 / 0.5 | sparse | LinearSVC C=0.5. Decision scores are softmaxed, which is why the weight is large | **1.649** | – |
| style_lgb | ~370 handcrafted stylometric features (lengths, punctuation rates, stop-word frequencies) | dense | LGBM, 400 trees | 0.068 | – |
| q3_8b_cat7+22 | `qwen-lm/qwen-3/Transformers/8b-base/1`, truncated to 23 layers | mean-pool of hidden_states[7] ‖ [22] | StandardScaler + LR C=0.0025 | 0.0 | 1.5×0.219 |
| q25_7b_cat6+20 | `qwen-lm/qwen2.5/Transformers/7b/1` | mean-pool L6 ‖ L20 | LR C=0.0025 | 0.011 | 1.5×0.141 |
| style4b | `qwen-lm/qwen-3/Transformers/4b-base/1` | **style-token pooling**: mean only over tokens whose decoded piece is an English stop word or pure punctuation; L4 ‖ L7 | LR | 0.0 | 1.5×0.083 |
| glm9b | zai-org/GLM-4-9B-0414, first 25 layers (kernel output `lucasboesen/name-that-llm-glm4-9b-trunc25`) | mean L8 ‖ L3; L24 is also extracted for clustering | LR | 0.0 | 1.5×0.110 |
| debv3l | frozen microsoft/deberta-v3-large, fp32 (kernel output `lucasboesen/name-that-llm-extract-encoders2`) | mean L24 ‖ mean L6 | LR C=0.005 | 0.151 | 0.25 |
| electl | frozen google/electra-large-discriminator (`lucasboesen/name-that-llm-electra-large-weights`) | mean L21 ‖ style-pool L24 | LR | 0.077 | – |
| desk | frozen desklib/ai-text-detector-v1.01, a DeBERTa-v3-large AI detector (`lucasboesen/name-that-llm-desklib-weights`) | mean L23 ‖ style L23 | LR | 0.187 | – |
| lora | Qwen3-8B-Base LoRA, plain head only (`lightningv08/ntl-lora24-fivefold-training`, job `final`) | 8 LayerNorm'd mean pools | linear head saved in readouts.npz | **0.282** | – |
| ultrabert | `pjmathematician/ultrabert-380m-ntl`: 5 fine-tuned UltraBERT-380M folds (dmis-lab code, trust_remote_code), softmax averaged | – | fine-tuned classifier | 0.126 | – |

Notes on the table:

- The four frozen LLM plain heads have weight about 0 in the final fit. They are still extracted because their centered twins carry real weight. The GLM L24 features also drive the test clustering.
- All frozen LM and encoder heads are refitted on train inside the notebook: StandardScaler, then LR. Train features are extracted at submission time, and that is most of the runtime.
- LM inputs are prefixed with the BOS token, `max_len` is 512, models are truncated to max(layer)+1, and weights are fp16 split across both T4s with `device_map=auto`.

### Transductive tricks in the 0.541 notebook

1. **Test-set standardization.** For each frozen head, test features are standardized with the *test* mean and std, not the train scaler. Reported CV effect: "+0.002 to +0.01, never negative".
2. **Adaptive centering** of test passages within recovered prompt clusters (§2).
3. **Partial Sinkhorn class balancing** of the final posterior, with alpha 0.5:
   ```python
   target = 0.5*P.sum(0) + 0.5*N/K
   # 50 iterations of: P *= target/P.sum(0); renormalize rows
   ```
   Reported effect: in-domain +0.002 to +0.007, including when class counts are unequal.

### Reported CV and LB history (Lucas's version comments)

| Version | What changed | CV / held-out | LB |
|---|---|---|---|
| v4 | – | CV 0.450 | 0.436 |
| v5 | – | CV 0.4927 | – |
| v6 | test standardization + Sinkhorn | – | – |
| v7 | + Qwen3-4B style head, + GLM-4-9B head | – | – |
| v8 | adaptive centering, *replace* mode | – | **0.499** |
| v9 | twins *added* as evidence; minimum cluster size 2 | – | – |
| v10 | GLM-mixed clustering; twin weight ×1.5 | – | – |
| v11 | + DeBERTa-v3-large head | – | – |
| v13 | + ELECTRA, + desklib | – | **0.532** |
| v14 | + LoRA plain head | held-out +0.0046 vs v13 | **0.541** |

The LoRA alone (5-fold OOF, plain head) scores 0.4816. CV is GroupKFold by `prompt_id`, with simulated test pools (§2.4).

---

## 2. "Adaptive centering": exactly what it does

### 2.1 Idea

Every training prompt was answered by all 18 models. The mean embedding of a prompt's answers is mostly *topic*. Subtracting it leaves the *style* offset of each answer.

The notebook trains a second, "twin" logistic regression for each frozen head on prompt-centered train features. At test time it centers each test passage by the mean of its **recovered** prompt cluster.

### 2.2 Training side (`fit_head`)

```python
Xc   = group_center(Etr, train['prompt_id'])        # subtract the per-prompt mean (≈18 rows per prompt)
csc  = StandardScaler().fit(Xc)
cclf = LogisticRegression(C=1.0*C).fit(csc.transform(Xc), y)
```

- What gets centered: the raw pooled hidden-state vector (the concatenated layers), *before* the scaler.
- Grouping: the true `prompt_id`. All ~18 answers to a prompt are centered together.
- Twins exist only for q3_8b, q25_7b, style4b, glm9b and debv3l.

**Possible improvement.** This is a train/test mismatch: the train twin always sees full 18-answer prompt groups, while test clusters are often of size 2–5. Two other pipelines centre training prompts within random chunks that match the test group sizes:

- qasminumber v5 uses chunks of 2/3/4/6, stacks them, and fits one LR on the stack.
- The Public D LoRA uses `chunk_sizes` 2,3,4.

### 2.3 Test side: forming groups

1. Compute D_tfidf: cosine distance on a content-word TF-IDF (`TfidfVectorizer(stop_words='english', sublinear_tf=True)`) fitted on the test texts only.
2. Compute D_glm: cosine distance between GLM-4-9B layer-24 mean-pooled embeddings, after centering by the test mean and L2-normalizing.
3. Average them: `D = (D_tfidf + D_glm) / 2`.
4. Cluster with scipy `linkage(method='average')` and cut with `fcluster(..., 0.60, criterion='distance')`. The older v9 cut was TF-IDF only at 0.85.
5. A row is an "adapt row" if its cluster size is between 2 and 18 (`ADAPT_MIN=2`, `ADAPT_MAX=18`). Singletons and clusters larger than 18 are left alone. Centering a singleton gives a zero vector, which was v1's failure.
6. On adapt rows, centered test features are `Ete[rows] − mean(Ete[rows in same cluster])`, then the twin scaler and LR are applied.
7. In **add** mode, the twin's log-probability is added on top of the plain head:
   ```python
   LOGIT[ADAPT_ROWS] += TWIN_W[key] * log(pc)
   ```
   The plain heads stay active for every passage.

The run log includes a clustering QA on 433 train passages sampled at 1–5 per prompt: pair precision 0.940, pair recall 0.772. Cluster sizes were 78 singletons, 176 rows in clusters of 2–3, and 179 in clusters of 4–18.

At threshold 0.60 with m=150 prompts the false-positive pair rate is about 0.012–0.059. TF-IDF alone at 0.85 has a false-positive pair rate of 0.053 and recall 0.80.

### 2.4 Simulated gains (pooled 150-prompt simulation)

The simulation keeps 1–2, 1–5, 3–8 or all 18 passages per held-out prompt.

| Change | 1–2 / prompt | 1–5 | 3–8 | all 18 |
|---|---|---|---|---|
| v8 replace≥4 vs plain | +0.000 | +0.019 | +0.034 | +0.056 |
| v9 add≥2 vs plain | **+0.024** | **+0.042** | +0.043 | +0.052 |
| v10 GLM-mixed grouping vs v9 | +0.003 to +0.008 | +0.008 | +0.02 | +0.02 |

So the gain depends on how many test passages each test prompt has. Nobody knows the hidden test's density. Two clues:

- qasminumber v5 says the hidden test was simulated as "2–3 passages per unseen prompt (the structure that matched every leaderboard result so far)".
- The rules say test class counts are unequal and the mix differs between public and private, which fits a random subsample rather than full 18-answer blocks.

If that is right, ~4,000 test passages correspond to roughly 1,300–2,000 test prompts.

---

## 3. lightningv08 Qwen3-8B LoRA (the 0.282-weight member)

Kernel: lightningv08/ntl-lora24-fivefold-training, run as a separate notebook. Settings: T4×2, internet off, model input `qwen-lm/qwen-3/Transformers/8b-base/1`.

The submission notebooks only *load* its output, by attaching the kernel output as an input: `/kaggle/input/notebooks/lightningv08/ntl-lora24-fivefold-training/lora24/final/`. That directory holds `adapter/`, `heads.safetensors`, `readouts.npz`, `tokenizer/` and `anchor_features.npz`, which is used for a parity check.

The training code is in `train_lora24.py`, embedded as a JSON string in cell 1. I extracted it.

**Model and LoRA:**
- Backbone: Qwen3-8B-Base truncated to **24 of 36 layers**, fp16 frozen, sdpa attention, gradient checkpointing.
- LoRA: r=16, alpha=32, dropout 0.05 on q, k, v, o, gate, up and down projections (PEFT). Trainable parameters are in fp32. No quantization.

**Representation and heads:**
- For each of layers [2,4,6,8,12,16,20,24], take the masked mean over tokens, excluding position 0 (the attention-sink token).
- Apply a separate `LayerNorm(4096)` to each pool and concatenate: 32,768 dimensions.
- Two linear heads (18 classes each) with dropout 0.1:
  - plain: `CE(plain(z), y)`
  - centered: `CE(centered(z − mean(z over the 3-answer same-prompt micro-batch)), y)`
- The loss is the sum of the two.

**Batching:**
- Micro-batch = 3 distinct answers to the **same prompt**. Each epoch shuffles answers within each prompt into triples and pads incomplete triples.
- Gradient accumulation 6, so the effective batch is 18.

**Optimizer and schedule:**
- AdamW: LoRA lr 1e-4, head lr 1e-3, weight decay 0.01.
- 10% warmup, then cosine decay. Gradient clip 1.0. GradScaler with initial scale 256.
- **3 fixed epochs**, no early stopping. Each fold is about 360 updates.

**Inputs:** `max_len` 447 plus an EOS token, with no BOS or special tokens. Texts run 40–250 words, so almost nothing is truncated.

**Jobs and runtime:**
- Six jobs: outer0–4 (5-fold prompt-disjoint) plus `final` on all 2,699 rows.
- Each T4 runs one independent worker process (`CUDA_VISIBLE_DEVICES`) with three jobs. No model is split across GPUs.
- Measured: about 9.9 s per update, about 1 hour per job on one T4, **2.96 hours total wall time** for 6 jobs on 2×T4 (log ends at 10,650 s).

**Results** (from `lora24_comparison.json`; recovered contexts = 1, 3 or 6 passages per prompt, TF-IDF grouping at 0.9, singletons fall back to plain):

| Setup | 1 context | 3 contexts | 6 contexts |
|---|---|---|---|
| plain head OOF | 0.4816 | 0.4816 | 0.4816 |
| centered head | 0.470 | 0.491 | **0.553** |
| 50/50 plain+centered | 0.478 | 0.539 | 0.549 |
| Gemma-tail "core" baseline | 0.483 | 0.549 | 0.578 |
| core + LoRA at 0.2 | 0.505 | 0.563 | 0.589 |

The 0.541 notebook uses only the **plain** LoRA head. The centered LoRA head is unused there; the hybrid notebooks (tamerlan / lightningv08 v4) do use it.

---

## 4. "Public D" (ahmedberatozer/name-that-llm-d, 0.522) and its children

This is a script kernel on T4×2. Inputs:
- models: `google/gemma-4/Transformers/gemma-4-12b/2`, `qwen-lm/qwen-3/Transformers/8b-base/1`, `qwen-lm/qwen-3-embedding/Transformers/0.6b/1`
- datasets: `bbucxi/qwen35-9b-safetensors` (Qwen3.5-9B), `ahmedberatozer/ntl-code`

### Members and weights

Blend weights come from 5-fold OOF. Nested CV: balanced accuracy 0.4978, NLL 1.376.

| Member | Weight |
|---|---|
| q35 | 0.326 |
| g12 | 0.125 |
| q8lora | 0.413 |
| tfidf | 0.249 |
| stylo_lr | 0.040 |
| stylo_lgb | 0.081 |

- **Frozen probes:**
  - Gemma-4-12B (hidden 3840, 48 layers): layers 2,4,6,8,10,12,14,16,20,24,28.
  - Qwen3.5-9B (hidden 4096, 32 layers): layers 2,3,4,5,6,17,18,19.
  - Each layer gets its own StandardScaler + LR (C=0.002), and log-probs are averaged over layers.
  - Mean pooling excludes token 0. Test standardization is on.
  - The heads are prefit and shipped in `ntl-code/heads__*.npz`.
- **LoRA:** checkpoint `q8c_L24_s42_full` from `ahmedberatozer/ntl-code`, trained off-Kaggle (paths say `../models/Qwen3-8B-Base`):
  - 24 layers, r16 / alpha32, all projections, pool layers 2..24, lr 1e-4, head lr 1e-3, 3 epochs, max_len 448, bs 9, accum 2.
  - Two heads (plain and centered), with the centered head trained on random same-prompt chunks of size **2, 3, 4**.
  - Inference merges the LoRA weights manually (W += BA·α/r) without peft; see `lora_infer.py`.
  - The dataset was updated today (2026-10-16 12:23). It now also holds seeds s2, s3 and s4, plus `deb_c_full`: a fine-tuned DeBERTa-v3-large two-head model (pool layers 3..24, lr 2e-5, LLRD 0.9, 4 epochs).
  - Its `train_args` include `ext_n: 0`, so the training script supports external data but it was not used.
- **4-view TF-IDF** + LR (C=50), stylometric LR and LGBM.

### Grouping and decoding

- **Test grouping:** Qwen3-Embedding-0.6B (SentenceTransformer) embeddings with the instruction `"Instruct: Identify the specific topic or question this passage answers\nQuery: "`. Link pairs with cosine > 0.92, take connected components, and break components larger than 6 into singletons.
- **Centered heads:** for grouped rows, `z += w_k·gw_k·(centred_k − plain_k)`, with gw = 1.0 for the probes and 0.5 for the LoRA.
- **Logit repulsion,** a soft one-model-per-prompt rule:
  ```python
  z[m] = z0[m] - 0.2 * (sum(z0[group]) - z0[m]) / (n - 1)
  ```
  Each passage is pushed away from the classes its group-mates favour.

### Children

- **kirilltym/test-conditional-author-decoding (0.520)** replaces repulsion with **exact distinct-author marginal decoding**:
  - For a group of g ≤ 6 passages, compute `q_ic = P(a_i = c | all authors distinct)`. This is a sum over all injective assignments, computed by a subset dynamic program over the g passages and K=18 authors in O(K·g·2^g) time.
  - Then take the argmax of the marginals. This is a soft, marginal version of the Hungarian assignment.
  - Development: frozen/CPU stack +0.006 (95% CI [0.004, 0.008]). On the 30 reserved prompts +0.009, with a CI that includes 0.
  - **Leaderboard: 0.520 vs 0.522. No transfer.**
- **qasminumber v5** uses the same decoder at temperature 0.25 on TF-IDF clusters of 2–6. Simulated gain at 1/2/3/4 passages per prompt: −0.001 / +0.002 / +0.007 / +0.014.
- **kirilltym/0-525-learned-token-attention (0.525):**
  - Trains a small head inside the notebook: Qwen3.5-9B layer-18 token states (all ≤512 tokens) → shared 4096→64 tanh MLP → mean ‖ attention-pooled → linear.
  - About 265k parameters, 20 epochs, AdamW 5e-4, label smoothing 0.05.
  - Blended as `0.90·F + 0.0821·A + L`, where F is the frozen/CPU part, A the token head and L the LoRA.
  - +0.003 on LB over Public D.
- **lightningv08/name-that-llm-qwen-lora-ensemble (0.531 v1):** `0.75 × D_logits / 1.234 + 0.25 × own_branch`.
  - own_branch: frozen Qwen3/Qwen3.5 readouts, **"tail" features** (Qwen3-8B layers 4/6/8/22 averaged over the *last 32 or 64 tokens*), text/style models, frozen encoders, and UltraBERT at 15%.
  - Its v4 (copied by tamerlanomralinov) adds the new LoRA24 (plain+centered 50/50) at weight 0.2 and the exact decoder on lexical groups of 2–6.
  - Readouts ship in `lightningv08/ntl-oof-q35-heads`, `ntl-tail-inference-public`, `ntl-exact-inference-artifacts` and `ntl-oof-validation-cache`.

---

## 5. Other notebooks worth knowing

- **qasminumber/name-that-llm-v5** trains its LoRAs **inside the submission notebook** (T4×2):
  - Gemma-4-12B, seed 42; Qwen3-8B-Base, seeds 42 and 7.
  - 24 layers, r16/α32, 3 epochs, lr 1e-4 / head 1e-3, batch 8 × accum 2, max_len 448.
  - The input-embedding matrix is trimmed to only the tokens that occur (Gemma vocab 262k → 20k), so Gemma-12B fits on a T4 (10.4 GiB).
  - Runtime: about 850 s per epoch for Qwen3-8B and about 1,050 s per epoch for Gemma-12B, on one T4 each. The whole notebook took 7,280 s, about 2 hours.
  - Held-out fold: Qwen3-8B LoRA 0.485 / 0.463 (seeds 42 / 7), Gemma-4-12B LoRA 0.472. They agree on only about 63% of passages, and combined score 0.498. Qwen3-14B (16 layers) scored 0.432. **5 epochs overfit (0.456).**
  - Frozen single-head scores: Qwen3.5-9B 0.468, Gemma-4-12B 0.465, Qwen3-8B 0.464, Qwen3-14B 0.461, SVM 0.372, 4-view TF-IDF 0.385.
- **rudrabhaskar 0.490:**
  - Frozen Qwen3-8B (layers 4/6/8/22) and Qwen3-14B (4/6/8/24), using mean pooling plus per-token std.
  - Prefit LR heads, some with a topic projection that removes top topic directions; hierarchical family→model heads.
  - Centering on TF-IDF clusters at 0.9; calibrated SVM; frozen DeBERTa.
  - "Selection CV 0.5327" vs LB 0.490, which shows how optimistic selection-reused CV is.
- **malikhammadfarooq prompt-clustered stylometry (CPU):**
  - TF-IDF agglomerative clustering, ARI 0.90–0.96 vs true prompts in CV.
  - Hard Hungarian assignment per cluster, then pseudo-labelling.
  - CV with full ~18-per-prompt groups: argmax 0.4427 → **assigned 0.4842** (oracle groups 0.4864).
  - Its leaderboard rank is low, which again suggests the hidden test does *not* contain full prompt blocks.
- **kingdamon "SOTA Hungarian" v6/v9:** CPU Ridge/SGD on TF-IDF with hard `linear_sum_assignment` on recursively split TF-IDF clusters (threshold 0.9, max size 18). Low leaderboard.

---

## 6. Does any notebook use the one-answer-per-model-per-prompt structure?

Yes, at three levels of strength:

1. **Group centering.** All top notebooks do this.
   - Lucas: frozen twins, TF-IDF+GLM average-linkage clusters, sizes 2–18, add mode.
   - Public D: embedding connected components with cap 6.
   - LoRA centered heads trained on same-prompt micro-batches of 2–4.
2. **Soft exclusion.** Public D's logit repulsion (α 0.2), and Kirill's exact distinct-author marginals (groups ≤ 6, subset DP). The marginals did not transfer to LB: 0.520 vs 0.522.
3. **Hard Hungarian assignment** (one model per passage in a cluster of ≤ 18). Only in low-ranked CPU notebooks. It gives +0.04 in CV with full prompt groups but evidently does not pay off on the hidden test.

**No notebook does a global constrained assignment** that models the test as an unknown number of prompts, each with a subset of models.

The likely reason all of these help less than expected is the test structure: probably only a few passages per prompt, sampled with unequal class counts. Centering and exclusion gains grow with group size.

The 0.541 Sinkhorn step is the only use of the global class marginal, and it is deliberately partial (alpha 0.5) because class counts are unequal.

---

## 7. External data seen

- **olegzholobov/name-that-llm-response-corpus** (394 MB):
  - Responses scraped from public sources (Speechmap, HF datasets, EQ-Bench, Artificial Analysis, and others) under the 18 competition labels.
  - 135k unique responses. **24,015 are short (40–250 words).** 1,902 exact prompts have responses from all 18 models.
  - Provenance and quality flags are included. Luna has a separate quarantined weak-label file.
  - Checked: no exact or near-duplicate overlap with train.
  - **No top notebook uses it.** Lucas notes that external data under genre shift collapsed predictions onto one class.
  - Kirill tried "a filtered external corpus" and did not select it.
- **seyitkaangunes/ntl-extra-api-passages** (CC BY 4.0):
  - 4,998 passages generated through the Claude API for 1,000 new prompts × 5 Claude labels, at lengths matched to the train distribution.
  - On a TF-IDF+SVC baseline, grouped-CV balanced accuracy rose from 0.372 to 0.388 at 50% Claude-class weight. Claude recalls rose, and GLM recall dropped.
  - A model trained only on train got 30% exact and 65% provider accuracy on these passages, vs about 38% and 70% in-domain. So the distribution is close but not identical.
- **pjmathematician/ntl-train-prompts-reconstructed:** reconstructed train prompt templates. Each prompt carries its own min–max word range; the domain mix is 25% physics and 20% biology, and so on.

---

## 8. Summary per requested notebook

| Notebook | Backbone(s) | LoRA | Trained where | CV vs LB | Inference tricks |
|---|---|---|---|---|---|
| lightningv08/0-541-lora-adaptive-centering | Qwen3-8B (models 8b-base/1), Qwen2.5-7B, Qwen3-4B, GLM-4-9B-0414 trunc25, DeBERTa-v3-large, ELECTRA-large, desklib detector, UltraBERT-380M ×5 | loads lightningv08 LoRA24 (r16/α32, 24L, 3 epochs, lr 1e-4) | LR heads fit in the notebook; LoRA, UltraBERT and encoder weights loaded from kernel outputs / datasets | CV ≈ 0.49 to 0.50 plain; held-out sims; LB 0.541 | test standardization, adaptive centering (add mode, mixed TF-IDF+GLM clusters at 0.60, sizes 2–18), Sinkhorn α 0.5 |
| lucasboesen/lora-x-adaptive-centering | same (identical code) | same | same | same | same |
| kirilltym/…content-ablation | same | same | same | "small uncertain gain" on 2 folds | function-word view 0.5→0.25 |
| lightningv08/name-that-llm-qwen-lora-ensemble | Public D (Gemma-4-12B, Qwen3.5-9B, Qwen3-8B LoRA, Qwen3-Embedding-0.6B) + Qwen3-8B/14B tail readouts + UltraBERT | Public D LoRA (q8c_L24_s42_full) | readouts prefit (datasets lightningv08/ntl-oof-q35-heads, ntl-tail-inference-public); D branch refits TF-IDF and stylometry | v1 LB 0.531 | 0.75/0.25 log blend, embedding grouping, centered heads, repulsion; v4 adds the exact decoder |
| rudrabhaskar/0-490 | Qwen3-8B-base, Qwen3-14B-base prefixes, DeBERTa (rudrabhaskar/name-that-llm-original-offline-models) | none | heads prefit (rudrabhaskar/name-that-llm-qwen-fusion-heads) | selection CV 0.533 vs LB 0.490 | TF-IDF cluster centering at 0.9, topic projection, geometric blend |

**GPU and runtime:** everything is T4×2 with internet off.

| Run | Wall time |
|---|---|
| 0.541 notebook, interactive (3 test rows) | ~34 min |
| 0.541 notebook, full scoring run | likely < 2 h |
| LoRA24 training (6 jobs, 2×T4) | 2.96 h |
| qasminumber v5 (3 LoRAs trained in-notebook + full inference) | ~2 h |

## 9. Takeaways for beating 0.541

1. **The LoRA is the strongest single member** (plain OOF ≈ 0.48–0.49). It is still only one member, and only its plain head is used in 0.541. Cheap wins:
   - Add the **centered LoRA head** on recovered groups (lightningv08 shows +0.05 to +0.07 at 3–6 contexts).
   - Add more seeds or other backbones: Public D now ships seeds s2/s3/s4/s42 and a two-head DeBERTa. A Gemma-4-12B LoRA agrees only ~63% with the Qwen LoRA.
   - Train LoRAs on **all 36 layers or on Qwen3.5-9B / Gemma-12B**. Nobody has published a LoRA on Qwen3.5-9B, which is the strongest frozen probe at 0.468.
2. **Fix the centered-twin mismatch in Lucas's heads** (train on 2/3/4/6 chunks, as in v5 and Public D) and merge the two families. Lucas's encoders and centering plus Public D's Gemma/Qwen3.5 probes and LoRA have never been combined in one public notebook. Of the two, only Public D uses Gemma-4-12B / Qwen3.5-9B probes; only Lucas uses GLM, DeBERTa, ELECTRA, desklib and UltraBERT.
3. **Use external data carefully.**
   - The response corpus offers 24k short labelled passages from more than 1,900 full 18-model prompts, about 9× the train set. Filtering, length matching and per-class weighting would be needed.
   - It is untried by the top notebooks, and it is the biggest lever for retraining the classifier.
   - Validate only on official-train prompt folds.
4. **Structured decoding:** the exact distinct-author marginals are theoretically right but gave no LB gain on Public D. Expect small gains unless test groups are larger. Keep the Sinkhorn step partial.
5. **Validate like Lucas and lightningv08:** prompt-grouped folds, simulated test pools with 1–2, 1–5 and 3–8 passages per prompt and unequal class counts, and test-only clustering inside the simulation.
