# Name That LLM: test plan, Oct 17 to Dec 13, 2026

Goal: get from 0.544 (best public score, v1) past the current leader (0.572) on the **private** leaderboard.
Month 3 (Dec 14 to Jan 13) is set aside for testing external data and choosing the final submissions.

**Budgets:**
- Kaggle GPU: about 30 h per week. A full 5-fold LoRA run costs about 6 GPU-h; a 2-fold quick test about 2.5 h.
- Submissions: 5 per day. Only submit when a gain has already shown up on held-out prompts.
- Claude usage: run tests as Kaggle jobs plus small local scripts, and avoid big agent fan-outs.

**Rule for every test:** judge it on held-out prompts first (5-fold GroupKFold by prompt). Use the leaderboard only to confirm.
Proceed only when the gain is at least +0.005 and holds in at least 4 of the 5 folds.

---

## Month 1: stronger and more varied models (Oct 17 to Nov 13)

### Week 1 (Oct 17 to 23)
| # | Test | Cost | Proceed if |
|---|---|---|---|
| 1 | **Gemma-4-12B LoRA** (training now): held-out score for the plain head, the centered head and their average; how much it adds to the blend; then build and submit v3 | ~6 GPU-h (running), 1 submission | held-out ≥ 0.46 and blend gain ≥ +0.005 |
| 2 | **Honest blend-weight refit**: cache train features for every frozen member in v1 (Qwen3-8B, Qwen2.5-7B, Qwen3-4B, GLM-4-9B, DeBERTa, ELECTRA, desklib), then refit all blend weights with nested validation on held-out predictions. Includes the Qwen LoRA's per-fold held-out predictions and the Gemma one. **Also test dropping UltraBERT**: its public weights have seen all of train, so its fixed weight is unvalidated. | ~1.5 GPU-h, CPU | refit beats the current weights on held-out prompts |
| 3 | **GLM-4-9B LoRA** (same lab as glm-5.3 and glm-5.3-flash, the hardest classes): smoke test, then full run | ~5–6 GPU-h | GLM recall improves and the blend gains |

### Week 2 (Oct 24 to 30)
| # | Test | Cost | Proceed if |
|---|---|---|---|
| 4 | **Qwen3.5-9B LoRA**: smoke test with speed check, then full run | ~6–8 GPU-h | held-out ≥ 0.47, adds to the blend |
| 5 | **Merge the members of the "Public D" notebook**: Gemma-4-12B and Qwen3.5-9B frozen-feature classifiers, the extra Qwen LoRA seeds and the two-head DeBERTa from `ahmedberatozer/ntl-code` | ~1 GPU-h for feature caches | nested blend gain |
| 6 | **Finish the grouping-method test**, small scope, run directly: embedding models (bge, e5, gte, Qwen3-Embedding-0.6B) against the current TF-IDF+GLM grouping, on realistic pools with distractors and 3 seeds. Explain the "recovered beats true" anomaly. | CPU | downstream gain over v1's grouping on fresh seeds |

### Weeks 3–4 (Oct 31 to Nov 13): improve the LoRA recipe (2-fold quick tests on the best base model)
| # | Test | Cost | Proceed if |
|---|---|---|---|
| 7 | LoRA rank: r16 vs r64 (α = r/2), FFN-only vs all layers | 2 × 2.5 GPU-h | +0.005 on held-out |
| 8 | Head learning rate 3e-4 and longer warmup (fixes the early loss spikes) | 2.5 GPU-h | plain head improves |
| 9 | **Contrastive loss (SupCon)**: same model = positive; other models answering the same prompt = hard negatives | 2.5 GPU-h | +0.005 |
| 10 | Length augmentation: prefix crops at sentence boundaries (p = 0.3) | 2.5 GPU-h | +0.003 |
| 11 | Depth: 24 vs 32 vs all layers (memory permitting) | 2.5 GPU-h | +0.005 |
| 12 | **Qwen3.8-27B 4-bit QLoRA** feasibility: one fold, timed (same lab as qwen3.8-max) | ~3 GPU-h | full run fits in < 12 h and fold score ≥ 0.48 |

### Added from external review (DeepSeek, 2026-10-17), screened against our evidence
| # | Test | Cost | Proceed if |
|---|---|---|---|
| 12a | **Position-aware pooling**: pool the first and last 32 tokens separately, max-pool, first/last-token states (frozen Qwen3-8B cache, then the CPU head test) | ~0.3 GPU-h + CPU | +0.005 over the mean-pool head |
| 12b | Label smoothing 0.05 in LoRA training (combine with #9 SupCon) | in the #9 run | — |
| 12c | Logit-lens layer contrast: $W_U(h^{(22)}-h^{(7)})$ restricted to style tokens (function words and punctuation) | CPU, needs $W_U$ | +0.003 |
| 12d | Pairwise experts routed by top-2 margin (extends the lab-gated MoE, idea by Md Muqtadir Fuad) | CPU | +0.005 |
| 12e | **Self-distilled LoRA** (Argon): same-prompt batches; loss adds $\lambda T^2\,\mathrm{KL}(\mathrm{sg}(p_{\text{cent}})\,\|\,p_{\text{single}})$ so the singleton head learns to match the prompt-centred head; plus topic-word cutout (mask 15% of high-IDF content tokens) | 2-fold run, ~2.5 GPU-h | singleton (plain) held-out +0.01 |
| 12f | Prior-aware balancing (EM prior, then Sinkhorn, then divide by $\hat\pi^\tau$) vs uniform partial Sinkhorn under skewed priors; nuisance whitening; CORAL | CPU (running) | +0.003 under skewed priors / +0.005 |
| 12g | Log-likelihood ratios across open LLMs from the same lab (GLM-4 / Gemma-4 / Qwen) plus token surprisal statistics | ~1 GPU-h (full depth) | +0.005 |
| 12h | Model-debiased kNN centering (subtract neighbours' expected model style before averaging) | CPU, pool simulation | +0.005 at 1–3 passages/prompt |
Rejected: damped test-prior balancing (pushing toward the estimated test prior lowers balanced accuracy, whatever that prior is); test-set-wide evidence summing; 8–10 epochs (overfits); PCA/NAP projection; kNN label propagation; entropy and reverse-cloze features. Tree models (LightGBM/ExtraTrees) on the stylegram TF-IDF instead of or beside the LinearSVC (E032: standalone 0.26–0.29 vs 0.385; blend +0.0015 at best, 3/5).

## Month 2: ensemble, robustness, inference (Nov 14 to Dec 13)

| # | Test | Cost | Proceed if |
|---|---|---|---|
| 13 | Retrain the winning recipe on the best 2–3 base models, full 5 folds | ~6 GPU-h each | held-out gain confirmed |
| 14 | More seeds, and averaging fold adapters into one ("fold soup") vs separate fold models | ~6 GPU-h | +0.003 |
| 15 | **Stacking**: a small meta-classifier on all held-out logits vs the weighted log-linear blend; nested validation | CPU | +0.005 nested |
| 16 | Class-specific bias tuning (nested) and partial class balancing (Sinkhorn α 0 / 0.25 / 0.5) under simulated unequal class mixes | CPU | robust across class-mix simulations |
| 17 | Grouping revisited with the best grouping method plus centered LoRA heads (soft only; no hard per-group constraints) | CPU + 1 submission | sim gain at 1–3 passages/prompt and leaderboard ≥ the previous version |
| 18 | Test-time augmentation: multi-crop averaging for the LoRA members | ~1 GPU-h | +0.003 |
| 19 | **Runtime check**: full notebook on a synthetic 4,000-passage test, must stay well under 9 h | ~3 GPU-h | < 6 h |
| 20 | Distillation: train 1–2 student LoRAs on the ensemble's soft labels | ~6 GPU-h | +0.003 or a big runtime saving |

**Ongoing, both months:** collect free public data (Kaggle datasets `olegzholobov/name-that-llm-response-corpus` and
`seyitkaangunes/ntl-extra-api-passages`, public eval logs). Log what's available; **don't test it until month 3**.

## Month 3 preview (Dec 14 to Jan 13)
- Test external data: CPU TF-IDF check first, then pretrain the LoRA on the external data and fine-tune on train (2 folds).
- Optional paid API pilot (~$20), only for the weakest classes, if free data isn't enough.
- Choose the 2 final submissions: (a) the best held-out blend; (b) a more conservative, diverse blend.

## Already tested (don't repeat)
| Idea | Result |
|---|---|
| Hard Hungarian / one-model-per-prompt assignment | hurts on the leaderboard |
| Push-apart within groups (a = 0.2) | leaderboard 0.536 vs 0.544 |
| Family / pair stylometric specialists | ≈0 or negative |
| Topic-direction removal (NAP/INLP), prototype head, test pseudo-labelling | negative |
| Company term added to the blend after the fact | ≈0 on the blend (move it into training instead) |
| Replacing the 4-view TF-IDF SVM (30 variants, NBSVM, char-LM) | none better; keep the current one |
| LoRA centered head as a 50/50 average on grouped rows (v1) | +0.003 on the leaderboard (0.541 → 0.544) |
