# Name That LLM: what the literature and past Kaggle competitions suggest

Date: 2026-10-16. Scope: published work on attributing text to an LLM and on model fingerprinting, plus winning write-ups from related Kaggle competitions. For each technique I list the reported gain and whether it should carry over to this competition.

The setting, briefly:
- 18 classes, including close siblings (gpt-5.6 luna/sol/terra; gemini 3.6/3.7/3.8-flash; five Claude models; glm-5.3 vs glm-5.3-flash).
- Training data: 2,699 passages (150 prompts × about 18 models). Passages run 40–250 words (median 101).
- Test prompts are unseen. About 4,000 test passages are available at inference, so transductive methods are possible.
- Constraints: Kaggle T4×2, 9 hours, no internet. Metric: balanced accuracy.
- Background: `research/top_notebooks.md` (current public state of the art, 0.541) and `research/forum.md`.

Kaggle write-ups were read in full with `kaggle competitions topic-messages <comp> <topic_id>`. Papers were read through arXiv HTML. Every number below is the source's own reported number, not something I re-measured.

---

## 0. Summary: what to change, in priority order

| # | Change | Main evidence | Expected gain here (my estimate) | Cost |
|---|---|---|---|---|
| 1 | **More in-domain labelled data.** Pretrain on the external 18-label corpus, then fine-tune on train, with length, format and prompt-style matching. | DAIGT 1st: "modelling approach was less important"; they had several single models above 0.970 because of the data mix. LMSYS 1st/3rd: post-pretraining and 500k pseudo-labels. Idiosyncrasies learning curve: accuracy keeps rising up to about 10k samples per task. FDLLM: about 4.5k/class gives 96.6% on 20 LLMs. | +0.02 to +0.06, but high variance: a genre shift once collapsed predictions (Lucas). | Medium. Data is already public; training is offline on Kaggle GPU. |
| 2 | **Stronger LoRA classifiers.** Larger and newer backbones (Qwen3.5-9B, Qwen3-14B, Gemma-4-12B, and Qwen3.8-27B, which is the same lab as qwen3.8-max); LoRA on all linear layers including the FFN; higher rank; several backbones and seeds. | Idiosyncrasies: an LLM2vec LoRA beats BERT/T5/GPT-2 by 5–7 points. FDLLM: FFN-targeted, r=256/α=128 is best; 7–9B backbones are within ~2 points of each other; 0.6B is 8 points worse. Jigsaw-ACRC 2025: Qwen3-14B > 8B > 4B, and the encoder (Ettin-400M) is 3 points behind. In this competition the Qwen and Gemma LoRAs agree on only 63% of passages. | +0.01 to +0.03 | Medium-high (GPU hours) |
| 3 | **Train for topic invariance, not only correct for it afterwards.** Same-prompt micro-batches with a group-centred head on random group sizes 2–6. A better centring estimator (shrinkage or alternating "crossed-effects" EM, §5.3). Projecting out the topic subspace (INLP/NAP), which also works for singletons. | Recursion 2019, 5th place: same-plate batch statistics took validation accuracy from 40%+ to 70%+. Wegmann 2022: content-controlled hard negatives give the most content-independent style embeddings. lightningv08: centred LoRA head +0.07 at 6 contexts. | +0.01 to +0.03, more with denser test groups | Low-medium |
| 4 | **Test-time pseudo-labelling / test-time training (TTT)** on the 4,000 test passages. Confident and distinct-within-group labels, then refit heads or run one more LoRA epoch. | Recursion 1st: progressive pseudo-labelling under a uniqueness constraint, single model 0.990 → 0.997. DAIGT 5th: test-time students +0.007 private. DAIGT 3rd: 1k confident pseudo-labels lifted the TF-IDF branch from 0.893 to 0.927 private. Jigsaw-ACRC 2025: LoRA TTT on Qwen3-14B inside the 12-hour window was the winning pattern. | +0.005 to +0.02 | Medium (fits in the 9 h) |
| 5 | **Hierarchical and contrastive objectives.** Lab head (6) + model head (18); supervised contrastive loss whose hard negatives are other models on the same prompt; specialists for sibling groups. | DeTeCtive: multi-level contrastive learning beats plain contrastive by +1.9 AvgRec and a RoBERTa CE classifier by +8.9 out of domain; kNN beats a linear head by +3.0. DA-MTL: an auxiliary detection head adds +1.6 to +2.7 F1 to attribution. Forum: lab-level BA 0.587 vs model-level 0.369. | +0.005 to +0.02 | Low |
| 6 | **Family-likelihood features.** Per-token log-prob, entropy and rank statistics, plus Binoculars-style pairwise ratios, under open siblings of the target labs: Qwen3.8-27B, Gemma-4, gpt-oss-20b, GLM-4. | SeqXGPT: 95.7 macro-F1 when the generators' own LMs are available. Sniffer: 44.7/67.5. FDLLM: GLTR and log-likelihood are almost useless on black-box generators (6.7 and 7.5 macro-F1). | 0 to +0.01, mostly at lab level | Medium (GPU inference) |
| 7 | **Length augmentation.** Prefix crops at sentence boundaries; multi-crop TTA. | DAIGT 5th: short-context students. Idiosyncrasies: accuracy grows with length up to ~256 tokens. | 0 to +0.01 | Low |
| 8 | **Ensembling and distillation.** OOF-fitted logit blend across backbones and families; LoRA fold soups; mutual learning. | LMSYS 1st: averaging the 5 fold LoRAs and distilling from 70B teachers. Jigsaw 6th: Deep Mutual Learning +0.004 private. | +0.005 to +0.015 | Low-medium |

**Things that will probably *not* help** (§9): a tokenizer or TF-IDF vocabulary fitted on test (that DAIGT trick worked because of test-specific typos); EM label-shift prior correction applied to the decision (wrong for balanced accuracy, §6.4); hard Hungarian assignment (the test does not contain full 18-model blocks); watermark detection (no keys); asking an LLM judge directly (M4GT: humans score below chance on multi-way attribution).

---

## 1. How hard the problem is, according to the literature

- **Sun et al. 2025, "Idiosyncrasies in Large Language Models"** (ICML 2025) — https://arxiv.org/abs/2502.12150
  - Five chat APIs (ChatGPT, Claude, Grok, Gemini, DeepSeek): **97.1%** accuracy with an LLM2vec embedder fine-tuned with LoRA, trained on 10k responses.
  - **Same family:** the four Qwen2.5 sizes reach only **59.8%** together, with pairwise accuracies of 63.1–85.5%. Pure sampling differences (greedy vs top-k) are near chance (50–60%).
    - This matches our hard sub-problems: the luna/sol/terra and gemini-flash trios.
  - **Data scaling (Fig. 3):** 10 samples gives 40.3%; accuracy keeps rising and converges near 10k samples. At 150 per class we are on the steep part of the curve, so **data is the main lever**.
  - **Length (Fig. 2):** accuracy rises with the number of tokens and saturates around 256 tokens. One token already gives about 50% for chat models in the 5-way task, so the opening words carry a lot of signal.
  - **What carries the signal:**
    - Word shuffle keeps 88.9% and letter shuffle drops to 38.9%, so it is lexical, word-level information.
    - Removing special characters costs 2 points.
    - Paraphrasing by another LLM still leaves 91.4%; summarising leaves 58.1%. Some of the signal is semantic: what a model chooses to say.
    - TF-IDF (1–2 grams) with LR gets 85.5%, against 95.1% for the fine-tuned embedder on the same no-special-character text. Neural is about +10 points at large N.
  - **OOD prompts:** training on UltraChat and testing on other prompt sets gives 89.9–98.9%, against 96.3% in-distribution. Unseen prompts hurt somewhat, but the signal generalises.
  - **Recipe:** LoRA r16, α32, dropout 0.05, lr 5e-5, 3 epochs, batch 8, first 512 tokens, mean pooling.
  - **Distillation leaves fingerprints:** SFT on another model's outputs makes text classifiable as that teacher. This is consistent with the forum finding that both GLMs are most often mistaken for claude-opus-5.
- **FDLLM (2025)**, black-box fingerprinting of 20 LLMs (90k samples) — https://arxiv.org/abs/2501.16029
  - Qwen2.5-7B-Instruct + LoRA + linear head: **96.56%** accuracy, against 74.6% for the best non-LLM baseline.
  - The 20 LLMs include siblings (GPT-4o vs 4o-mini; GLM4 / GLM4-Flash / GLM4-Plus; Qwen-turbo vs Qwen2.5-14B). The most common confusions are GLM4 ↔ GLM4-Flash and Qwen2.5 ↔ Qwen-turbo.
  - **Out of domain** (Quora QA): accuracy falls to **50.3%**. Unseen prompts or domains are the real difficulty.
  - **Few-shot addition** of new models (50 samples each) still gives about 95.6% accuracy.
  - **Zero-shot statistics** on 20 classes: GLTR 6.7 macro-F1, log-likelihood 7.5, DetectGPT 3.1.
  - Backbone and LoRA details are in §2.
- **McGovern et al. 2024, "Your LLMs Are Leaving Fingerprints"** — https://arxiv.org/abs/2405.14057
  - Character 3–5 and word 2–4 n-grams plus POS n-grams with simple classifiers. F1 0.936 for ChatGPT and 0.920 for Claude on OUTFOX.
  - Fingerprints persist within a model family, which makes siblings hard to separate.
- **Suzuki et al. 2025, "Natural Fingerprints of LLMs"** — https://arxiv.org/abs/2504.14871
  - Models trained on *identical data* can still be told apart. Seed, size and optimiser settings each leave traces.
  - So sibling separation is possible in principle, but the differences are subtle.
- **System attribution, Oct 2026** — https://arxiv.org/abs/2610.00253
  - 5 deployed systems (GPT-5.2, Gemini 3 Flash ± search, Grok, Perplexity), prompt-grouped CV.
  - **Character 2–4 TF-IDF + LR gives 97.8%** on 800-character prefixes; length alone gives only 31.7%.
  - Between distinct labs a simple character n-gram model is almost perfect. Siblings (Gemini with vs without search; Grok vs GPT on one subset) produce most of the errors.
- **M4GT-Bench** (ACL 2024), multi-way generator detection with 7 classes — https://arxiv.org/abs/2402.11175
  - RoBERTa 97.0% in-domain, but **36.6% on an unseen domain (arXiv)**.
  - GLTR + LR/SVM 45–50%. Stylistic SVM 35%.
  - Human annotators are **below random**.
  - davinci-003 and ChatGPT, the most similar pair, are confused most.
- **Przystalski et al. 2025**, stylometry on short samples — https://arxiv.org/abs/2507.00838
  - StyloMetrix and n-gram features with LightGBM reach MCC 0.87 on 7 classes for 10-sentence Wikipedia-style texts.
- **SemEval-2024 Task 8, subtask B** (6-way: human + 5 generators) — https://arxiv.org/abs/2404.14183
  - Winner AISPACE **90.85%**: an ensemble of fine-tuned encoders (encoder-only beat decoder-only for them) with class-balanced weighted CE and soft voting.
  - Runner-up 86.96%.
- **Surveys:** Huang, Chen & Shu, "Authorship Attribution in the Era of LLMs" — https://arxiv.org/abs/2408.08946
  - Frames LLM-to-model attribution as one of four problems.
  - Names domain generalisation, which here means unseen prompts, as the central open challenge.

**Main takeaway.** Separating labs is easy given enough data. Separating siblings is hard (60–85% pairwise even at large N). Unseen prompts or domains cost a lot (FDLLM 96 → 50; M4GT 97 → 37). We are in the hardest corner on all three counts: siblings, unseen prompts and only 150 samples per class. So the biggest gains should come from (a) more data, (b) explicit topic invariance and (c) dedicated sibling discrimination.

---

## 2. Backbones for style fingerprinting

| Source | Comparison | Result |
|---|---|---|
| Idiosyncrasies (Table 6) | ELMo / BERT / T5 / GPT-2 fully fine-tuned vs LLM2vec (decoder LLM) + LoRA | 90.8 / 91.1 / 90.5 / 92.1 vs **97.1** (chat). Base-model task: 69.8 / 66.0 / 67.9 / 80.2 vs **87.3**. The decoder LLM gains most on the hard task. |
| FDLLM (Table IV) | Qwen2.5-7B-Instr / GLM4-9B / DeepSeek-7B / Qwen3-8B / Qwen3-0.6B, all with LoRA | 96.56 / 95.12 / 94.16 / 94.62 / 88.38. Among 7–9B models the choice matters about ±1–2 points; dropping to 0.6B costs 8. |
| FDLLM LoRA ablation | rank; target modules | **r=256, α=128 (α = r/2) best** (93.28 macro-F1 in the sweep). **FFN modules (gate/up/down) beat attention-only**; gate alone gives 92.22. |
| DAIGT 2024 1st | Mistral-7B QLoRA vs DeBERTa-v3-large vs Ghostbuster vs custom-vocabulary DeBERTa-small | 0.984 / 0.963 / 0.974 / 0.943 private. **The LLM with LoRA was the best single model.** LoRA r=64, α=16, on QKVO. |
| LMSYS 2024 1st–4th | Gemma-2-9B vs Llama-3-8B vs Llama-3.1 | Gemma-2-9B "easily the best" (3rd). Reward-model initialisation beat chat-model initialisation. Disabling Gemma-2 soft-capping gave +0.002. |
| Jigsaw-ACRC 2025 1st | Qwen3-14B / Qwen2.5-14B / Qwen3-8B / Qwen3-4B / Llama3.1-8B / Ettin-400M encoder | 0.9239 / 0.9232 / 0.9236 / 0.9198 / 0.9202 / **0.8944** private. **Larger is better; encoders are clearly behind.** Unsloth fits a 14B LoRA on one 16 GB T4. |
| SemEval-24 8B (AISPACE) | encoder vs decoder vs encoder-decoder | Encoder-only best. This was 2023-era generators with large data and many samples per class. |
| This competition | Qwen3-8B LoRA vs ModernBERT ("laya") vs frozen probes | LoRA plain OOF about 0.48. ModernBERT poor (forum 749832). Frozen Qwen3.5-9B probe 0.468, Gemma-4-12B 0.465, Qwen3-14B 0.461. A 16-layer-truncated Qwen3-14B LoRA scored 0.432, probably because of the truncation. |

**Recommendations for backbones:**
- Keep decoder LLMs with LoRA as the core. Try:
  - **Qwen3.5-9B** (the best frozen probe, and nobody has trained a LoRA on it yet).
  - **Gemma-4-12B** (diversity; it is also the Gemini-family proxy).
  - **Qwen3-14B with all 40 layers.**
  - **Qwen3.8-27B with QLoRA.** It is mirrored on Kaggle (`dangkhoa2016/qwen-qwen3-8-27b`, `ugvfpdcuwfnh/qwen3-8-27b` in BF16; `mikedan7/qwen3-8-27b-fp8-official`). It belongs to the same lab and generation as `qwen3.8-max`. Train it on Kaggle GPU with Unsloth 4-bit: a 27B model needs about 15 GB in 4-bit, so it fits one T4 at short sequence lengths, but it is slow.
- Change the LoRA recipe toward the FDLLM findings:
  - Target **all linear layers including gate/up/down** (lightningv08 already does).
  - Try **r=64–128 with α=r/2**. The current r16/α32 is the opposite α ratio.
  - Keep a 3-epoch budget; 5 epochs overfit in qasminumber.
- Use **reward-model or detector initialisation** as a "pretrain on a related task" stage. LMSYS 3rd and 5th: initialising from reward models trained on UltraFeedback gave up to 0.020 log loss.
  - Our analogue: first fine-tune on Oleg's 135k-response 18-label corpus (all lengths, cropped), then on train. This is the same "post-pretrain then fine-tune" pattern as LMSYS 1st, DAIGT 2nd and Jigsaw 3rd.
- Encoders (DeBERTa-v3-large, ELECTRA, desklib) remain useful as **diverse, low-weight ensemble members**. Their 0.541 blend weights are 0.08–0.19. They should not be the core.
- **Pooling:** multi-layer mean pooling from early-to-middle layers (lightningv08 uses layers 2–24). This fits the Idiosyncrasies finding that the signal is mostly lexical and word-level. Keep excluding the attention-sink token.

---

## 3. Per-token log-prob and perplexity features from open LMs

| Method | Idea | Reported | Applies here? |
|---|---|---|---|
| **Ghostbuster** (NAACL 2024) https://arxiv.org/abs/2305.15047 | Token probabilities under weak LMs (unigram, trigram, ada, davinci); search over vector and scalar combinations; LR | 99.0 F1 binary; +5.9 F1 over DetectGPT/GPTZero. DAIGT 1st reimplemented it with Llama-7B and TinyLlama (100 features: min/max/mean/quantiles/var of 4 LMs and 6 ratios; SVM+RF): **0.974 private**, the second-best single model | Built for binary human-vs-AI. Fine-grained attribution needs the *generator family's* likelihoods (below). Cheap to add as a 100-feature block. |
| **Binoculars** (ICML 2024) https://arxiv.org/abs/2401.12070 | Perplexity divided by cross-perplexity between an observer/performer LM pair | >90% TPR at 0.01% FPR for ChatGPT, zero-shot | Binary only. The *pairwise ratio* idea is useful as features across family LMs (e.g. log-ppl under Qwen vs under Gemma). |
| **Smaller LMs are better zero-shot detectors** (EACL 2024) https://arxiv.org/abs/2305.09859 | Cross-detection: score text from model A with model B | OPT-125M is the best universal cross-detector, 0.07 AUC below self-detection on average. GPT-4: AUC 0.90 with OPT-125M vs 0.71 with OPT-6.7B | Implies **self-likelihood (same-family LM) gives the strongest signal**, and small LMs are reasonable generic scorers. |
| **Sniffer / origin tracing** https://arxiv.org/abs/2304.14072 | For each pair of known open models, the fraction of tokens where one assigns lower perplexity; linear classifier | Multi-class macro-F1 44.7 sentence-level / 67.5 document-level (from SeqXGPT's table) | Same pairwise-discrepancy idea; modest on its own. |
| **LLMDet** https://arxiv.org/abs/2305.15004 | Proxy perplexity from a dictionary of salient n-gram next-token probabilities, per candidate model | Source tracing among open models | Needs the candidate models' probabilities. We only have siblings. |
| **SeqXGPT** (EMNLP 2023) https://arxiv.org/abs/2310.08903 | Token log-prob "waves" from GPT-2-xl, GPT-Neo, GPT-J and LLaMA-7B, aligned to words; 5-layer CNN + 2-layer transformer | **95.7 / 94.2 macro-F1** for 6-way (sentence/document) when the generators *are* the white-box models; OOD 92.8, where Seq-RoBERTa gets 60.6 | Strong only in the white-box case. The architecture (sequence model over per-token log-prob, entropy and rank channels) is a cheap head to try on our family LMs. |
| **GLTR** (8th place in DAIGT) https://arxiv.org/abs/1906.04043 | Token-rank histogram (top-10/100/1000) | DAIGT: PPL+GLTR with GPT-2-large and 800k training texts, **0.956 private** (binary) | Weak for multi-way. M4GT: GLTR-LR 45%, GLTR-SVM 50% on 7-way vs RoBERTa 97%. FDLLM: 6.7 F1 on 20 classes. |

**How this transfers.** The generators are closed frontier models, so we cannot compute their likelihoods. But for 4 of the 6 labs, **open siblings exist on Kaggle Models**:
- Qwen: `qwen-lm/qwen-3`, Qwen3.5, and the Qwen3.8-27B mirrors.
- Google: `google/gemma-4`.
- OpenAI: `danielhanchen/gpt-oss-20b`. MXFP4 on T4 needs fp32 workarounds or bnb-8bit.
- ZAI: GLM-4-9B, already used by Lucas.

GLM-5.3-Flash itself is open (Kaggle `jeannkouagou/glm-5-3-flash-nvfp4`), but it is reported at about 320B MoE, and NVFP4 needs Blackwell, so it is infeasible on T4.

Following the self-detection finding (Mireshghallah; Sun et al. on SFT inheritance), a Qwen3.8 text should get *relatively* higher likelihood under Qwen3.8-27B than under Gemma-4. So build per-passage features:
- mean, std and quantiles of token log-prob, entropy and rank under each family LM;
- pairwise differences or ratios between LMs, Binoculars-style;
- optionally the SeqXGPT-style CNN over aligned per-word channels.

Feed these into the stack.

The hidden-state probes already capture part of this, since hidden states determine the next-token distribution. So the expected gain is small, mostly at lab level (glm vs claude-opus-5, qwen vs others). One prompt-scoring pass of a 9–27B model over 6.7k × ~150 tokens on T4 takes roughly 0.5–1.5 h per LM.

---

## 4. Contrastive and metric learning

- **DeTeCtive** (NeurIPS 2024) — https://arxiv.org/abs/2410.20964
  - Multi-level SimCLR-style loss: human-vs-AI, then company, then model series, then specific model, with α=β=γ=1. Added to CE.
  - Inference is **kNN over a training-embedding database** (FAISS).
  - Results:
    - Beats plain contrastive learning: 96.15 vs 94.22 AvgRec on Deepfake cross-domain and cross-model.
    - Beats a linear classification head on the same encoder: 93.19.
    - Beats a RoBERTa CE classifier: 87.30 (+8.85).
    - Removing any single level costs 0.9–1.3.
    - Fine-grained attribution (TuringBench, 19 LLMs): 82.75 vs 81.73 accuracy for RoBERTa-multinomial, so +1.0.
  - **Training-Free Incremental Adaptation (TFIA):** adding OOD texts to the kNN database lifted unseen-domain AvgRec **82.6 → 89.6**.
  - *Transfer:* our labels have a natural hierarchy (lab → model; also an "OpenAI 5.6 series" level). A lab-level positive plus a model-level positive is a direct port. kNN/prototype heads over a test-augmented database are a transductive bonus.
- **FAID** (2025) — https://arxiv.org/abs/2505.14271
  - Multi-level contrastive learning at family level, XLM-R encoder, fuzzy kNN (K=20) with test samples added to a temporary vector database.
  - Family-level accuracy 0.80–0.91.
  - Confirms that kNN over a contrastively trained space generalises better to unseen generators.
- **Two Birds with One Stone / DA-MTL** (2025) — https://arxiv.org/abs/2508.14190
  - Shared encoder with a detection head and an attribution head, λ = 0.5/0.5. Average attribution F1 +1.6 to +2.7 across 9 datasets; up to +6.5 for XLM-R on MULTITuDE.
  - *Transfer:* add a **lab-level auxiliary head** (6 classes), plus possibly a "series" head (gpt-5.6 trio; gemini-flash trio), to the LoRA model.
- **Wegmann et al. 2022, "Same Author or Just Same Topic?"** — https://arxiv.org/abs/2204.04907
  - Contrastive style embeddings trained with **content-controlled pairs** (same conversation, different author) are the most content-independent; better than domain control or no control.
  - *Transfer:* our training set is perfectly content-controlled, since every prompt is answered by all 18 models. Use **same-prompt passages from other models as hard negatives**, and **same-model passages from other prompts as positives**. Then the loss can only be reduced through style. This is the training-time counterpart of prompt centring.
- **Gunel et al. 2021, SCL for fine-tuning** (ICLR) — https://arxiv.org/abs/2011.01403
  - CE + supervised contrastive term. Significant few-shot gains over RoBERTa-large and more robustness to label noise. Gains shrink as data grows.
  - We are in the few-shot regime: 150 per class.
- **Rivera Soto et al. 2024, few-shot detection with style representations** (ICLR) — https://arxiv.org/abs/2401.06712
  - LUAR/UAR style encoders trained on 5M Reddit authors identify an LLM from N=5–10 support documents. pAUC 0.87 at N=5 and 0.96 at N=10 (single target); 0.91 pAUC at 32 tokens.
  - Controlling for topic and length matters: ProtoNet pAUC 0.765 without control vs 0.844 with both.
  - *Transfer:* a frozen LUAR embedding (Kaggle-uploadable, about 82M parameters) is a cheap, **topic-robust**, diverse ensemble member, and prototypes per class are natural.
- **LMSYS / DAIGT practice:** DAIGT 1st trained an embedding model with supervised contrastive loss for kNN over test essays (the "Ahmet" weak-labelling variant scored 0.971 private).

**Recommended loss for the LoRA model:**

`L = CE_model + 0.5·CE_lab + λ·SupCon` (λ ≈ 0.1–0.5, τ ≈ 0.1)

- Use the L2-normalised pooled vector.
- Each batch holds several prompts × several models. In SupCon, positives = same model; hard negatives = same prompt, other model.
- Keep the **group-centred CE head** (§5) on the same batch.

Expected gain: +0.5 to +2 points (DeTeCtive and DA-MTL magnitudes). Cheap to try because the batch structure already exists in lightningv08's same-prompt micro-batching.

---

## 5. Removing the topic (prompt) nuisance

### 5.1 Evidence

- **Recursion Cellular Image Classification (Kaggle 2019).** This is the clearest analogy: the "plate" or batch effect plays the role of our "prompt" effect.
  - 5th place, "AdaBN [domain==plate]": batches drawn from the same plate, with test batch statistics per plate. Validation accuracy: random batches **40%+** → same cell type 50%+ → same experiment 60%+ → **same plate 70%+**.
  - 1st place: per-plate class exclusion + Linear Sum Assignment + progressive pseudo-labelling (§6).
  - https://www.kaggle.com/c/recursion-cellular-image-classification/discussion/110362 , /110543
- **This competition:**
  - Centred twins (Lucas): +0.024 at 1–2 passages per prompt, +0.052 at 18.
  - Centred LoRA head (lightningv08): 0.470 / 0.491 / 0.553 at 1 / 3 / 6 contexts, vs plain 0.4816.
  - Topic projection (rudrabhaskar).
  - A function-word view (Lucas). This is the "text distortion" idea from cross-topic authorship attribution (Stamatatos): mask content words and keep function words and punctuation.
- **Speaker and face verification:** PLDA (Prince & Elder 2007, https://mlanthology.org/iccv/2007/prince2007iccv-probabilistic) separates identity variation from nuisance variation (pose, channel). Nuisance attribute projection removes the subspace of channel variability.
- **INLP** (Ravfogel et al. 2020, https://arxiv.org/abs/2004.07667) repeatedly trains linear classifiers for an attribute and projects onto their null-space, removing that attribute.

### 5.2 What is wrong with current centring

Write x = a_model + b_prompt + ε.
- Centring a test passage by the mean of its recovered cluster of size n gives x_i − mean = (a_i − ā_cluster) + (ε_i − ε̄).
- For small n, ā_cluster contains a fraction 1/n of the passage's *own* author vector. That shrinks the signal by (1 − 1/n) and mixes in the authors of the other members. This is why centring hurts at 1 context (0.470 < 0.482) and only pays off at larger n.
- Lucas's train twins see n = 18 groups; test clusters are usually 2–5 in size. lightningv08 and Public D already train on 2–4 chunks to reduce this mismatch.

### 5.3 Better estimators (new; none of them is in public notebooks)

1. **Shrinkage centring.** Subtract n/(n+κ) · cluster mean, with κ tuned on simulated pools. Singletons are left almost untouched, and large groups are close to fully centred.
2. **Alternating (EM-like) crossed-effects estimate:**
   - Estimate the prompt offset as b̂ = mean over the cluster of (x_j − Σ_c p(c|x_j)·â_c), where â_c are the class means learned on prompt-centred train features.
   - Reclassify x_i − b̂, and iterate 2–3 times.
   - This removes the author component from the topic estimate instead of subtracting it twice.
   - With distinct-author marginals (Kirill's decoder), p(c|x_j) can come from the joint, which also handles "one model per prompt" softly.
   - This is the two-way (model × prompt) random-effects model, i.e. PLDA with a crossed nuisance factor.
3. **Projecting out the topic subspace (INLP/NAP).**
   - Compute the 150 train prompt means, do PCA, keep the top k directions, and project them out of every feature vector before the heads. Alternatively, run INLP with a prompt-ID or topic classifier.
   - **No test clustering is needed**, so it helps singletons too. It complements group centring.
   - With external data (1,900+ full 18-model prompt groups in Oleg's corpus) the topic subspace can be estimated much better than from 150 prompts.
4. **Train the LoRA end-to-end** with a group-centred head on random group sizes 2–6, drawn from the realistic distribution of test cluster sizes, rather than only 3-tuples. Optionally add a set-transformer layer over the group so the model learns "style relative to siblings".
   - Recursion 1st tried "reference images" for the same idea and failed.
   - But here the siblings share the *content*, which makes the contrast far more informative.
   - Try this after items 1–3, which are cheap.

---

## 6. Transductive tricks (4,000 test passages at inference)

### 6.1 Test-time clustering of prompts
Already done in the 0.541 notebook: TF-IDF + GLM average-linkage, pair precision 0.94, recall 0.77; or Qwen3-Embedding connected components.

Possible improvements:
- Use the **prompt-recovery instruction embedding** ("Identify the specific topic...").
- Add a **mutual-kNN** graph and let cluster sizes be anything from 1 to 18.

Recursion's lesson: performance depends on getting the group right. Measure purity on simulated pools.

### 6.2 Pseudo-labelling and test-time training

| Source | Recipe | Gain |
|---|---|---|
| Recursion 1st | **Progressive** PL. Each epoch, add the K most confident test predictions (margin top1−top2) whose class is not already taken on that plate; keep only the most confident when two collide; final decode with LSA | Single model 0.98997 → **0.99700** private |
| DAIGT 1st | DeBERTa-small trained *at inference* with a train+test tokenizer, MLM on train+test, and test pseudo-labels from Mistral | 0.943 private on its own; diversity for the ensemble |
| DAIGT 3rd | Add 1k test rows where the transformer ensemble is confident (p<0.01 or >0.99) to the TF-IDF training set | TF-IDF branch 0.893 → **0.927** private |
| DAIGT 5th | Teacher soft-labels the test set → fine-tune short-context students (128/256 characters) on random test chunks → average overlapping chunks | 0.970 → **0.977** private. *Iterative hard PL was dropped as too high-variance.* |
| DAIGT 21st | Iterative top/bottom-X PL (4–5 rounds) | Public +0.007, private mixed (0.922–0.957); a cautionary example |
| LMSYS 2nd / 3rd / 4th | Ensemble soft-labels 30k–500k *external* unlabelled examples, then retrain | 3rd: "PL was huge"; Gemma2-RM + PL 0.880 LB |
| Jigsaw-ACRC 2025 1st / 3rd / 6th | LoRA "test-time training" of Qwen3-14B / 8B / Qwen2.5-14B **inside the notebook** on test-provided examples (Unsloth on T4) | Winning pattern. DML co-training +0.004 private |

**Recipe for this competition (fits in about 3 h on T4×2):**
1. Run the full ensemble on the test set.
2. Select pseudo-labels: confidence > τ and margin > m, and **distinct within the recovered cluster** (Recursion-style collision rule). Use soft targets (ensemble probabilities, temperature ≈ 1), not hard labels.
3. Cheap version (CPU, minutes): refit the frozen-feature LR heads (Lucas's 9 heads) on train + PL, with PL weight ≈ 0.3–0.5.
4. Medium version: one extra LoRA epoch on train + PL from the trained adapter. lightningv08 measured about 850 s per epoch on 2.7k rows for Qwen3-8B, so roughly 25–35 min.
5. Make it progressive, as in Recursion: add 30% → 50% → 70% of the test set over 2–3 rounds.

Accuracy matters: at a base BA of about 0.55, the top-30% confidence slice is probably 0.8–0.9 accurate. Check this in prompt-grouped simulations before trusting it.

### 6.3 Constrained assignment
- Recursion 1st: LSA with per-plate exclusion of impossible classes. All 277 classes appear on each plate, so the constraint is exact.
- **Here:** hard Hungarian assignment helps only with full 18-model groups (CPU notebook: 0.4427 → 0.4842 in CV). It fails on the LB because the test is evidently **not** full blocks: the class counts are unequal.
- Exact distinct-author marginals (Kirill) gave +0.006 in CV and nothing on the LB.
- Keep a **soft** constraint only: marginals or repulsion, with the strength tuned on simulations of 1–5 passages per prompt.

### 6.4 Class priors under balanced accuracy (important theory point)
- Balanced accuracy is (1/K) Σ_y ∫ 1[ŷ(x)=y] p(x|y) dx. So the BA-optimal decision is **ŷ = argmax_y p(x|y)**, the class-conditional likelihood, *whatever the test prior*.
- Training is balanced (150 per class), so a calibrated p_train(y|x) is already ∝ p(x|y).
- **Re-weighting by an EM-estimated test prior** (Saerens 2002; Alexandari et al. 2020, https://arxiv.org/abs/1901.06852, the state of the art for label shift) would *lower* BA. It moves decisions toward the large classes.
- What helps instead is removing **class-specific bias** from the classifier: Alexandari's "bias-corrected calibration" terms, or partial Sinkhorn.
  - The 0.541 Sinkhorn with α=0.5 toward uniform marginals gave +0.002 to +0.007.
  - A principled variant: Sinkhorn toward the **EM-estimated true marginals** π̂ rather than uniform. If the classifier were unbiased, predicted counts would match the true counts, so the per-class offsets this finds are pure bias corrections.
  - Test both in simulations with unequal class counts. The public and private mixes differ, so never hard-code a prior.
- **TIM** (Boudiaf et al. NeurIPS 2020, https://arxiv.org/abs/2008.11297): transductive mutual-information maximisation, i.e. a conditional-entropy term plus a marginal-entropy term. It is +2–5% over the best few-shot methods, including under domain shift and with many classes.
  - It can be applied to the final linear heads over the test set in seconds. Its marginal term is the balanced-accuracy-friendly version of Sinkhorn.

### 6.5 kNN / label propagation over a test-augmented database
- DeTeCtive TFIA (+7 AvgRec on unseen domains) and FAID (fuzzy kNN with test samples added).
- Label spreading over a kNN graph of *prompt-centred* embeddings of train + test.
- A cheap transductive member for the blend.

---

## 7. Data: external data, generation and augmentation

- **Data dominated the wins in DAIGT.**
  - 1st: 160k mixed examples. "Each of our individual models would have been in the gold medal range" because of the data.
  - 2nd: pretrained on 500k pairs, then fine-tuned on in-domain data, to 0.967/0.983.
  - 5th: 1.7M documents.
- **LMSYS:** 3rd generated 500k responses with vLLM and soft-labelled them. 1st did a one-epoch post-pretraining on 33k extra examples.
- **Here:**
  - Oleg's corpus: 24,015 responses at 40–250 words, 1,902 complete 18-model prompt groups, 135k responses in total. Luna labels are noisy.
  - seyitkaan's Claude API data: 5k passages, +0.016 on a TF-IDF baseline.
  - The reconstructed prompt templates (pjm) allow matched generation. API generation is a rules grey zone; publish anything generated.
- **How to use external data safely** (lessons from Lucas's collapse and from M4GT and FDLLM domain-shift numbers):
  1. Filter to a single paragraph with no markdown, 40–250 words, and English prose. Strip `**bold**` wrappers the way the host did.
  2. Pretrain → fine-tune: one epoch on external data, then on train (LMSYS 1st, DAIGT 2nd, Jigsaw 3rd). This beats mixing, because the final stage is pure in-domain.
  3. Balance classes per stage. Weight Luna lower, or drop the Luna quarantine file.
  4. Exploit the complete 18-model prompt groups for **group-centred and SupCon training**. This is exactly the structure §4 and §5 need, and it is 13× more prompts than train. More prompts also give a better topic subspace for INLP/NAP.
  5. Validate *only* on train-prompt folds.
- **Length augmentation (random cropping):**
  - Evidence:
    - Idiosyncrasies: accuracy rises with length up to ~256 tokens, and even the first token is informative.
    - DAIGT 5th: students trained on 128- and 256-character chunks, with overlapping-chunk averaging at inference, worked well as test-time students.
    - System-attribution (2026): 800-character prefixes lose nothing versus full text.
  - Caveat: test passages are *complete* generations with a target length, not crops. A truncated 250-word answer is not the same as a model's native 60-word answer.
  - Recipe:
    - With p≈0.3 per sample per epoch, replace the text with a **prefix ending at a sentence boundary**, length uniform in [40, n] words. Keep the opening; it is the most informative part.
    - Optionally use **suffix crops** (drop the first 1–2 sentences) with p≈0.1, so the model also learns ending style.
    - Never crop below 40 words.
    - At inference, average the full text with 1–2 prefix crops of long passages (TTA), at low weight.
  - Expected gain is small (0 to +1 point). It is mainly a regulariser for 2.7k samples. Check it on the CV.
- **Do not** use paraphrase or back-translation augmentation. Rewriting changes the "author" (Idiosyncrasies: paraphrased text is still classifiable, which means it carries the rewriter's style) and adds label noise.

---

## 8. Ensembling and distillation

- **Logit blending with OOF-fitted weights.** 0.541 already fits L-BFGS weights on seed-0 OOF. DAIGT 1st used rank averaging, which is not needed for a multi-class argmax.
- **Diversity:** the Qwen3-8B and Gemma-4-12B LoRAs agree on only 63% of passages, and combining them gives 0.498 vs 0.485 / 0.472 single (qasminumber). Prefer **different backbones** over more seeds.
- **LoRA fold soup** (LMSYS 1st): average the LoRA weights of the 5 fold models into one adapter. Inference cost is 1× instead of 5×, and the result is usually close to the 5-model average.
- **Distillation** (LMSYS 1st: Llama3-70B and Qwen2-72B teachers → Gemma-2-9B student, "the most important aspect"):
  - Teachers could be Qwen3.8-27B or Qwen3-32B QLoRA, trained on Kaggle T4 as single slow jobs.
  - Or use the whole ensemble as the teacher: soft-label train OOF, test and external unlabelled text, then distil into one or two LoRA students.
  - This pays off mainly if inference time becomes the constraint, which it is not yet (0.541 runs in < 2 h).
- **Deep Mutual Learning** (Jigsaw 6th): co-train 2–3 peer LoRAs with a symmetric KL term. +0.004 private each, and the DML ensemble beat independently trained models (0.9271 vs 0.925).
- **Hierarchical blending:** p(model) = p(lab) · p(model | lab), with **sibling specialists** for {luna, sol, terra}, {gemini-3.6/3.7/3.8-flash}, {5 Claude models}, {glm-5.3, glm-5.3-flash}.
  - Train each specialist on within-lab data with **prompt-matched pair or triplet inputs**: the difference of embeddings for the same prompt answered by sibling A vs sibling B.
  - Idiosyncrasies and FDLLM show that siblings are where the errors concentrate. The forum notes that about half of all errors stay inside a lab.

---

## 9. Prior Kaggle competitions: what won and whether it transfers

### LLM - Detect AI Generated Text (2023–24, binary, AUC)
- **1st** (rbiswasfc / nbroad / ubamba98) — https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/473295 , code https://github.com/rbiswasfc/llm-detect-ai
  - 160k-example data mix (40k human), built from many LLMs with diverse decoding (contrastive search, high temperature/top-k), LLMs instruction-tuned on Persuade, and augmentations.
  - Single models (private): Mistral-7B QLoRA (r=64, QKVO) **0.984**; Ghostbuster with Llama-7B + TinyLlama **0.974**; DeBERTa-v3-large ranking **0.963**; train+test tokenizer + MLM + PL DeBERTa-small **0.943**; SupCon embedding kNN weak labelling **0.971**.
  - Rank-average ensemble, weighted toward Mistral.
  - *Transfer:* the "data first" lesson and the LLM-LoRA backbone transfer. The test-vocabulary tokenizer does **not** (see below).
- **2nd** (wowfattie) — /discussion/470395
  - DeBERTa-v3-large pretrained on 500k SlimPajama human/AI pairs: 0.916 / 0.967. Fine-tuned on in-domain data: 0.94 / 0.98. Ensemble: **0.967 / 0.983**.
  - *Transfer:* the pretrain → fine-tune staging.
- **3rd** (evgeniimaslov2) — /discussion/470333
  - 12 DeBERTa-v3-large models trained on up to 1.2M generated continuations from about 35 open LLMs with random sampling parameters, plus TF-IDF.
  - Hard-example mining loop: add the 500+500 worst-predicted samples per iteration.
  - 1k confident test pseudo-labels for the TF-IDF branch: private 0.893 → 0.927.
  - Per-prompt UMAP distance post-processing: small gain.
  - *Transfer:* confident pseudo-labels for the sparse and linear heads; hard-example mining over external data.
- **5th** (jsday96) — /discussion/470093
  - 1.7M documents. Teacher → **test-time short-context students** with overlapping-chunk averaging: 0.970 → 0.977.
  - Iterative hard PL discarded as unstable.
  - Dropout disabled entirely: DeBERTa StableDropout behaved oddly.
  - *Transfer:* soft-label test-time distillation and the crop idea.
- **8th** (abdullahmeda) — /discussion/470224
  - GPT-2 PPL + GLTR features with a voting classifier on 800k texts: **0.956 private** (from 0.656 with 44k texts, so data mattered).
  - **Training the tokenizer on the test vocabulary hurt.**
  - The public test-vocabulary TF-IDF pipelines crashed to about 0.89 private.
  - *Transfer:* log-prob features need large data. Test-fitted vocabulary is a public-LB overfit.
- **21st** (asalhi) — /discussion/470148
  - Iterative self-training on test with BPE and TF-IDF: public 0.975 → 0.984, private only 0.922–0.957.
  - *Transfer:* a warning. Iterative hard PL can overfit the public LB.
- **Ahmet's unsupervised approach** — /discussion/465156
  - Binary-count kNN density ratio, "30th best match to each class", 0.91 with no training data.
  - *Transfer:* kNN density-ratio scoring is a cheap transductive member.
- **Why the test-tokenizer/TF-IDF trick does not apply here:** it worked in DAIGT because the hidden test had host-injected typos and character obfuscations that a test-fitted BPE vocabulary captures. Our passages are clean frontier-LLM text whose topics are unseen. Test-specific n-grams are mostly *topic* words, which are noise for us. Fitting IDF on train+test is harmless, but expect about 0 gain.

### LMSYS Chatbot Arena Human Preference (2024, 3-way, log loss)
- **1st** (sayoulala) — /lmsys-chatbot-arena/discussion/527629 , https://github.com/shyoulala/LMSYS_BlackPearl
  - Post-pretrain 1 epoch on extra data.
  - 5-fold Llama3-70B and Qwen2-72B QLoRA teachers (r=64, α=128, all linear) → **distil logits into Gemma-2-9B**.
  - Average the 5 fold LoRAs, GPTQ 8-bit, TTA.
  - "The most important aspect is distillation using larger models."
- **2nd** (tascj) — /discussion/527685
  - Full fine-tuning (no LoRA) of Gemma-2-9B / 27B / ArmoRM-8B, prompt-grouped CV.
  - **Full A/B swap augmentation with gradients accumulated in the same step: +0.003.**
  - 240k pseudo-labelled external examples, stage 2 → stage 3.
  - Custom MLP head, re-initialised.
- **3rd** (Tenenholtz + conjuring92) — /discussion/527766
  - Start from **reward-model checkpoints** instead of chat models.
  - 500k generated and soft-labelled examples ("PL was huge").
  - Gemma-2 soft-capping off: +0.002.
  - An auxiliary model-identity loss did not help.
- **5th** (Team Danube) — /discussion/527669
  - Pre-training Gemma-2-9B on UltraFeedback reward data: "**up to 20 points** on CV and LB" (0.020 log loss).
- *Transfer:*
  - Gemma or Qwen 9–14B LoRA as the core.
  - Task-related pretraining before in-domain fine-tuning.
  - Large-scale soft PL.
  - Fold-LoRA averaging.
  - Swap-style symmetry augmentation: our analogue is permuting the members of same-prompt groups, which is already implicit.

### LLM Prompt Recovery (2024)
- 1st and 2nd exploited the sentence-T5 metric (the "lucrarea" token ≈ `</s>` in embedding space; mean-prompt brute force). Little transfers.
- Minor transferable points:
  - 2nd: an LLM trained to directly regress the target embedding with a cosine loss, i.e. metric learning.
  - A small validation set built to correlate with the LB.
- https://www.kaggle.com/competitions/llm-prompt-recovery/discussion/494343 , /494497

### Jigsaw - Agile Community Rules (2025)
- 1st (wowfattie, /jigsaw-agile-community-rules/discussion/613305):
  - Qwen3-14B 0.9239, Qwen3-8B 0.9236, Qwen2.5-14B 0.9232, Ettin-400M encoder 0.8944 private; ensemble 0.9293.
  - Unsloth makes 14B LoRA training fit one T4.
  - Loss on the answer token only.
- 3rd (/613324): pretrain the LoRA on public data, then **TTT inside the 12 h inference window**. Embeddings of the fine-tuned LLM are fed to LGBM/SVC.
- 6th (/613150): **Deep Mutual Learning**, +0.004 private per model.
- *Transfer:* training LoRAs on Kaggle T4 inside the submission notebook is a proven pattern. That enables TTT with test pseudo-labels. Qwen3-14B > 8B.

### Recursion Cellular Image Classification (2019): structure-exploitation precedent
- Per-plate AdaBN: validation 40% → 70%.
- LSA under per-plate class constraints.
- Progressive pseudo-labelling with a uniqueness rule: 0.990 → 0.997.
- https://www.kaggle.com/c/recursion-cellular-image-classification/discussion/110543 , /110362 , /102905
- *Transfer:* the group-normalisation and progressive-PL ideas transfer. The hard LSA requires the constraint to be exact, which it is not here.

### 2026 competitions similar to this one
- None found besides Name That LLM itself (deadline 2027-01-06, $50k).
- The 2022 ML Model Attribution Challenge (MLMAC, https://arxiv.org/abs/2302.06716) attributed fine-tuned models to base models by *querying* them. That is active fingerprinting, as in LLMmap (https://arxiv.org/abs/2407.15847: 95% on 42 LLM versions with 8 queries). It does not apply to passive classification of fixed passages.

---

## 10. Not applicable, or negative

- **Watermark detection** (Kirchenbauer et al.; SynthID-Text in Gemini; "Who wrote this code?" code watermarking): needs the provider's secret keys. Sending text to a detector service is also forbidden (data must not leave).
- **Active fingerprinting** (LLMmap, Hide-and-Seek): we cannot query the models.
- **Zero-shot LLM-as-judge attribution:** M4GT human annotators are below chance; general LLM embeddings without fine-tuning are unreliable for attribution (2025 studies).
- **EM label-shift prior correction on the decision:** hurts balanced accuracy (§6.4).
- **Hard Hungarian assignment:** the test does not have full 18-model blocks.
- **Test-fitted tokenizer/vocabulary:** DAIGT-specific artefact; 8th place found it harmful.
- **Paraphrase or back-translation augmentation:** introduces another model's style as label noise.
- **Mamba or other unusual architectures:** DAIGT 5th found Mamba-790m dragged the private LB down.

---

## 11. Proposed v2 architecture (if we are willing to replace the pipeline)

**Stage A: offline training (Kaggle GPU, weights shipped as datasets)**
1. **Backbones:** Qwen3.5-9B, Gemma-4-12B and Qwen3-14B, all layers; optionally Qwen3.8-27B with QLoRA as a teacher or member.
   - LoRA on q, k, v, o, gate, up and down.
   - r=64, α=32 (try 128/64).
   - fp16 base, fp32 adapters.
2. **Inputs:** raw text; no chat template.
3. **Stage A0 (pretrain):** 1 epoch on filtered external 18-label data.
   - Oleg short plus cropped long passages, and seyitkaan Claude data. Luna down-weighted.
   - Batches are made of same-prompt groups.
4. **Stage A1 (fine-tune):** 3 epochs on train with group batches of random size 2–6 from the same prompt, plus 20–30% prefix-crop augmentation.
5. **Heads:**
   - plain-18;
   - centred-18 (minus the group mean, or a shrinkage version);
   - lab-6 auxiliary;
   - SupCon projection (positives = same model, hard negatives = same prompt).
6. **5 prompt-grouped folds** for OOF, plus a full-data model. Make fold LoRA soups for inference.

**Stage B: inference notebook (T4×2, ≤ 9 h)**
1. **Features and probabilities:**
   - LoRA models: plain + centred + lab heads.
   - Frozen probes + LR heads (Lucas / Public D), with topic-subspace projection.
   - TF-IDF / SVM / stylometry.
   - Optional family-LM log-prob features.
2. **Test grouping:** prompt-recovery embedding + TF-IDF; clusters of size 1–18.
3. **Group adaptation:** shrinkage or alternating crossed-effects centring. Centred heads added as evidence (add mode).
4. **First-pass blend:** OOF-fitted logit weights, then a hierarchical lab × model refinement with sibling specialists.
5. **Test-time PL:** confident, distinct-in-cluster, soft labels.
   - Refit the LR heads (CPU).
   - Optionally one LoRA epoch on train + PL (Unsloth, one model per T4).
   - Re-predict.
6. **Decoding:** soft distinct-author marginals within clusters; partial Sinkhorn toward the EM-estimated marginals, or a TIM marginal term. Pick the variant by simulation with unequal class counts.

**Validation protocol** (as in `top_notebooks.md`, mandatory):
- Prompt-grouped 5-fold CV.
- Simulated test pools with 1–2, 1–5 and 3–8 passages per prompt and **unequal class counts**.
- Clustering run on the test pool only, inside the simulation.
- Report BA per pool type and the lab-level BA.
- Tune nothing on the public LB.

**Suggested experiment order (cheapest first):**
1. Topic-subspace projection on the frozen heads, plus shrinkage and alternating centring (CPU, hours).
2. Confident-PL refit of the frozen and linear heads (CPU).
3. TIM / EM-marginal Sinkhorn decoding (CPU).
4. LoRA recipe changes (FFN, rank, aux lab head, SupCon, group sizes 2–6, crop augmentation) on Qwen3-8B. About 1 h per fold on a T4.
5. External-data pretraining stage, then fine-tune.
6. New backbones (Qwen3.5-9B, Gemma-4-12B, Qwen3-14B).
7. Family-LM log-prob features.
8. In-notebook LoRA TTT.
9. Distillation or DML, if time allows.

---

## References (URLs)

Papers:
- Sun et al., Idiosyncrasies in Large Language Models (ICML 2025): https://arxiv.org/abs/2502.12150
- FDLLM: A Dedicated Detector for Black-Box LLMs Fingerprinting (2025): https://arxiv.org/abs/2501.16029
- Abburi et al., Generative AI Text Classification using Ensemble LLM Approaches (AuTexTification model attribution; ensembles of PLM probabilities + classical classifier): https://arxiv.org/abs/2309.07755
- Merkhofer et al., Machine Learning Model Attribution Challenge (2023): https://arxiv.org/abs/2302.06716
- McGovern et al., Your Large Language Models Are Leaving Fingerprints (2024/25): https://arxiv.org/abs/2405.14057
- Suzuki et al., Natural Fingerprints of Large Language Models (2025): https://arxiv.org/abs/2504.14871
- System Attribution in LLM Brand Recommendations (2026): https://arxiv.org/abs/2610.00253
- Wang et al., M4GT-Bench (ACL 2024): https://arxiv.org/abs/2402.11175
- SemEval-2024 Task 8 overview: https://arxiv.org/abs/2404.14183 ; AISPACE: https://aclanthology.org/2024.semeval-1.212
- Guo et al., DeTeCtive (NeurIPS 2024): https://arxiv.org/abs/2410.20964
- FAID, fine-grained AI-text detection (2025): https://arxiv.org/abs/2505.14271
- Two Birds with One Stone, multi-task detection and attribution (2025): https://arxiv.org/abs/2508.14190
- Verma et al., Ghostbuster (NAACL 2024): https://arxiv.org/abs/2305.15047
- Hans et al., Binoculars (ICML 2024): https://arxiv.org/abs/2401.12070
- Mireshghallah et al., Smaller LMs are Better Zero-shot MGT Detectors (EACL 2024): https://arxiv.org/abs/2305.09859
- Li et al., Origin Tracing and Detecting of LLMs (Sniffer): https://arxiv.org/abs/2304.14072
- Wu et al., LLMDet: https://arxiv.org/abs/2305.15004
- Wang et al., SeqXGPT (EMNLP 2023): https://arxiv.org/abs/2310.08903
- Gehrmann et al., GLTR: https://arxiv.org/abs/1906.04043
- Abassy et al., LLM-DetectAIve (EMNLP 2024 demo): https://arxiv.org/abs/2408.04284
- Pasquini et al., LLMmap (USENIX Sec 2025): https://arxiv.org/abs/2407.15847
- Huang, Chen, Shu, Authorship Attribution in the Era of LLMs (survey): https://arxiv.org/abs/2408.08946
- Rivera Soto et al., Few-Shot Detection of MGT using Style Representations (ICLR 2024): https://arxiv.org/abs/2401.06712
- Wegmann et al., Same Author or Just Same Topic? (RepL4NLP 2022): https://arxiv.org/abs/2204.04907
- Gunel et al., Supervised Contrastive Learning for PLM Fine-tuning (ICLR 2021): https://arxiv.org/abs/2011.01403
- Boudiaf et al., Transductive Information Maximization (NeurIPS 2020): https://arxiv.org/abs/2008.11297
- Alexandari et al., Max-Likelihood with Bias-Corrected Calibration for Label Shift (ICML 2020): https://arxiv.org/abs/1901.06852
- Ravfogel et al., INLP / Null It Out (ACL 2020): https://arxiv.org/abs/2004.07667
- Prince & Elder, PLDA (ICCV 2007): https://mlanthology.org/iccv/2007/prince2007iccv-probabilistic
- Przystalski et al., Stylometry recognizes human and LLM texts in short samples (2025): https://arxiv.org/abs/2507.00838
- Barlas & Stamatatos, Cross-domain AA with pre-trained LMs (2020): https://pmc.ncbi.nlm.nih.gov/articles/PMC7256385
- Zhang et al., Deep Mutual Learning: https://arxiv.org/abs/1706.00384

Kaggle write-ups:
- DAIGT 1st (short): https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470121 ; (full) /discussion/473295 ; code https://github.com/rbiswasfc/llm-detect-ai
- DAIGT 2nd: /discussion/470395 · 3rd: /discussion/470333 · 5th: /discussion/470093 · 8th: /discussion/470224 · 21st: /discussion/470148 · Ahmet unsupervised: /discussion/465156 · BPE tokenizer: /discussion/458522
- LMSYS 1st: https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527629 · 2nd: /527685 · 3rd: /527766 · 4th: /529067 · 5th: /527669
- LLM Prompt Recovery 1st: https://www.kaggle.com/competitions/llm-prompt-recovery/discussion/494343 · 2nd: /494497
- Jigsaw Agile Community Rules 1st: https://www.kaggle.com/competitions/jigsaw-agile-community-rules/discussion/613305 · 3rd: /613324 · 6th: /613150
- Recursion Cellular 1st: https://www.kaggle.com/c/recursion-cellular-image-classification/discussion/110543 · 5th (AdaBN by plate): /110362 · plate structure: /102905

Kaggle Models checked for family-likelihood LMs (exact refs):
- `danielhanchen/gpt-oss-20b`
- `google/gemma-4`, `google/gemma-3`
- `qwen-lm/qwen-3`, `qwen-lm/qwen-3-5`
- `dangkhoa2016/qwen-qwen3-8-27b`, `ugvfpdcuwfnh/qwen3-8-27b`, `mikedan7/qwen3-8-27b-fp8-official`
- `marquis03/glm-4`
- `jeannkouagou/glm-5-3-flash-nvfp4` (infeasible on T4)
- `helium990/kimi-k3` (infeasible)
- `answer-ai/modernbert`
