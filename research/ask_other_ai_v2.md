# Prompt for other AI models (status 2026-10-19)

Paste everything below the line.

---

I am competing in the Kaggle code competition **"Name That LLM"** and I am stuck. Please think hard and give concrete,
testable advice. Facts below are measured; do not re-suggest things listed as already tested unless you explain why a
different variant would behave differently.

## Task and constraints
- 18-class attribution: which LLM wrote a 40-250-word English passage. Labels: gpt-5.5, gpt-5.6-luna/sol/terra,
  gpt-6-astra, gemini-3.1-pro-preview, gemini-3.6/3.7/3.8-flash, claude-fable-5, claude-opus-4-6, claude-opus-4-8,
  claude-opus-5, claude-sonnet-5, qwen3.8-max, glm-5.3, glm-5.3-flash, kimi-k3. Metric: balanced accuracy.
- Train: 2,699 passages = 150 prompts x 18 models. Hidden test: ~4,000 passages from UNSEEN prompts (public/private split).
  The current #1 believes the hidden test has few passages per prompt.
- Code competition: inference notebook on 2x T4 (16 GB each, fp16 only), <= 9 h, internet off. Training may happen
  anywhere (we can rent an A100/H100). External data must be public or "minimal cost". Deadline 2027-01-13 (3 months left).
- Our resources: Kaggle GPU ~30 h/week (T4), about $300 total, free Kaggle Benchmarks proxy ~$10/day for 10 of the 18
  models (gpt-5.5, gpt-5.6 luna/sol/terra, gemini-3.6/3.8-flash, claude opus-4-6/4-8/5, sonnet-5), Gemini API free tier
  (20 requests/day per model).

## Leaderboard now
#1 0.650, then 0.607, 0.607, 0.597, 0.595; 58 teams >= 0.56. Best public notebook 0.563. **We are at 0.552 (rank ~362/1,597).**

## What the public 0.563 notebook is (we forked it)
Log-linear blend: style-aware TF-IDF (char 3-5, word 1-3, word-shape, function-word views) + LinearSVC (weight 1.5),
stylometric LightGBM, frozen-LLM heads (Qwen3-8B, Qwen2.5-7B, Qwen3.5-9B, GLM-4-9B layer features), encoders,
three Qwen3-8B LoRA classifiers (24 layers, multi-layer pooling), token-attention heads; then test-time "adaptive
prompt-centering" on inferred test clusters, partial Sinkhorn class balancing, and within-family gates (a family-only
SVC re-picks the label when the blend predicts that family). 0.552 -> 0.559 came from adding ~5.5k "in-style" external
passages (same prompt bank, "one paragraph of about N words", only prompts answered by >= 12 labels) to the TF-IDF SVC;
-> 0.563 from family gates + a Qwen2.5-7B LoRA trained on train + in-style passages (that LoRA is private).

## The #1 team's public notes
A single Qwen3-14B-Base LoRA (r16, all linear layers, classification head, 3 epochs, max_len 384, no extra data,
trained off-Kaggle) scores 0.466 public alone. Test clustering + Hungarian assignment gave +0.10 in grouped CV but
LOWERED public LB (0.353 -> 0.344 -> 0.329 with per-cluster centering). "More passages from the 18 models help."

## Our measured results (prompt-grouped 5-fold CV; "strong" = a 37-head frozen OOF blend at 0.49)
- Our Gemma-4-12B LoRA (24 of 48 layers): OOF 0.469; + 3.2k external passages 0.478; GLM-4-9B LoRA same recipe 0.4735.
  Gemma + GLM together add +0.045 on the strong blend (3/3 folds). **But on the public LB, adding two Gemma LoRAs +
  gates to the 0.552 notebook gave exactly 0.552 (CV promised ~+0.04).** Earlier: Gemma on a 0.541 base +0.003 LB.
- Within-family gates with external data: +0.017 CV.
- In-style pool in the TF-IDF SVC: 0.3845 -> 0.409 standalone. Adding SpeechMap (225k off-protocol API answers to
  controversial questions) HURTS (-> 0.390). Adding ~1.1k of our own Benchmarks generations (same protocol, 10 labels):
  no change. Threshold 10 vs 12 labels per prompt: no change.
- Tree models (LightGBM/ExtraTrees) on the TF-IDF: much worse (0.26-0.29 vs 0.385), no blend gain.
- Text "spectrograms" (position x band shares at char / word / sentence / token level): weak alone (char 0.26), no
  blend gain. Within-family contrast: Claude siblings differ in ; and : rates (opus-5 1.86x semicolons vs family),
  GPT-5.6 siblings in em-dash and apostrophe rates; **the three Gemini flash models and the two GLM models show no
  reliable character-level difference at all** (GLM within-pair CV 0.497 = chance).
- Sibling separability ceilings we measured: GLM pair ~0.50-0.56, GPT-5.6 trio ~0.56, Gemini 3.7 vs 3.8 near chance
  (3.7 is served as 3.8 on the current Gemini API, i.e. aliased).
- External data available: public in-style pool ~5.5k, prvsiyan 3.2k (CC BY), our Benchmarks ~1k (growing ~500/day),
  Oleg's corpus (36k paragraphs but 90% from 3 labels), SpeechMap (225k, off-protocol).
- API drift: today's Gemini APIs use em-dashes in ~30% of passages vs ~0% in the competition data, so freshly generated
  data may not match the test distribution.

## Where we are stuck
1. CV gains from new model members do not transfer to the public LB; we cannot tell which changes are real.
2. Siblings inside a lab (GLM pair, Gemini flash trio, GPT-5.6 trio, Claude opus 4-6/4-8/5) are the main error mass.
3. The gap to #1 (0.65 vs 0.55) is far larger than anything incremental we have found.

## Questions (answer each)
1. What is the most likely recipe behind 0.60-0.65, given the evidence above?
2. With $300, rentable GPUs and 3 months, which plan maximises the PRIVATE LB? Be specific: backbone, LoRA config,
   how many in-style passages per label, which labels to generate first, generation protocol (and how to handle
   API drift vs the competition data), number of seeds/folds.
3. How do we build a validation that tracks the LB when the test has unseen prompts and few passages per prompt?
4. Concrete methods for sibling separation that go beyond n-grams (likelihood ratios under sibling-specific
   adapters, contrastive training, pairwise heads, length/temperature artefacts, tokenizer-level cues, ...)?
5. Should we remove the transductive parts (adaptive centering, Sinkhorn balancing) from the public notebook?
6. What are we wasting time on? What should we stop?

## Answer format
A ranked list. For each item: idea | why it should work here | expected public-LB gain | cost (GPU-hours, $) |
cheapest test that would falsify it | main risk. Then a 2-week plan. Say explicitly where you are guessing.
