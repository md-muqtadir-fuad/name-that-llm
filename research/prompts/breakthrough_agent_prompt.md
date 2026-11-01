# Task for you: find what closes the gap from 0.66 to 0.74 in Kaggle "Name That LLM"

You are a senior ML researcher reviewing a Kaggle team's position. We are stuck and want an independent diagnosis and a
concrete plan, not a list of generic ideas. Everything below is measured unless marked *claim* (another team's report)
or *est* (our estimate). Read it all before answering.

## 1. The competition

- Task: attribute a short passage (40-250 words, median ~100) to one of 18 LLMs. Metric: balanced accuracy (macro recall).
- Labels (6 labs): Anthropic claude-fable-5, claude-opus-4-6, claude-opus-4-8, claude-opus-5, claude-sonnet-5;
  Google gemini-3.1-pro-preview, gemini-3.6-flash, gemini-3.7-flash, gemini-3.8-flash; OpenAI gpt-5.5, gpt-5.6-luna,
  gpt-5.6-sol, gpt-5.6-terra, gpt-6-astra; Zhipu glm-5.3, glm-5.3-flash; Moonshot kimi-k3; Alibaba qwen3.8-max.
- Train: 2,699 rows = 150 prompts x 18 models (one answer missing). Essay-style prompts. Length was set per request
  (prompt + model explain only ~6% of length variance), so length is not a useful feature.
- Hidden test: ~4,000 rows, ~40% public LB. Few passages per prompt (2-3 inferred: every prompt-grouping trick loses on
  the LB, and OOF scored on 2 answers per prompt correlates 0.91 with the LB vs 0.13 for all 18, *claim*). Organisers:
  class counts are unequal and public/private mixes differ.
- Code competition: Kaggle notebook, 2x T4 (15 GB each), 9 h limit, no internet. External data and models are allowed
  if public by the deadline (2027-01-13; own trained weights may stay private). 5 submissions/day.
- Our compute: Kaggle 30 GPU-h/week (T4x2), Google Colab, 4-core CPU locally. Kaggle Benchmarks: $10/day of generation
  through 12 of the 18 labels (NOT available: claude-fable-5, gemini-3.7-flash, glm-5.3, glm-5.3-flash, kimi-k3,
  qwen3.8-max). Paid API generation is currently off (owner's decision; tell us if the case for it is strong).

## 2. Leaderboard

- Top of public LB: 0.737 (2026-11-01); 0.703 on 2026-10-25; gold cut-off was ~0.66 then.
- Best public notebook: 0.664. Re-running identical code moves the public score by up to 0.009 (*claim*, many reruns).
- Our submissions: 0.544 -> ... -> 0.639 (a copy of a public notebook) ; +our word-BiLSTM 0.637 ; +our Gemma plain
  readout 15% 0.635 ; one timeout (cause found: the notebook metadata carried a CPU docker image).
- Next candidate (not yet submitted): v12 = public 0.664 + our Gemma two-stage member (see 4).

## 3. The best public stack (0.664), what it contains

Log-linear blend of member log-probs, then damped Sinkhorn (strength 0.45, target = 0.65 predicted marginal + 0.35 train
prior), then a 37-mark typography likelihood. Members (hand weights):
- LoRA24 two-stage members (stage 1 on a ~84k public pool, stage 2 on host folds; 24-27 kept layers, mean-pooled 8 layers,
  linear heads), each averaged over 5 fold adapters + a full-data adapter: Qwen3-14B-Base w3, Qwen3-8B-Base w2,
  Llama-3.1-8B w1. Each also has a "twin" head (prompt-centred features on estimated test clusters, weight 0.35 x w) and
  two have a shared-prompt Gaussian head (0.70, 0.44).
- Nishant two-stage Qwen3-14B (w2.5), Qwen2.5-14B (w1), DeBERTa-v3-large (w1).
- 4-view n-gram LinearSVC trained on train + a 198k public "open pool" (w12), TF-IDF LR (0.25), topic-masked TF-IDF (0.10).
- 15% "complementary stack" (frozen-feature heads, family gates).
- ATLAS 20%: class-conditional likelihood scoring with a Qwen3-8B bundle on rows with top-2 margin < 0.3, under a hard
  time deadline (fills the remaining 9 h budget).
- Test clusters: TF-IDF content words + capped average linkage at cosine 0.15.

## 4. Our honest out-of-fold measurements (this week)

The public "--sim" mode leaks (it averages fold adapters that saw the held-out rows). We rebuilt honest per-row OOF on
the adapters' own fold split (verified: each fold adapter scores ~0.60-0.64 on its held-out fold vs 0.88-0.93 in-sample)
and simulate the test: random groups of 2-3 answers per prompt, twin/Gaussian heads on those groups or on infer.py's own
estimated clusters. Balanced accuracy, single members: Qwen3-14B 0.635, Qwen3-8B 0.615, Llama-8B 0.593, SVC 0.504,
TF-IDF 0.421, topic-masked 0.322; our Gemma-4-12B two-stage (24 layers, same recipe): plain 0.578, prompt-chunk-centred
0.563 (folds 0-2 only).
Blend of the members we can score honestly (all but Nishant x3, DeBERTa, comp stack, ATLAS, typography): 0.666.
- NNLS-fitted weights: -0.004 vs the hand weights (1/12 draws better). The hand weights are near-optimal.
- Leave-one-out: SVC -0.011, Qwen3-14B twin -0.004, other twins -0.002, Gaussian heads ~-0.001, plain heads ~0.
- Sinkhorn strength 0-0.45 equal on uniform and skewed class mixes; 1.0 loses 0.002-0.017 under skew.
- Our Gemma as an extra member (plain w2 + centred twin 2.8): +0.011 (31/40 draws, ~4 SE) with estimated clusters.
  Up-weighting the existing twins (control) gives 0. Plain Gemma alone ~+0.003 (matches its earlier LB loss).

## 5. Error structure

- A prompt-grouped TF-IDF baseline: 0.344; 55% of its errors stay inside the right lab.
- Pairwise separability on paired prompts (same prompt, which-is-which, token n-grams, 1,600-2,500 pool prompts):
  gemini-3.7 vs 3.8-flash 0.53, glm-5.3 vs flash 0.55, claude-fable-5 vs glm-5.3 0.58 (cross-lab!), gpt-5.6-sol vs
  terra 0.61, luna vs sol 0.64, gemini-3.6 vs 3.7 0.70, gpt-5.5 vs terra 0.71, gemini-3.6 vs 3.8 0.73,
  opus-4-8 vs sonnet-5 0.75; cross-lab references 0.97.
- The 6 labels without Benchmarks data have the lowest recall (0.24 vs 0.39 in the baseline).
- Topic dominates representations: >99% of TF-IDF nearest neighbours answer the same prompt; prompt-centring helps only
  with many answers per prompt (with 2 it hurts, with 3 it is neutral).
- Typography separates labs: Claude/GLM closed em dashes and straight apostrophes, GPT curly apostrophes/quotes, Gemini
  almost no em dashes.

## 6. Already tried (with result) - do not re-propose without a specific reason it would differ

Ours (OOF on a 37-member bundle unless noted): CNN/BiLSTM two-stage (+0.016 OOF, -0.002 LB); POS/affect features
(-0.002); one-shot style-transfer log-probs (+0.005); uncertainty-routed weights (0); sibling specialists / pairwise
experts / 3-way specialist (~0, fixes = breaks); attention-BiLSTM for gemini-3.7/3.8 (0.52); surface-format features
(below blend); label-noise hypothesis for fable/glm (rejected); prompt residualisation (-0.002 to -0.006); CORAL /
transductive whitening (only works with full prompt groups); hard Hungarian assignment (loses on LB); NNLS weights on the
0.664 members (-0.004). Public (*claim*, LB deltas vs base): Ministral-3-8B member -0.010/-0.016, Qwen3.5-27B swap
-0.013, ModernBERT -0.006, second ATLAS -0.001, Response Corpus added to classifiers -0.011 to -0.024, any Hungarian
-0.017 to -0.028, lower cluster threshold -0.006; Llama removal -0.005 to -0.008 (Llama helps).
Rejected on reasoning: GRPO/DPO for a classifier (reduces to a margin loss; target pairs are near chance).

## 7. Data available

Public: open pool 247k passages (198k after excluding train-prompt regenerations; 56 sources), pool_v4 84k, darragh
1.5k (CC0), prvsiyan 5.5k, alessandrolair 3.9k (Benchmarks), olegzholobov Response Corpus 24k (different task mix).
Ours: ~2,000 Benchmarks passages on our own cleaned real-user prompts (12 labels), ~2,300 Benchmarks passages on public
prompts, 720-passage drift panel. Gemini free API (20 requests/day/model; gemini-3.7-flash is silently served as 3.8).

## 8. What we want from you

1. **Diagnosis.** Why is the best public stack at 0.664 while the top is 0.737? List the hypotheses you find plausible
   (e.g. data volume or match for specific labels, generation protocol match, backbone size or full fine-tuning,
   test-time / transductive use of the 4,000 test rows, other features, evaluation blind spots in our harness), each
   with: mechanism, evidence for and against from the numbers above, and what it implies about the 6 unavailable labels
   and the near-chance pairs.
2. **Ranking.** Rank them by expected LB gain per cost under our constraints (T4x2, 30 GPU-h/week, 9 h inference,
   no internet at inference, public-data rule, 10 weeks left).
3. **Decisive tests.** For the top 3-5, the cheapest experiment that would confirm or kill each, designed for our honest
   OOF harness (per-row OOF on a fixed prompt split, test-like 2-3 answer groups), with GPU/CPU hours and the acceptance
   threshold. Say what result would make you drop it.
4. **Stop list.** What we should stop doing.
5. **Blind spots.** Anything in our evaluation that could make honest OOF disagree with the LB.

Be concrete and quantitative. Prefer one well-argued idea over ten generic ones. If an idea needs paid API data, say what
it would cost and which labels it would cover.

## 9. If you have access to the repository (I:\venv_python\that_llm)

Useful files: `research/community_2026-11-01.md` (public notebooks, LB deltas), `experiments/honest/` (honest-OOF sim:
`stack_sim.py`, `stack_sim_groups.py`, outputs in `out/`), `paper/experiments.csv` (all experiments), `research/bottleneck/`
(sibling analyses), `nb/eda/build.py` (EDA), `research/TEST_PLAN.md`, `research/kernels_2026-11-01/` (public notebooks).
Rules: read only; do not open `.env*`, `kaggle.json` or `tools/kbench/.env.init`; do not push, submit, or run Kaggle
commands; write your report to `research/agent_reports/breakthrough_<your-name>.md` and nothing else.
