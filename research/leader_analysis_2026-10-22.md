# How the top teams reach 0.63-0.69 (analysis, 2026-10-22)

Leaderboard: #1 Hod Carriers 0.688, #2 Vishak K Bhat 0.667, #3 Adisak Sukul 0.658, then 0.644, 0.637, 0.634 x2;
public notebooks top out at 0.607. Ours: 0.556 (v6a).

## Evidence
| Source | What it shows |
|---|---|
| nishantdahal "Two-stage LoRA on the public pool" (0.592, checkpoints now public: `nishantdahal/ntl-two-stage-lora`) | Stage 1 LoRA on ~35k public passages only, stage 2 two epochs on host train: stage 1 alone = host-only LoRA, stage 2 adds ~9 CV points. Qwen3-8B 24L + Qwen3-14B 27L, 8-depth pooling. Single biggest public jump (0.563 -> 0.592). |
| prvsiyan "The Prompt Is Not the Author" (0.569) | Shared-prompt Gaussian heads: explicit model of the prompt component on inferred test groups. |
| kirilltym 0.594 / 0.595 / 0.600 | 85% two-stage ensemble + 15% old Gaussian feature stack, one final full Sinkhorn; gates dropped ("no gain"). |
| Adamski "Teach the Model + 10% ATLAS" (0.607) | ATLAS likelihood adapter at 10% weight: +0.007 on top of 0.600 (third time ATLAS helps). |
| sangam21 "Two-Stage LoRA + Gaussian + Full Hungarian" (0.604+) | Hard one-model-per-prompt assignment on inferred test groups helps on top of two-stage. |
| Vishak K Bhat (#2, 0.667) public launcher `v1-vishak` | 11-line notebook running a private `infer.py` + `blend.json`; attached backbones **OLMo-2-13B + Qwen3-8B-Base + Qwen3-14B-Base**; override `center_beta=0.0` (his own centering switched off). |
| Our v6e (E045) | Adaptive test-cluster centering is worth +0.019 on the hidden test: the test has same-prompt groups. |
| Forum 750706 / 750768 | Train = 150 full rounds of 18 models; test has unequal per-model counts and unseen prompts (probably subsampled rounds) -> "at most one per model per prompt" constraints are plausible. |
| Forum 751294 / 751011 | Teams are generating paid-API data and asking to publish it under rule 2.6; the top scores likely include large self-generated sets (not public yet). |
| Forum 751370 | Gemini 3.7 requests routed to 3.8 (also in our own logs): the 3.7/3.8 pair is near-inseparable for everyone, and fresh 3.7 data is impossible. |

## Most likely recipe of 0.65-0.69 (inference, ranked by evidence)
1. **Two-stage LoRA on big in-style data** (public 35k pool + their own generations), several seeds and folds. Evidence:
   public 0.563 -> 0.592 from this alone; #1's own note that "more passages help".
2. **Backbone diversity beyond Qwen**: OLMo-2-13B next to Qwen3-8B/14B (#2's attached models). Different pretraining
   corpora -> decorrelated errors (our Gemma+GLM pair: agreement 0.59, +0.012 over one LoRA).
3. **Prompt-structure exploitation on the test**: Gaussian shared-prompt heads, centering on inferred groups, Hungarian
   "one model per prompt" assignment (+0.019 for centering on our LB; Hungarian 0.600 -> 0.604+ publicly).
4. **Small likelihood member (ATLAS 10%)**: +0.002 to +0.007, consistent across four notebooks.
5. **Final balancing (full Sinkhorn)** for balanced accuracy; family gates dropped.
Gap 0.607 (public) -> 0.688 (#1) is most plausibly item 1 at larger scale (more generated data, more backbones/seeds)
plus item 2; we have no evidence of any single secret trick.

## Our position and plan
- Fastest +0.05: fork the public 0.607 notebook (`nb/v7`, all inputs public). Blocked until the weekly GPU quota resets.
- Then, one change per submission on top of v7: + our Gemma-ext / GLM-ext LoRAs; + OLMo-2-13B two-stage LoRA;
  wider adaptive centering / Hungarian.
- Train two-stage LoRAs on OLMo-2-13B and Qwen3-14B (research/next_steps.md), on Colab or a rented GPU while the
  Kaggle quota is exhausted.
- Data: keep the daily Benchmarks generation; completing the prompt grid for the 7-8 missing labels (paid, ~$35-45)
  remains the open spending decision.
