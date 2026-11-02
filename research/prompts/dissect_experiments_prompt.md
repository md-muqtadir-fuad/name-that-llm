# Task: dissect our recent experiments - root cause of each result and a concrete fix

You have read access to the repository `I:\venv_python\that_llm` (Kaggle "Name That LLM": attribute a 40-250 word
passage to one of 18 LLMs; metric balanced accuracy; train 2,699 rows = 150 prompts x 18 models; hidden test ~4,000 rows
with only 2-3 passages per prompt). For each experiment below, explain WHY it came out the way it did (root cause, with
evidence you computed), and what fix or next experiment follows. We need a mechanism, not a restatement of the number.

## Ground rules (read first; reports that break them will be discarded)

1. **Verify before you claim.** Every factual statement carries one label: `verified` (you ran code or read the exact line;
   give the command, file:line or number), `not verified`, or `estimate`. Never call something false without a check.
2. **Test-like regime only.** The test has 2-3 passages per prompt. Any grouping, centring or clustering must be done on
   one random group of 2-3 answers per prompt (or a partition of each prompt into 2-3 groups). Never cluster or centre
   all 18 answers of a prompt together: that is the oracle regime and inflates every centred head (an earlier report did
   this: median cluster size 18, +0.021 that shrinks to +0.010 test-like).
3. **Honest OOF only.** Public fold adapters: the seeded split (`RandomState(0)` shuffle of sorted prompt ids,
   fold = i % 5); each row is read only by the adapter that did not train on it. Our own members: our `GroupKFold(5)`.
4. **Paired comparisons with uncertainty.** Report differences as paired means over draws/folds with a standard error and
   a "better in k/n" count. A difference below ~2 SE is not a result.
5. **Subset vs full stack.** Our harness scores only the members with honest OOF (q14v5, q8v5, ll8v5 with twin and
   Gaussian heads; tfidf, distort, svc4). It cannot score n14b, q25_14, DeBERTa, the comp stack or ATLAS. Do not present
   subset numbers as leaderboard predictions.
6. **Gemma is dropped** (owner's decision). Do not propose Gemma fixes; Gemma data may be used only as evidence about
   general mechanisms.
7. **Scope.** Read anything except `.env*`, `kaggle.json`, `tools/kbench/.env.init`. Write scratch code only under
   `experiments/agent_scratch/` and your report to `research/agent_reports/dissect_<your-name>.md`. Do not push, submit,
   run Kaggle commands or long local jobs (> 10 min); if a test needs more, describe it instead.

## Where things are

- Honest OOF inputs: `research/honest_oof/gpu/honest_oof_q8_ll8.npz`, `honest_oof_q14.npz` (per member: `_plain`
  logits, `_rc` / `_bc` centred-head projection and bias, `_U` / `_m` / `_lam` Gaussian head, `_Pall` / `_Rall` every
  adapter on every row); `research/honest_oof/cpu_members_oof.npz` (tfidf / distort / svc4 log-probs);
  `research/honest_oof/cpu_own/cache/sim*_members.npz` (E068 variant).
- Harness and experiments: `experiments/honest/` - `stack_sim.py` (E066), `stack_sim_groups.py` (E067/E067b),
  `e068_compare.py`, `e069_bias.py`, `e070_chunk_head.py`, `e071_typography.py`, `e072_layers.py`; outputs in `out/`.
  The blend replicates the public `infer.py` (`nb/honest_cpu/infer_0664.py`): sum of weighted member log-probs,
  divided by the member count, damped Sinkhorn 0.45 toward 0.65 x predicted + 0.35 x train prior.
- Logs: `paper/experiments.csv` (all results), `paper/activity_log.csv`, `paper/DECISIONS.md`,
  `research/agent_reports/verification_2026-11-02.md` (what was checked in four earlier agent reports).
- Public stack and community evidence: `research/community_2026-11-01.md`.

## Experiments to dissect

| Id | What | Result (verified by us) |
|---|---|---|
| E065 | honest OOF of the 0.664 CPU members (svc4 on the 198k open pool, tfidf, distort) | svc4 0.504, tfidf 0.421, distort 0.322 |
| E066 | honest blend of the scorable members at the 0.664 hand weights vs NNLS-fitted weights | hand 0.6656, NNLS 0.6617 (-0.004, 1/12 draws) |
| E066 | leave-one-out on the hand weights | svc4 -0.011, q14v5 twin -0.004, other twins -0.002, Gaussian ~-0.001, q14v5 plain +0.0015 |
| E066 | Sinkhorn strength 0 / 0.3 / 0.45 / 0.6 / 1.0, uniform and Dirichlet(1, 3) class mixes | 0-0.45 equal; 1.0 worst under skew |
| E068 | our 3,333 Benchmarks passages (12 labels) added to the CPU members' training data | blend +0.0010 (se 0.0005); unserved-label recall down |
| E069 | per-class bias offset learned on 4 folds | ~0: gemini-3.7 recall +0.066, 3.8 -0.067; fable +0.06, glm-5.3 -0.034 |
| E070 / E070b | centred head trained on 2-3 chunks vs whole prompts; drop of each twin from whole-prompt to 2-3-group centring | chunks +0.026 on frozen features; all twins drop 0.07-0.08 |
| E071 | the 0.664 typography likelihood and its strength | +0.0005 at any scale / clip |
| E072 | frozen Phi-4-14B, OLMo-2-13B, Qwen3-8B: every layer, probe + gain on the honest blend | running; results in `experiments/honest/out/e072_layers.csv` when done |
| LB v12 | 0.664 public stack + Gemma row | public LB 0.658 (base author reported 0.664 for the stack without Gemma; identical code varies by up to 0.009) |

## Questions

1. For each row: the root cause of the outcome, shown with a computation (for example: which classes moved, which rows
   changed prediction, how correlated a member's errors are with the blend's, whether a gain is concentrated in a few
   classes or prompts).
2. Across experiments: is there one common reason why every post-hoc change to this stack lands at ~0 (weights, bias,
   typography, extra data)? Test your explanation (e.g. member error correlation, how often the blend's top-2 margin is
   small, how many errors are inside near-chance sibling pairs and what the ceiling is if those pairs were coin flips).
3. Fixes: for each fix you propose, give the mechanism it attacks, the cheapest decisive test in this harness (CPU minutes,
   or GPU hours with a design), the acceptance threshold (>= 2 SE, >= 3/4 of draws), and what result would kill it.
4. Blind spots: anything in the harness that could make it disagree with the leaderboard.

## Report format

1. One table per experiment: claim | label (verified / not verified / estimate) | evidence.
2. A ranked fix list: mechanism | test | cost | threshold | kill condition.
3. A short list of what you could not verify, and why.
