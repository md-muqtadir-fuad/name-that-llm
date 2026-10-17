# Experimental protocol (fixed from 2026-10-17)

Every experiment, generation batch and submission is logged in `paper/experiments.csv` via `paper/log.py`.
Nothing is reported in the paper unless it has a log row. Results files referenced by a row must not be deleted.

## 1. Evaluation (fixed)
- Data: `data/train.csv` (2,699 passages, 150 prompts, 18 labels). Metric: balanced accuracy (mean per-class recall).
- Folds: `sklearn.model_selection.GroupKFold(n_splits=5)` on `prompt_id`, default ordering (no shuffling).
  Robustness check for any claimed gain: a second split = GroupKFold on prompts shuffled with seed 1.
- Reported for every result: OOF balanced accuracy, delta vs the stated baseline, folds with positive delta (k/5).
  Confidence interval: paired prompt bootstrap (1,000 resamples of the 150 prompts, seed 0), 95% percentile CI.
- Acceptance rule for a method: delta >= +0.005, positive in >= 4/5 folds, and CI lower bound > 0.
- Simulated test pools (transductive methods): held-out rows sampled 1-3 per prompt, seeds 100-103.
  Caveat (logged finding): pools built from complete held-out prompt blocks are near-oracle for covariance methods.
- Reference "strong" base for blend-level tests: public prompt-disjoint OOF `research/ext_oof/q35heads`.

## 2. Seeds
- numpy / python `random` / sklearn `random_state`: 0 unless stated. LoRA training: seed 42 (fixed adapter init).
- Generation length targets: `word_range(prompt_id, model)` = seeded lognormal (mu 4.6, sigma 0.45) clipped [40, 250].

## 3. Generation pipeline (fixed)
- Prompts: public `seyitkaangunes/ntl-extra-api-passages/prompts.csv` (1,000) in order; batch = window [START, START+N).
- User message only, no system prompt, provider defaults (no temperature / thinking parameters).
  Instruction suffix: "Write it as a single self-contained paragraph with no headings or lists. (about N words)".
- Every row keeps: prompt_id, label, requested model id, served model id (where the API returns one), prompt text,
  target N, raw response, timestamp, route, cost (where available). Served-id mismatch => row quarantined.
- Raw downloads are kept unmodified under `research/ext_data/<source>/<batch>/`; cleaning is a separate, logged step.
- Data must be published as a public Kaggle dataset before any submission uses it.

## 4. Drift panel (fixed, re-run monthly)
- `tools/kbench/ntl_gen_task_drift.py` is FROZEN (never edited): prompts g0003-g0062, the 12 Kaggle-Benchmarks
  labels, original range-format instruction "(lo-hi words)", per-prompt range. Month 0 = batch b001 (2026-10-17).
- Re-run on the 1st-3rd of each month with BATCH='drift_YYYYMM'; compare per-label statistics (em-dash share,
  curly apostrophe share, sentence length, words, type-token ratio) and a fixed classifier's accuracy across months.

## 5. Ablation and sensitivity (final system)
- Ablation: remove one component at a time from the final blend (each member, twins/centering, gates, balancing,
  test standardisation, external data), report OOF delta + CI, and the LB score where a submission exists.
- Dry sensitivity (CPU, cached OOF only, no retraining): perturb each blend weight x{0.5, 0.75, 1.25, 1.5}, gate
  lambda, Sinkhorn alpha {0, 0.25, 0.5}, LR C x{0.5, 2}, twin weights; report OOF range. Plus class-mix sensitivity:
  Dirichlet(alpha in {1, 3, 10}) resampled class mixes, 200 draws.
