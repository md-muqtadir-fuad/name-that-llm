# Decision log (append-only)

| Date | Decision | Reason |
|---|---|---|
| 2026-10-16 | Evaluate only with prompt-grouped CV (GroupKFold by prompt_id) | random splits leak topic; CV tracks LB within ~0.014 |
| 2026-10-17 | Drop hard/soft one-model-per-group constraints | LB 0.544 -> 0.536 (E002) |
| 2026-10-17 | Paid API data deferred; free sources first (Kaggle Benchmarks $10/day, Gemini free tier) | user budget decision |
| 2026-10-17 | Skip claude-fable-5 generation | cost ($0.016/passage); not on Benchmarks proxy |
| 2026-10-17 | Prompt source = seyitkaangunes prompts.csv (1,000, public, disjoint from train) | reproducible; matches seyit's Claude passages |
| 2026-10-17 | Length instruction switched from shared range "(lo-hi words)" to per-generation "(about N words)" | prvsiyan pilot: shared range makes length model-dependent (80% vs 1% in train) |
| 2026-10-17 | Served-model-id check mandatory; mismatches quarantined | gemini-3.7-flash served as 3.8 on the Developer API |
| 2026-10-17 | Drift panel frozen = b001 file (range format, g0003-g0062, 12 labels); re-run monthly | time-series of API register drift |
| 2026-10-17 | External data used for within-lab gates first, not raw 18-way training | register drift vs host data (prvsiyan table; seyit forum post) |
| 2026-10-17 | Generated data published as public Kaggle dataset before any submission uses it | competition rule 2.6 |
| 2026-10-17 | Repo private on GitHub until 2027-01-13 | public code sharing must be on Kaggle during the competition |
| 2026-10-17 | b001 discarded; every kbench prompt runs in kbench.chats.new(orphan=True) with max_completion_tokens=4000 | prompt() accumulates chat history; default max tokens reserve $3.2/call |
| 2026-10-17 | Keep lab gates (claude, gemini, gpt), not data-driven cluster gates | E023 cluster gates <= lab gates |
| 2026-10-18 | Daily Benchmarks batches use 10 priority models; gemini-3.1-pro-preview and gpt-6-astra dropped (drift panel keeps all 12, weekly) | $10/day ran out at ~540 passages; these two are expensive and already easy classes |
| 2026-10-18 | Ablation in 3 stages: continuous CV with/without; OOF-harness ablation + sensitivity at final freeze; late-submission ablation on the private test after the deadline | private-test ablation is only possible after the competition; final-submission choice needs stage 2 |
| 2026-10-27 | **Gemma stop rule (user):** v11b = public 0.651 + Gemma-4-12B two-stage as a proper MODELS row (centred twin, log-space blend, w 1.0) is Gemma's last submission test. If v11b does not beat the 0.651 base on the public LB, Gemma is dropped as a submission member (paper result only) and no further GPU goes to Gemma. | v9 LSTM -0.002, v10b Gemma plain 15% -0.004 on the 0.639 stack |
| 2026-11-01 | Drift panel month 0 = drift_202611 (frozen file unchanged, g0003-g0062, range format, 720/720 ok) | b001 is chat-contaminated and mostly failed; b002 uses another window and format so it cannot be this panel's month 0 (next_steps had that wrong) |
| 2026-11-01 | Gemma decision moves off v11b: v11b is built on the 0.651 base (Sinkhorn 1.0, hard guards + Hungarian, `--post 1`), the same base that timed out as v11a. Gemma enters a submission only if it gets real weight in an honest NNLS next to the 0.664 members (seeded-split OOF) and raises that OOF; otherwise it is dropped with no submission spent | a v11b submission would most likely time out like v11a and test nothing; the honest-OOF gate is stricter than one noisy LB reading (+-0.009) |
| 2026-11-02 | **No paper unless the method is ours (user).** Paper-only work stopped: monthly drift panel, final-freeze ablation and sensitivity grid. All effort goes to the score; the paper reopens only if an own method beats the 0.664 base on honest OOF and on the LB | our LB best (0.639) is a public notebook's score; our additions lowered it |
| 2026-11-02 | GPU kernels and submissions use the GPU byod image `gcr.io/kaggle-private-byod/python@sha256:2757e0c7...` (as every GPU-scored 0.66x public notebook); `tools/check_meta.py` now fails on a `gcr.io/kaggle-images/*` (CPU) image with enable_gpu | those images give torch no GPU while the T4 session is charged; v11a copied avazaku's pulled metadata (CPU image 17b2...) and most likely ran its hidden rerun on CPU -> timeout (not the ATLAS budget) |
| 2026-11-02 | Honest OOF (seeded split, test-like 2-3 groups, infer.py estimated clusters) on the 0.664 members we can score: keep the hand weights (NNLS -0.004); Sinkhorn 0-0.45 equal, never 1.0; Gemma two-stage (plain w2 + centred twin 2.8) +0.011 (31/40 draws, ~4 se) -> passes the Gemma gate; v12 = 0.664 + Gemma row | E065-E067b |
| 2026-11-02 | **Gemma dropped completely (user):** no Gemma in submissions, no Gemma retraining, no further Gemma proposals. v12 (0.664 base + Gemma) scored 0.658 public | user judged it useless after v10b 0.635 and v12 0.658; honest-OOF gain (+0.011) did not show on the LB |
