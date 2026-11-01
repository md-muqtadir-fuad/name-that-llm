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
