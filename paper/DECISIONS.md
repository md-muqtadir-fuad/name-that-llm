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
