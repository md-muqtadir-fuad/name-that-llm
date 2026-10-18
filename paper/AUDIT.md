# Code / analysis audit (append-only)

| Date | Where | Issue | Impact | Status |
|---|---|---|---|---|
| 2026-10-17 | tools/kbench/ntl_gen_task.py (b001) | `llm.prompt()` appends to one running chat -> passage k saw k-1 earlier Q&A; growing context hit 403 quota | b001 contaminated | fixed (chats.new per prompt, max tokens); b001 excluded (E024) |
| 2026-10-17 | nb/v4/v4.py (kernel v1, v2) | `\b` written as backspace in gate token_pattern | Gemini/GPT gate features wrong | fixed in kernel v3 before any submission |
| 2026-10-17 | tools/gemini_collect.py | gemini-3.7-flash served as 3.8 | 8 mislabeled rows | relabelled; served-version check added |
| 2026-10-17 | experiments/gates/gate_test.py | binary LinearSVC returns 1-D decision | crash (GLM gate) | fixed |
| 2026-10-17 | experiments/papers/argon_tests.py | sampled more rows than available | crash | fixed |
| 2026-10-17 | experiments/papers/moe.py | within-lab accuracy printout used global labels vs local indices | display only | noted; MoE results unaffected |
| 2026-10-17 | experiments/papers/twhiten.py | unused variable line | none | noted |
| 2026-10-17 | CORAL / transductive whitening | held-out folds are complete prompt blocks -> near-oracle covariance | gains unproven | logged as unproven (E014, E015) |
| 2026-10-17 | gate_test lambda | chosen on the reporting folds | mild optimism | same lambda best for all labs; CI reported |
| 2026-10-17 | prvsiyan passages_clean.csv | checked for train_regen rows / train openings | none found (splits new, proxy_v1 only) | OK |
| 2026-10-17 | prvsiyan proxy_v1 (516 rows) | generated via Kaggle Benchmarks proxy, possibly same multi-turn pattern | possible contamination | gate test re-run without proxy_v1 (pending) |
| 2026-10-17 | prvsiyan proxy_v1 (gate re-run E027) | without proxy_v1: claude/gemini unchanged, GPT gate +0.0052 -> +0.0011; combined +0.0174 -> +0.0137 | rows help on clean CV; kept in v4 (CV is the judge), caveat logged | resolved |
| 2026-10-18 | v4 kernel v3 | Kaggle assigned no GPU (gpus 0); ran 12 h on CPU | v4 never scored | v5 asserts NGPU>=1 at start |
| 2026-10-18 | git commit 3805b91 | nb/lora_gemma_ext/full_out (incl. train_plus_ext.csv = competition train text) committed and pushed to the PRIVATE repo; .gitignore only covered nb/lora_gemma/full_out | data-security rule | untracked in 4th commit after; ignore now covers nb/**/full_out; history purge pending user OK |
| 2026-10-18 | ntl_gen_task.py param fallback | errors mentioning max_output_tokens (e.g. quota reservation) were treated as wrong-parameter and overwritten by a TypeError | 183 b002 errors mislabeled | fallback only on TypeError unexpected keyword / 400 unsupported param |
