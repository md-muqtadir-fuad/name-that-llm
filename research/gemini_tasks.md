# Tasks for Gemini (copy one prompt at a time)

Order: Task 1 -> Task 2 (needs Task 1 output). Task 3 is independent (needs web search; AI Studio with Grounding or Deep Research).
Each task ends with a short SUMMARY.md so Claude only reads that file.

---------------------------------------------------------------------------------------------------------------------
## Shared rules (paste at the top of Task 1 and Task 2)

You are working in the local repo I:\venv_python\that_llm (Kaggle competition "Name That LLM": 18-class attribution of
40-250-word passages, metric balanced accuracy, hidden test uses unseen prompts).
Rules:
- Python: I:\venv_python\that_llm\.venv\Scripts\python.exe. Install nothing unless unavoidable (say which).
- NEVER print, copy or commit kaggle.json, .env, .env.kbench, tools/kbench/.env.init.
- NEVER put competition text (data/train.csv, data/test.csv) or external passage text into git. Text outputs go to
  research/ext_data/... (git-ignored). Only code and small aggregate CSVs/markdown may be committed.
- Do NOT git commit, do NOT push anything to Kaggle, do NOT submit. Leave changes in the working tree for review.
- CV protocol (fixed): GroupKFold(5) on data/train.csv grouped by prompt_id, default order, no shuffling.
  Acceptance rule for a gain: delta >= +0.005, >= 4/5 folds positive.
- Log every experiment row with paper/log.py: `from log import log_row` (fields: id, category, description, config,
  metric, baseline, value, delta, folds_pos, ci95, files, notes). Log every new dataset as one line in paper/data_log.csv
  (same columns as existing lines).
- Label set (exact strings): gpt-5.5, gpt-5.6-luna, gpt-5.6-sol, gpt-5.6-terra, gpt-6-astra, gemini-3.1-pro-preview,
  gemini-3.6-flash, gemini-3.7-flash, gemini-3.8-flash, claude-fable-5, claude-opus-4-6, claude-opus-4-8, claude-opus-5,
  claude-sonnet-5, qwen3.8-max, glm-5.3, glm-5.3-flash, kimi-k3.
- If something is ambiguous, choose the conservative option and write the choice in SUMMARY.md. Do not invent results.

---------------------------------------------------------------------------------------------------------------------
## Task 1 - Unified external corpus

[paste Shared rules]

Goal: build ONE clean external training corpus from every external source we have, with per-source provenance.

Sources (inspect each file's columns first; print shape + columns + 2 truncated rows):
1. research/ext_data/prvsiyan/passages_clean.csv            (column `model` = label)
2. research/ext_data/speechmap/paragraphs.csv               (label, variant, refusal, para_idx, question_id, text).
   Keep refusal==0, variant!='reasoning', para_idx==0.
3. research/ext_data/kbench/b002/**/ntl_gen_b002_*.csv      (llm, prompt_id, response, error). Keep error empty and
   response non-empty. Map llm -> label: strip provider prefix ("openai/", "google/", "anthropic/"), strip "@default",
   "-default", "-2026-04-23"; claude-opus-4-6 stays claude-opus-4-6. EXCLUDE batch b001 entirely (contaminated).
4. research/ext_data/gemini_api/responses.csv               (check columns; keep only rows whose served model
   version matches the label; gemini-3.7-flash rows do not exist - it is aliased to 3.8 on the API).
5. research/ext_data/extra_api/ntl_extra_passages_claude.csv (check columns/labels).
6. research/ext_data/oleg/external_short_wide.parquet        (check what it is; include only if it has the 18 labels).
7. Public Kaggle datasets - download with `.venv\Scripts\kaggle.exe datasets download -d <slug> -p research/ext_data/<short> --unzip`:
   lucasboesen/ntl-free-proxy-generations (short: lucas), abdullah1afify/ntl-generations (abdullah),
   dophuchung/ntl-proxy-generations (dophuchung). Record each dataset's license from its metadata.

Normalise every row to: source, label, prompt_key (source prompt/question id; prefix with source), text, words,
served_model (if known else ''), license.
Filters (same for all sources): label in the 18 labels; strip markdown **bold**/__/`; single paragraph (no blank
lines, no lines starting with #, -, *, digits+'.', '>', '|'); 40 <= words <= 250; ends with . ! ? or closing quote;
not a refusal (regex: I can't / I cannot / I won't / I'm sorry, but / I apologize, but in the first 400 chars).
Dedup: exact (lowercased, whitespace-collapsed) and near-duplicate (word 8-gram Jaccard > 0.8, use datasketch MinHash
if installed, else a simple 8-gram hash index); keep the first by source priority kbench > gemini_api > lucas >
abdullah > dophuchung > prvsiyan > extra_api > oleg > speechmap.
Leakage guard: drop any external row with word 8-gram Jaccard > 0.5 to ANY row of data/train.csv or data/test.csv.
Prompt-balance flag: add column `balanced` = 1 if, within the same source and prompt_key, there are rows from >= 3
labels covering >= 2 labs (lab = gpt/gemini/claude/qwen/glm/kimi), else 0. (Reason: prompts answered only by Claude
create a topic -> label shortcut.)

Outputs:
- research/ext_data/unified/ext_all.parquet  (git-ignored)
- experiments/ext_unified/build.py            (the code; rerunnable)
- experiments/ext_unified/out/counts.csv      (rows = label, cols = source, plus total and balanced_total)
- experiments/ext_unified/out/drops.csv       (per source: raw, after each filter, after dedup, after leakage)
- one paper/data_log.csv line per source
- experiments/ext_unified/out/SUMMARY.md (<= 25 lines: totals per label, smallest labels, sources dropped and why,
  licenses, any surprises)

---------------------------------------------------------------------------------------------------------------------
## Task 2 - Value of each source (needs Task 1)

[paste Shared rules]

Goal: measure which external sources help, using the existing harness experiments/speechmap/eval.py.
Make experiments/ext_unified/eval.py as a copy of experiments/speechmap/eval.py with these changes only:
- external rows come from research/ext_data/unified/ext_all.parquet (columns text, label, source, balanced).
- cap: at most 500 rows per (source, label), sampled with random_state=0.
- sets: none | PV (prvsiyan only, reference) | ALL | ALL_balanced (balanced==1) | ALL_minus_<source> for each source
  with >= 200 rows | ALL_no_SM (everything except speechmap).
- Test A (18-way SVC, w_ext=1.0, member weight grid as in eval.py) and Test B (gates, lam 1.5) for every set.
- Keep the strong base blend, calibration, folds, SVC spec and gate code byte-identical to eval.py.
- Log ids E033_<test>_<set>. Results to experiments/ext_unified/out/eval_results.csv.
It is CPU-only and may take 1-3 h; run it in the background and check it finishes.
SUMMARY.md (experiments/ext_unified/out/SUMMARY.md, append a section <= 20 lines): table of set -> A delta, A folds,
B delta, B folds, and which source removals hurt (= that source helps). Reference numbers to reproduce first:
eval.py with SM+PV gave A +0.0204 (4/5) and B +0.0107 (5/5) on base 0.4916; if PV-only does not reproduce
A +0.0133 within 0.002, stop and report.

---------------------------------------------------------------------------------------------------------------------
## Task 3 - Related-work scan for the paper (web search; independent)

You are helping check the novelty of a paper. Search arXiv, ACL Anthology, OpenReview, Google Scholar and Kaggle
competition write-ups. Only list items you actually found with a working link; never invent titles, authors or ids.
If unsure, mark verified=n.

Our candidate claims:
C1 Prompt-grouped CV vs a hidden test: transductive/test-distribution methods (prompt-centering, CORAL/whitening,
   balancing, test clustering) give large CV gains (+0.06..+0.19 balanced accuracy) that vanish or go negative on the
   hidden test in LLM authorship attribution.
C2 API identity drift: the same model name served over months changes style measurably (e.g. em-dash rate 0 -> ~0.3),
   and some API model ids are silently aliased to other versions; we keep a served-id-verified monthly series.
C3 Value of external/third-party generated data for LLM attribution under prompt/topic shift, incl. a topic->label
   shortcut when some prompts are answered by only one model family.
C4 Stylistic similarity across labs (e.g. one lab's model resembling another lab's) visible in attribution confusion.

Seed items to verify and place (do not assume they say what I guess):
- "Idiosyncrasies in Large Language Models" (Sun et al., 2025)
- ATLAS, arXiv 2610.19583 (class-conditional LoRA likelihood attribution)
- "How is ChatGPT's behavior changing over time?" (Chen, Zaharia, Zou, 2023)
- "Model Equality Testing: Which model is this API serving?" (Gao, Liang, Guestrin, 2024)
- work on auditing model substitution / model identity in LLM APIs (2025)
- FDLLM / LLM fingerprinting / authorship attribution of LLM text (Huang et al. 2024 survey "Authorship attribution in
  the era of LLMs")
- test-time adaptation / CORAL failures under label/prompt shift; CV leakage with grouped data (Kaufman et al. leakage)
- Kaggle "LLM - Detect AI Generated Text" (2023/24) winning write-ups (CV-vs-LB lessons)

Search queries to run (add more as needed): "LLM authorship attribution which model generated", "model attribution
LLM generated text fingerprint", "LLM API drift over time", "silent model update API", "model substitution detection
API", "prompt confounding authorship attribution", "topic leakage authorship attribution", "transductive domain
adaptation cross-validation optimistic", "CORAL test-time adaptation fails", "Kaggle LLM detect AI generated text
solution".

Output: ONE CSV (comma-separated, quoted) with header
id,title,authors,year,venue_or_arxiv,url,topic,what_it_shows,claim_overlap,overlap_level,verified
- topic: attribution_method | prompt_confound | transductive_cv | api_drift | aliasing | ext_data_shift | lineage |
  competition_writeup
- claim_overlap: C1..C4 or none; overlap_level: none | partial | full (full = already shows our claim)
- what_it_shows: one sentence, factual
Then below the CSV, <= 15 lines: for each of C1-C4, verdict (novel / partially covered / already done) with the ids
that cover it, and the gap we could still own.
Save as paper/related_work.csv (the CSV) and paper/related_work_notes.md (the verdict lines).
