# Repository tree

`[ignored]` = not in git; re-create it with `bash tools/fetch_data.sh` or the command named in [DATA_SOURCES.md](DATA_SOURCES.md).
Branch for ongoing work: `experiments/fuad`.

```
name-that-llm/
├── README.md              project overview, protocol summary
├── DATA_SOURCES.md        where every git-ignored file comes from and how to fetch it
├── TREE.md                this file
├── requirements.txt       venv freeze (Python 3.11.9; torch CPU wheel for local work)
├── LICENSE, NOTICE        Apache-2.0; credits for reused public notebook code
│
├── paper/                 everything the paper needs (all experiments are logged here)
│   ├── PROTOCOL.md        fixed CV / seeds / generation protocol / drift panel / ablation schedule
│   ├── experiments.csv    one row per experiment (E001...), written via log.py
│   ├── data_log.csv       one row per dataset / generation batch
│   ├── DECISIONS.md       decisions and why
│   ├── AUDIT.md           bugs found and fixes
│   ├── related_work.csv   literature scan (+ related_work_notes.md: novelty verdicts, links to verify)
│   └── log.py             log_row(...) helper
│
├── nb/                    Kaggle notebooks (source .py + pushed .ipynb + kernel-metadata.json)
│   ├── v1/ v3/ v4/ v5/    submitted inference notebooks (public LB: v1 0.544, v3 0.547, v5 0.552)
│   ├── v6/                v6a = public 0.563 notebook + our Gemma-ext LoRA (make_v6.py builds it); v6b dropped
│   ├── lora_gemma/        Gemma-4-12B LoRA trainer (train only)
│   ├── lora_gemma_ext/    Gemma-4-12B LoRA trainer (train + prvsiyan rows)
│   ├── lora_glm_ext/      GLM-4-9B LoRA trainer (train + prvsiyan rows)
│   ├── featcache*/ surpcache/   GPU feature-cache kernels for offline CPU experiments
│   └── */out, */full_out  [ignored] kernel outputs (kaggle kernels output mdmuqtadirfuad/<kernel>)
│
├── experiments/           local (CPU) experiments; each writes out/*.csv|log and a row in paper/experiments.csv
│   ├── gates/             within-lab gates (Claude / Gemini / GPT) with external data
│   ├── speechmap/         SpeechMap parser + value-of-data eval
│   ├── oleg/              Oleg corpus paragraph view
│   ├── ext_unified/       unified external corpus build + eval (Gemini tasks 1-2)
│   ├── instyle/           0.563 in-style stylegram scheme: SpeechMap / our Benchmarks data / label thresholds
│   ├── trees/             LightGBM / ExtraTrees vs LinearSVC (rejected)
│   ├── papers/            paper-method tests (residualize, MoE, CORAL, whitening, ...)
│   ├── grouping/ local/   test-clustering / assignment experiments
│   ├── style/ lexical/    char-LM surprisal, writeprints, stylometric specialists
│   ├── analysis/          model-family graph (research/model_graph.*)
│   └── gemma_oof.py       LoRA OOF evaluation on the strong blend
│
├── tools/
│   ├── fetch_data.sh      re-download all git-ignored data
│   ├── kbench/            Kaggle Benchmarks generation: ntl_gen_task.py, daily.py (+ state.json), drift task
│   ├── gemini_collect.py  Gemini API free-tier collection (+ gemini_daily.cmd for Task Scheduler)
│   ├── chat_collect.py    manual chat-app collection helper
│   └── download_priority_datasets.py   Gemini's benchmark-dataset downloader
│
├── research/              notes and pulled material
│   ├── SYNTHESIS.md       results table + what works / what does not
│   ├── TEST_PLAN.md       planned tests and the rejected list
│   ├── forum.md, top_notebooks.md, other_notebooks.md, papers.md, lexical_properties.md, model_graph.md
│   ├── gemini_tasks.md    prompts for work delegated to Gemini
│   ├── datasets_meta/     metadata/READMEs of external datasets (small)
│   ├── ext_data/          [ignored] external + our generated text (prvsiyan, seyit, lucas, abdullah, dophuchung,
│   │                      oleg, speechmap, kbench, gemini_api, unified); only SOURCE.txt files are tracked
│   ├── ext_oof/           [ignored] strong OOF bundle (lightningv08/ntl-oof-q35-heads)
│   ├── kernels*/ kernel_outputs/   [ignored] pulled public notebooks and their outputs
│   └── forum_raw*/        [ignored] raw discussion JSON
│
├── data/                  [ignored] competition data (train.csv, test.csv, sample_submission.parquet)
├── external_dataset/      [ignored] raw downloads: speechmap-data (GitHub, LFS), eqbench, lechmazur, hf_priority
├── comp_info/             [ignored] saved competition pages and rules
├── .venv/                 [ignored] Python environment
└── kaggle.json, .env, .env.kbench, tools/kbench/.env.init   [ignored] SECRETS - never commit or share
```
