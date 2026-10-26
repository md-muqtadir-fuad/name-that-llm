# Repository tree (updated 2026-10-26)

`[ignored]` = not in git; re-create it with `bash tools/fetch_data.sh` or the command named in [DATA_SOURCES.md](DATA_SOURCES.md).
Branch for ongoing work: `experiments/fuad`. Current plan and open tasks: [research/next_steps.md](research/next_steps.md).

```
name-that-llm/
├── README.md              project overview, protocol summary
├── DATA_SOURCES.md        where every git-ignored file comes from and how to fetch it
├── TREE.md                this file
├── TEAMMATE_SETUP.md      prompt that sets up a teammate's copy and verifies it
├── requirements.txt       venv freeze (Python 3.11.9; torch CPU wheel for local work)
├── LICENSE, NOTICE        Apache-2.0; credits for reused public notebook code
│
├── paper/                 everything the paper needs (all experiments are logged here)
│   ├── PROTOCOL.md        fixed CV / seeds / generation protocol / drift panel / ablation schedule
│   ├── experiments.csv    one row per experiment (E001-E053, LB_v8, LB_v9, ...), written via log.py
│   ├── activity_log.csv   every state-changing action (push, submit, dataset, run), via `log.py --act`
│   ├── data_log.csv       one row per dataset / generation batch (b002.., o001..)
│   ├── DECISIONS.md, AUDIT.md   decisions and why; bugs found and fixes
│   ├── related_work.csv   literature scan (+ related_work_notes.md)
│   └── log.py             log_row(...) and log_activity(...) helpers
│
├── nb/                    Kaggle notebooks / kernels (source + kernel-metadata.json); all PRIVATE until 2027-01-13
│   ├── v1/ v3/ v4/ v5/ v6/  earlier submitted notebooks (public LB v1 0.544 ... v6a 0.556)
│   ├── v7/                fork of public 0.607 (never submitted; superseded by v8)
│   ├── v8/                private copy of public 0.639 (kirilltym); public LB 0.639 = current best
│   ├── v9/                v8 + word-BiLSTM at 10% (make_v9.py, lstm_cell.py); public LB 0.637 (rejected)
│   ├── colab_twostage/    two-stage LoRA trainer: parts/header.py + parts/launcher.py -> build_trainer.py ->
│   │                      twostage_train.py (backbones olmo2-7b, qwen3-8b-base, gemma4-12b, llama31-8b); Colab notebook
│   ├── gemma_twostage/    build.py: Kaggle kernels that run twostage_train.py in sessions (smoke_*, ntl-gemma-ts-s1/s2,
│   │                      ntl-llama-ts-s1/...); stage 1 = pool_v4 + darragh run-1, stage 2 = host folds 0-2 + final
│   ├── llama_download/    CPU kernel: Llama-3.1-8B from Hugging Face into its output (needs Kaggle secret HF_TOKEN)
│   ├── lora_twostage/     Qwen3-8B two-stage run (E048)
│   ├── lora_gemma/ lora_gemma_ext/ lora_glm_ext/   earlier host-only LoRA trainers
│   ├── cnn_lstm/          CPU kernels: byte-CNN / word-BiLSTM CV (E050) and the final LSTM (lstm-final)
│   ├── oneshot/ pos_affect/ backbone_screen/   kernels for E052, E051, E047
│   ├── featcache*/ surpcache/   GPU feature-cache kernels for offline CPU experiments
│   └── */out, */full_out  [ignored] kernel outputs (kaggle kernels output mdmuqtadirfuad/<kernel>)
│
├── experiments/           local (CPU) experiments; each writes out/*.csv|log and a row in paper/experiments.csv
│   ├── cnn_lstm/          E050 byte-CNN / word-BiLSTM (two-stage on pool_v4)
│   ├── importance/        E049 exact Shapley of blend sources per label / family
│   ├── pos_affect/        E051 parts of speech with meaning (sentiment, gender, hedges)
│   ├── oneshot/           E052 one-shot style-transfer log-probs (training-free)
│   ├── robust/            E053 Sinkhorn balancing vs test class mix
│   ├── backbones/         E047 frozen-backbone screening eval
│   ├── spectro/           E037-E039, E046 text spectrograms and model signatures
│   ├── rl_featsel/        E041 RL feature-set search (rejected)
│   ├── hier/              E040, E042, E043 review falsification tests
│   ├── gates/ speechmap/ oleg/ ext_unified/ instyle/ trees/ papers/ grouping/ local/ style/ lexical/ analysis/
│   └── gemma_oof.py       LoRA OOF evaluation on the strong blend
│
├── tools/
│   ├── fetch_data.sh      re-download all git-ignored data
│   ├── kbench/            Kaggle Benchmarks generation: ntl_gen_task.py, daily.py (`--own` = own prompt set),
│   │                      state.json (used prompt ids, retries), ntl_gen_task_drift.py (FROZEN drift panel)
│   ├── prompts/           clean_prompts.py: scraped Reddit / Quora / Stack Exchange titles -> prompts.csv
│   ├── gemini_collect.py, chat_collect.py, download_priority_datasets.py
│
├── research/              notes and pulled material
│   ├── next_steps.md      CURRENT plan and open tasks
│   ├── TEST_PLAN.md       planned tests and the rejected list; SYNTHESIS.md results and method ideas
│   ├── leader_analysis_2026-10-22.md, leader_update_2026-10-24.md   leaderboard / public-notebook analyses
│   ├── own_prompts_sources.md   provenance of the own prompt set (sources, counts, licensing)
│   ├── importance/ pos_affect/ spectro/ backbones/   figures and summaries of E049 / E051 / E037+ / E047
│   ├── forum.md, top_notebooks.md, other_notebooks.md, papers.md, model_graph.md, twostage_SUMMARY.md
│   ├── ext_data/          [ignored] external + our generated text: prvsiyan, extra_api (seyit), lucas, abdullah,
│   │                      dophuchung, oleg, speechmap, twostage (pool_v1, pool_v4), darragh_run1, kbench (b*, o*),
│   │                      own_prompts (raw scrapes, prompts.csv, clean_report.csv), own_prompts_ds, lstm_final_ds
│   ├── ext_oof/           [ignored] strong OOF bundle (lightningv08/ntl-oof-q35-heads)
│   ├── kernels*/ kernel_outputs/   [ignored] pulled public notebooks and our kernel outputs
│   └── forum_raw*/        [ignored] raw discussion JSON
│
├── data/                  [ignored] competition data (train.csv, test.csv, sample_submission.parquet)
├── external_dataset/      [ignored] raw downloads: speechmap-data, eqbench, lechmazur, hf_priority
├── comp_info/             [ignored] saved competition pages and rules
├── .venv/                 [ignored] Python environment
└── kaggle.json, .env, .env.kbench, .env.hf, tools/kbench/.env.init   [ignored] SECRETS - never commit or share
```
