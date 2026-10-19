# Data sources (everything git does not track)

Git tracks code, notebooks, logs and small result CSVs only. Everything below is git-ignored and is re-created with
`bash tools/fetch_data.sh` (needs your own `~/.kaggle/kaggle.json`, `git`, `git-lfs`, and the project venv).
Secrets are never shared: each person uses their own `kaggle.json`, `.env` (Gemini key) and Benchmarks login.

| Local path | Source | How it is fetched | License / notes |
|---|---|---|---|
| `data/` | Kaggle competition `name-that-llm` | `kaggle competitions download` | competition rules; never commit or share outside the team |
| `research/ext_oof/q35heads/` | Kaggle dataset `lightningv08/ntl-oof-q35-heads` | `kaggle datasets download` | public; prompt-disjoint OOF log-probs of 37 heads = the "strong" CV base |
| `research/ext_data/prvsiyan/` | Kaggle dataset `prvsiyan/prvsiyans-extended-name-that-llm-dataset` | `kaggle datasets download` | CC BY 4.0 |
| `research/ext_data/extra_api/` | Kaggle dataset `seyitkaangunes/ntl-extra-api-passages` | `kaggle datasets download` | CC BY 4.0; Claude passages + the 1,000-prompt bank (`prompts.csv`) |
| `research/ext_data/lucas/` | Kaggle dataset `lucasboesen/ntl-free-proxy-generations` | `kaggle datasets download` | CC BY 4.0 |
| `research/ext_data/abdullah/` | Kaggle dataset `abdullah1afify/ntl-generations` | `kaggle datasets download` | CC BY 4.0 |
| `research/ext_data/dophuchung/` | Kaggle dataset `dophuchung/ntl-proxy-generations` | `kaggle datasets download` | CC BY 4.0 |
| `research/ext_data/oleg/` | Kaggle dataset `olegzholobov/name-that-llm-response-corpus` | `kaggle datasets download` (4 files) | mixed per-source licenses (see its provenance file); we use license-known rows only |
| `external_dataset/speechmap-data/` | GitHub `xlr8harder/speechmap-data` | `git clone` + `git lfs pull` (target-model files) | Apache-2.0; provider output terms may apply |
| `research/ext_data/speechmap/paragraphs.csv` | derived | `python experiments/speechmap/parse.py` | from the clone above |
| `research/ext_data/oleg/oleg_paragraphs.parquet` | derived | `python experiments/oleg/parse.py` | from the Oleg files above |
| `research/ext_data/unified/` | derived | `python experiments/ext_unified/build.py` | Gemini task 1 |
| `research/ext_data/speechmap/kaggle_ds/` | ours (built from SpeechMap) | Kaggle dataset `mdmuqtadirfuad/ntl-speechmap-paragraphs` (PRIVATE, share with team) | Apache-2.0 source |
| `research/ext_data/kbench/` | ours: Kaggle Benchmarks proxy generations (b002, b002r, b003, ...) | Kaggle dataset `mdmuqtadirfuad/ntl-own-generations` (PRIVATE, to be created) | b001 is contaminated, never use |
| `research/ext_data/gemini_api/` | ours: Gemini API free tier (`tools/gemini_collect.py`) | same private dataset as kbench | served-model-id verified |
| `research/kernels*/`, `research/kernel_outputs/` | public notebooks / their outputs | `kaggle kernels pull <ref>` / `kaggle kernels output <ref>` | reading only; refs are in `research/top_notebooks.md`, `other_notebooks.md` |
| `nb/*/full_out/`, `nb/*/out/` | our Kaggle kernels' outputs | `kaggle kernels output mdmuqtadirfuad/<kernel>` | needs team access to our kernels |
| `research/ext_data/hf/` | Hugging Face `Qwen/Qwen3-0.6B-Base`, `HuggingFaceTB/SmolLM2-135M` | `huggingface-cli download` | small local test models only |
| `external_dataset/eqbench*`, `lechmazur_debate/`, `hf_priority/` | see `external_dataset/download_manifest.csv` (URLs per file) | Gemini's downloader (`tools/download_priority_datasets.py`) | evaluated, not used in submissions |

Rules reminder: team members only (merge teams on Kaggle before sharing anything private); any external data used by a
submission must be public or meet the "minimal cost" standard before the deadline.
