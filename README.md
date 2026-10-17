# name-that-llm

Fine-grained authorship attribution of short LLM-written passages. This is the code and experiment record for
the Kaggle competition [Name That LLM](https://www.kaggle.com/competitions/name-that-llm): assign each
40–250-word paragraph to one of 18 frontier models (OpenAI, Google, Anthropic, Qwen, ZAI, Moonshot) on prompts not
seen in training, scored by balanced accuracy.

The repository also tracks a research study, run from October 2026 to January 2027, on:
- **prompt confounding** in attribution: topic dominates representations, and group-based corrections that work in
  cross-validation fail on real test sets;
- **sibling-model separability** within a lab;
- **API register drift**: the same model versions write measurably differently over time, and some versions are
  silently aliased;
- **scaling of generated training data** for attribution.

## Layout

| Path | Contents |
|---|---|
| `nb/` | Kaggle notebooks and training kernels (submission notebooks `v1`, `v3`; Gemma-4-12B LoRA trainer; feature-cache kernels) |
| `experiments/` | Offline CPU experiments on cached features and out-of-fold predictions; logs in `*/out/` |
| `tools/` | Data generation: Kaggle Benchmarks task (`kbench/`), Gemini API collector, manual chat collector |
| `paper/` | `PROTOCOL.md` (fixed evaluation and generation protocol), `experiments.csv` (one row per experiment), `log.py` |
| `research/` | Literature and notebook reviews, test plan, synthesis, lexical analysis |

## Protocol (summary)

- Evaluation uses 5-fold GroupKFold by `prompt_id` (never random splits), with a fixed seed of 0.
- Every result reports balanced accuracy, its delta against a stated baseline, folds positive (k/5), and a paired
  prompt-bootstrap 95% CI.
- A method is accepted only if the gain is at least +0.005, positive in at least 4 of 5 folds, and the CI excludes 0.
- Generated data records the prompt, target length, requested and served model id, timestamp, route and cost.
  Rows with a served-id mismatch are quarantined.
- A frozen drift panel is re-run monthly. See [`paper/PROTOCOL.md`](paper/PROTOCOL.md).

## Setup

```bash
python -m venv .venv && .venv/Scripts/pip install kaggle pandas pyarrow scikit-learn lightgbm kaggle-benchmarks
# Kaggle API token: ~/.kaggle/kaggle.json   (not in the repo)
# Optional keys in .env (not in the repo):  GEMINI_API_KEY=...
kaggle competitions download name-that-llm -p data && unzip data/name-that-llm.zip -d data
```

Model training and inference run on Kaggle (T4×2, internet off). Kernels are pushed with
`kaggle kernels push -p nb/<dir>`.

## Data and rules

- Competition data (CC BY-SA 4.0) is **not** included; download it from Kaggle after accepting the rules.
- External data comes only from public, free sources (Kaggle datasets, Kaggle Benchmarks model proxy, free API
  tiers). Generated passages will be published as a public Kaggle dataset before any submission uses them.
- During the competition this repository is private, as the rules require. It becomes public after the final deadline
  (2027-01-13).

## License

Apache License 2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE) for third-party attributions.
