# Teammate setup prompt

Paste everything below the line into Claude Code (or Gemini CLI) opened in an EMPTY folder where the project should live.

---

You are setting up my local copy of my team's Kaggle pipeline for the competition **"Name That LLM"**
(https://www.kaggle.com/competitions/name-that-llm): 18-class attribution of 40-250-word passages to the LLM that wrote
them, metric balanced accuracy, code competition (2xT4, <= 9 h, internet off), deadline 2027-01-13. I am an official
member of the team. Reproduce my teammate's pipeline exactly, verify it, and report back. Do not improvise around
failures: stop and tell me what failed and the exact error.

## 0. Check prerequisites (report each as OK / MISSING)
- `git`, `git lfs` (run `git lfs install`), Python **3.11** (the project uses 3.11.9), `bash` (Git Bash on Windows).
- About 15 GB free disk.
- My Kaggle API token at `~/.kaggle/kaggle.json` (Windows: `C:\Users\<me>\.kaggle\kaggle.json`), and I have accepted the
  competition rules on the Kaggle website. Test: `kaggle competitions files name-that-llm` must list files.
- GitHub access to the private repo `md-muqtadir-fuad/name-that-llm`.
If anything is missing, stop and tell me what to do.

## 1. Clone and branch
```
git clone https://github.com/md-muqtadir-fuad/name-that-llm.git
cd name-that-llm
git checkout experiments/fuad
git switch -c experiments/<my-first-name>      # my own working branch; never commit directly to experiments/fuad
```

## 2. Python environment
```
python -m venv .venv
# Windows:  .venv/Scripts/pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
# Linux/macOS: .venv/bin/pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```
If a pinned package fails to install on my OS, install the nearest available version, and list every substitution in
the report. Some tools hard-code Windows paths (`.venv/Scripts/kaggle.exe`, e.g. `tools/kbench/daily.py`); on
Linux/macOS change them locally to `.venv/bin/kaggle` but do not commit that change.

## 3. Data (everything git does not track)
Read `DATA_SOURCES.md` and `TREE.md` first. Then run:
```
bash tools/fetch_data.sh
```
It downloads the competition data, the strong OOF bundle, the public external datasets (prvsiyan, seyit, Lucas,
abdullah, dophuchung, Oleg), the two-stage pools (`pool_v1`, `pool_v4`) and darragh's run-1 generations, clones SpeechMap
(Git LFS) and rebuilds the derived paragraph files. Four private datasets (`mdmuqtadirfuad/ntl-speechmap-paragraphs`,
`ntl-own-generations`, `ntl-own-prompts`, `ntl-lstm-final`) download only if they have been shared with my Kaggle
account; "skip" for them is acceptable, report it. The raw scraped prompt files (`research/ext_data/own_prompts/raw/`)
are never shared: they stay on my teammate's machine.
On Windows, if a dataset folder ends up empty, re-run that download with an absolute `-p` path (Kaggle CLI path bug).

## 4. Our trained LoRA outputs (Kaggle kernel outputs; adapter weights are skipped, OOF files are enough)
```
kaggle kernels output mdmuqtadirfuad/ntl-lora-gemma12b-train     -p nb/lora_gemma/full_out     --file-pattern '^(?!.*adapter_model\.safetensors).*$'
kaggle kernels output mdmuqtadirfuad/ntl-lora-gemma12b-ext-train -p nb/lora_gemma_ext/full_out --file-pattern '^(?!.*adapter_model\.safetensors).*$'
kaggle kernels output mdmuqtadirfuad/ntl-lora-glm9b-ext-train    -p nb/lora_glm_ext/full_out   --file-pattern '^(?!.*adapter_model\.safetensors).*$'
```
If access is denied, the kernels have not been shared with me yet; report it and continue.
Newer runs (two-stage, pool_v4): `ntl-gemma-ts-s2` (Gemma-4-12B; fold OOF files under `ts/lora_twostage/fold*/`) and,
later, `ntl-llama-ts-s2` (Llama-3.1-8B). Fetch them the same way, skipping `adapter_model.safetensors` and
`lora_state.pt`. Llama weights come from the kernel `ntl-llama31-8b-weights` (Hugging Face download); never re-run it
with your own token unless asked.

## 5. Verify (numbers must match to 3 decimals)
```
python experiments/gemma_oof.py nb/lora_gemma_ext/full_out/lora_gemma
#   expect: gemma plain 0.4778, centred 0.4667, half 0.5241; strong blend 0.4938
python experiments/gemma_oof.py nb/lora_glm_ext/full_out/lora_gemma
#   expect: plain 0.4735, half 0.5272; strong blend 0.4938
RUN_ID=CHECK python experiments/instyle/stylegram_extra_cv.py pool      # ~15 min CPU
#   expect: pool standalone 0.4093
git checkout -- paper/ experiments/instyle/      # discard the CHECK log rows; they are not real experiments
```
A mismatch means a different data version or package version: report the numbers, do not "fix" the code.

## 6. Read, in this order, and give me a 10-line summary of where the project stands
`research/next_steps.md` (current plan), `README.md`, `TREE.md`, `paper/PROTOCOL.md`,
`research/leader_update_2026-10-24.md`, `research/TEST_PLAN.md` (planned + rejected), `research/SYNTHESIS.md`, the last 30
rows of `paper/experiments.csv`, `paper/DECISIONS.md`.

## 7. Working rules (apply to everything you do in this repo from now on)
- **Secrets:** never print, copy or commit `kaggle.json`, `.env`, `.env.kbench`, `.env.hf`, `tools/kbench/.env.init`. I use
  my own Kaggle token, Gemini key, Hugging Face token and Kaggle Benchmarks login.
- **Privacy:** every notebook, kernel and dataset stays PRIVATE until the deadline (2027-01-13). Never publish anything.
- **Data:** never commit competition data or external passage text (`data/`, `research/ext_data/`, `external_dataset/`
  are git-ignored; run `git status` before every commit). Never share private code or data outside the team.
- **Evaluation protocol (fixed):** `GroupKFold(5)` on `data/train.csv` grouped by `prompt_id`, default order, no
  shuffling. A change is accepted only if delta >= +0.005 and >= 4/5 folds improve. Test-structure tricks: soft
  adaptive centering on inferred same-prompt clusters helps (+0.019 on the public LB, v6e), hard per-group constraints
  hurt (soft repulsion -0.008, E002). Public-LB differences of +-0.002 are inside noise.
- **Logging:** every experiment gets a row via `paper/log.py` (`log_row(...)`) using ids **T001, T002, ...** (the E-series
  is my teammate's); every new dataset or generation batch gets a line in `paper/data_log.csv`.
- **Kaggle notebooks:** push to Kaggle only after a local or smoke test; copy `docker_image` from
  `nb/v6/a/kernel-metadata.json` (without the GPU image the notebook silently runs on CPU for 12 h).
  Submissions share the team's 5/day: never submit without agreeing with my teammate first.
- **Data generation:** `tools/kbench/daily.py --own` takes the next unused prompt ids of the own prompt set and records
  them in `tools/kbench/state.json`. To avoid duplicates: agree with my teammate before running, `git pull` first so
  `state.json` is current, and commit + push `state.json` right after the run. Each Kaggle account has its own
  Benchmarks credit, so two people running on different days doubles the output. Batch b001 is contaminated and must
  never be used; the drift panel task (`ntl_gen_task_drift.py`) is frozen and run only by my teammate.
- **Git:** commit on my branch, open pull requests into `experiments/fuad`, never force-push.

## 8. Report
Prerequisites status, what was downloaded (folders + sizes), anything skipped and why, package substitutions,
verification numbers vs expected, and the 10-line project summary.
