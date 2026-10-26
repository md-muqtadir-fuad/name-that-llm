#!/usr/bin/env bash
# Re-create the git-ignored data folders listed in DATA_SOURCES.md. Run from the repo root:  bash tools/fetch_data.sh
# Needs: your own ~/.kaggle/kaggle.json (competition rules accepted), git, git-lfs, the project venv (.venv).
set -euo pipefail
cd "$(dirname "$0")/.."
K=./.venv/Scripts/kaggle.exe; [ -x "$K" ] || K=./.venv/bin/kaggle; [ -x "$K" ] || K=kaggle
PY=./.venv/Scripts/python.exe; [ -x "$PY" ] || PY=./.venv/bin/python

ds() {  # ds <owner/slug> <local dir> [file ...]
  mkdir -p "$2"
  if [ $# -gt 2 ]; then for f in "${@:3}"; do "$K" datasets download -d "$1" -f "$f" -p "$2" --unzip -q || "$K" datasets download -d "$1" -f "$f" -p "$2" -q; done
  else "$K" datasets download -d "$1" -p "$2" --unzip -q; fi
  for z in "$2"/*.zip; do [ -f "$z" ] && (cd "$2" && unzip -o -q "$(basename "$z")" && rm "$(basename "$z")"); done
  echo "ok  $1 -> $2"
}

mkdir -p data && "$K" competitions download name-that-llm -p data -q && (cd data && unzip -o -q name-that-llm.zip) && echo "ok  competition data"
ds lightningv08/ntl-oof-q35-heads research/ext_oof/q35heads
ds prvsiyan/prvsiyans-extended-name-that-llm-dataset research/ext_data/prvsiyan
ds seyitkaangunes/ntl-extra-api-passages research/ext_data/extra_api
ds lucasboesen/ntl-free-proxy-generations research/ext_data/lucas
ds abdullah1afify/ntl-generations research/ext_data/abdullah
ds dophuchung/ntl-proxy-generations research/ext_data/dophuchung
ds olegzholobov/name-that-llm-response-corpus research/ext_data/oleg \
   external_short_wide.parquet external_all_wide.parquet external_cell_metadata.parquet external_cell_provenance.parquet
# two-stage LoRA stage-1 pools (public) and newer in-style generations
ds nishantdahal/ntl-two-stage-lora research/ext_data/twostage pool_v1.parquet
ds nishantdahal/ntl-handwriting-blend research/ext_data/twostage pool_v4.parquet
ds darragh1walsh/ntl-kaggle-ai-gen-run-1 research/ext_data/darragh_run1 ntl-kaggle-ai-gen-1.csv ntl-kaggle-ai-gen-2.csv
# our private datasets (work only after they are shared with your Kaggle account)
ds mdmuqtadirfuad/ntl-speechmap-paragraphs research/ext_data/speechmap/kaggle_ds || echo "skip: ntl-speechmap-paragraphs not shared with you"
ds mdmuqtadirfuad/ntl-own-generations research/ext_data/own || echo "skip: ntl-own-generations not shared with you (or not created yet)"
ds mdmuqtadirfuad/ntl-own-prompts research/ext_data/own_prompts_ds || echo "skip: ntl-own-prompts not shared with you"
ds mdmuqtadirfuad/ntl-lstm-final research/ext_data/lstm_final_ds || echo "skip: ntl-lstm-final not shared with you"

# SpeechMap raw responses (Git LFS); only the target-model files
if [ ! -d external_dataset/speechmap-data ]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone --filter=blob:none https://github.com/xlr8harder/speechmap-data external_dataset/speechmap-data
fi
git -C external_dataset/speechmap-data lfs pull --include "responses/us_hard_*" && echo "ok  speechmap-data"

# derived files
"$PY" experiments/speechmap/parse.py
"$PY" experiments/oleg/parse.py
echo "done. See DATA_SOURCES.md for anything not fetched here (public notebooks, our kernel outputs, HF test models)."
