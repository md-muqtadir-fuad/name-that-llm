"""Download and log priority benchmark datasets for Name That LLM competition.
All downloaded files and logs are saved inside:
  dataset_downloaded_and_experimentted by_gemini/
"""
import csv
import datetime
import glob
import json
import os
import pathlib
import sys
import time
import zipfile

import requests

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOWNLOAD_DIR = ROOT / "dataset_downloaded_and_experimentted by_gemini"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

LOG_MD = DOWNLOAD_DIR / "DOWNLOAD_LOG.md"
MANIFEST_CSV = DOWNLOAD_DIR / "download_manifest.csv"

# Initialize manifest if not existing
if not MANIFEST_CSV.exists():
    with open(MANIFEST_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "benchmark", "target_model", "file_name", "relative_path",
            "size_mb", "num_rows", "source_url", "notes"
        ])


def log_entry(benchmark, target_model, file_name, rel_path, size_mb, num_rows, source_url, notes):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(MANIFEST_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            now, benchmark, target_model, file_name, rel_path,
            f"{size_mb:.2f}", num_rows, source_url, notes
        ])
    
    # Also append to human-readable Markdown log
    entry_md = f"### [{now}] {benchmark}: `{file_name}`\n"
    entry_md += f"- **Target Model(s)**: {target_model}\n"
    entry_md += f"- **Relative Path**: `{rel_path}`\n"
    entry_md += f"- **Size**: {size_mb:.2f} MB | **Rows/Items**: {num_rows}\n"
    entry_md += f"- **Source URL**: {source_url}\n"
    entry_md += f"- **Notes**: {notes}\n\n"
    with open(LOG_MD, "a", encoding="utf-8") as f:
        f.write(entry_md)


def is_valid_zip(path):
    try:
        with zipfile.ZipFile(path, "r") as z:
            return z.testzip() is None
    except Exception:
        return False


def download_file(url, dest_path, chunk_size=1024 * 1024):
    dest_path = pathlib.Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    if dest_path.exists() and dest_path.stat().st_size > 0:
        if dest_path.suffix == ".zip":
            if is_valid_zip(dest_path):
                print(f"  [Skip] Already valid zip: {dest_path.name} ({dest_path.stat().st_size / 1e6:.2f} MB)")
                return dest_path.stat().st_size
            else:
                print(f"  [Corrupted] Removing incomplete zip: {dest_path.name}")
                dest_path.unlink()
        else:
            print(f"  [Skip] Already exists: {dest_path.name} ({dest_path.stat().st_size / 1e6:.2f} MB)")
            return dest_path.stat().st_size
    
    temp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")
    print(f"  [Downloading] {url} -> {dest_path.name}")
    t0 = time.time()
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    total_len = resp.headers.get("content-length")
    total_bytes = int(total_len) if total_len else None
    
    downloaded = 0
    with open(temp_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_bytes:
                    pct = (downloaded / total_bytes) * 100
                    mb = downloaded / (1024 * 1024)
                    speed = mb / max(time.time() - t0, 0.001)
                    sys.stdout.write(f"\r    {mb:.1f}/{total_bytes / 1e6:.1f} MB ({pct:.1f}%) @ {speed:.2f} MB/s")
                    sys.stdout.flush()
    print()
    if temp_path.exists():
        temp_path.replace(dest_path)
    return downloaded


def step1_eqbench_cw_v3():
    print("\n=== STEP 1: EQ-Bench Creative Writing v3 ===")
    eq_dir = DOWNLOAD_DIR / "eqbench_cw_v3"
    eq_dir.mkdir(parents=True, exist_ok=True)
    url = "https://raw.githubusercontent.com/EQ-bench/creative-writing-bench/main/creative_bench_runs.zip"
    zip_dest = eq_dir / "creative_bench_runs.zip"
    
    download_file(url, zip_dest)
    json_path = eq_dir / "creative_bench_runs.json"
    if not json_path.exists():
        print("  Extracting creative_bench_runs.zip...")
        with zipfile.ZipFile(zip_dest, "r") as z:
            z.extractall(eq_dir)
            
    if json_path.exists():
        size_mb = json_path.stat().st_size / (1024 * 1024)
        print(f"  creative_bench_runs.json exists ({size_mb:.2f} MB)")
        # Inspect models
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        models = list(data.keys())
        total_runs = sum(len(v) if isinstance(v, list) else 1 for v in data.values())
        print(f"  Found {len(models)} models, ~{total_runs} total runs")
        log_entry(
            benchmark="EQ-Bench Creative Writing v3",
            target_model="16 models (gpt-5.5/5.6/6, gemini-3.1/3.6/3.7/3.8, claude-fable/opus4.6/4.8/5/sonnet5, glm-5.3, kimi-k3)",
            file_name="creative_bench_runs.json",
            rel_path=str(json_path.relative_to(DOWNLOAD_DIR)),
            size_mb=size_mb,
            num_rows=total_runs,
            source_url=url,
            notes="32 identical creative writing prompts x 3 iterations; ~900-2100 words per story"
        )


def step2_speechmap_raw():
    print("\n=== STEP 2: SpeechMap (llm-compliance) Target Models ===")
    sm_dir = DOWNLOAD_DIR / "speechmap"
    sm_dir.mkdir(parents=True, exist_ok=True)
    
    # 16 target models with non-empty files in llm-compliance/responses
    targets = [
        ("gpt-5.5", "us_hard_openai_gpt-5.5.jsonl"),
        ("gpt-5.6-luna", "us_hard_openai_gpt-5.6-luna.jsonl"),
        ("gpt-5.6-sol", "us_hard_openai_gpt-5.6-sol.jsonl"),
        ("gpt-5.6-terra", "us_hard_openai_gpt-5.6-terra.jsonl"),
        ("gpt-6-astra", "us_hard_openai_gpt-6-astra.jsonl"),
        ("gemini-3.1-pro-preview", "us_hard_google_gemini-3.1-pro-preview.jsonl"),
        ("gemini-3.6-flash", "us_hard_google_gemini-3.6-flash.jsonl"),
        ("gemini-3.7-flash", "us_hard_google_gemini-3.7-flash.jsonl"),
        ("gemini-3.8-flash", "us_hard_google_gemini-3.8-flash.jsonl"),
        ("claude-fable-5", "us_hard_anthropic_claude-fable-5.jsonl"),
        ("claude-opus-4-6", "us_hard_anthropic_claude-opus-4.6.jsonl"),
        ("claude-opus-4-8", "us_hard_anthropic_claude-opus-4.8.jsonl"),
        ("claude-opus-5", "us_hard_anthropic_claude-opus-5.jsonl"),
        ("claude-sonnet-5", "us_hard_anthropic_claude-sonnet-5.jsonl"),
        ("qwen3.8-max", "us_hard_qwen_qwen3.8-max.jsonl"),
        ("glm-5.3-flash", "us_hard_z-ai_glm-5.3-flash.jsonl"),
    ]
    
    base_url = "https://raw.githubusercontent.com/xlr8harder/llm-compliance/main/responses"
    for model_name, fn in targets:
        url = f"{base_url}/{fn}"
        dest = sm_dir / fn
        try:
            download_file(url, dest)
            sz_mb = dest.stat().st_size / (1024 * 1024)
            # Count rows
            num_rows = 0
            with open(dest, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.strip():
                        num_rows += 1
            print(f"    -> {fn}: {num_rows} rows ({sz_mb:.2f} MB)")
            log_entry(
                benchmark="SpeechMap (llm-compliance)",
                target_model=model_name,
                file_name=fn,
                rel_path=str(dest.relative_to(DOWNLOAD_DIR)),
                size_mb=sz_mb,
                num_rows=num_rows,
                source_url=url,
                notes="2,120 shared political/social prompts. Text in response.choices[0].message.content"
            )
        except Exception as e:
            print(f"    Error downloading {fn}: {e}")


def safe_extract_zip(zip_path, target_dir, allowed_prefixes=None, skip_prefixes=None):
    target_dir = pathlib.Path(target_dir)
    extracted = 0
    with zipfile.ZipFile(zip_path, "r") as z:
        for member in z.infolist():
            fn = member.filename
            if skip_prefixes and any(fn.startswith(sp) for sp in skip_prefixes):
                continue
            if allowed_prefixes and not any(fn.startswith(ap) for ap in allowed_prefixes):
                continue
            try:
                z.extract(member, target_dir)
                extracted += 1
            except Exception as e:
                # Skip long path errors or individual extraction errors
                pass
    return extracted


def step3_lechmazur_debate():
    print("\n=== STEP 3: Lech Mazur Debate Benchmark ===")
    deb_dir = DOWNLOAD_DIR / "lechmazur_debate"
    deb_dir.mkdir(parents=True, exist_ok=True)
    zip_dest = deb_dir / "debate_main.zip"
    url = "https://github.com/lechmazur/debate/archive/refs/heads/main.zip"
    
    download_file(url, zip_dest)
    extracted_flag = deb_dir / ".extracted"
    if not extracted_flag.exists():
        print("  Extracting transcripts and judgments from debate_main.zip...")
        count = safe_extract_zip(
            zip_dest,
            deb_dir,
            skip_prefixes=["debate-main/reports/debate_matchup_judgments/"]
        )
        print(f"  Extracted {count} files.")
        with open(extracted_flag, "w") as f:
            f.write("ok")
            
    sz_mb = zip_dest.stat().st_size / (1024 * 1024)
    transcripts = list(deb_dir.glob("**/transcripts/*.json*"))
    print(f"  Debate archive extracted: {len(transcripts)} transcript files ({sz_mb:.2f} MB)")
    log_entry(
        benchmark="Lech Mazur Debate Benchmark",
        target_model="13 models (Fable 5, Opus 5, Kimi K3, GPT-5.6 Sol, GLM-5.3, Opus 4.8, Astra, Sonnet 5, Qwen 3.8 Max, GPT-5.5, Gemini 3.8/3.7/3.1)",
        file_name="debate_main.zip",
        rel_path=str(zip_dest.relative_to(DOWNLOAD_DIR)),
        size_mb=sz_mb,
        num_rows=len(transcripts),
        source_url=url,
        notes="683 policy/empirical/moral debate questions, 5 turns per debater, structured explanatory paragraphs"
    )


def step4_eqbench4():
    print("\n=== STEP 4: EQ-Bench 4 ===")
    eq4_dir = DOWNLOAD_DIR / "eqbench4"
    eq4_dir.mkdir(parents=True, exist_ok=True)
    zip_dest = eq4_dir / "eqbench4_main.zip"
    url = "https://github.com/EQ-bench/eqbench4/archive/refs/heads/main.zip"
    
    download_file(url, zip_dest)
    extracted_flag = eq4_dir / ".extracted"
    if not extracted_flag.exists():
        print("  Extracting eqbench4_main.zip...")
        count = safe_extract_zip(zip_dest, eq4_dir)
        print(f"  Extracted {count} files.")
        with open(extracted_flag, "w") as f:
            f.write("ok")
            
    sz_mb = zip_dest.stat().st_size / (1024 * 1024)
    transcripts = list(eq4_dir.glob("**/*.json*"))
    print(f"  EQ-Bench 4 archive extracted: {len(transcripts)} files ({sz_mb:.2f} MB)")
    log_entry(
        benchmark="EQ-Bench 4",
        target_model="11 models (Opus 5, Fable 5, Kimi K3, GPT-5.5, Opus 4.8, GPT-5.6 Sol/Terra/Luna, Sonnet 5, Opus 4.6, Gemini 3.1 Pro)",
        file_name="eqbench4_main.zip",
        rel_path=str(zip_dest.relative_to(DOWNLOAD_DIR)),
        size_mb=sz_mb,
        num_rows=len(transcripts),
        source_url=url,
        notes="120 interpersonal coaching / advice scenarios, 1-4 short explanatory paragraphs per turn"
    )


def step5_hf_priority():
    print("\n=== STEP 5: Hugging Face Priority Datasets ===")
    hf_dir = DOWNLOAD_DIR / "hf_priority"
    hf_dir.mkdir(parents=True, exist_ok=True)
    
    from huggingface_hub import hf_hub_download
    
    # 1. best-distill/glm-5.3-flash-distillation-chat (crucial for glm-5.3-flash)
    print("  Downloading best-distill/glm-5.3-flash-distillation-chat...")
    try:
        f = hf_hub_download(
            repo_id="best-distill/glm-5.3-flash-distillation-chat",
            filename="data/train-00000-of-00001.parquet",
            repo_type="dataset",
            local_dir=hf_dir / "glm-5.3-flash"
        )
        p = pathlib.Path(f)
        sz_mb = p.stat().st_size / (1024 * 1024)
        print(f"    -> {p.name}: {sz_mb:.2f} MB")
        log_entry(
            benchmark="HuggingFace: best-distill/glm-5.3-flash-distillation-chat",
            target_model="glm-5.3-flash",
            file_name=p.name,
            rel_path=str(p.relative_to(DOWNLOAD_DIR)),
            size_mb=sz_mb,
            num_rows=18112,
            source_url="https://huggingface.co/datasets/best-distill/glm-5.3-flash-distillation-chat",
            notes="18,112 rows of multi-paragraph explanatory English science/math articles from GLM-5.3-Flash"
        )
    except Exception as e:
        print(f"    Error: {e}")

    # 2. saidutta69/kimi-k3-distillation-clean (crucial for kimi-k3)
    print("  Downloading saidutta69/kimi-k3-distillation-clean...")
    try:
        f = hf_hub_download(
            repo_id="saidutta69/kimi-k3-distillation-clean",
            filename="data/train-00000-of-00001.parquet",
            repo_type="dataset",
            local_dir=hf_dir / "kimi-k3"
        )
        p = pathlib.Path(f)
        sz_mb = p.stat().st_size / (1024 * 1024)
        print(f"    -> {p.name}: {sz_mb:.2f} MB")
        log_entry(
            benchmark="HuggingFace: saidutta69/kimi-k3-distillation-clean",
            target_model="kimi-k3",
            file_name=p.name,
            rel_path=str(p.relative_to(DOWNLOAD_DIR)),
            size_mb=sz_mb,
            num_rows=19700,
            source_url="https://huggingface.co/datasets/saidutta69/kimi-k3-distillation-clean",
            notes="19.7k clean single-model Kimi-K3 responses on science/reasoning tasks"
        )
    except Exception as e:
        print(f"    Error: {e}")


if __name__ == "__main__":
    t_start = time.time()
    print(f"Starting prioritized dataset downloads into:\n  {DOWNLOAD_DIR}")
    step1_eqbench_cw_v3()
    step2_speechmap_raw()
    step3_lechmazur_debate()
    step4_eqbench4()
    step5_hf_priority()
    print(f"\nAll steps completed in {time.time() - t_start:.1f}s.")
    print(f"Manifest written to: {MANIFEST_CSV}")
    print(f"Detailed Markdown log written to: {LOG_MD}")
