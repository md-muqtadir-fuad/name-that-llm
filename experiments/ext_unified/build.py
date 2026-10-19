"""Build the unified, provenance-preserving external corpus.

All passage-bearing inputs and outputs are ignored data. The inspection preview is
also written only below research/ext_data/unified/.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXT = ROOT / "research" / "ext_data"
OUT = ROOT / "experiments" / "ext_unified" / "out"
UNIFIED = EXT / "unified"
LABELS = [
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
    "qwen3.8-max", "glm-5.3", "glm-5.3-flash", "kimi-k3",
]
PRIORITY = ["kbench", "gemini_api", "lucas", "abdullah", "dophuchung", "prvsiyan", "extra_api", "oleg", "speechmap"]
OUTPUT_COLUMNS = ["source", "label", "prompt_key", "text", "words", "served_model", "license", "balanced"]
LICENSES = {
    "kbench": "Kaggle Benchmarks proxy output; no separate output license stated",
    "gemini_api": "Google Gemini API output; provider terms apply",
    "lucas": "CC BY 4.0 (Kaggle metadata)",
    "abdullah": "CC BY 4.0 (Kaggle metadata)",
    "dophuchung": "CC BY 4.0 (Kaggle metadata)",
    "prvsiyan": "CC BY 4.0 (Kaggle metadata; upstream provider terms may apply)",
    "extra_api": "CC BY 4.0 (dataset documentation; provider output terms may apply)",
    "oleg": "Other (Kaggle metadata; mixed source licenses, consult upstream provenance)",
    "speechmap": "Apache-2.0 source dataset; provider output terms may apply",
}
STAGES = ["raw", "after_label", "after_source_filter", "after_text_clean", "after_word_range", "after_ending", "after_refusal", "after_dedup", "after_leakage"]
REFUSAL_RE = re.compile(r"\b(?:I\s+can't|I\s+cannot|I\s+won't|I'm\s+sorry,\s+but|I\s+apologize,\s+but)\b", re.I)
START_RE = re.compile(r"^(?:#|[-*]|\d+\.|>|\|)")
ENDING_RE = re.compile(r"[.!?\"'’”»]\s*$")
MARKDOWN_RE = re.compile(r"\*\*(.*?)\*\*|__(.*?)__|`([^`]*)`", re.S)


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False, keep_default_na=False)


def clean_text(value: object) -> str:
    text = str(value or "")
    text = MARKDOWN_RE.sub(lambda m: next((g for g in m.groups() if g is not None), ""), text)
    return re.sub(r"\s+", " ", text).strip()


def words(text: str) -> list[str]:
    return re.findall(r"\S+", text.casefold())


def shingles(text: str) -> set[int]:
    ws = words(text)
    if len(ws) < 8:
        return set()
    # Stable 64-bit hashes keep the exact inverted index compact and reproducible.
    return {int.from_bytes(hashlib.blake2b(" ".join(ws[i:i + 8]).encode("utf-8"), digest_size=8).digest(), "big")
            for i in range(len(ws) - 7)}


def normalize_label(value: object) -> str:
    label = str(value or "").strip()
    label = re.sub(r"^(?:openai|google|anthropic)/", "", label, flags=re.I)
    for suffix in ("@default", "-default", "-2026-04-23"):
        if label.endswith(suffix):
            label = label[:-len(suffix)]
    return label


def source_records() -> dict[str, list[dict]]:
    records: dict[str, list[dict]] = {name: [] for name in PRIORITY}

    p = EXT / "kbench" / "b002"
    for path in sorted(p.rglob("ntl_gen_b002_*.csv")):
        d = read_csv(path)
        for row in d.to_dict("records"):
            records["kbench"].append({"source": "kbench", "label": normalize_label(row.get("llm", "")),
                "prompt_key": row.get("prompt_id", ""), "text": row.get("response", ""),
                "served_model": row.get("llm", ""), "_error": row.get("error", "")})

    path = EXT / "gemini_api" / "responses.csv"
    if path.exists():
        for row in read_csv(path).to_dict("records"):
            label = str(row.get("model_version", "")).strip()
            records["gemini_api"].append({"source": "gemini_api", "label": label,
                "prompt_key": row.get("prompt_id", ""), "text": row.get("response_clean", ""),
                "served_model": row.get("model_version", ""), "_served_ok": label == str(row.get("model", "")).strip() or label == str(row.get("api_model", "")).strip()})

    for source, path, label_col, text_col, prompt_col, served_col in [
        ("lucas", EXT / "lucas" / "gen_passages.csv", "model", "text", "prompt_id", "model_slug"),
        ("abdullah", EXT / "abdullah" / "ntl_generations.csv", "label", "text", "prompt_id", "model_seen"),
        ("dophuchung", EXT / "dophuchung" / "ntl_proxy_generations.csv", "label", "text", "prompt_id", "model_seen"),
        ("prvsiyan", EXT / "prvsiyan" / "passages_clean.csv", "model", "text", "prompt_id", "served_model_id"),
        ("extra_api", EXT / "extra_api" / "ntl_extra_passages_claude.csv", "model", "text", "prompt_id", ""),
    ]:
        if not path.exists():
            continue
        for row in read_csv(path).to_dict("records"):
            records[source].append({"source": source, "label": normalize_label(row.get(label_col, "")),
                "prompt_key": row.get(prompt_col, "") if prompt_col else "",
                "text": row.get(text_col, ""), "served_model": row.get(served_col, "") if served_col else ""})

    path = EXT / "speechmap" / "paragraphs.csv"
    if path.exists():
        for row in read_csv(path).to_dict("records"):
            records["speechmap"].append({"source": "speechmap", "label": normalize_label(row.get("label", "")),
                "prompt_key": row.get("question_id", ""), "text": row.get("text", ""),
                "served_model": row.get("served_model", ""), "_refusal": str(row.get("refusal", "")),
                "_variant": str(row.get("variant", "")), "_para_idx": str(row.get("para_idx", ""))})

    path = EXT / "oleg" / "external_short_wide.parquet"
    if path.exists():
        d = pd.read_parquet(path)
        if set(LABELS).issubset(d.columns):
            for row in d.to_dict("records"):
                prompt = str(row.get("prompt", ""))
                prompt_key = str(row.get("id", "")) or hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
                for label in LABELS:
                    value = row.get(label, "")
                    if not pd.isna(value) and str(value).strip():
                        records["oleg"].append({"source": "oleg", "label": label, "prompt_key": prompt_key,
                            "text": value, "served_model": ""})

    return records


def add_prefix(records: dict[str, list[dict]]) -> None:
    for source, rows in records.items():
        for row in rows:
            row["prompt_key"] = f"{source}:{str(row.get('prompt_key', '')).strip()}"


def inspect_inputs(records: dict[str, list[dict]]) -> None:
    """Store required schema/shape/two truncated row previews in ignored ext_data."""
    paths = [
        EXT / "prvsiyan" / "passages_clean.csv", EXT / "speechmap" / "paragraphs.csv",
        *sorted((EXT / "kbench" / "b002").rglob("ntl_gen_b002_*.csv")),
        EXT / "gemini_api" / "responses.csv", EXT / "extra_api" / "ntl_extra_passages_claude.csv",
        EXT / "oleg" / "external_short_wide.parquet", EXT / "lucas" / "gen_passages.csv",
        EXT / "abdullah" / "ntl_generations.csv", EXT / "dophuchung" / "ntl_proxy_generations.csv",
    ]
    lines = []
    for path in paths:
        if not path.exists():
            lines.append(f"{path.relative_to(ROOT).as_posix()}: MISSING")
            continue
        d = pd.read_parquet(path) if path.suffix == ".parquet" else read_csv(path)
        preview = d.head(2).copy()
        for col in preview.columns:
            if preview[col].dtype == object:
                preview[col] = preview[col].map(lambda x: (str(x)[:120] + "…") if len(str(x)) > 120 else str(x))
        lines.append(f"{path.relative_to(ROOT).as_posix()}\nshape={d.shape}; columns={list(d.columns)}\n2 truncated rows={preview.to_dict('records')}\n")
    (UNIFIED / "source_inspection.txt").write_text("\n".join(lines), encoding="utf-8")


def filter_records(records: dict[str, list[dict]]) -> tuple[list[dict], dict[str, dict[str, int]]]:
    filtered: list[dict] = []
    drops = {s: {stage: 0 for stage in STAGES} for s in PRIORITY}
    for source in PRIORITY:
        rows = records[source]
        d = drops[source]
        d["raw"] = len(rows)
        for row in rows:
            if row["label"] not in LABELS:
                continue
            d["after_label"] += 1
            if source == "kbench" and (str(row.get("_error", "")).strip() or not str(row.get("text", "")).strip()):
                continue
            if source == "speechmap" and (row.get("_refusal") != "0" or row.get("_variant") == "reasoning" or row.get("_para_idx") != "0"):
                continue
            if source == "gemini_api" and not row.get("_served_ok", False):
                continue
            d["after_source_filter"] += 1
            text = clean_text(row.get("text", ""))
            if not text:
                continue
            # A normalized single line may not hide a source's paragraph/list structure.
            original = str(row.get("text", ""))
            lines = [line.strip() for line in original.splitlines() if line.strip()]
            if re.search(r"\n\s*\n", original) or any(START_RE.match(line) for line in lines):
                continue
            d["after_text_clean"] += 1
            count = len(words(text))
            if not 40 <= count <= 250:
                continue
            d["after_word_range"] += 1
            if not ENDING_RE.search(text):
                continue
            d["after_ending"] += 1
            if REFUSAL_RE.search(text[:400]):
                continue
            d["after_refusal"] += 1
            filtered.append({"source": source, "label": row["label"],
                "prompt_key": row["prompt_key"], "text": text, "words": count,
                "served_model": str(row.get("served_model", "") or ""), "license": LICENSES[source],
                "balanced": 0})
    return filtered, drops


class ExactShingleIndex:
    """Inverted word-8-gram hash index for exact Jaccard candidate verification."""
    def __init__(self):
        self.by_gram: dict[int, list[int]] = defaultdict(list)
        self.sets: list[set[int]] = []

    def matches(self, grams: set[int], threshold: float) -> bool:
        if not grams:
            return False
        common = Counter()
        for gram in grams:
            common.update(self.by_gram.get(gram, ()))
        for idx, inter in common.items():
            union = len(grams) + len(self.sets[idx]) - inter
            if union and inter / union > threshold:
                return True
        return False

    def add(self, grams: set[int]) -> None:
        idx = len(self.sets)
        self.sets.append(grams)
        for gram in grams:
            self.by_gram[gram].append(idx)


def deduplicate(rows: list[dict], drops: dict[str, dict[str, int]]) -> list[dict]:
    seen_text: set[str] = set()
    index = ExactShingleIndex()
    kept = []
    for row in rows:
        norm = re.sub(r"\s+", " ", row["text"].casefold()).strip()
        grams = shingles(row["text"])
        if norm in seen_text or index.matches(grams, 0.8):
            continue
        seen_text.add(norm)
        index.add(grams)
        kept.append(row)
        drops[row["source"]]["after_dedup"] += 1
    return kept


def build_leakage_index() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pairs_g, pairs_r, row_sizes = [], [], []
    rowid = 0
    for path in [ROOT / "data" / "train.csv", ROOT / "data" / "test.csv"]:
        if not path.exists():
            continue
        for chunk in pd.read_csv(path, usecols=["text"], chunksize=2048, keep_default_na=False):
            for text in chunk["text"].astype(str):
                gs = shingles(text)
                row_sizes.append(len(gs))
                pairs_g.extend(gs)
                pairs_r.extend([rowid] * len(gs))
                rowid += 1
    if not pairs_g:
        raise FileNotFoundError("data/train.csv and data/test.csv are required for leakage guard")
    grams = np.asarray(pairs_g, dtype=np.uint64)
    refs = np.asarray(pairs_r, dtype=np.uint32)
    order = np.argsort(grams, kind="mergesort")
    return grams[order], refs[order], np.asarray(row_sizes, dtype=np.uint32)


def leaks(grams: set[int], sorted_g: np.ndarray, sorted_r: np.ndarray, row_sizes: np.ndarray, threshold: float = 0.5) -> bool:
    if not grams:
        return False
    query = np.fromiter(grams, dtype=np.uint64)
    starts = np.searchsorted(sorted_g, query, side="left")
    ends = np.searchsorted(sorted_g, query, side="right")
    shared = Counter()
    for start, end in zip(starts, ends):
        if end > start:
            shared.update(sorted_r[start:end].tolist())
    for row_id, inter in shared.items():
        union = len(grams) + int(row_sizes[row_id]) - inter
        if union and inter / union > threshold:
            return True
    return False


def apply_leakage(rows: list[dict], drops: dict[str, dict[str, int]]) -> list[dict]:
    sorted_g, sorted_r, row_sizes = build_leakage_index()
    kept = []
    for row in rows:
        if leaks(shingles(row["text"]), sorted_g, sorted_r, row_sizes):
            continue
        kept.append(row)
        drops[row["source"]]["after_leakage"] += 1
    return kept


def add_balance(rows: list[dict]) -> None:
    lab = {label: label.split("-", 1)[0] for label in LABELS}
    groups: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        groups[(row["source"], row["prompt_key"])].add(row["label"])
    for row in rows:
        labels = groups[(row["source"], row["prompt_key"])]
        if len(labels) >= 3 and len({lab[x] for x in labels}) >= 2:
            row["balanced"] = 1


def write_outputs(rows: list[dict], drops: dict[str, dict[str, int]]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    UNIFIED.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    frame.to_parquet(UNIFIED / "ext_all.parquet", index=False)
    counts = frame.pivot_table(index="label", columns="source", values="text", aggfunc="count", fill_value=0)
    counts = counts.reindex(LABELS, fill_value=0).reindex(columns=PRIORITY, fill_value=0)
    counts["total"] = counts.sum(axis=1)
    balanced = frame[frame["balanced"] == 1].groupby("label").size()
    counts["balanced_total"] = balanced.reindex(LABELS, fill_value=0).astype(int)
    counts.to_csv(OUT / "counts.csv", index_label="label")
    pd.DataFrame.from_dict(drops, orient="index").reindex(PRIORITY).rename_axis("source").reset_index().to_csv(OUT / "drops.csv", index=False)
    append_data_log(frame, drops)
    write_summary(frame, drops)


def append_data_log(frame: pd.DataFrame, drops: dict[str, dict[str, int]]) -> None:
    log_path = ROOT / "paper" / "data_log.csv"
    with log_path.open("r", encoding="utf-8", newline="") as f:
        old_lines = f.readlines()
    fields = next(csv.reader([old_lines[0].rstrip("\r\n")]))
    date = "2026-10-18"
    dataset_ids = {
        "kbench": "ext_unified_kbench", "gemini_api": "ext_unified_gemini_api", "lucas": "ext_unified_lucas",
        "abdullah": "ext_unified_abdullah", "dophuchung": "ext_unified_dophuchung", "prvsiyan": "ext_unified_prvsiyan",
        "extra_api": "ext_unified_extra_api", "oleg": "ext_unified_oleg", "speechmap": "ext_unified_speechmap",
    }
    routes = {"kbench": "Kaggle Benchmarks proxy, batch b002 only", "gemini_api": "Gemini API; served-version checked",
        "lucas": "lucasboesen/ntl-free-proxy-generations", "abdullah": "abdullah1afify/ntl-generations",
        "dophuchung": "dophuchung/ntl-proxy-generations", "prvsiyan": "prvsiyan extended dataset",
        "extra_api": "seyitkaangunes/ntl-extra-api-passages", "oleg": "olegzholobov/name-that-llm-response-corpus",
        "speechmap": "xlr8harder/llm-compliance SpeechMap"}
    new_rows = []
    for source in PRIORITY:
        sub = frame[frame.source == source]
        labels = sub.label.nunique()
        prompts = sub.prompt_key.nunique()
        path = f"research/ext_data/unified/ext_all.parquet"
        note = f"unified pipeline; {drops[source]['after_leakage']} retained after filters/dedup/leakage; {LICENSES[source]}"
        values = {"date": date, "batch_id": dataset_ids[source], "source": routes[source], "route": "external generations",
                  "labels": str(labels), "prompts": str(prompts), "n_rows": str(len(sub)), "format": "normalized parquet",
                  "status": "included" if len(sub) else "no rows retained", "path": path, "notes": note}
        new_rows.append(values)
    # Preserve legacy lines byte-for-byte. Some earlier rows are ragged CSV, so
    # parsing/re-serializing the whole log would shift existing fields.
    kept_lines = [old_lines[0]]
    for line in old_lines[1:]:
        cells = next(csv.reader([line.rstrip("\r\n")]))
        if len(cells) > 1 and cells[1] in dataset_ids.values():
            continue
        kept_lines.append(line if line.endswith(("\n", "\r")) else line + "\n")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    for row in new_rows:
        writer.writerow(row)
    with log_path.open("w", encoding="utf-8", newline="") as f:
        f.writelines(kept_lines)
        f.write(buffer.getvalue())


def write_summary(frame: pd.DataFrame, drops: dict[str, dict[str, int]]) -> None:
    label_counts = frame.groupby("label").size().reindex(LABELS, fill_value=0).sort_values()
    lines = ["# Unified external corpus", "", f"Retained **{len(frame):,}** passages from {frame.source.nunique()} sources after shared filters, priority dedup, and train/test 8-gram leakage checks.", "", "## Counts by label"]
    for i in range(0, len(LABELS), 2):
        pair = LABELS[i:i + 2]
        lines.append("; ".join(f"{label}: {int(label_counts[label])}" for label in pair))
    lines += ["", "## Smallest labels", ", ".join(f"{label} ({int(label_counts[label])})" for label in label_counts.index[:5]), "", "## Sources, drops, and licenses"]
    for i in range(0, len(PRIORITY), 3):
        group = []
        for source in PRIORITY[i:i + 3]:
            why = "included" if drops[source]["after_leakage"] else "zero survived"
            group.append(f"{source}: {drops[source]['after_leakage']}/{drops[source]['raw']} ({why}); {LICENSES[source]}")
        lines.append(" | ".join(group))
    lines += ["## Choices and surprises", "- Only kbench b002 was used; b001 remains excluded as contaminated. No listed source was wholly dropped; Oleg was reduced by the shared single-paragraph/length/dedup filters. Gemini API has no 3.7 served rows.",
              "- Oleg Kaggle metadata says `Other` and describes mixed licenses. `datasketch` was unavailable; exact 8-gram hash indexing was used. Two-row previews are in ignored `research/ext_data/unified/source_inspection.txt`."]
    (OUT / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    UNIFIED.mkdir(parents=True, exist_ok=True)
    records = source_records()
    inspect_inputs(records)
    add_prefix(records)
    rows, drops = filter_records(records)
    # Dedup and leakage are applied in source-priority order.
    rows = deduplicate(rows, drops)
    rows = apply_leakage(rows, drops)
    add_balance(rows)
    write_outputs(rows, drops)
    print(f"retained={len(rows)}; sources={len(PRIORITY)}; outputs={OUT.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
