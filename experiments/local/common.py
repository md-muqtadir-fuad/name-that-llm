"""Shared helpers for local CPU experiments (Name That LLM)."""
from __future__ import annotations

import os
import re
import string
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT = ROOT / "experiments" / "local" / "out"
OUT.mkdir(parents=True, exist_ok=True)

LAB = {
    "gpt-5.5": "openai", "gpt-5.6-luna": "openai", "gpt-5.6-sol": "openai",
    "gpt-5.6-terra": "openai", "gpt-6-astra": "openai",
    "gemini-3.1-pro-preview": "google", "gemini-3.6-flash": "google",
    "gemini-3.7-flash": "google", "gemini-3.8-flash": "google",
    "claude-fable-5": "anthropic", "claude-opus-4-6": "anthropic",
    "claude-opus-4-8": "anthropic", "claude-opus-5": "anthropic",
    "claude-sonnet-5": "anthropic",
    "qwen3.8-max": "qwen", "glm-5.3": "zhipu", "glm-5.3-flash": "zhipu",
    "kimi-k3": "moonshot",
}
CLASSES = sorted(LAB)  # alphabetical, same as the public bundles
K = len(CLASSES)
C2I = {c: i for i, c in enumerate(CLASSES)}


def load_train() -> pd.DataFrame:
    df = pd.read_csv(DATA / "train.csv")
    df["y"] = df["model"].map(C2I).astype(int)
    df["g"] = df["prompt_id"].astype("category").cat.codes.astype(int)
    df["lab"] = df["model"].map(LAB)
    return df


def folds(df: pd.DataFrame, n_splits: int = 5, seed: int = 42):
    """Prompt-disjoint folds (shuffled GroupKFold). Returns fold id per row."""
    rng = np.random.RandomState(seed)
    prompts = np.array(sorted(df["prompt_id"].unique()))
    rng.shuffle(prompts)
    fid = {p: i % n_splits for i, p in enumerate(prompts)}
    return df["prompt_id"].map(fid).values


def bacc(y, pred) -> float:
    return float(balanced_accuracy_score(y, pred))


def softmax(z, axis=1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def log_softmax(z, axis=1):
    z = z - z.max(axis=axis, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=axis, keepdims=True))


# ---------------------------------------------------------------- stylometry
FUNC_WORDS = set(
    "a an the and or but if then so because as of in on at to for from by with about into over "
    "under between through during without within is are was were be been being am do does did "
    "have has had it its this that these those there here we you they he she i me my our your "
    "their his her them us not no yes can could would should will shall may might must also just "
    "only even still yet very more most less much many such what which who whom whose when where "
    "why how all any each every both either neither some few other another than too".split()
)

_WORD = re.compile(r"[A-Za-z][A-Za-z'’\-]*")


def stylo_features(texts) -> pd.DataFrame:
    rows = []
    for t in texts:
        words = _WORD.findall(t)
        nw = max(len(words), 1)
        nc = max(len(t), 1)
        sents = [s for s in re.split(r"(?<=[.!?])\s+", t.strip()) if s]
        ns = max(len(sents), 1)
        lw = [w.lower() for w in words]
        first = lw[0] if lw else ""
        r = {
            "n_words": len(words),
            "n_chars": len(t),
            "n_sents": len(sents),
            "words_per_sent": len(words) / ns,
            "sent_len_std": float(np.std([len(_WORD.findall(s)) for s in sents])) if sents else 0.0,
            "mean_word_len": float(np.mean([len(w) for w in words])) if words else 0.0,
            "type_token": len(set(lw)) / nw,
            "n_paragraphs": t.count("\n\n") + 1,
            "has_newline": int("\n" in t),
            "em_dash_pw": t.count("—") / nw,
            "en_dash_pw": t.count("–") / nw,
            "spaced_hyphen_pw": t.count(" - ") / nw,
            "double_hyphen_pw": t.count("--") / nw,
            "semicolon_pw": t.count(";") / nw,
            "colon_pw": t.count(":") / nw,
            "comma_pw": t.count(",") / nw,
            "paren_pw": t.count("(") / nw,
            "question_pw": t.count("?") / nw,
            "exclam_pw": t.count("!") / nw,
            "ellipsis_pw": (t.count("...") + t.count("…")) / nw,
            "curly_dq_pw": (t.count("“") + t.count("”")) / nw,
            "straight_dq_pw": t.count('"') / nw,
            "curly_apos_pw": t.count("’") / nw,
            "straight_apos_pw": t.count("'") / nw,
            "uses_curly_apos": int("’" in t),
            "uses_straight_apos": int("'" in t),
            "uses_em_dash": int("—" in t),
            "uses_en_dash": int("–" in t),
            "markdown_bold": int("**" in t),
            "markdown_italic": int(bool(re.search(r"(?<![*\w])\*[^*\s][^*]*\*(?!\*)", t))) + int(bool(re.search(r"(?<!\w)_[^_\s][^_]*_(?!\w)", t))),
            "markdown_any": int(bool(re.search(r"(\*\*|__|^#|^\s*[-*] |`)", t, re.M))),
            "digit_pc": sum(ch.isdigit() for ch in t) / nc,
            "upper_pc": sum(ch.isupper() for ch in t) / nc,
            "non_ascii_pc": sum(ord(ch) > 127 for ch in t) / nc,
            "func_word_pc": sum(w in FUNC_WORDS for w in lw) / nw,
            "contraction_pw": sum(("'" in w or "’" in w) for w in words) / nw,
            "first_the": int(first == "the"),
            "first_imagine": int(first == "imagine"),
            "first_in": int(first == "in"),
            "first_a": int(first in ("a", "an")),
            "first_when": int(first == "when"),
            "ends_period": int(t.rstrip().endswith(".")),
            "ends_question": int(t.rstrip().endswith("?")),
            "you_pw": sum(w in ("you", "your", "you're", "you’re") for w in lw) / nw,
            "we_pw": sum(w in ("we", "our", "us") for w in lw) / nw,
            "i_pw": sum(w in ("i", "my", "me") for w in lw) / nw,
            "not_x_but_y": len(re.findall(r"\bnot\b[^.;]{0,60}\bbut\b", t, re.I)) / ns,
            "isnt_its": len(re.findall(r"(isn['’]t|is not)[^.]{0,80}[;,—-]\s*it['’]s", t, re.I)),
            "rule_of_three": len(re.findall(r"\b\w+, \w+,? and \w+\b", t)) / ns,
        }
        rows.append(r)
    return pd.DataFrame(rows)
