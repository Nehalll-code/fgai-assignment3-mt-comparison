"""Download, clean, subset and split the OPUS-100 English-Hindi corpus.

The script is deliberately deterministic: the same seed always yields the same
50k/2k/500 split, so every one of the four systems sees exactly the same
training data and is scored on exactly the same test sentences.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

from src import config as C

DEVANAGARI = re.compile(r"[ऀ-ॿ]")
WHITESPACE = re.compile(r"\s+")
# Leftover markup / placeholders that OPUS subtitle data is full of.
JUNK = re.compile(r"(https?://|www\.|<[^>]{1,20}>|\{\d+\}|&[a-z]{2,6};)")


def _normalise(text: str) -> str:
    """NFC-normalise, collapse whitespace, strip surrounding quotes/dashes."""
    text = unicodedata.normalize("NFC", text)
    text = WHITESPACE.sub(" ", text).strip()
    text = text.strip("-–— ")
    return text


def _devanagari_ratio(text: str) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    return sum(bool(DEVANAGARI.match(ch)) for ch in letters) / len(letters)


def clean_pair(src: str, tgt: str) -> tuple[str, str] | None:
    """Return a cleaned (source, target) pair, or ``None`` to drop it.

    Filters, in order:
      1. normalisation and empty / over-long removal,
      2. obvious markup, URLs and entity junk,
      3. length-ratio outliers (mis-aligned pairs),
      4. targets that are not actually Devanagari Hindi (OPUS-100 en-hi
         contains a noticeable number of romanised or untranslated rows),
      5. pairs where source and target are identical (untranslated copies).
    """
    src, tgt = _normalise(src), _normalise(tgt)

    if not (C.MIN_CHARS <= len(src) <= C.MAX_CHARS):
        return None
    if not (C.MIN_CHARS <= len(tgt) <= C.MAX_CHARS):
        return None
    if JUNK.search(src) or JUNK.search(tgt):
        return None

    s_words, t_words = len(src.split()), len(tgt.split())
    if s_words == 0 or t_words == 0:
        return None
    ratio = max(s_words / t_words, t_words / s_words)
    if ratio > C.MAX_LEN_RATIO:
        return None

    if _devanagari_ratio(tgt) < C.MIN_DEVANAGARI_RATIO:
        return None
    if DEVANAGARI.search(src):          # source should be Latin English
        return None
    if src.lower() == tgt.lower():
        return None

    return src, tgt


def _clean_split(rows, limit: int, seen: set[str]) -> list[dict]:
    """Clean and de-duplicate ``rows``, stopping once ``limit`` pairs are kept."""
    kept: list[dict] = []
    for row in rows:
        tr = row["translation"]
        cleaned = clean_pair(tr[C.SRC_LANG], tr[C.TGT_LANG])
        if cleaned is None:
            continue
        src, tgt = cleaned
        key = src.lower()
        if key in seen:                  # de-duplicate across all three splits
            continue
        seen.add(key)
        kept.append({"src": src, "tgt": tgt})
        if len(kept) >= limit:
            break
    return kept


def build_splits(force: bool = False) -> dict[str, list[dict]]:
    """Build (or load from cache) the train / validation / test splits."""
    C.ensure_dirs()
    paths = {name: C.DATA_DIR / f"{name}.jsonl" for name in ("train", "val", "test")}

    if not force and all(p.exists() for p in paths.values()):
        splits = {name: _read_jsonl(p) for name, p in paths.items()}
        if all(splits[n] for n in splits):
            print(f"[data] loaded cached splits from {C.DATA_DIR}")
            _report(splits)
            return splits

    from datasets import load_dataset

    print(f"[data] downloading {C.DATASET_ID} ({C.DATASET_CONFIG}) ...")
    raw = load_dataset(C.DATASET_ID, C.DATASET_CONFIG)
    print(f"[data] raw sizes: " + ", ".join(f"{k}={len(v)}" for k, v in raw.items()))

    seen: set[str] = set()
    splits: dict[str, list[dict]] = {}

    # The held-out splits are cleaned first so that the test set can never be
    # starved of material by the (much larger) training subset.
    for name, hf_split, limit in (
        ("test", "test", C.P.n_test),
        ("val", "validation", C.P.n_val),
        ("train", "train", C.P.n_train),
    ):
        # Dataset.shuffle permutes the arrow index and is then read
        # sequentially; shuffling a Python index list instead would mean
        # half a million random row lookups.
        ds = raw[hf_split].shuffle(seed=C.SEED)
        splits[name] = _clean_split(ds, limit, seen)

    for name, rows in splits.items():
        _write_jsonl(paths[name], rows)

    # The exact test set is committed to the repository so the reported scores
    # can be reproduced sentence for sentence.
    _write_jsonl(C.RESULTS_DIR / "test_set_en_hi.jsonl", splits["test"])

    _report(splits)
    return splits


def _report(splits: dict[str, list[dict]]) -> None:
    print("[data] split sizes: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))
    stats = corpus_stats(splits)
    for name, s in stats.items():
        print(
            f"[data]   {name:<5} src_words={s['src_mean_words']:.1f} "
            f"tgt_words={s['tgt_mean_words']:.1f} src_chars={s['src_mean_chars']:.1f} "
            f"tgt_chars={s['tgt_mean_chars']:.1f}"
        )


def corpus_stats(splits: dict[str, list[dict]]) -> dict[str, dict]:
    """Descriptive statistics used in the dataset section of the README."""
    out = {}
    for name, rows in splits.items():
        if not rows:
            out[name] = {}
            continue
        sw = [len(r["src"].split()) for r in rows]
        tw = [len(r["tgt"].split()) for r in rows]
        sc = [len(r["src"]) for r in rows]
        tc = [len(r["tgt"]) for r in rows]
        out[name] = {
            "pairs": len(rows),
            "src_mean_words": sum(sw) / len(sw),
            "tgt_mean_words": sum(tw) / len(tw),
            "src_mean_chars": sum(sc) / len(sc),
            "tgt_mean_chars": sum(tc) / len(tc),
            "src_max_words": max(sw),
            "tgt_max_words": max(tw),
            "src_vocab": len({w.lower() for r in rows for w in r["src"].split()}),
            "tgt_vocab": len({w for r in rows for w in r["tgt"].split()}),
        }
    return out


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


if __name__ == "__main__":
    import sys

    build_splits(force="--force" in sys.argv)
