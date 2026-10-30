"""Metrics, timing and result serialisation.

All four systems are scored with the same sacrebleu configuration on the same
500 test sentences, and inference time is always measured over that whole test
set with the same generation batch size.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

from src import config as C


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

def score(hypotheses: list[str], references: list[str]) -> dict:
    """Corpus BLEU and chrF++ for one system."""
    import sacrebleu

    hyps = [h if h.strip() else "." for h in hypotheses]
    refs = [list(references)]
    bleu_metric = sacrebleu.BLEU(tokenize=C.BLEU_TOKENIZE)
    chrf_metric = sacrebleu.CHRF(word_order=C.CHRF_WORD_ORDER)
    bleu = bleu_metric.corpus_score(hyps, refs)
    chrf = chrf_metric.corpus_score(hyps, refs)
    return {
        "bleu": round(bleu.score, 2),
        "chrf": round(chrf.score, 2),
        "bleu_signature": str(bleu_metric.get_signature()),
        "chrf_signature": str(chrf_metric.get_signature()),
        "empty_outputs": sum(1 for h in hypotheses if not h.strip()),
    }


def sentence_scores(hypotheses: list[str], references: list[str]) -> list[float]:
    """Per-sentence chrF++, used to surface the best and worst translations."""
    import sacrebleu

    metric = sacrebleu.CHRF(word_order=C.CHRF_WORD_ORDER)
    return [metric.sentence_score(h or ".", [r]).score
            for h, r in zip(hypotheses, references)]


def timed(fn, *args, **kwargs):
    """Run ``fn`` and return (result, wall-clock seconds)."""
    import torch

    if C.DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    if C.DEVICE == "cuda":
        torch.cuda.synchronize()
    return result, time.perf_counter() - t0


# --------------------------------------------------------------------------- #
# Result rows
# --------------------------------------------------------------------------- #

def make_row(name: str, meta: dict, hypotheses: list[str], references: list[str],
             inference_time_s: float, extra: dict | None = None) -> dict:
    s = score(hypotheses, references)
    n = max(len(hypotheses), 1)
    row = {
        "model": name,
        "architecture": C.ARCHITECTURE.get(name, "?"),
        "adaptation": meta.get("adaptation", ""),
        "checkpoint": meta.get("model_id", meta.get("checkpoint", "")),
        "total_params_m": round(meta.get("total_params", 0) / 1e6, 2),
        "trainable_params_m": round(meta.get("trainable_params", 0) / 1e6, 3),
        "bleu": s["bleu"],
        "chrf": s["chrf"],
        "train_time_s": round(meta.get("train_time_s") or 0.0, 1),
        "inference_time_s": round(inference_time_s, 1),
        "ms_per_sentence": round(1000 * inference_time_s / n, 1),
        "test_sentences": n,
        "empty_outputs": s["empty_outputs"],
    }
    row.update(extra or {})
    row["_bleu_signature"] = s["bleu_signature"]
    row["_chrf_signature"] = s["chrf_signature"]
    return row


def save_translations(name: str, sources: list[str], references: list[str],
                      hypotheses: list[str]) -> Path:
    C.ensure_dirs()
    slug = _slug(name)
    path = C.TRANSLATIONS_DIR / f"{slug}.csv"
    chrf = sentence_scores(hypotheses, references)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["idx", "source_en", "reference_hi", "hypothesis_hi", "sent_chrf"])
        for i, (s, r, h, c) in enumerate(zip(sources, references, hypotheses, chrf)):
            w.writerow([i, s, r, h, round(c, 2)])
    print(f"[eval] wrote {path.relative_to(C.ROOT)} ({len(hypotheses)} rows)")
    return path


def write_metrics(rows: list[dict], merge: bool = True) -> tuple[Path, Path]:
    """Write metrics.csv/.md/.json.

    With ``merge``, rows from an earlier run for systems that are *not* in this
    run are carried forward, so the study can be completed over several
    sessions (``--stages scratch mt5`` now, the decoders later) without losing
    the earlier numbers.
    """
    C.ensure_dirs()
    if merge:
        rows = _merge_rows(rows, C.RESULTS_DIR / "metrics.json", "model")
    order = {name: i for i, name in enumerate(C.MODEL_ORDER)}
    rows = sorted(rows, key=lambda r: order.get(r["model"], 99))

    csv_path = C.RESULTS_DIR / "metrics.csv"
    public = [k for k in rows[0] if not k.startswith("_")] if rows else []
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=public, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    md_path = C.RESULTS_DIR / "metrics.md"
    md_path.write_text(markdown_table(rows), encoding="utf-8")

    json_path = C.RESULTS_DIR / "metrics.json"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[eval] wrote {csv_path.relative_to(C.ROOT)}, "
          f"{md_path.relative_to(C.ROOT)}, {json_path.relative_to(C.ROOT)}")
    return csv_path, md_path


def _merge_rows(fresh: list[dict], previous_json: Path, key: str) -> list[dict]:
    """Keep previous rows whose ``key`` was not produced by this run."""
    if not previous_json.exists():
        return fresh
    try:
        old = json.loads(previous_json.read_text(encoding="utf-8"))
    except Exception:
        return fresh
    produced = {r[key] for r in fresh}
    carried = [r for r in old if r.get(key) not in produced]
    if carried:
        print(f"[eval] carrying forward {len(carried)} row(s) from the previous run: "
              + ", ".join(sorted({r[key] for r in carried})))
    return fresh + carried


def markdown_table(rows: list[dict]) -> str:
    cols = [
        ("model", "Model"),
        ("architecture", "Architecture"),
        ("adaptation", "Adaptation"),
        ("total_params_m", "Params (M)"),
        ("trainable_params_m", "Trained params (M)"),
        ("bleu", "BLEU"),
        ("chrf", "chrF++"),
        ("train_time_s", "Train time (s)"),
        ("inference_time_s", "Inference (s)"),
        ("ms_per_sentence", "ms / sentence"),
    ]
    out = ["| " + " | ".join(h for _, h in cols) + " |",
           "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(k, "")) for k, _ in cols) + " |")
    return "\n".join(out) + "\n"


def write_samples(per_model: dict[str, list[str]], sources: list[str],
                  references: list[str], n: int = 15) -> Path:
    """A side-by-side sample file, the raw material for the ethics discussion."""
    C.ensure_dirs()
    path = C.RESULTS_DIR / "sample_translations.md"
    names = [m for m in C.MODEL_ORDER if m in per_model]
    lines = [
        f"# Sample {C.SRC_LANG_NAME}->{C.TGT_LANG_NAME} translations",
        "",
        f"First {n} sentences of the held-out test set, same inputs for every system.",
        "",
    ]
    for i in range(min(n, len(sources))):
        lines += [f"### {i + 1}. `{sources[i]}`", "",
                  f"- **Reference**: {references[i]}"]
        for name in names:
            hyp = per_model[name][i] if i < len(per_model[name]) else ""
            lines.append(f"- **{name}**: {hyp or '_(empty)_'}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[eval] wrote {path.relative_to(C.ROOT)}")
    return path


def _slug(name: str) -> str:
    keep = [c.lower() if c.isalnum() else "_" for c in name]
    slug = "".join(keep)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")
