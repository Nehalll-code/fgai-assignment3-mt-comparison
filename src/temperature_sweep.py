"""Sampling-temperature analysis.

Greedy decoding is the right default for translation, but the assignment asks
for a temperature analysis and it is a genuinely informative experiment here: it
shows how much of each system's quality depends on the sharpness of its output
distribution.  A well-adapted model keeps most of its BLEU as the temperature
rises; a weakly adapted one degrades quickly into fluent-but-wrong output.

The sweep runs on a fixed prefix of the test set (cheaper than the full set,
identical sentences for every system and every temperature) and always includes
greedy decoding, recorded as temperature 0.0, for reference.

Each system is swept while it is still in memory, immediately after it has been
scored, so that a 16 GB GPU never has to hold two large models at once.
"""

from __future__ import annotations

import csv
from pathlib import Path

from src import config as C
from src import evaluate as E

CSV_PATH = C.RESULTS_DIR / "temperature_sweep.csv"
FIELDS = ["model", "architecture", "temperature", "decoding", "bleu", "chrf",
          "empty_outputs", "mean_output_chars", "distinct_output_ratio",
          "inference_time_s", "sentences"]


def sweep_subset(splits: dict[str, list[dict]]) -> tuple[list[str], list[str]]:
    subset = splits["test"][: C.P.n_temp_sweep]
    return [r["src"] for r in subset], [r["tgt"] for r in subset]


def sweep_one(name: str, translate_fn, splits: dict[str, list[dict]],
              temperatures: tuple | None = None) -> list[dict]:
    """Decode the sweep subset at each temperature for a single system.

    ``translate_fn`` must accept ``(sentences, temperature)``.
    """
    sources, references = sweep_subset(splits)
    temps = [0.0] + list(temperatures or C.P.temperatures)
    rows: list[dict] = []
    for temp in temps:
        hyps, secs = E.timed(translate_fn, sources, temp)
        s = E.score(hyps, references)
        rows.append({
            "model": name,
            "architecture": C.ARCHITECTURE.get(name, "?"),
            "temperature": temp,
            "decoding": "greedy" if temp == 0.0 else "sampling",
            "bleu": s["bleu"],
            "chrf": s["chrf"],
            "empty_outputs": s["empty_outputs"],
            "mean_output_chars": round(sum(len(h) for h in hyps) / max(len(hyps), 1), 1),
            "distinct_output_ratio": round(len(set(hyps)) / max(len(hyps), 1), 3),
            "inference_time_s": round(secs, 1),
            "sentences": len(sources),
        })
        print(f"[temp] {name:<30} T={temp:<4} BLEU={s['bleu']:>6.2f} "
              f"chrF++={s['chrf']:>6.2f}  ({secs:.0f}s)")
    return rows


def write_sweep(rows: list[dict], merge: bool = True) -> Path:
    """Write the sweep, carrying forward systems this run did not sweep."""
    C.ensure_dirs()
    if merge and CSV_PATH.exists():
        produced = {r["model"] for r in rows}
        carried = [r for r in load_sweep() if r["model"] not in produced]
        if carried:
            print(f"[temp] carrying forward {len(carried)} row(s) from the "
                  "previous run")
        rows = rows + carried
    order = {name: i for i, name in enumerate(C.MODEL_ORDER)}
    rows = sorted(rows, key=lambda r: (order.get(r["model"], 99), r["temperature"]))
    with CSV_PATH.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    (C.RESULTS_DIR / "temperature_sweep.md").write_text(
        _markdown(rows), encoding="utf-8")
    print(f"[temp] wrote {CSV_PATH.relative_to(C.ROOT)} ({len(rows)} rows)")
    return CSV_PATH


def _markdown(rows: list[dict]) -> str:
    cols = [("model", "Model"), ("temperature", "Temperature"),
            ("bleu", "BLEU"), ("chrf", "chrF++"),
            ("mean_output_chars", "Mean output chars"),
            ("distinct_output_ratio", "Distinct outputs"),
            ("empty_outputs", "Empty")]
    out = ["| " + " | ".join(h for _, h in cols) + " |",
           "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(k, "")) for k, _ in cols) + " |")
    return "\n".join(out) + "\n"


def load_sweep() -> list[dict]:
    if not CSV_PATH.exists():
        return []
    with CSV_PATH.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in ("temperature", "bleu", "chrf", "mean_output_chars",
                  "distinct_output_ratio"):
            if r.get(k) not in (None, ""):
                r[k] = float(r[k])
    return rows
