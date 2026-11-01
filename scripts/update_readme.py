"""Fill the generated blocks of README.md (and docs/responsible_ai.md) from results/.

The prose in the README is written by hand; every number in it is injected from
the files the pipeline produces, so the report can never drift from the run that
actually happened.  Re-run this after any run:

    python scripts/update_readme.py

Blocks are delimited by ``<!-- BEGIN:name -->`` / ``<!-- END:name -->``.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "results"
PLACEHOLDER = (
    "> **Not populated yet.** Run the pipeline (`python -m src.run_all`, or the\n"
    "> notebook on a GPU runtime) and then `python scripts/update_readme.py`.\n"
)


def _read_csv(name: str) -> list[dict]:
    path = RESULTS / name
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _read_json(name: str) -> dict:
    path = RESULTS / name
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _table(rows: list[dict], cols: list[tuple[str, str]]) -> str:
    out = ["| " + " | ".join(h for _, h in cols) + " |",
           "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(k, "")) for k, _ in cols) + " |")
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- #
# Block builders
# --------------------------------------------------------------------------- #

def _compute_time(info: dict) -> str:
    """How long the run took, in whichever terms the run actually recorded.

    The notebook drives one stage per cell, so there is no single wall clock
    spanning them and it records 0.  The per-stage train and inference timings
    are measured either way, so sum those instead of reporting a zero.
    """
    wall = info.get("wall_clock_s") or 0.0
    if wall:
        return f"total wall clock {wall / 60:.1f} min"
    rows = _read_csv("metrics.csv")
    measured = sum(float(r.get("train_time_s") or 0)
                   + float(r.get("inference_time_s") or 0) for r in rows)
    if not measured:
        return "run time not recorded"
    return (f"{measured / 3600:.1f} h of measured training and inference "
            "across the six systems")


def block_run_banner() -> str:
    info = _read_json("run_info.json")
    if not info:
        return PLACEHOLDER
    device = info.get("gpu") or info.get("device", "?")
    warn = ""
    if info.get("profile") != "full":
        warn = (f"\n> **These numbers come from the `{info.get('profile')}` profile**, a tiny "
                "CPU pipeline check that uses randomly initialised stand-in checkpoints. "
                "They verify that the code runs; they are not meaningful translation "
                "scores. Re-run the `full` profile on a GPU to replace them.\n")
    stats = info.get("corpus_stats", {}).get("test", {})
    return (
        f"Run completed {info.get('finished_utc', '?')} on **{device}** "
        f"({info.get('dtype')}), profile `{info.get('profile')}`, "
        f"{_compute_time(info)}.\n"
        f"Splits: "
        + ", ".join(f"{k}={v}" for k, v in info.get("split_sizes", {}).items())
        + f". Test-set mean length: {stats.get('src_mean_words', 0):.1f} English words "
          f"/ {stats.get('tgt_mean_words', 0):.1f} Hindi words.\n"
        f"Metric signatures: `{info.get('bleu_signature', '?')}` and "
        f"`{info.get('chrf_signature', '?')}`.\n" + warn
    )


def block_results_table() -> str:
    rows = _read_csv("metrics.csv")
    if not rows:
        return PLACEHOLDER
    return _table(rows, [
        ("model", "Model"),
        ("architecture", "Architecture"),
        ("adaptation", "Adaptation"),
        ("total_params_m", "Params (M)"),
        ("trainable_params_m", "Params trained (M)"),
        ("bleu", "BLEU"),
        ("chrf", "chrF++"),
        ("train_time_s", "Train time (s)"),
        ("inference_time_s", "Inference, full test set (s)"),
        ("ms_per_sentence", "ms / sentence"),
    ])


def block_temperature_table() -> str:
    rows = _read_csv("temperature_sweep.csv")
    if not rows:
        return PLACEHOLDER
    return _table(rows, [
        ("model", "Model"),
        ("temperature", "Temperature"),
        ("decoding", "Decoding"),
        ("bleu", "BLEU"),
        ("chrf", "chrF++"),
        ("mean_output_chars", "Mean chars"),
        ("distinct_output_ratio", "Distinct outputs"),
        ("empty_outputs", "Empty"),
    ])


def block_temperature_summary() -> str:
    rows = _read_csv("temperature_sweep.csv")
    if not rows:
        return PLACEHOLDER
    by_model: dict[str, list[dict]] = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r)
    lines = []
    for model, items in by_model.items():
        items.sort(key=lambda r: float(r["temperature"]))
        greedy = next((i for i in items if float(i["temperature"]) == 0.0), items[0])
        hottest = items[-1]
        # "Best sampled" must exclude greedy, or it reports T=0 as the winner.
        sampled = [i for i in items if float(i["temperature"]) > 0.0]
        best = max(sampled, key=lambda r: float(r["chrf"])) if sampled else None
        g, h = float(greedy["chrf"]), float(hottest["chrf"])
        drop = (g - h) / g * 100 if g else 0.0
        part = ""
        if best is not None:
            verdict = "still below greedy" if float(best["chrf"]) < g else "above greedy"
            part = (f"best sampled setting T={float(best['temperature']):.1f} "
                    f"(chrF++ {float(best['chrf']):.2f}, {verdict}); ")
        lines.append(
            f"- **{model}** - greedy chrF++ {g:.2f}; {part}"
            f"at T={float(hottest['temperature']):.1f} chrF++ falls to {h:.2f}, "
            f"a {drop:.0f}% relative loss."
        )
    return "\n".join(lines) + "\n"


def block_error_table() -> str:
    rows = _read_csv("error_analysis.csv")
    if not rows:
        return PLACEHOLDER
    return _table(rows, [
        ("system", "System"),
        ("sentences", "n"),
        ("empty_pct", "Empty %"),
        ("latin_leak_pct", "Latin-script leak %"),
        ("number_mismatch_pct", "Number mismatch %"),
        ("repetition_pct", "Degenerate repetition %"),
        ("length_blowup_pct", "Over-long output %"),
        ("refusal_or_meta_pct", "Refusal / meta %"),
        ("source_copy_pct", "Source copied %"),
    ])


def block_best_model() -> str:
    rows = _read_csv("metrics.csv")
    info = _read_json("run_info.json")
    if not rows:
        return PLACEHOLDER
    best_q = max(rows, key=lambda r: float(r["chrf"]))
    best_bleu = max(rows, key=lambda r: float(r["bleu"]))
    cheapest = min((r for r in rows if float(r["train_time_s"]) > 0),
                   key=lambda r: float(r["train_time_s"]), default=None)
    fastest = min(rows, key=lambda r: float(r["ms_per_sentence"]))

    def eff(r: dict) -> float:
        return float(r["chrf"]) / max(float(r["total_params_m"]), 1e-9)

    best_eff = max(rows, key=eff)
    warn = ""
    if info.get("profile") != "full":
        warn = ("\n> Taken from the smoke profile, so this names the best *stand-in*, "
                "not a real winner. Re-run on a GPU.\n")
    lines = [
        f"- **Best overall quality: {best_q['model']}** - chrF++ "
        f"{best_q['chrf']}, BLEU {best_q['bleu']} "
        f"({best_q['architecture']}, {best_q['adaptation']}, "
        f"{best_q['total_params_m']}M parameters).",
        f"- Best BLEU: {best_bleu['model']} ({best_bleu['bleu']}).",
        f"- Fastest at inference: {fastest['model']} "
        f"({fastest['ms_per_sentence']} ms/sentence).",
        f"- Best quality per parameter: {best_eff['model']} "
        f"({eff(best_eff):.3f} chrF++ per million parameters).",
    ]
    if cheapest is not None:
        lines.insert(3, f"- Cheapest adaptation: {cheapest['model']} "
                        f"({cheapest['train_time_s']}s of training).")
    return "\n".join(lines) + "\n" + warn


def block_dataset_stats() -> str:
    info = _read_json("run_info.json")
    stats = info.get("corpus_stats")
    if not stats:
        return PLACEHOLDER
    rows = []
    for split in ("train", "val", "test"):
        s = stats.get(split) or {}
        if not s:
            continue
        rows.append({
            "split": split,
            "pairs": s.get("pairs"),
            "src_mean_words": f"{s.get('src_mean_words', 0):.1f}",
            "tgt_mean_words": f"{s.get('tgt_mean_words', 0):.1f}",
            "src_mean_chars": f"{s.get('src_mean_chars', 0):.1f}",
            "tgt_mean_chars": f"{s.get('tgt_mean_chars', 0):.1f}",
            "src_vocab": s.get("src_vocab"),
            "tgt_vocab": s.get("tgt_vocab"),
        })
    return _table(rows, [
        ("split", "Split"), ("pairs", "Pairs"),
        ("src_mean_words", "Mean EN words"), ("tgt_mean_words", "Mean HI words"),
        ("src_mean_chars", "Mean EN chars"), ("tgt_mean_chars", "Mean HI chars"),
        ("src_vocab", "EN word types"), ("tgt_vocab", "HI word types"),
    ])


def block_checkpoints() -> str:
    info = _read_json("run_info.json")
    ck = info.get("checkpoints")
    if not ck:
        return PLACEHOLDER
    rows = [{"model": k, "checkpoint": f"`{v['checkpoint']}`",
             "adaptation": v["adaptation"],
             "note": "open fallback used" if v.get("is_fallback") else ""}
            for k, v in ck.items()]
    return _table(rows, [("model", "System"), ("checkpoint", "Checkpoint actually loaded"),
                         ("adaptation", "Adaptation"), ("note", "Note")])


BLOCKS = {
    "run_banner": block_run_banner,
    "results_table": block_results_table,
    "temperature_table": block_temperature_table,
    "temperature_summary": block_temperature_summary,
    "error_table": block_error_table,
    "best_model": block_best_model,
    "dataset_stats": block_dataset_stats,
    "checkpoints": block_checkpoints,
}


def fill(path: Path) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    changed = False
    for name, builder in BLOCKS.items():
        pattern = re.compile(
            rf"(<!-- BEGIN:{name} -->\n)(.*?)(<!-- END:{name} -->)", re.DOTALL)
        if not pattern.search(text):
            continue
        body = builder()
        if not body.endswith("\n"):
            body += "\n"
        text, n = pattern.subn(lambda m: m.group(1) + body + m.group(3), text)
        changed = changed or bool(n)
    path.write_text(text, encoding="utf-8")
    return changed


def main() -> int:
    targets = [ROOT / "README.md", ROOT / "docs" / "responsible_ai.md"]
    for path in targets:
        if fill(path):
            print(f"[readme] updated generated blocks in {path.relative_to(ROOT)}")
        else:
            print(f"[readme] no blocks found in {path.name} (skipped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
