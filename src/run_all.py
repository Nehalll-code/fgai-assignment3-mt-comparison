"""End-to-end driver: build the data, train/adapt every system, evaluate, plot.

Each system is trained, scored on the full test set, swept over decoding
temperatures and run through the responsible-AI probes while it is still
resident in memory, and is then released before the next one loads.  That keeps
peak memory to one model at a time, so the whole pipeline fits on a single free
Colab T4.

    python -m src.run_all                       # everything
    python -m src.run_all --stages scratch mt5  # a subset
    FGAI_SMOKE=1 python -m src.run_all          # tiny CPU smoke test

``run_stage`` is the unit the notebook calls, one cell per system, so the
notebook and the command line run exactly the same code.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
import time
from datetime import datetime, timezone

import torch

from src import analysis
from src import config as C
from src import data
from src import decoder_only as DO
from src import evaluate as E
from src import mt5_finetune as MT5
from src import plots
from src import temperature_sweep as TS
from src import transformer_scratch as SCRATCH

ALL_STAGES = ["scratch", "mt5", "llama", "gemma", "llama_prompt", "gemma_prompt"]
NAMES = {
    "scratch": "Transformer (scratch)",
    "mt5": "mT5-small (fine-tuned)",
    "llama": "Llama-3.2-1B (LoRA)",
    "gemma": "Gemma-2-2B (LoRA)",
    "llama_prompt": "Llama-3.2-1B (3-shot prompt)",
    "gemma_prompt": "Gemma-2-2B (3-shot prompt)",
}
# Systems included in the sampling-temperature sweep.  The prompted variants are
# left out: they are a contrast on adaptation method, not on decoding.
SWEEP_STAGES = {"scratch", "mt5", "llama", "gemma"}


def free_gpu() -> None:
    """Release the accelerator between stages so only one model is resident."""
    gc.collect()
    if C.DEVICE == "cuda":
        torch.cuda.empty_cache()


def banner(text: str) -> None:
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72, flush=True)


# --------------------------------------------------------------------------- #
# One system at a time
# --------------------------------------------------------------------------- #

def build_system(stage: str, splits: dict, shots: list[dict],
                 force: bool = False):
    """Train or load one system and return ``(translate_fn, meta)``.

    ``translate_fn`` always has the signature ``(sentences, temperature)`` so
    every downstream consumer - scoring, the temperature sweep, the probes - is
    architecture-agnostic.
    """
    if stage == "scratch":
        res = SCRATCH.train_scratch(splits, force=force)
        model, tok = res["model"], res["tokenizer"]

        def translate(sents, temperature=0.0):
            return SCRATCH.translate(model, tok, sents, temperature=temperature)

        meta = {**res, "adaptation": "trained from scratch",
                "model_id": f"custom {C.P.num_layers}+{C.P.num_layers} layers, "
                            f"d_model={C.P.d_model}, joint BPE {C.P.vocab_size}"}

    elif stage == "mt5":
        res = MT5.finetune_mt5(splits, force=force)
        model, tok = res["model"], res["tokenizer"]

        def translate(sents, temperature=0.0):
            return MT5.translate(model, tok, sents, temperature=temperature)

        meta = {**res, "adaptation": "full fine-tuning", "model_id": C.P.mt5_id}

    elif stage in ("llama", "gemma"):
        res = DO.finetune_decoder(stage, splits, force=force)
        model, tok = res["model"], res["tokenizer"]

        def translate(sents, temperature=0.0):
            return DO.translate(model, tok, sents, temperature=temperature)

        meta = {**res, "adaptation": f"LoRA r={C.P.lora_r}"}

    elif stage in ("llama_prompt", "gemma_prompt"):
        res = DO.load_prompted(stage.split("_")[0])
        model, tok = res["model"], res["tokenizer"]

        def translate(sents, temperature=0.0):
            return DO.translate(model, tok, sents, shots=shots,
                                temperature=temperature)

        meta = {**res, "adaptation": "3-shot prompting (no weight update)"}

    else:
        raise ValueError(f"unknown stage {stage!r}")

    return translate, meta


def run_stage(stage: str, splits: dict, shots: list[dict], *,
              force: bool = False, do_sweep: bool = True,
              do_probes: bool = True) -> dict:
    """Build, score, sweep and probe one system, then release it."""
    name = NAMES[stage]
    banner(f"[{stage}] {name}")

    sources = [r["src"] for r in splits["test"]]
    references = [r["tgt"] for r in splits["test"]]

    translate, meta = build_system(stage, splits, shots, force=force)

    print(f"[{stage}] translating {len(sources)} test sentences (greedy) ...")
    hypotheses, secs = E.timed(translate, sources)
    row = E.make_row(name, meta, hypotheses, references, secs)
    E.save_translations(name, sources, references, hypotheses)
    print(f"[{stage}] BLEU={row['bleu']} chrF++={row['chrf']} "
          f"train={row['train_time_s']}s inference={row['inference_time_s']}s "
          f"({row['ms_per_sentence']} ms/sentence)")

    sweep = TS.sweep_one(name, translate, splits) \
        if (do_sweep and stage in SWEEP_STAGES) else []
    probes = analysis.run_probes(name, translate) if do_probes else []

    out = {
        "stage": stage,
        "name": name,
        "row": row,
        "hypotheses": hypotheses,
        "sweep": sweep,
        "probes": probes,
        "history": meta.get("history") or [],
        "checkpoint": {
            "checkpoint": meta.get("model_id", ""),
            "is_fallback": bool(meta.get("is_fallback", False)),
            "adaptation": meta["adaptation"],
        },
    }
    del translate, meta
    free_gpu()
    return out


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def finalise(results: list[dict], splits: dict, shots: list[dict],
             stages: list[str], wall_clock_s: float) -> dict:
    """Write every metrics file, plot and the run manifest."""
    banner("Results")
    sources = [r["src"] for r in splits["test"]]
    references = [r["tgt"] for r in splits["test"]]

    rows = [r["row"] for r in results]
    per_model = {r["name"]: r["hypotheses"] for r in results}
    sweep = [s for r in results for s in r["sweep"]]
    probes = [p for r in results for p in r["probes"]]
    histories = {r["name"]: r["history"] for r in results}

    if rows:
        E.write_metrics(rows)
        print("\n" + E.markdown_table(rows))
        E.write_samples(per_model, sources, references)
    if sweep:
        TS.write_sweep(sweep)
    if probes:
        analysis.write_probes(probes)
    analysis.profile_translations()
    if rows:
        plots.plot_all(rows, sweep, histories)

    run_info = {
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "wall_clock_s": round(wall_clock_s, 1),
        "profile": C.P.name,
        "device": C.DEVICE,
        "gpu": torch.cuda.get_device_name(0) if C.DEVICE == "cuda" else None,
        "dtype": str(C.DTYPE).split(".")[-1],
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": _version("transformers"),
        "peft": _version("peft"),
        "sacrebleu": _version("sacrebleu"),
        "dataset": {"id": C.DATASET_ID, "config": C.DATASET_CONFIG,
                    "url": C.DATASET_URL},
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "corpus_stats": data.corpus_stats(splits),
        "bleu_tokenize": C.BLEU_TOKENIZE,
        "bleu_signature": rows[0].get("_bleu_signature") if rows else None,
        "chrf_signature": rows[0].get("_chrf_signature") if rows else None,
        "few_shot_examples": shots,
        "checkpoints": {r["name"]: r["checkpoint"] for r in results},
        "stages": stages,
    }
    path = C.RESULTS_DIR / "run_info.json"
    path.write_text(json.dumps(run_info, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    print(f"[run] wrote {path.relative_to(C.ROOT)}")
    print(f"[run] total wall clock {wall_clock_s / 60:.1f} min")
    return run_info


def _version(package: str) -> str | None:
    try:
        from importlib.metadata import version
        return version(package)
    except Exception:
        return None


# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the full MT comparison.")
    ap.add_argument("--stages", nargs="*", default=ALL_STAGES, choices=ALL_STAGES)
    ap.add_argument("--force", action="store_true",
                    help="retrain even if a checkpoint exists")
    ap.add_argument("--force-data", action="store_true",
                    help="rebuild the dataset splits")
    ap.add_argument("--no-sweep", action="store_true",
                    help="skip the sampling-temperature analysis")
    ap.add_argument("--no-probes", action="store_true",
                    help="skip the responsible-AI probe sentences")
    args = ap.parse_args(argv)

    C.ensure_dirs()
    banner(f"FGAI Assignment 3 - {C.SRC_LANG_NAME} to {C.TGT_LANG_NAME} MT study")
    print(C.describe())
    if C.DEVICE == "cpu" and not C.SMOKE:
        print("WARNING: no CUDA device found. The full profile is not practical on "
              "CPU; run with FGAI_SMOKE=1 for a pipeline check, or use a GPU "
              "runtime.", flush=True)

    splits = data.build_splits(force=args.force_data)
    shots = DO.pick_shots(splits, k=3)

    t0 = time.perf_counter()
    results = []
    for stage in args.stages:
        results.append(run_stage(stage, splits, shots, force=args.force,
                                 do_sweep=not args.no_sweep,
                                 do_probes=not args.no_probes))
    finalise(results, splits, shots, args.stages, time.perf_counter() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
