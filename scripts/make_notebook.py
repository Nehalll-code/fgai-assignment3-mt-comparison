"""Generate notebooks/FGAI_Assignment3_MT_Comparison.ipynb.

The notebook is generated rather than hand-edited so it can never drift from
the modules in ``src/``: every cell calls the same functions the command-line
driver calls.  Regenerate with:

    python scripts/make_notebook.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "FGAI_Assignment3_MT_Comparison.ipynb"
REPO_URL = "https://github.com/Nehalll-code/fgai-assignment3-mt-comparison.git"
REPO_DIR = "fgai-assignment3-mt-comparison"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip("\n"))


CELLS: list[nbf.NotebookNode] = []

CELLS.append(md(f"""
# PolyGlot-4 — English→Hindi Machine Translation across Four Generative Architectures

**Foundations of Generative AI (ICT4403) — Assignment 3**
Nehal Verma · 230953144

This notebook runs the complete study end to end:

| Part | What happens |
|---|---|
| (i) | Describe and preprocess OPUS-100 `en-hi`, split it, then train the from-scratch Transformer, fine-tune mT5, and adapt Llama-3.2-1B and Gemma-2-2B |
| (ii) | Score all systems with BLEU, chrF++, training time, inference time and parameter count on the **same** 500 test sentences |
| (iii) | Profile failure modes, run the responsible-AI probe sets, and inspect the translations |
| extra | Sampling-temperature sweep for every trained system |

### Before you run

1. **Runtime → Change runtime type → T4 GPU.** The full profile needs a CUDA GPU.
2. Accept the licences for the two gated checkpoints:
   * <https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct>
   * <https://huggingface.co/google/gemma-2-2b-it>
3. Create a **read** token at <https://huggingface.co/settings/tokens>, then in
   Colab click the 🔑 key icon in the left sidebar and add a secret named
   **`HF_TOKEN`** with notebook access enabled.
4. **Runtime → Run all.** Expect roughly 1.5–3 hours on a T4.

If a licence or the token is missing, the decoder-only stages fall back to an
open checkpoint of the same family and record that fact in
`results/run_info.json` — the notebook will not crash, but the report will say
a stand-in was used.

Repository: {REPO_URL}
"""))

CELLS.append(md("## 0. Environment"))

CELLS.append(code("""
# Confirm a GPU is attached. If this prints nothing useful, change the runtime
# type to T4 GPU before continuing.
!nvidia-smi || echo "NO GPU - the full profile will not be practical"
"""))

CELLS.append(code("""
%pip install -q "transformers>=4.44" "datasets>=2.19" "peft>=0.11" \\
    "sacrebleu>=2.4" "sentencepiece>=0.2" "protobuf>=4.25" "accelerate>=0.30"
"""))

CELLS.append(code(f"""
import os, sys, subprocess
from pathlib import Path

REPO_URL = "{REPO_URL}"
REPO_DIR = Path("/content/{REPO_DIR}")

if REPO_DIR.exists():
    subprocess.run(["git", "-C", str(REPO_DIR), "pull", "--ff-only"], check=False)
else:
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)], check=True)

os.chdir(REPO_DIR)
sys.path.insert(0, str(REPO_DIR))
print("working directory:", Path.cwd())
"""))

CELLS.append(code("""
# Hugging Face token for the two gated checkpoints.
try:
    from google.colab import userdata
    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
    print("HF_TOKEN loaded from Colab secrets")
except Exception as exc:
    print(f"No Colab secret found ({type(exc).__name__}).")
    print("Set os.environ['HF_TOKEN'] = '...' manually, or accept the open-checkpoint fallback.")
"""))

CELLS.append(code("""
import importlib
import pandas as pd
from IPython.display import Markdown, Image, display

from src import config as C
from src import data, analysis, evaluate as E, plots
from src import temperature_sweep as TS
from src import decoder_only as DO
from src import run_all as R

print(C.describe())
print("instruction given to both decoder-only models:")
print(" ", C.INSTRUCTION)
"""))

CELLS.append(md("""
## Part (i) a — Dataset: description, preprocessing and splits

`data.build_splits()` downloads OPUS-100 `en-hi`, applies the cleaning chain
described in the README (NFC normalisation → length and markup filtering →
length-ratio filtering → Devanagari script validation → cross-split
de-duplication), and caches deterministic 20k / 1k / 500 splits. The held-out
splits are cleaned first so the test set is never starved of material, and the
exact test set is written to `results/test_set_en_hi.jsonl`.
"""))

CELLS.append(code("""
splits = data.build_splits()
stats = pd.DataFrame(data.corpus_stats(splits)).T
display(stats)
"""))

CELLS.append(code("""
# A look at the actual data the models will see.
display(pd.DataFrame(splits["train"][:8]))
print("\\nFew-shot examples used by the prompted decoder-only systems:")
shots = DO.pick_shots(splits, k=3)
display(pd.DataFrame(shots))
"""))

CELLS.append(md("""
## Part (i) b — Build, train and evaluate each system

Every system goes through the identical `run_stage` call, which:

1. trains or loads the system,
2. translates the **same** 500 test sentences with greedy decoding and times it,
3. scores BLEU and chrF++ and saves every hypothesis,
4. runs the sampling-temperature sweep (trained systems only),
5. runs the responsible-AI probe set,
6. releases the model from GPU memory before the next cell.

`results` accumulates the per-system output. Run the cells in order; if you need
to re-run one, re-run this next cell first to reset the accumulator.
"""))

CELLS.append(code("""
results = []        # reset the accumulator
"""))

STAGE_DOCS = {
    "scratch": """
### System 1 — basic Transformer encoder–decoder, trained from scratch

3+3 layers, `d_model=256`, 4 heads, tied embeddings, and a **joint byte-level
BPE vocabulary of 16k types learned on the training subset only**. No
pretraining of any kind, which makes this the control condition: whatever the
other three systems score above this is the value of pretraining plus scale.
""",
    "mt5": """
### System 2 — mT5-small, encoder–decoder, fully fine-tuned

`google/mt5-small` (~300M parameters, pretrained on mC4 across 101 languages).
Translation is framed with the T5-style prefix `translate English to Hindi:`,
and **all** parameters are updated. Trained in fp32 because mT5 overflows in
fp16 — a real cost of this model that shows up in its measured training time.
""",
    "llama": """
### System 3 — Llama-3.2-1B-Instruct, decoder-only, LoRA

LoRA rank 16 on all attention and MLP projections, with the loss masked to the
Hindi continuation only. Llama-3's tokeniser is English-weighted, so Devanagari
fragments into many pieces — watch the inference time, not just the score.
""",
    "gemma": """
### System 4 — Gemma-2-2B-IT, decoder-only, LoRA

Identical recipe to System 3, so the difference measured here is the checkpoint,
not the method. Gemma's 256k vocabulary covers Indic script considerably better
than Llama-3's, which is the clearest mechanism by which it could win.
""",
    "llama_prompt": """
### Contrast A — Llama-3.2-1B with 3-shot prompting, no weight update

Same checkpoint as System 3 with **zero** training. The gap between this and
System 3 separates "what the base model already knows about Hindi" from "what
task adaptation added".
""",
    "gemma_prompt": """
### Contrast B — Gemma-2-2B-IT with 3-shot prompting, no weight update

The same contrast for System 4.
""",
}

for stage, doc in STAGE_DOCS.items():
    CELLS.append(md(doc))
    CELLS.append(code(f"""
results.append(R.run_stage("{stage}", splits, shots))
display(pd.DataFrame([results[-1]["row"]]).T)
"""))

CELLS.append(md("""
## Part (ii) — Evaluation, comparison and figures

`finalise` writes every metrics file, the sample translations, the temperature
sweep, the failure-mode profile, the probe outputs and all five figures, plus
`results/run_info.json` recording the device, library versions and exact
sacrebleu signatures.
"""))

CELLS.append(code("""
import time
run_info = R.finalise(results, splits, shots,
                      [r["stage"] for r in results], wall_clock_s=0.0)
"""))

CELLS.append(code("""
display(Markdown("### Headline results\\n\\n" +
                 (C.RESULTS_DIR / "metrics.md").read_text(encoding="utf-8")))
display(Markdown(f"BLEU signature: `{run_info['bleu_signature']}`  \\n"
                 f"chrF++ signature: `{run_info['chrf_signature']}`"))
"""))

CELLS.append(code("""
for fig in ["quality_bleu_chrf.png", "compute_cost.png", "quality_vs_size.png",
            "loss_curves.png"]:
    path = C.PLOTS_DIR / fig
    if path.exists():
        display(Markdown(f"**{fig}**"))
        display(Image(filename=str(path)))
"""))

CELLS.append(md("""
### Sampling-temperature analysis

Temperature rescales the logits before the softmax: 0 is greedy, 1 samples from
the model's own distribution, above 1 flattens it. For translation this is
mostly downside — but the **shape** of each curve is diagnostic. A curve that
stays flat before falling belongs to a model whose probability mass is
concentrated on the right token; a curve that drops immediately belongs to one
whose arg-max is right but whose distribution underneath it is not. Rising mean
output length is the hallucination signature: at high temperature the
end-of-sequence token becomes less likely, so the model keeps generating content
that was never in the source.
"""))

CELLS.append(code("""
sweep = TS.load_sweep()
if sweep:
    df = pd.DataFrame(sweep)
    display(df.pivot_table(index="temperature", columns="model",
                           values=["bleu", "chrf"]))
    display(Image(filename=str(C.PLOTS_DIR / "temperature_sweep.png")))
"""))

CELLS.append(md("""
## Part (iii) — Examining the translations: responsible and ethical AI

Two instruments, because corpus BLEU cannot see any of these failures: a fluent
translation that assigns the wrong gender, drops a digit from a dosage or
silently localises a cultural reference scores *well*.

**1. Automatic failure-mode profiling** over all 500 hypotheses per system:
empty output, Latin-script leakage, number mismatch against the source,
degenerate repetition, over-long output (the shape of hallucinated additions),
refusal/meta-commentary leaking out of an instruction-tuned model, and verbatim
source copying.

**2. Fixed probe sets.** Hindi marks gender on verbs, so an English sentence
with no gender cue forces every model to commit — which makes its default
visible. Further probes test culture-specific references, numeric and
named-entity fidelity, and clinically/legally worded sensitive content.

The full discussion is in [`docs/responsible_ai.md`](docs/responsible_ai.md).
"""))

CELLS.append(code("""
display(Markdown("### Failure-mode profile (% of the 500 test sentences)\\n\\n" +
                 (C.RESULTS_DIR / "error_analysis.md").read_text(encoding="utf-8")))
"""))

CELLS.append(code("""
# Concrete instances of each flag, and the worst-scoring sentence per system.
ex = pd.read_csv(C.RESULTS_DIR / "error_examples.csv")
display(ex.head(30))
"""))

CELLS.append(code("""
display(Markdown((C.RESULTS_DIR / "responsible_ai_probes.md").read_text(encoding="utf-8")))
"""))

CELLS.append(code("""
display(Markdown((C.RESULTS_DIR / "sample_translations.md").read_text(encoding="utf-8")))
"""))

CELLS.append(md("""
## Finish: inject the numbers into the README and download the results

`scripts/update_readme.py` rewrites the generated blocks of `README.md` and
`docs/responsible_ai.md` from the files in `results/`, so no number in the
report is ever typed by hand. Download `fgai_results.zip`, copy its contents
over `results/` in the local clone, re-run the script, and commit.
"""))

CELLS.append(code("""
!python scripts/update_readme.py
"""))

CELLS.append(code("""
!zip -qr /content/fgai_results.zip results README.md docs/responsible_ai.md
try:
    from google.colab import files
    files.download("/content/fgai_results.zip")
except Exception as exc:
    print(f"Download the zip from the file browser ({type(exc).__name__}).")
"""))

CELLS.append(md("""
### After this notebook

1. Download `fgai_results.zip` (the cell above).
2. In the local clone, replace `results/` with the version from the zip and copy
   over the updated `README.md` and `docs/responsible_ai.md`.
3. `python scripts/update_readme.py` (no-op if the zip already contains them).
4. `git add -A && git commit -m "results: full GPU run" && git push`
5. `python scripts/make_submission_doc.py` to rebuild the submission .docx/.pdf
   with the final repository link, then upload the PDF to the LMS.
"""))


def main() -> int:
    nb = nbf.v4.new_notebook(cells=CELLS)
    nb.metadata.update({
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": [], "toc_visible": True},
        "accelerator": "GPU",
    })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUT)
    print(f"[notebook] wrote {OUT.relative_to(ROOT)} ({len(CELLS)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
