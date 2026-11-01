# Submission checklist

The assignment's five steps, mapped to what exists and what is left to do.

**Status: steps 1–4 are complete.** The full study ran on a Kaggle T4 and
finished 2026-11-01T01:25:48Z — 7.6 h of measured training and inference over
all six systems, no fallback checkpoints. The only step left is step 5, the
LMS upload, which has to be done by hand.

## Step 1 — Complete the implementation ✅

| Requirement | Where | Status |
|---|---|---|
| Describe and preprocess the parallel dataset | `src/data.py`, README §2 | ✅ |
| Train / validation / test split | 20k / 1k / 500, seeded, cached | ✅ |
| Basic Transformer (encoder–decoder) from scratch | `src/transformer_scratch.py` | ✅ |
| mT5 fine-tuned (encoder–decoder) | `src/mt5_finetune.py` | ✅ |
| LLaMA (decoder-only), fine-tuned or prompted | `src/decoder_only.py` — both | ✅ |
| Gemma (decoder-only), fine-tuned or prompted | `src/decoder_only.py` — both | ✅ |
| Translations for the **same** test set, comparable settings | `results/translations/*.csv` — 6 files × 500 sentences | ✅ |
| BLEU, chrF, training time, inference time, parameter count | `results/metrics.csv` | ✅ |
| Encoder–decoder vs decoder-only comparison | README §6 | ✅ |
| Responsible / ethical AI discussion | `docs/responsible_ai.md`, README §8 | ✅ |
| Program executes correctly and produces the reported results | 28/28 unit tests; full run completed on a T4 | ✅ |

Results, as reported in README §5 — 500 held-out test sentences, greedy
decoding, identical settings across systems:

| Model | BLEU | chrF++ |
|---|---|---|
| Transformer (scratch) | 5.48 | 18.34 |
| mT5-small (fine-tuned) | 9.22 | 21.66 |
| Llama-3.2-1B (LoRA) | 14.96 | 32.33 |
| **Gemma-2-2B (LoRA)** | **20.31** | **38.40** |
| Llama-3.2-1B (3-shot prompt) | 0.72 | 8.39 |
| Gemma-2-2B (3-shot prompt) | 8.55 | 25.73 |

### Reproducing the run

Open `notebooks/FGAI_Assignment3_MT_Comparison.ipynb` on **Kaggle** or **Google
Colab** — it detects which one it is on — attach a **T4** and *Run all*.

On Kaggle switch **Internet** to **On** in the right-hand panel; it is off by
default and the notebook cannot download anything without it. Save the notebook
with **Save & Run All** rather than relying on the draft session: draft output
is wiped when the session ends, and `/kaggle/working` goes with it.

**No Hugging Face token or licence approval is needed** — Llama-3.2 and Gemma-2
load from ungated mirrors of the same weights. Then:

1. Download `fgai_results.zip` — from the version's **Output** tab on Kaggle,
   or from the last cell on Colab.
2. `python scripts/import_results.py <path-to>/fgai_results.zip` — installs the
   results, re-injects every number into `README.md` and
   `docs/responsible_ai.md`, and rebuilds the .docx/.pdf with the
   best-performing model line.
3. `git add -A && git commit -m "results: full GPU run" && git push`

## Step 2 — Upload to GitHub ✅

* Repository: <https://github.com/Nehalll-code/fgai-assignment3-mt-comparison> — **public**, so no evaluator access grant is needed.
* Implementation as `.py` sources **and** a Jupyter notebook. ✅
* Supporting files, results, plots and generated-text samples under `results/`. ✅ 5 plots, 6 translation files, metrics, sweep, probes
* Dataset source link given in the README and in the submission document. ✅
* `README.md` with all eight required sections. ✅

README section coverage, in the order the assignment lists them:

| Required section | README |
|---|---|
| problem statement and objective | §1 |
| dataset description and source link | §2 |
| brief description of the models implemented | §3 |
| execution steps and required dependencies | §4 |
| results obtained | §5 |
| model comparison and evaluation | §6 |
| sampling-temperature analysis | §7 |
| conclusion identifying the best-performing model | §9 |

## Step 3 — Word document ✅

`docs/NehalVerma_230953144_FGAI_Assignment3.docx`, containing student name,
registration number, assignment title, title of the developed system, dataset
name and link, and the GitHub repository link.

## Step 4 — Convert to PDF ✅

`docs/NehalVerma_230953144_FGAI_Assignment3.pdf` (one page). Rebuild both with
`python scripts/make_submission_doc.py`.

## Step 5 — Upload to the LMS ⏳ *the only step left*

Upload **`docs/NehalVerma_230953144_FGAI_Assignment3.pdf`** — that single PDF
is the LMS deliverable; everything else is reached through the repository link
inside it. It carries the final best-performing-model line (Gemma-2-2B LoRA,
BLEU 20.31, chrF++ 38.4), so the copy in the repo is the one to upload.
