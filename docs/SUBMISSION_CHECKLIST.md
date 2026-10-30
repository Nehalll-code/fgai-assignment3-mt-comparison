# Submission checklist

The assignment's five steps, mapped to what exists and what is left to do.

## Step 1 — Complete the implementation ✅ / ⏳

| Requirement | Where | Status |
|---|---|---|
| Describe and preprocess the parallel dataset | `src/data.py`, README §2 | ✅ |
| Train / validation / test split | 20k / 1k / 500, seeded, cached | ✅ |
| Basic Transformer (encoder–decoder) from scratch | `src/transformer_scratch.py` | ✅ |
| mT5 fine-tuned (encoder–decoder) | `src/mt5_finetune.py` | ✅ |
| LLaMA (decoder-only), fine-tuned or prompted | `src/decoder_only.py` — both | ✅ |
| Gemma (decoder-only), fine-tuned or prompted | `src/decoder_only.py` — both | ✅ |
| Translations for the **same** test set, comparable settings | `results/translations/*.csv` | ⏳ needs the GPU run |
| BLEU, chrF, training time, inference time, parameter count | `src/evaluate.py` → `results/metrics.csv` | ⏳ needs the GPU run |
| Encoder–decoder vs decoder-only comparison | README §6 | ✅ prose, ⏳ numbers |
| Responsible / ethical AI discussion | `docs/responsible_ai.md`, README §8 | ✅ |
| Program executes correctly and produces the reported results | smoke profile verified end to end on CPU | ✅ |

**The one remaining task is the full run on a GPU.** Open
`notebooks/FGAI_Assignment3_MT_Comparison.ipynb` in Google Colab, set the
runtime to **T4 GPU**, add the `HF_TOKEN` secret, and *Run all* (~2–3 hours).
Then:

1. Download `fgai_results.zip` from the last cell.
2. Unzip it over the local clone, replacing `results/`, `README.md` and
   `docs/responsible_ai.md`.
3. `python scripts/update_readme.py` (idempotent).
4. `python scripts/make_submission_doc.py` — rebuilds the .docx/.pdf, now
   including the best-performing model line.
5. `git add -A && git commit -m "results: full GPU run" && git push`

Before the run, accept the two licences:

* <https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct>
* <https://huggingface.co/google/gemma-2-2b-it>

## Step 2 — Upload to GitHub ✅

* Repository: <https://github.com/Nehalll-code/fgai-assignment3-mt-comparison> — **public**, so no evaluator access grant is needed.
* Implementation as `.py` sources **and** a Jupyter notebook. ✅
* Supporting files, results, plots and generated-text samples under `results/`. ⏳ after the GPU run
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

## Step 5 — Upload to the LMS

Upload **`docs/NehalVerma_230953144_FGAI_Assignment3.pdf`** — that single PDF
is the LMS deliverable; everything else is reached through the repository link
inside it. Re-generate it after the GPU run so it carries the final
best-performing-model line.
