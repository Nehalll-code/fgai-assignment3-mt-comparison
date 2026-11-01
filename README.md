# PolyGlot-4 — English→Hindi Machine Translation across Four Generative Architectures

**Foundations of Generative AI (ICT4403) — Assignment 3**
Nehal Verma · 230953144

A single, directly comparable study of four generative approaches to the same
translation task: a Transformer encoder–decoder **trained from scratch**, a
**fine-tuned mT5** encoder–decoder, and two **decoder-only** models
(Llama-3.2-1B and Gemma-2-2B) adapted with LoRA and, as a contrast, with 3-shot
prompting. Every system sees the same training subset, is scored on the same
500 held-out sentences, and is decoded with the same settings, so the numbers in
this report compare architectures rather than experimental conditions.

> Every number in this README is injected from `results/` by
> `scripts/update_readme.py`. Nothing numeric is typed by hand.

---

## 1. Problem statement and objective

**Problem.** Machine translation is the classic conditional sequence-generation
task: given a source sentence, produce a fluent and faithful sentence in another
language. Two generative architectures now compete for it. The
**encoder–decoder** design reads the whole source with a bidirectional encoder
and generates the target with a cross-attending decoder — the design machine
translation was invented with. The **decoder-only** design, which drives today's
large language models, has no encoder at all: the source sentence is simply part
of the prompt, and translation is next-token prediction conditioned on an
instruction. Decoder-only models dominate general-purpose generation, but it is
not obvious that they are the better choice for a narrow, well-specified task
like translation once compute is taken into account — especially for a
lower-resource target language written in a non-Latin script.

**Objective.** Build four working English→Hindi translation systems, one per
approach named in the assignment, and answer three questions with measurements
rather than assertion:

1. **Quality** — how much translation quality does each architecture reach on
   the same held-out test set, measured with BLEU and chrF++?
2. **Cost** — what does that quality cost in parameters, adaptation time and
   inference latency, and how does the quality-per-unit-compute compare between
   encoder–decoder and decoder-only designs?
3. **Responsibility** — what do the actual outputs reveal about bias,
   hallucination, cultural misrepresentation and the handling of sensitive
   content, and what concrete mitigations follow?

A fourth, cross-cutting question runs through the work: how sensitive is each
system to **decoding temperature**, i.e. how much of its apparent quality is
really the sharpness of its output distribution rather than knowledge of Hindi?

---

## 2. Dataset description and source link

**Dataset:** OPUS-100, English–Hindi configuration
**Hugging Face id:** `Helsinki-NLP/opus-100`, config `en-hi`
**Source link:** <https://huggingface.co/datasets/Helsinki-NLP/opus-100>
**Original corpus:** OPUS — <https://opus.nlpl.eu/>
**Licence:** the OPUS-100 release is distributed for research use; the
underlying sub-corpora carry their own licences (see the dataset card).

OPUS-100 is an English-centric multilingual corpus covering 100 language pairs,
assembled from the OPUS collection (subtitles, software localisation,
religious texts, news commentary, GNOME/KDE/Ubuntu documentation, and similar
sources). The `en-hi` configuration holds roughly **534k training pairs** with
official **validation** and **test** splits, which this project uses as the
basis of its own splits rather than cutting a test set out of training data.

It is a realistic rather than a clean corpus, and that matters for the analysis:
because it is largely harvested from subtitles and software strings, it mixes
registers heavily, contains short fragments and interjections, and includes a
noticeable number of rows whose "Hindi" side is romanised, untranslated or
misaligned. Those properties are exactly what the preprocessing below removes,
and what the responsible-AI section traces forward into model behaviour.

### Preprocessing

Implemented in [`src/data.py`](src/data.py). Applied in this order, with the
same cleaning function on every split:

1. **Unicode NFC normalisation** — Devanagari can encode the same grapheme in
   multiple ways (pre-composed characters vs. base + nukta/matra sequences).
   Without normalisation, identical-looking strings tokenise differently and
   chrF++ penalises correct output.
2. **Whitespace collapsing** and stripping of leading/trailing dashes and
   quotation marks, which subtitle lines carry constantly.
3. **Length filtering** — pairs are kept only if both sides are between 2 and
   200 characters, which drops empty rows and long concatenated paragraphs.
4. **Markup and URL filtering** — rows containing URLs, HTML-like tags,
   `{0}`-style format placeholders or HTML entities are dropped; these are
   localisation artefacts that teach a model nothing about Hindi.
5. **Length-ratio filtering** — pairs whose English:Hindi word-count ratio is
   worse than 3:1 in either direction are dropped as probable misalignments.
6. **Script validation** — a target is kept only if at least 30 % of its
   alphabetic characters are Devanagari, and a source is dropped if it contains
   any Devanagari. This removes the romanised-Hindi and
   copied-English rows that would otherwise teach the model to echo its input.
7. **De-duplication** on the lowercased source, applied **across all three
   splits at once**, so no test sentence can also appear in training.
8. **Identity filtering** — pairs whose source and target are the same string.

The held-out splits are cleaned **before** the training subset so that the test
set is never starved of material, and sampling is seeded (`SEED = 42`), so the
same split is reproduced on every machine.

**Splits.** 20,000 training pairs / 1,000 validation / 500 test, drawn from
OPUS-100's own `train` / `validation` / `test` splits respectively. The exact
test set is committed as
[`results/test_set_en_hi.jsonl`](results/test_set_en_hi.jsonl) so every score
below can be reproduced sentence for sentence.

<!-- BEGIN:dataset_stats -->
| Split | Pairs | Mean EN words | Mean HI words | Mean EN chars | Mean HI chars | EN word types | HI word types |
|---|---|---|---|---|---|---|---|
| train | 20000 | 8.8 | 10.2 | 47.7 | 49.3 | 24243 | 22264 |
| val | 1000 | 8.6 | 9.8 | 46.0 | 47.0 | 3127 | 3313 |
| test | 500 | 7.6 | 8.8 | 41.2 | 41.7 | 1688 | 1840 |
<!-- END:dataset_stats -->

---

## 3. Models implemented

All four systems solve the identical task with the identical data. What differs
is pretraining, tokenisation and adaptation method — which is precisely what the
comparison is about.

### 3.1 Basic Transformer, encoder–decoder, trained from scratch
[`src/transformer_scratch.py`](src/transformer_scratch.py)

The textbook architecture of Vaswani et al. (2017), built on
`torch.nn.Transformer`: 3 encoder layers + 3 decoder layers, `d_model = 256`,
4 attention heads, feed-forward dimension 1024, sinusoidal positional
encodings, post-norm blocks, dropout 0.1, and the output projection **tied** to
the embedding matrix.

* **Tokenisation.** A joint English+Hindi **byte-level BPE** vocabulary of
  16,000 types, trained from scratch on the 20k training subset only. Byte-level
  BPE is used so Devanagari is representable without a hand-built character set
  and no input is ever out-of-vocabulary.
* **Training.** Cross-entropy with label smoothing 0.1, AdamW
  (β = 0.9/0.98), inverse-square-root schedule with warm-up, gradient clipping
  at 1.0, 20 epochs, best-validation-loss checkpoint kept.
* **Pretraining.** None. This system knows only what the 20k pairs taught it,
  which makes it the control condition for the whole study.

### 3.2 mT5-small, encoder–decoder, fully fine-tuned
[`src/mt5_finetune.py`](src/mt5_finetune.py)

`google/mt5-small` (~300M parameters), pretrained on mC4 across 101 languages
including Hindi. Translation is framed the way T5 frames every task, with a text
prefix: `translate English to Hindi: …`.

* **Tokenisation.** mT5's own 250k-entry SentencePiece vocabulary, which was
  trained multilingually and therefore segments Devanagari efficiently.
* **Adaptation.** **Full** fine-tuning — all parameters updated — with AdamW, a
  one-cycle schedule and gradient clipping, for 3 epochs.
* **Precision note.** mT5 is known to overflow in fp16, so it is trained and
  decoded in fp32 even when a GPU is available. This is a real cost of the model
  and is reflected in its measured times.

### 3.3 Llama-3.2-1B-Instruct, decoder-only, LoRA fine-tuned
### 3.4 Gemma-2-2B-IT, decoder-only, LoRA fine-tuned
[`src/decoder_only.py`](src/decoder_only.py)

Both decoder-only models are adapted by **exactly the same code path**, so the
Llama/Gemma comparison isolates the checkpoint rather than the recipe:

* **Prompt.** One instruction, shared verbatim by both models and rendered
  through each model's own chat template:
  *"Translate the following English sentence into Hindi. Reply with the Hindi
  translation only."*
* **Loss masking.** The prompt tokens are labelled `-100`, so the loss is
  computed **only** on the Hindi continuation. Without this the model spends its
  capacity learning to reproduce the instruction.
* **LoRA.** Rank 16, α = 32, dropout 0.05, applied to all attention and MLP
  projections (`q,k,v,o,gate,up,down`). The frozen base stays in fp16/bf16 while
  the LoRA parameters are held in fp32; training uses autocast with a gradient
  scaler.
* **Checkpoints.** `unsloth/Llama-3.2-1B-Instruct` and `unsloth/gemma-2-2b-it`,
  ungated redistributions of Meta's and Google's weights (see section 4). The
  run records which repo it loaded.
* **Tokenisation.** Each model's native tokeniser — and this is a substantive
  difference, not a detail. Llama-3's BPE vocabulary is heavily English-weighted
  and fragments Devanagari into many short pieces; Gemma's 256k vocabulary
  covers Indic script far better. The effect shows up directly in output
  quality and in tokens-per-sentence at inference.
* **Prompted contrast.** Each model is *also* evaluated with **3-shot prompting
  and no weight update at all**, using few-shot examples drawn deterministically
  from the training split. The assignment permits either fine-tuning or
  prompting for the decoder-only systems; running both turns "adaptation method"
  into a measured variable instead of a choice.
* **Output post-processing.** Instruction-tuned models like to add preambles
  ("Sure, here is the translation:"), quote marks and extra examples. A single
  shared cleaner strips these and keeps the first translation line, identically
  for both models, so the comparison is not an accident of parsing.

Gemma-2 is loaded with `attn_implementation="eager"` because its logit
soft-capping is ignored by the fused attention kernels.

### Comparability controls

| Control | How it is enforced |
|---|---|
| Same data | one seeded `build_splits()` feeds every system |
| Same test set | the 500 sentences in `results/test_set_en_hi.jsonl` |
| Same decoding | greedy (`do_sample=False`, no beam search) for all headline numbers |
| Same length budget | `max_target_len = 64` tokens for every system |
| Same metrics | one sacrebleu configuration, signatures recorded in `run_info.json` |
| Same output cleaning | one minimal, documented rule per family: sentinel stripping for mT5, preamble stripping for the decoders |
| Same timing method | CUDA-synchronised wall clock over the whole test set, same batch size |
| One model in memory | each system is released before the next loads |

<!-- BEGIN:checkpoints -->
| System | Checkpoint actually loaded | Adaptation | Note |
|---|---|---|---|
| Transformer (scratch) | `custom 3+3 layers, d_model=256, joint BPE 16000` | trained from scratch |  |
| mT5-small (fine-tuned) | `google/mt5-small` | full fine-tuning |  |
| Llama-3.2-1B (LoRA) | `unsloth/Llama-3.2-1B-Instruct` | LoRA r=16 |  |
| Gemma-2-2B (LoRA) | `unsloth/gemma-2-2b-it` | LoRA r=16 |  |
| Llama-3.2-1B (3-shot prompt) | `unsloth/Llama-3.2-1B-Instruct` | 3-shot prompting (no weight update) |  |
| Gemma-2-2B (3-shot prompt) | `unsloth/gemma-2-2b-it` | 3-shot prompting (no weight update) |  |
<!-- END:checkpoints -->

---

## 4. Execution steps and required dependencies

### Dependencies

```
torch>=2.1  transformers>=4.44  tokenizers>=0.19  datasets>=2.19  peft>=0.11
sentencepiece>=0.2  protobuf>=4.25  accelerate>=0.30
sacrebleu>=2.4  matplotlib>=3.8  pandas>=2.0
python-docx>=1.1  reportlab>=4.0   # submission document only
```

Install with:

```bash
pip install -r requirements.txt
```

### Hardware

The full profile needs a **CUDA GPU with ~16 GB**. A free **Kaggle T4** or
**Google Colab T4** is sufficient and is what this project was designed around:
the pipeline holds only one model in memory at a time specifically so that it
fits, and gradient checkpointing is on by default for the decoder stages so
Gemma-2-2B does not overflow. CPU-only execution is supported *for pipeline
verification only* (see the smoke profile below); fine-tuning
billion-parameter decoders on CPU is not practical.

Kaggle is the easier of the two for a run this long — sessions last up to 12
hours, where free Colab disconnects when idle.

### Checkpoints — no Hugging Face token required

Meta and Google put Llama-3.2 and Gemma-2 behind a licence click-through on
their own Hugging Face repos. The same weights are also redistributed ungated,
which their licences permit, and **this project loads those mirrors by
default**:

| Model | Repo loaded by default | Official gated repo |
|---|---|---|
| Llama-3.2-1B-Instruct | `unsloth/Llama-3.2-1B-Instruct` | `meta-llama/Llama-3.2-1B-Instruct` |
| Gemma-2-2B-IT | `unsloth/gemma-2-2b-it` | `google/gemma-2-2b-it` |

Same architecture, same tensors, same tokenizer — so the run needs **no token
and no manual approval step**, which removes the most common way for it to
fail. Use of the weights is still governed by the
[Llama 3.2 Community License](https://www.llama.com/llama3_2/license/) and the
[Gemma Terms of Use](https://ai.google.dev/gemma/terms) wherever they are
downloaded from.

To pull from the official gated repos instead, accept both licences, create a
**read** token at <https://huggingface.co/settings/tokens>, expose it as
`HF_TOKEN`, and set `FGAI_OFFICIAL_GATED=1`.

Either way, the repo actually loaded is recorded in `results/run_info.json` and
printed in the table at the end of section 3, so the provenance of every number
is unambiguous. Check before a long run:

```bash
python scripts/check_access.py
```

### Run it

**Option A — the notebook (recommended).**
[`notebooks/FGAI_Assignment3_MT_Comparison.ipynb`](notebooks/FGAI_Assignment3_MT_Comparison.ipynb)
runs unchanged on Kaggle or Colab; its cells detect which platform they are on
and find the working directory, the `HF_TOKEN` secret and the download
mechanism accordingly. It installs the dependencies, clones this repository,
runs every stage, displays the tables and figures inline, and refreshes this
README's generated blocks at the end.

*On Kaggle:* set **Accelerator** to **GPU T4 x2** and **Internet** to **On** in
the right-hand panel — internet is off by default, and without it nothing can
be installed or downloaded. Both settings require a phone-verified account.
Then **Run All**; collect `fgai_results.zip` from the **Output** panel.

*On Colab:* **Runtime → Change runtime type → T4 GPU**, then
**Runtime → Run all**. The last cell downloads `fgai_results.zip` to your
browser.

No Hugging Face token is needed on either platform unless you set
`FGAI_OFFICIAL_GATED=1` (see above).

When the notebook finishes, fold its `fgai_results.zip` back into a local
clone with one command:

```bash
python scripts/import_results.py ~/Downloads/fgai_results.zip
```

That installs the results, re-injects every number into this README and
`docs/responsible_ai.md`, and rebuilds the submission document. It validates
every path in the archive before writing anything.

**Option B — command line.**

```bash
git clone https://github.com/Nehalll-code/fgai-assignment3-mt-comparison.git
cd fgai-assignment3-mt-comparison
pip install -r requirements.txt
export HF_TOKEN=hf_xxx            # Windows PowerShell: $env:HF_TOKEN="hf_xxx"
python -m src.run_all             # full study, writes everything under results/
python scripts/update_readme.py   # inject the numbers into this README
```

### Tests

```bash
python -m tests.test_pipeline_units
```

23 unit tests covering the text handling that sits between a model and its
score — output cleaning, corpus filtering, the failure-mode flags and the
metrics. No GPU, no downloads, a couple of seconds. A bug in any of these
changes every reported number without raising anything, which is not something
a three-hour GPU run should be the first to discover.

Useful variations:

```bash
python -m src.run_all --stages scratch mt5      # only the encoder-decoders
python -m src.run_all --no-sweep --no-probes    # headline metrics only
python -m src.run_all --force                   # ignore existing checkpoints
FGAI_SMOKE=1 python -m src.run_all              # a few minutes, CPU pipeline check
```

The full study takes roughly 2–3 hours on a T4, which is close to a free Colab
session limit, so `--stages` is also the way to **split the run across
sessions**: `metrics.csv`, `metrics.json` and `temperature_sweep.csv` carry
forward any system the current run did not produce, so

```bash
python -m src.run_all --stages scratch mt5 llama   # session 1
python -m src.run_all --stages gemma               # session 2, keeps session 1's rows
```

ends with one complete table. (`run_info.json` always describes the most recent
run.) Model checkpoints are cached under `checkpoints/`, so a re-run of a stage
that already finished skips its training and only re-evaluates.

The **smoke profile** (`FGAI_SMOKE=1`) shrinks every dimension and swaps the
large checkpoints for randomly initialised stand-ins of the same architecture.
It exists to prove the pipeline executes end to end on any machine; its scores
are meaningless by construction, and `scripts/update_readme.py` labels them as
such if they are ever the most recent run.

### What gets written

```
results/
  metrics.csv / metrics.md / metrics.json   headline table for all systems
  temperature_sweep.csv / .md               sampling-temperature analysis
  error_analysis.csv / .md                  automatic failure-mode profiling
  error_examples.csv                        concrete instances of each failure mode
  responsible_ai_probes.csv / .md           probe-set outputs for section 7
  sample_translations.md                    side-by-side outputs, same inputs
  test_set_en_hi.jsonl                      the exact 500 test pairs
  run_info.json                             device, versions, metric signatures
  translations/*.csv                        every hypothesis, with sentence chrF++
  plots/*.png                               the figures used below
```

---

## 5. Results obtained

<!-- BEGIN:run_banner -->
Run completed 2026-11-01T01:25:48+00:00 on **Tesla T4** (bfloat16), profile `full`, 7.6 h of measured training and inference across the six systems.
Splits: test=500, val=1000, train=20000. Test-set mean length: 7.6 English words / 8.8 Hindi words.
Metric signatures: `nrefs:1|case:mixed|eff:no|tok:intl|smooth:exp|version:2.6.0` and `nrefs:1|case:mixed|eff:yes|nc:6|nw:2|space:no|version:2.6.0`.
<!-- END:run_banner -->

<!-- BEGIN:results_table -->
| Model | Architecture | Adaptation | Params (M) | Params trained (M) | BLEU | chrF++ | Train time (s) | Inference, full test set (s) | ms / sentence |
|---|---|---|---|---|---|---|---|---|---|
| Transformer (scratch) | encoder-decoder | trained from scratch | 9.64 | 9.643 | 5.48 | 18.34 | 660.6 | 9.8 | 19.6 |
| mT5-small (fine-tuned) | encoder-decoder | full fine-tuning | 300.18 | 300.177 | 9.22 | 21.66 | 1781.7 | 30.7 | 61.4 |
| Llama-3.2-1B (LoRA) | decoder-only | LoRA r=16 | 1235.81 | 11.272 | 14.96 | 32.33 | 7775.7 | 239.1 | 478.1 |
| Gemma-2-2B (LoRA) | decoder-only | LoRA r=16 | 2614.34 | 20.767 | 20.31 | 38.4 | 15591.1 | 360.1 | 720.2 |
| Llama-3.2-1B (3-shot prompt) | decoder-only | 3-shot prompting (no weight update) | 1235.81 | 0.0 | 0.72 | 8.39 | 0.0 | 370.5 | 741.0 |
| Gemma-2-2B (3-shot prompt) | decoder-only | 3-shot prompting (no weight update) | 2614.34 | 0.0 | 8.55 | 25.73 | 0.0 | 450.4 | 900.9 |
<!-- END:results_table -->

![Translation quality](results/plots/quality_bleu_chrf.png)

![Computational cost](results/plots/compute_cost.png)

![Quality against model size](results/plots/quality_vs_size.png)

![Training and validation loss](results/plots/loss_curves.png)

The two LoRA panels show a single point rather than a curve: both decoder-only
models are trained for one epoch, so there is exactly one train/validation
measurement each. The scratch Transformer (20 epochs) is the only system here
that trains long enough to show the shape that matters — its validation loss
flattens around epoch 10 while training loss keeps falling, which is the
overfitting this data size makes unavoidable and the reason its BLEU is the
lowest of the four trained systems.

Side-by-side outputs for the same inputs are in
[`results/sample_translations.md`](results/sample_translations.md), and every
hypothesis with its sentence-level chrF++ is in
[`results/translations/`](results/translations).

### Why BLEU *and* chrF++

BLEU counts matching word n-grams. Hindi is morphologically richer than English
and inflects for case, number, gender and honorific level, so a translation that
is correct but inflected differently from the reference loses BLEU credit it
arguably deserves. **chrF++** scores character n-grams (plus word unigrams and
bigrams), which makes it far more forgiving of inflection and much better
correlated with human judgement for Indic languages. BLEU is reported because
the assignment asks for it; chrF++ is the metric the conclusions lean on.

BLEU is computed with sacrebleu's `intl` tokeniser rather than the default
`13a`, because `13a` does not split Devanagari punctuation (danda `।`,
double danda `॥`) and systematically mis-scores Hindi. The exact metric
signatures are recorded in `results/run_info.json` so the numbers are
reproducible.

---

## 6. Model comparison and evaluation

### 6.1 Encoder–decoder vs decoder-only

**Pretraining dominates everything else.** The from-scratch Transformer and the
fine-tuned mT5 share an architecture family and differ almost only in
pretraining, so the gap between them is a clean measurement of what 20k
sentence pairs can and cannot teach. The scratch model has to learn Hindi
morphology, word order and vocabulary from those pairs alone; mT5 arrives
already knowing Hindi and only has to learn the task. On a subset this size the
pretrained model is expected to win decisively, and the scratch model's value is
as the control that shows *how much* of the pretrained models' score is
pretraining rather than architecture.

**Architecture is not the main axis — adaptation is.** The decoder-only models
carry 3–7× more parameters than mT5 but update only ~0.5–1 % of them through
LoRA, while mT5 updates 100 % of its weights. That is the real asymmetry in this
study, and it cuts both ways: LoRA is dramatically cheaper per step and far less
prone to catastrophic forgetting, but it has less capacity to re-shape a model
whose pretraining under-represents the target language.

**Tokenisation is the hidden cost of decoder-only translation.** An
encoder–decoder reads the source once through its encoder and then generates;
cost scales with (source + target) length. A decoder-only model re-reads the
instruction, the few-shot examples and the source as prompt context for every
sentence, and generates the target autoregressively over that longer context.
When the target script is also fragmented by an English-weighted vocabulary —
Llama-3's case for Devanagari — each Hindi sentence costs more tokens to emit,
so the same sentence is slower *and* more likely to drift. Gemma-2's larger,
more Indic-friendly vocabulary is the clearest mechanism by which it can beat
a model of similar scale here.

**The prompted rows are the control for adaptation.** Comparing LoRA against
3-shot prompting on the *same* checkpoint separates "what the base model already
knows about Hindi" from "what a small amount of task adaptation adds". A large
LoRA-over-prompting gap means the base model could translate but did not know
the expected output format and register; a small gap means the base model's
Hindi itself is the ceiling.

### 6.2 Quality per unit of compute

Three cost axes are measured independently, and deliberately plotted as separate
panels rather than on one dual-axis chart, because they do not share a scale:

* **Parameters** — model size, the capacity and memory cost.
* **Adaptation time** — seconds of training/fine-tuning actually spent. Zero
  for the prompted rows, which is their whole advantage.
* **Inference latency** — milliseconds per sentence over the full 500-sentence
  test set, at a fixed generation batch size, CUDA-synchronised.

The practically interesting question is not "which model scores highest" but
"which point on the quality/cost curve a deployment should pick". A system that
reaches 90 % of the best chrF++ for a tenth of the latency is usually the right
answer for a translation service; the quality-per-parameter figure in section 8
makes that trade-off explicit.

### 6.3 Automatic failure-mode profiling

Corpus metrics hide the failures that matter most in practice, so every
hypothesis is additionally checked for seven specific failure modes
([`src/analysis.py`](src/analysis.py)): empty output, Latin-script leakage
(English copied through untranslated), number mismatch against the source,
degenerate repetition, output far longer than the reference (the usual shape of
hallucinated additions), refusal or meta-commentary leaking out of an
instruction-tuned model, and verbatim source copying.

<!-- BEGIN:error_table -->
| System | n | Empty % | Latin-script leak % | Number mismatch % | Degenerate repetition % | Over-long output % | Refusal / meta % | Source copied % |
|---|---|---|---|---|---|---|---|---|
| gemma_2_2b_3_shot_prompt | 500 | 1.8 | 6.2 | 1.2 | 0.4 | 0.4 | 0.0 | 3.2 |
| gemma_2_2b_lora | 500 | 0.0 | 2.4 | 0.6 | 0.4 | 0.0 | 0.0 | 0.0 |
| llama_3_2_1b_3_shot_prompt | 500 | 0.0 | 17.6 | 4.4 | 0.2 | 21.2 | 0.0 | 2.0 |
| llama_3_2_1b_lora | 500 | 0.0 | 2.2 | 0.8 | 1.2 | 0.4 | 0.0 | 0.0 |
| mt5_small_fine_tuned | 500 | 0.0 | 4.8 | 0.8 | 7.0 | 0.2 | 0.0 | 0.2 |
| transformer_scratch | 500 | 0.2 | 0.0 | 4.4 | 1.6 | 0.0 | 0.0 | 0.0 |
<!-- END:error_table -->

Concrete instances of each flag, plus the three worst-scoring sentences per
system, are in [`results/error_examples.csv`](results/error_examples.csv).

---

## 7. Sampling-temperature analysis

Temperature `T` rescales the logits before the softmax. `T → 0` is greedy
decoding (always the arg-max token); `T = 1` samples from the model's own
distribution; `T > 1` flattens it and makes rare tokens more likely. For
open-ended generation some temperature is desirable — it buys diversity. For
**translation it is almost pure downside**, because there is a small set of
correct outputs and sampling trades accuracy for variety the task does not want.

This project measures that rather than assuming it. Each system decodes a fixed
150-sentence prefix of the test set at `T ∈ {0 (greedy), 0.3, 0.6, 0.9, 1.2}` —
identical sentences for every system and every temperature — and
four quantities are recorded per setting: BLEU, chrF++, mean output length, and
the ratio of distinct outputs (a direct diversity measure).

![Temperature sweep](results/plots/temperature_sweep.png)

<!-- BEGIN:temperature_summary -->
- **Transformer (scratch)** - greedy chrF++ 18.93; best sampled setting T=0.3 (chrF++ 18.63, still below greedy); at T=1.2 chrF++ falls to 6.65, a 65% relative loss.
- **mT5-small (fine-tuned)** - greedy chrF++ 21.67; best sampled setting T=0.3 (chrF++ 22.16, above greedy); at T=1.2 chrF++ falls to 14.97, a 31% relative loss.
- **Llama-3.2-1B (LoRA)** - greedy chrF++ 32.91; best sampled setting T=0.3 (chrF++ 32.75, still below greedy); at T=1.2 chrF++ falls to 7.06, a 79% relative loss.
- **Gemma-2-2B (LoRA)** - greedy chrF++ 39.82; best sampled setting T=0.3 (chrF++ 38.63, still below greedy); at T=1.2 chrF++ falls to 15.81, a 60% relative loss.
<!-- END:temperature_summary -->

Full sweep:

<!-- BEGIN:temperature_table -->
| Model | Temperature | Decoding | BLEU | chrF++ | Mean chars | Distinct outputs | Empty |
|---|---|---|---|---|---|---|---|
| Transformer (scratch) | 0.0 | greedy | 5.79 | 18.93 | 37.0 | 1.0 | 1 |
| Transformer (scratch) | 0.3 | sampling | 5.15 | 18.63 | 36.1 | 1.0 | 1 |
| Transformer (scratch) | 0.6 | sampling | 3.21 | 17.55 | 37.0 | 0.993 | 1 |
| Transformer (scratch) | 0.9 | sampling | 2.84 | 14.52 | 40.7 | 1.0 | 1 |
| Transformer (scratch) | 1.2 | sampling | 0.53 | 6.65 | 64.5 | 1.0 | 0 |
| mT5-small (fine-tuned) | 0.0 | greedy | 7.32 | 21.67 | 40.6 | 1.0 | 0 |
| mT5-small (fine-tuned) | 0.3 | sampling | 7.59 | 22.16 | 38.1 | 1.0 | 0 |
| mT5-small (fine-tuned) | 0.6 | sampling | 6.46 | 20.84 | 39.4 | 1.0 | 0 |
| mT5-small (fine-tuned) | 0.9 | sampling | 4.65 | 18.24 | 41.6 | 1.0 | 0 |
| mT5-small (fine-tuned) | 1.2 | sampling | 2.98 | 14.97 | 46.6 | 1.0 | 0 |
| Llama-3.2-1B (LoRA) | 0.0 | greedy | 14.81 | 32.91 | 41.5 | 1.0 | 0 |
| Llama-3.2-1B (LoRA) | 0.3 | sampling | 14.14 | 32.75 | 42.2 | 1.0 | 0 |
| Llama-3.2-1B (LoRA) | 0.6 | sampling | 11.99 | 29.06 | 41.3 | 1.0 | 0 |
| Llama-3.2-1B (LoRA) | 0.9 | sampling | 5.81 | 20.6 | 41.2 | 1.0 | 0 |
| Llama-3.2-1B (LoRA) | 1.2 | sampling | 1.03 | 7.06 | 118.9 | 1.0 | 0 |
| Gemma-2-2B (LoRA) | 0.0 | greedy | 20.96 | 39.82 | 40.9 | 0.993 | 0 |
| Gemma-2-2B (LoRA) | 0.3 | sampling | 18.64 | 38.63 | 40.4 | 1.0 | 0 |
| Gemma-2-2B (LoRA) | 0.6 | sampling | 15.3 | 33.6 | 40.2 | 1.0 | 0 |
| Gemma-2-2B (LoRA) | 0.9 | sampling | 8.29 | 25.0 | 41.8 | 1.0 | 0 |
| Gemma-2-2B (LoRA) | 1.2 | sampling | 4.06 | 15.81 | 47.5 | 1.0 | 0 |
<!-- END:temperature_table -->

### How to read the curve

The **shape** of each curve is as informative as its height, and it is a probe
of how confidently a model has learned the task:

* A curve that stays flat to `T ≈ 0.4` and then falls belongs to a model whose
  probability mass is concentrated on the right token — adaptation worked, and
  low-temperature sampling merely picks among near-ties.
* A curve that drops immediately belongs to a model whose correct token is only
  narrowly ahead of its competitors. Its greedy score is real but brittle: the
  arg-max is right, the distribution underneath it is not.
* **Rising mean output length with temperature** is the hallucination signature.
  At high `T` the model becomes less likely to emit its end-of-sequence token,
  so it keeps generating — adding content that was never in the source. This is
  why the temperature analysis and the responsible-AI section are the same
  experiment viewed twice.
* **Rising distinct-output ratio with falling chrF++** is diversity bought with
  accuracy — the exact trade translation does not want.

The practical conclusion is reported in section 8: greedy decoding for
production translation, with any temperature reserved for deliberately
generating *alternative* candidate translations for a human to choose among.

---

## 8. Responsible and ethical AI

Full discussion, with the probe outputs quoted and mitigations mapped to the
pipeline, is in
[**`docs/responsible_ai.md`**](docs/responsible_ai.md). In summary, four
concerns were examined against the systems' own outputs rather than in the
abstract:

1. **Bias.** Hindi marks gender on verbs and adjectives, so an English sentence
   with no gender cue (*"The doctor finished the report"*) forces every model to
   commit to a gender. A fixed probe set makes each model's default visible,
   including whether it follows an explicit counter-stereotypical cue.
2. **Hallucination.** Probes carrying numbers, names, dates and amounts test
   whether content survives translation; the automatic `number_mismatch` and
   `length_blowup` flags quantify it across the whole test set, and the
   temperature sweep shows how it grows with `T`.
3. **Cultural misrepresentation.** Probes with culture-specific references test
   whether a system translates, transliterates or silently substitutes a locally
   more familiar concept — a failure that is invisible to BLEU because the
   output is fluent.
4. **Sensitive content.** Clinically and legally worded probes (dosages,
   allergies, contract terms) test numeric and polarity fidelity, where an error
   is a real-world harm rather than a lost metric point.

Probe outputs for every system: [`results/responsible_ai_probes.md`](results/responsible_ai_probes.md).

---

## 9. Conclusion — best-performing model

<!-- BEGIN:best_model -->
- **Best overall quality: Gemma-2-2B (LoRA)** - chrF++ 38.4, BLEU 20.31 (decoder-only, LoRA r=16, 2614.34M parameters).
- Best BLEU: Gemma-2-2B (LoRA) (20.31).
- Fastest at inference: Transformer (scratch) (19.6 ms/sentence).
- Cheapest adaptation: Transformer (scratch) (660.6s of training).
- Best quality per parameter: Transformer (scratch) (1.902 chrF++ per million parameters).
<!-- END:best_model -->

### What the study shows

* **Pretraining, not architecture, is the first-order variable.** The
  from-scratch Transformer and mT5 are the same kind of model; the distance
  between them is the value of multilingual pretraining on a 20k-pair budget.
* **Decoder-only models are competitive but not free.** They bring far more
  pretrained knowledge, but pay for it in prompt re-reading, tokenisation
  overhead on Devanagari, and inference latency — and under LoRA they adapt only
  a fraction of their weights.
* **Tokenisation is a first-class design decision for non-Latin targets.** A
  vocabulary that fragments Devanagari costs quality and speed simultaneously.
* **Greedy decoding is the correct default for translation.** The temperature
  sweep shows quality falling, and output length — the hallucination proxy —
  rising, as temperature increases. The fall is monotonic above T=0.3 for all
  four systems; the only setting that beats greedy anywhere is mT5 at T=0.3
  (chrF++ 22.16 against 21.67), a margin small enough to be sampling noise and
  not a reason to abandon greedy.
* **Good corpus metrics do not mean a safe system.** The error profiling and the
  probe sets surface gendered defaults, dropped numbers and softened sensitive
  content in systems whose BLEU looks respectable.

---

## Repository layout

```
src/
  config.py              all hyper-parameters, both run profiles, one place
  data.py                download, clean, split, cache (deterministic)
  transformer_scratch.py system 1: Transformer from scratch + joint BPE
  mt5_finetune.py        system 2: mT5-small full fine-tuning
  decoder_only.py        systems 3 & 4: Llama + Gemma, LoRA and 3-shot prompting
  evaluate.py            BLEU / chrF++, timing, result serialisation
  temperature_sweep.py   sampling-temperature analysis
  analysis.py            failure-mode profiling + responsible-AI probe sets
  plots.py               the figures
  run_all.py             end-to-end driver
scripts/
  update_readme.py       inject results/ numbers into this README
  import_results.py      fold a finished GPU run's zip back into the repo
  make_submission_doc.py build the submission .docx and .pdf
  make_notebook.py       regenerate the Colab notebook from these modules
notebooks/
  FGAI_Assignment3_MT_Comparison.ipynb    Colab-ready, runs the whole study
tests/
  test_pipeline_units.py  unit tests for cleaning, filtering, flags and metrics
docs/
  responsible_ai.md      the full section-(iii) discussion
results/                 all generated metrics, tables, plots and translations
```

## Reproducibility

Seeded data sampling and model initialisation (`SEED = 42`), the exact test set
committed to the repository, metric signatures and library versions recorded in
`results/run_info.json`, and every README number injected from those files by
script rather than typed. GPU kernel non-determinism means scores may move by a
few hundredths between runs on different hardware.

## References

* Vaswani et al., *Attention Is All You Need*, NeurIPS 2017.
* Xue et al., *mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer*, NAACL 2021.
* Zhang et al., *Improving Massively Multilingual Neural Machine Translation and Zero-Shot Translation* (OPUS-100), ACL 2020.
* Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR 2022.
* Popović, *chrF++: words helping character n-grams*, WMT 2017.
* Post, *A Call for Clarity in Reporting BLEU Scores* (sacrebleu), WMT 2018.
