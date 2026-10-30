"""Central configuration for the English->Hindi machine-translation comparison.

Every module reads its hyper-parameters from here so that the four systems are
trained and evaluated under directly comparable settings.

Two run profiles exist:

* ``full``  - the profile used for the numbers reported in the README.
              Needs a CUDA GPU (a free Colab T4 is enough).
* ``smoke`` - a tiny CPU-only profile used to prove the pipeline executes
              end to end.  It swaps all three pretrained checkpoints for
              randomly initialised stand-ins of the same architecture, so the
              scores it produces are meaningless by construction.  It also
              writes to ``data_smoke/``, ``checkpoints_smoke/`` and
              ``results_smoke/`` so it can never touch a real run's output.

Select the smoke profile with the environment variable ``FGAI_SMOKE=1``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]

SMOKE = os.environ.get("FGAI_SMOKE", "0") == "1"


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


DEVICE = _device()
# bf16 is only safe on Ampere+ ; the free Colab T4 is Turing, so prefer fp16 there.
if DEVICE == "cuda":
    DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
else:
    DTYPE = torch.float32


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #

DATASET_ID = "Helsinki-NLP/opus-100"
DATASET_CONFIG = "en-hi"
DATASET_URL = "https://huggingface.co/datasets/Helsinki-NLP/opus-100"
SRC_LANG, TGT_LANG = "en", "hi"
SRC_LANG_NAME, TGT_LANG_NAME = "English", "Hindi"

# Cleaning thresholds (see src/data.py for how each one is applied).
MIN_CHARS = 2
MAX_CHARS = 200
MAX_LEN_RATIO = 3.0          # reject pairs whose length ratio is wilder than this
MIN_DEVANAGARI_RATIO = 0.30  # reject "Hindi" targets that are mostly not Devanagari


@dataclass
class Profile:
    """Sizes and hyper-parameters that differ between the full and smoke runs."""

    name: str

    # split sizes
    n_train: int
    n_val: int
    n_test: int
    n_temp_sweep: int

    # scratch Transformer
    vocab_size: int
    d_model: int
    nhead: int
    num_layers: int
    dim_feedforward: int
    dropout: float
    scratch_epochs: int
    scratch_batch_size: int
    scratch_lr: float

    # mT5
    mt5_id: str
    mt5_epochs: int
    mt5_batch_size: int
    mt5_lr: float

    # decoder-only
    llama_id: str
    gemma_id: str
    decoder_epochs: int
    decoder_batch_size: int
    decoder_grad_accum: int
    decoder_lr: float
    lora_r: int
    lora_alpha: int
    lora_dropout: float

    # generation
    max_source_len: int
    max_target_len: int
    gen_batch_size: int
    temperatures: tuple = (0.2, 0.4, 0.6, 0.8, 1.0, 1.2)
    fallback_llama_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    fallback_gemma_id: str = "google/gemma-2-2b-it"


FULL = Profile(
    name="full",
    # 20k pairs is the largest subset for which all four systems - including a
    # LoRA pass over Gemma-2-2B - finish inside one free Colab T4 session.
    # Every system sees the same 20k, which is what keeps the comparison honest.
    n_train=20_000,
    n_val=1_000,
    n_test=500,
    n_temp_sweep=150,
    vocab_size=16_000,
    d_model=256,
    nhead=4,
    num_layers=3,
    dim_feedforward=1_024,
    dropout=0.1,
    scratch_epochs=20,
    scratch_batch_size=64,
    scratch_lr=5e-4,
    mt5_id="google/mt5-small",
    mt5_epochs=3,
    mt5_batch_size=16,
    mt5_lr=3e-4,
    llama_id="meta-llama/Llama-3.2-1B-Instruct",
    gemma_id="google/gemma-2-2b-it",
    decoder_epochs=1,
    decoder_batch_size=4,
    decoder_grad_accum=4,
    decoder_lr=2e-4,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    max_source_len=64,
    max_target_len=64,
    gen_batch_size=16,
    temperatures=(0.3, 0.6, 0.9, 1.2),
)

SMOKE_PROFILE = Profile(
    name="smoke",
    n_train=400,
    n_val=64,
    n_test=24,
    n_temp_sweep=8,
    vocab_size=2_000,
    d_model=64,
    nhead=2,
    num_layers=2,
    dim_feedforward=128,
    dropout=0.1,
    scratch_epochs=2,
    scratch_batch_size=16,
    scratch_lr=1e-3,
    # A randomly initialised tiny mT5 as well, so the smoke run takes minutes
    # rather than half an hour of CPU time on the real 300M checkpoint.
    mt5_id="hf-internal-testing/tiny-random-mt5",
    mt5_epochs=1,
    mt5_batch_size=4,
    mt5_lr=3e-4,
    # Randomly initialised stand-ins: correct architectures, ~2 MB each, no gate.
    llama_id="hf-internal-testing/tiny-random-LlamaForCausalLM",
    gemma_id="hf-internal-testing/tiny-random-Gemma2ForCausalLM",
    decoder_epochs=1,
    decoder_batch_size=2,
    decoder_grad_accum=1,
    decoder_lr=2e-4,
    lora_r=4,
    lora_alpha=8,
    lora_dropout=0.0,
    max_source_len=32,
    max_target_len=32,
    gen_batch_size=4,
    temperatures=(0.4, 1.0),
)

P: Profile = SMOKE_PROFILE if SMOKE else FULL

# The two profiles keep entirely separate data, checkpoints and results, so a
# quick smoke run can never overwrite real results or feed 400 cached training
# pairs to a full run.
_SUFFIX = "" if P.name == "full" else f"_{P.name}"
DATA_DIR = ROOT / f"data{_SUFFIX}"
RESULTS_DIR = ROOT / f"results{_SUFFIX}"
CHECKPOINT_DIR = ROOT / f"checkpoints{_SUFFIX}"
PLOTS_DIR = RESULTS_DIR / "plots"
TRANSLATIONS_DIR = RESULTS_DIR / "translations"

SEED = 42


def _dtype_kwarg_name() -> str:
    """transformers 4.54 renamed ``from_pretrained(torch_dtype=)`` to ``dtype=``.

    Picking the name by version keeps the notebook working on both old and new
    releases; guessing wrong would silently load a 2B model in fp32.
    """
    try:
        from transformers import __version__ as v
        parts = tuple(int(p) for p in v.split(".")[:2] if p.isdigit())
        return "dtype" if parts >= (4, 54) else "torch_dtype"
    except Exception:
        return "torch_dtype"


DTYPE_KEY = _dtype_kwarg_name()

# Prompt used for both decoder-only models, identical wording so that the
# comparison measures the model and not the prompt.
INSTRUCTION = (
    f"Translate the following {SRC_LANG_NAME} sentence into {TGT_LANG_NAME}. "
    f"Reply with the {TGT_LANG_NAME} translation only."
)

# sacrebleu settings.  'intl' splits on Unicode punctuation and therefore
# handles Devanagari danda/double-danda correctly, unlike the default '13a'.
BLEU_TOKENIZE = "intl"
CHRF_WORD_ORDER = 2  # chrF++

MODEL_ORDER = [
    "Transformer (scratch)",
    "mT5-small (fine-tuned)",
    "Llama-3.2-1B (LoRA)",
    "Gemma-2-2B (LoRA)",
    "Llama-3.2-1B (3-shot prompt)",
    "Gemma-2-2B (3-shot prompt)",
]

ARCHITECTURE = {
    "Transformer (scratch)": "encoder-decoder",
    "mT5-small (fine-tuned)": "encoder-decoder",
    "Llama-3.2-1B (LoRA)": "decoder-only",
    "Gemma-2-2B (LoRA)": "decoder-only",
    "Llama-3.2-1B (3-shot prompt)": "decoder-only",
    "Gemma-2-2B (3-shot prompt)": "decoder-only",
}


def ensure_dirs() -> None:
    for d in (DATA_DIR, RESULTS_DIR, PLOTS_DIR, TRANSLATIONS_DIR, CHECKPOINT_DIR):
        d.mkdir(parents=True, exist_ok=True)


def describe() -> str:
    return (
        f"profile={P.name} device={DEVICE} dtype={str(DTYPE).split('.')[-1]} "
        f"train={P.n_train} val={P.n_val} test={P.n_test}"
    )
