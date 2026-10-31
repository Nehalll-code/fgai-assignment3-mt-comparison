"""Unit tests for the parts of the pipeline that are easy to get quietly wrong.

These cover the text handling that sits between a model and its score - output
cleaning, corpus filtering and the failure-mode flags. A bug in any of them
changes every reported number without raising anything, which is exactly the
kind of bug a 3-hour GPU run should not be the first thing to reveal.

Run with pytest, or directly:

    python -m tests.test_pipeline_units
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import analysis, data  # noqa: E402
from src import decoder_only as DO  # noqa: E402
from src import evaluate as E  # noqa: E402
from src import mt5_finetune as MT5  # noqa: E402

NL = "\n"


# --------------------------------------------------------------------------- #
# Decoder-only output cleaning
# --------------------------------------------------------------------------- #

def test_strips_single_preamble():
    assert DO.clean_output("Sure, here is the translation: नमस्ते") \
        == "नमस्ते"


def test_strips_stacked_preambles():
    """An anchored pattern only removes the leading preamble; models stack them."""
    assert DO.clean_output("Sure! Here is the translation:" + NL + "वह आया") \
        == "वह आया"


def test_preamble_on_its_own_line_keeps_the_translation():
    """Selecting the first line before stripping would keep the preamble."""
    assert DO.clean_output("Here is the Hindi translation:" + NL + NL + "वह आया") \
        == "वह आया"


def test_stops_at_the_next_example():
    text = "पहली" + NL + "English: the next sentence"
    assert DO.clean_output(text) == "पहली"


def test_strips_wrapping_quotes_and_fences():
    assert DO.clean_output('"नमस्ते"') == "नमस्ते"
    assert DO.clean_output("```वह आया```") == "वह आया"


def test_empty_generation_stays_empty():
    assert DO.clean_output("") == ""
    assert DO.clean_output("   " + NL + "  ") == ""


def test_does_not_eat_a_real_translation():
    """Nothing resembling a preamble: the text must survive untouched."""
    original = "वह ठीक है।"
    assert DO.clean_output(f"  {original}  ") == original


# --------------------------------------------------------------------------- #
# mT5 output cleaning
# --------------------------------------------------------------------------- #

def test_mt5_strips_span_sentinels():
    assert MT5.clean_output("<extra_id_0> नमस्ते") == "नमस्ते"
    assert MT5.clean_output("<extra_id_0>") == ""
    assert MT5.clean_output("a <extra_id_17> b") == "a b"


def test_mt5_keeps_sentence_final_punctuation():
    """Stripping a trailing stop would silently change every BLEU score."""
    assert MT5.clean_output("वह ठीक है।") == "वह ठीक है।"
    assert MT5.clean_output("वह ठीक है.") == "वह ठीक है."


# --------------------------------------------------------------------------- #
# Corpus cleaning
# --------------------------------------------------------------------------- #

def test_keeps_a_good_pair():
    assert data.clean_pair("How are you?", "आप कैसे हैं?") is not None


def test_rejects_romanised_or_untranslated_targets():
    """OPUS-100 en-hi contains rows whose 'Hindi' side is not Devanagari."""
    assert data.clean_pair("How are you?", "aap kaise hain?") is None
    assert data.clean_pair("How are you?", "How are you?") is None


def test_rejects_markup_and_placeholders():
    assert data.clean_pair("Visit http://example.com", "यहाँ देखें") is None
    assert data.clean_pair("Open {0} now", "अब खोलें") is None


def test_rejects_length_ratio_outliers():
    long_hi = " ".join(["शब्द"] * 20)
    assert data.clean_pair("Hello there", long_hi) is None


def test_normalises_whitespace_and_stray_dashes():
    cleaned = data.clean_pair("-  How   are  you? ", " आप  कैसे हैं? ")
    assert cleaned == ("How are you?", "आप कैसे हैं?")


# --------------------------------------------------------------------------- #
# Failure-mode flags
# --------------------------------------------------------------------------- #

def test_number_mismatch_detects_a_dropped_digit():
    src = "Take two tablets every 6 hours, up to 8 a day."
    good = "हर 6 घंटे, दिन में 8 तक।"
    bad = "हर 6 घंटे।"
    assert analysis.flags(src, good, good)["number_mismatch"] is False
    assert analysis.flags(src, good, bad)["number_mismatch"] is True


def test_number_mismatch_accepts_devanagari_digits():
    """A correctly transliterated number is not an error."""
    src = "Only 3 samples passed."
    hyp = "केवल ३ नमूने पास हुए।"
    assert analysis.flags(src, hyp, hyp)["number_mismatch"] is False


def test_latin_leak_only_fires_when_the_reference_has_no_latin():
    src = "Hello there"
    assert analysis.flags(src, "नमस्ते", "Hello there")["latin_leak"] is True
    # A reference that legitimately contains Latin (a brand name) must not flag.
    assert analysis.flags(src, "Nokia फोन", "Nokia फोन")["latin_leak"] is False


def test_repetition_detects_degenerate_loops():
    assert analysis.flags("x", "y", "क्या " * 8)["repetition"] is True
    assert analysis.flags("x", "y", "यह एक सामान्य वाक्य है")["repetition"] is False


def test_source_copy_and_empty():
    assert analysis.flags("Hello there.", "नमस्ते", "hello there")["source_copy"] is True
    assert analysis.flags("Hello", "नमस्ते", "   ")["empty"] is True


def test_refusal_detection():
    hyp = "I'm sorry, I cannot translate medical instructions."
    assert analysis.flags("Take two tablets.", "दो", hyp)["refusal_or_meta"] is True


# --------------------------------------------------------------------------- #
# Scoring and serialisation
# --------------------------------------------------------------------------- #

def test_score_handles_empty_hypotheses_without_crashing():
    s = E.score(["", "नमस्ते"], ["नमस्ते", "नमस्ते"])
    assert s["empty_outputs"] == 1
    assert 0.0 <= s["chrf"] <= 100.0
    assert "tok:intl" in s["bleu_signature"]


def test_perfect_translation_scores_100():
    # Both sentences need at least four tokens, or BLEU's 3- and 4-gram
    # precisions are empty and even an exact match does not reach 100.
    refs = ["यह एक सामान्य परीक्षण वाक्य है",
            "वह कल सुबह घर वापस आया था"]
    s = E.score(refs, refs)
    assert s["bleu"] > 99.0, s["bleu"]
    assert s["chrf"] > 99.0, s["chrf"]


def test_training_schedule_keeps_the_effective_batch_identical():
    """Whatever the per-model batch, both decoders must train under one recipe.

    If the effective batch differed between Llama and Gemma, the comparison
    would be measuring the schedule as well as the checkpoint.
    """
    from src import config as C

    effective = C.P.decoder_batch_size * C.P.decoder_grad_accum
    for params in (1.2e9, 2.6e9, 1e6, 7e9):
        batch, accum, _ = DO.training_schedule(params)
        assert batch * accum == effective, (params, batch, accum, effective)
        assert batch >= 1 and accum >= 1


def test_training_schedule_only_checkpoints_the_large_model():
    _, _, small = DO.training_schedule(1.24e9)   # Llama-3.2-1B
    _, _, large = DO.training_schedule(2.61e9)   # Gemma-2-2B
    assert small is False
    assert large is True


def test_training_schedule_gives_the_small_model_the_bigger_batch():
    small_batch, _, _ = DO.training_schedule(1.24e9)
    large_batch, _, _ = DO.training_schedule(2.61e9)
    assert small_batch >= large_batch


def test_torchao_guard_neutralises_a_raising_availability_check():
    """peft raises instead of returning False when torchao is too old.

    That kills every LoRA build on images like Kaggle's, after the checkpoint
    has already downloaded. The guard must turn it into a plain False.
    """
    from peft import import_utils
    from peft.tuners.lora import torchao as lora_torchao

    original = (import_utils.is_torchao_available,
                lora_torchao.is_torchao_available)

    def raises_like_kaggle():
        raise ImportError("Found an incompatible version of torchao. Found "
                          "version 0.10.0, but only versions above 0.16.0 are "
                          "supported")

    try:
        import_utils.is_torchao_available = raises_like_kaggle
        lora_torchao.is_torchao_available = raises_like_kaggle

        DO.skip_torchao_dispatch()

        # The binding the dispatcher actually calls must now be harmless.
        assert lora_torchao.is_torchao_available() is False
        assert import_utils.is_torchao_available() is False
    finally:
        import_utils.is_torchao_available = original[0]
        lora_torchao.is_torchao_available = original[1]


def test_torchao_guard_leaves_a_healthy_check_alone():
    from peft import import_utils
    from peft.tuners.lora import torchao as lora_torchao

    original = (import_utils.is_torchao_available,
                lora_torchao.is_torchao_available)
    sentinel = lambda: True  # noqa: E731
    try:
        import_utils.is_torchao_available = sentinel
        lora_torchao.is_torchao_available = sentinel
        DO.skip_torchao_dispatch()
        assert lora_torchao.is_torchao_available is sentinel
    finally:
        import_utils.is_torchao_available = original[0]
        lora_torchao.is_torchao_available = original[1]


def test_slug_is_filesystem_safe():
    assert E._slug("Llama-3.2-1B (3-shot prompt)") == "llama_3_2_1b_3_shot_prompt"


def _main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:
            failed += 1
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
