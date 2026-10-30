"""Systems 3 and 4: Llama-3.2-1B-Instruct and Gemma-2-2B-IT (decoder-only).

Both models are adapted the same way so that the comparison between them, and
against the two encoder-decoder systems, is fair:

* identical instruction wording (``config.INSTRUCTION``),
* identical LoRA configuration, learning rate and number of epochs,
* loss computed on the Hindi continuation only (the prompt is masked out),
* identical greedy decoding and identical output post-processing.

A 3-shot prompted (no-training) variant of each model is also provided, because
the assignment allows either fine-tuning or prompting for the decoder-only
architectures and contrasting the two adaptation methods is informative.
"""

from __future__ import annotations

import os
import re
import time

import torch
from torch.utils.data import DataLoader, Dataset

from src import config as C

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]

# Prefixes a chatty instruct model likes to add in front of the translation.
_PREAMBLE = re.compile(
    r"^\s*(sure[,!.]?|here(\s+is|'s)?( the)?( hindi)?( translation)?\s*[:\-]?|"
    rf"{C.TGT_LANG_NAME}\s*[:\-]|translation\s*[:\-])\s*",
    re.IGNORECASE,
)


def hf_token() -> str | None:
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        if os.environ.get(key):
            return os.environ[key]
    return None


# --------------------------------------------------------------------------- #
# Prompt construction
# --------------------------------------------------------------------------- #

def _user_turn(src: str, shots: list[dict] | None = None) -> str:
    lines = [C.INSTRUCTION, ""]
    for shot in shots or []:
        lines += [f"{C.SRC_LANG_NAME}: {shot['src']}",
                  f"{C.TGT_LANG_NAME}: {shot['tgt']}", ""]
    lines += [f"{C.SRC_LANG_NAME}: {src}", f"{C.TGT_LANG_NAME}:"]
    return "\n".join(lines)


def build_prompt(tok, src: str, shots: list[dict] | None = None) -> str:
    """Render the instruction with the model's own chat template when it has one."""
    user = _user_turn(src, shots)
    template = getattr(tok, "chat_template", None)
    if template:
        return tok.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True,
        )
    return user + " "


def clean_output(text: str) -> str:
    """Reduce a free-form generation to a single translation line.

    Order matters here.  The preamble has to go *before* a line is selected,
    because an instruct model often puts its preamble on its own line and the
    translation on the next one - taking the first line first would throw the
    translation away and keep "Here is the translation:".  And the preamble is
    stripped repeatedly, because models stack them ("Sure! Here is the
    translation:") and an anchored pattern only removes the leading one.
    """
    text = text.strip().strip("`").strip()

    # A model that keeps going starts a new example; everything from there is
    # not part of this translation.
    for stop in (f"\n{C.SRC_LANG_NAME}:", f"\n{C.TGT_LANG_NAME}:"):
        if stop in text:
            text = text.split(stop)[0]

    for _ in range(3):
        stripped = _PREAMBLE.sub("", text).strip()
        if stripped == text:
            break
        text = stripped

    # First line with actual content: a preamble may have left a blank one.
    text = next((line.strip() for line in text.split("\n") if line.strip()), "")

    if len(text) > 1 and text[0] == text[-1] and text[0] in "\"'“”":
        text = text[1:-1].strip()
    return text


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load_causal(model_id: str, for_training: bool):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    token = hf_token()
    tok = AutoTokenizer.from_pretrained(model_id, token=token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kwargs: dict = {"token": token, C.DTYPE_KEY: C.DTYPE}
    # Gemma-2 uses logit soft-capping, which the fused SDPA/flash kernels ignore.
    if "gemma-2" in model_id.lower():
        kwargs["attn_implementation"] = "eager"
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).to(C.DEVICE)
    model.config.use_cache = not for_training
    return model, tok


def resolve_model_id(which: str) -> tuple[str, bool]:
    """Return (model_id, is_fallback).

    The gated ``meta-llama``/``google`` checkpoints need an accepted licence and
    a token.  If neither is present we fall back to an open checkpoint of the
    same family so the notebook still runs, and the README records which was
    used.
    """
    from huggingface_hub import model_info

    primary = C.P.llama_id if which == "llama" else C.P.gemma_id
    fallback = C.P.fallback_llama_id if which == "llama" else C.P.fallback_gemma_id
    if primary == fallback:
        return primary, False
    try:
        model_info(primary, token=hf_token())
        return primary, False
    except Exception as exc:  # gated, missing token, or offline
        print(f"[{which}] cannot access {primary} ({type(exc).__name__}); "
              f"falling back to {fallback}")
        return fallback, True


# --------------------------------------------------------------------------- #
# Supervised fine-tuning data
# --------------------------------------------------------------------------- #

class SFTDataset(Dataset):
    """Tokenised prompt+answer pairs with the prompt masked out of the loss."""

    def __init__(self, rows: list[dict], tok):
        self.rows = rows
        self.tok = tok

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        r = self.rows[i]
        prompt = build_prompt(self.tok, r["src"])
        prompt_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        answer_ids = self.tok(r["tgt"], add_special_tokens=False)["input_ids"]
        answer_ids = answer_ids[: C.P.max_target_len] + [self.tok.eos_token_id]
        budget = C.P.max_source_len * 3 + 48
        prompt_ids = prompt_ids[-budget:]
        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids
        return {"input_ids": input_ids, "labels": labels}


def make_sft_collate(tok):
    def collate(batch: list[dict]) -> dict:
        width = max(len(b["input_ids"]) for b in batch)
        pad_id = tok.pad_token_id
        input_ids, labels, attn = [], [], []
        for b in batch:
            gap = width - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [pad_id] * gap)
            labels.append(b["labels"] + [-100] * gap)
            attn.append([1] * len(b["input_ids"]) + [0] * gap)
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attn),
        }

    return collate


# --------------------------------------------------------------------------- #
# LoRA fine-tuning
# --------------------------------------------------------------------------- #

def _adapter_size(peft_model) -> int:
    """Adapter parameter count, for a reloaded adapter whose weights are frozen."""
    return sum(p.numel() for n, p in peft_model.named_parameters() if "lora_" in n)


def finetune_decoder(which: str, splits: dict[str, list[dict]],
                     force: bool = False) -> dict:
    """LoRA fine-tune Llama or Gemma for English->Hindi translation."""
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(C.SEED)
    C.ensure_dirs()
    model_id, is_fallback = resolve_model_id(which)
    out_dir = C.CHECKPOINT_DIR / f"{which}_lora"
    cached = out_dir.exists() and not force

    # The base is loaded exactly once. Attaching a cached adapter to a second
    # copy would put two 5 GB models on the GPU at the same time.
    model, tok = load_causal(model_id, for_training=not cached)
    base_params = sum(p.numel() for p in model.parameters())

    if cached:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, out_dir)
        trainable = sum(p.numel() for p in model.parameters()
                        if p.requires_grad) or _adapter_size(model)
        tpath = out_dir / "train_time.txt"
        train_time = float(tpath.read_text()) if tpath.exists() else float("nan")
        print(f"[{which}] loaded LoRA adapter ({train_time:.0f}s of training)")
        return {"model": model, "tokenizer": tok, "history": [],
                "model_id": model_id, "is_fallback": is_fallback,
                "total_params": base_params, "trainable_params": trainable,
                "train_time_s": train_time}

    present = {n.split(".")[-1] for n, _ in model.named_modules()}
    targets = [t for t in LORA_TARGETS if t in present] or ["q_proj", "v_proj"]
    lora_cfg = LoraConfig(
        r=C.P.lora_r, lora_alpha=C.P.lora_alpha, lora_dropout=C.P.lora_dropout,
        bias="none", task_type="CAUSAL_LM", target_modules=targets,
    )
    model = get_peft_model(model, lora_cfg)
    # LoRA weights are kept in fp32 even when the frozen base is fp16/bf16.
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()

    # Gradient checkpointing costs roughly a third of the step time and buys
    # several GB of activation memory.  On a 16 GB T4 that is the difference
    # between Gemma-2-2B fitting and a three-hour run dying with an OOM, so it
    # is on by default whenever a GPU is in use.  ``enable_input_require_grads``
    # is required for it to work with a frozen base and LoRA adapters.
    if C.DEVICE == "cuda":
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{which}] {model_id}: base={base_params / 1e6:.0f}M, "
          f"LoRA trainable={trainable / 1e6:.2f}M "
          f"({100 * trainable / base_params:.2f}% of base)")

    meta = {"model_id": model_id, "is_fallback": is_fallback,
            "total_params": base_params, "trainable_params": trainable}

    loader = DataLoader(SFTDataset(splits["train"], tok),
                        batch_size=C.P.decoder_batch_size, shuffle=True,
                        collate_fn=make_sft_collate(tok))
    val_loader = DataLoader(SFTDataset(splits["val"], tok),
                            batch_size=C.P.decoder_batch_size, shuffle=False,
                            collate_fn=make_sft_collate(tok))

    optimiser = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=C.P.decoder_lr)
    accum = C.P.decoder_grad_accum
    total_steps = max(1, (len(loader) // accum) * C.P.decoder_epochs)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=C.P.decoder_lr, total_steps=total_steps, pct_start=0.05)
    use_fp16 = C.DEVICE == "cuda" and C.DTYPE == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    autocast = (torch.autocast("cuda", dtype=C.DTYPE) if C.DEVICE == "cuda"
                else torch.autocast("cpu", enabled=False))

    history = []
    t0 = time.perf_counter()
    for epoch in range(1, C.P.decoder_epochs + 1):
        model.train()
        running, seen, opt_steps = 0.0, 0, 0
        optimiser.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader, 1):
            batch = {k: v.to(C.DEVICE) for k, v in batch.items()}
            with autocast:
                loss = model(**batch).loss
            scaler.scale(loss / accum).backward()
            running += loss.item()
            seen += 1
            if step % accum == 0 or step == len(loader):
                scaler.unscale_(optimiser)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                scaler.step(optimiser)
                scaler.update()
                optimiser.zero_grad(set_to_none=True)
                if opt_steps < total_steps - 1:
                    scheduler.step()
                opt_steps += 1
            if step % (50 * accum) == 0:
                print(f"[{which}] epoch {epoch} step {step}/{len(loader)} "
                      f"loss={running / seen:.4f}")
        val = _val_loss(model, val_loader, autocast)
        history.append({"epoch": epoch, "train_loss": running / max(seen, 1),
                        "val_loss": val})
        print(f"[{which}] epoch {epoch}/{C.P.decoder_epochs} "
              f"train_loss={running / max(seen, 1):.4f} val_loss={val:.4f}")
    train_time = time.perf_counter() - t0

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    (out_dir / "train_time.txt").write_text(f"{train_time:.3f}")
    print(f"[{which}] LoRA fine-tuned in {train_time:.0f}s")

    # Back to inference settings: the KV cache is what makes generation fast,
    # and checkpointing is pointless once there are no activations to keep.
    if C.DEVICE == "cuda":
        model.gradient_checkpointing_disable()
    model.config.use_cache = True
    meta["train_time_s"] = train_time
    return {"model": model, "tokenizer": tok, "history": history, **meta}


@torch.no_grad()
def _val_loss(model, loader, autocast) -> float:
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        batch = {k: v.to(C.DEVICE) for k, v in batch.items()}
        with autocast:
            loss = model(**batch).loss
        total += loss.item()
        n += 1
    model.train()
    return total / max(n, 1)


def load_prompted(which: str) -> dict:
    """Load the un-adapted instruct model for the 3-shot prompting baseline."""
    model_id, is_fallback = resolve_model_id(which)
    model, tok = load_causal(model_id, for_training=False)
    total = sum(p.numel() for p in model.parameters())
    print(f"[{which}-prompt] {model_id}: {total / 1e6:.0f}M parameters, no training")
    return {"model": model, "tokenizer": tok, "model_id": model_id,
            "is_fallback": is_fallback, "total_params": total,
            "trainable_params": 0, "train_time_s": 0.0, "history": []}


def pick_shots(splits: dict[str, list[dict]], k: int = 3) -> list[dict]:
    """Deterministically pick short, clean few-shot examples from the training set."""
    candidates = [r for r in splits["train"]
                  if 4 <= len(r["src"].split()) <= 12][:500]
    candidates = candidates or splits["train"][:k]
    step = max(1, len(candidates) // max(k, 1))
    return [candidates[i * step] for i in range(min(k, len(candidates)))]


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #

@torch.no_grad()
def translate(model, tok, sentences: list[str], shots: list[dict] | None = None,
              batch_size: int | None = None, temperature: float = 0.0) -> list[str]:
    """Batched left-padded generation; temperature == 0 means greedy."""
    model.eval()
    batch_size = batch_size or C.P.gen_batch_size
    original_side = tok.padding_side
    tok.padding_side = "left"
    out: list[str] = []
    try:
        for start in range(0, len(sentences), batch_size):
            chunk = sentences[start: start + batch_size]
            prompts = [build_prompt(tok, s, shots) for s in chunk]
            enc = tok(prompts, return_tensors="pt", padding=True,
                      add_special_tokens=False).to(C.DEVICE)
            gen_kwargs = dict(max_new_tokens=C.P.max_target_len + 16,
                              pad_token_id=tok.pad_token_id)
            if temperature and temperature > 0:
                gen_kwargs.update(do_sample=True, temperature=float(temperature),
                                  top_p=1.0, top_k=0)
            else:
                gen_kwargs.update(do_sample=False)
            ids = model.generate(**enc, **gen_kwargs)
            new_tokens = ids[:, enc["input_ids"].shape[1]:]
            for text in tok.batch_decode(new_tokens, skip_special_tokens=True):
                out.append(clean_output(text))
    finally:
        tok.padding_side = original_side
    return out
