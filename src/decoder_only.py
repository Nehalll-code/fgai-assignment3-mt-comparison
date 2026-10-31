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


def can_access(repo_id: str, token: str | None = None) -> tuple[bool, str]:
    """Can this token actually *download* ``repo_id``? Returns (ok, reason).

    A gated repository is public metadata with private files, so ``model_info``
    succeeds on Llama and Gemma even with no token at all and is useless as an
    access check.  ``auth_check`` is the call that enforces the gate; on hubs
    too old to have it, fetching one small file does the same job.
    """
    token = token if token is not None else hf_token()
    try:
        from huggingface_hub import auth_check
    except ImportError:
        auth_check = None

    try:
        if auth_check is not None:
            auth_check(repo_id, token=token)
        else:
            from huggingface_hub import hf_hub_download
            hf_hub_download(repo_id, "config.json", token=token)
        return True, ""
    except Exception as exc:
        return False, type(exc).__name__


def resolve_model_id(which: str) -> tuple[str, bool]:
    """Return (model_id, is_fallback).

    The gated ``meta-llama``/``google`` checkpoints need an accepted licence and
    a token.  If either is missing we fall back to an open checkpoint of the
    same family so the run still finishes, and the README records which was
    used.
    """
    primary = C.P.llama_id if which == "llama" else C.P.gemma_id
    fallback = C.P.fallback_llama_id if which == "llama" else C.P.fallback_gemma_id
    if primary == fallback:
        return primary, False

    ok, reason = can_access(primary)
    if ok:
        return primary, False
    print(f"[{which}] cannot access {primary} ({reason}); "
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

BIG_MODEL_PARAMS = 1.5e9


def training_schedule(base_params: int) -> tuple[int, int, bool]:
    """Pick (batch_size, grad_accum, is_big) for a decoder of this size.

    Gradient checkpointing buys several GB of activation memory and costs
    roughly a third of the step time.  Gemma-2-2B on a 16 GB T4 needs that
    trade; a 1B model has memory to spare and would only pay the cost.  The
    same reasoning applies to the batch size - the smaller model takes twice
    the batch, which halves the number of forward/backward passes and keeps the
    GPU better fed.

    The accumulation factor moves with the batch size so the **effective**
    batch is unchanged.  That matters: it is what keeps the optimisation recipe
    identical between the two decoder-only systems, so the comparison still
    measures the checkpoint rather than the schedule.  The batch is only
    doubled when it still divides the effective batch exactly.
    """
    effective = C.P.decoder_batch_size * C.P.decoder_grad_accum
    big = base_params > BIG_MODEL_PARAMS
    batch_size = C.P.decoder_batch_size
    if not big and effective % (batch_size * 2) == 0:
        batch_size *= 2
    return batch_size, max(1, effective // batch_size), big


def skip_torchao_dispatch() -> None:
    """Stop peft's LoRA dispatcher from raising on an out-of-date torchao.

    ``peft.import_utils.is_torchao_available`` *raises* ImportError when torchao
    is installed but older than peft's minimum, rather than reporting False.
    Every LoRA module build goes through a dispatcher that calls it, so on an
    image with that combination - Kaggle ships torchao 0.10.0 against a peft
    wanting >= 0.16 - ``get_peft_model`` dies, and it dies *after* the 2.5 GB
    checkpoint has downloaded.

    Nothing in this project uses torchao-quantised layers, so the correct answer
    for this pipeline is "not available", which makes the dispatcher skip. The
    dispatch module binds the name directly (``from peft.import_utils import
    is_torchao_available``), so that binding is the one that has to be replaced.
    """
    try:
        from peft import import_utils
        from peft.tuners.lora import torchao as lora_torchao
    except Exception:
        return

    try:
        import_utils.is_torchao_available()
        return                      # healthy, or genuinely absent - leave it be
    except ImportError as exc:
        print(f"[peft] {exc}")
        print("[peft] skipping the torchao LoRA dispatcher; it is unused here")
    except Exception:
        return

    import_utils.is_torchao_available = lambda: False
    lora_torchao.is_torchao_available = lambda: False


def _adapter_size(peft_model) -> int:
    """Adapter parameter count, for a reloaded adapter whose weights are frozen."""
    return sum(p.numel() for n, p in peft_model.named_parameters() if "lora_" in n)


def finetune_decoder(which: str, splits: dict[str, list[dict]],
                     force: bool = False) -> dict:
    """LoRA fine-tune Llama or Gemma for English->Hindi translation."""
    from peft import LoraConfig, get_peft_model

    skip_torchao_dispatch()
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

    # Gradient checkpointing buys several GB of activation memory and costs
    # roughly a third of the step time.  Gemma-2-2B on a 16 GB T4 needs that
    # trade; a 1B model has memory to spare and only pays the cost, so the
    # decision follows the model's size rather than applying to everything.
    # ``enable_input_require_grads`` is required for checkpointing to work with
    # a frozen base and LoRA adapters.
    batch_size, accum, big = training_schedule(base_params)
    if C.DEVICE == "cuda" and big:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{which}] {model_id}: base={base_params / 1e6:.0f}M, "
          f"LoRA trainable={trainable / 1e6:.2f}M "
          f"({100 * trainable / base_params:.2f}% of base)")
    print(f"[{which}] batch={batch_size} x accum={accum} "
          f"(effective {batch_size * accum}), "
          f"gradient checkpointing {'on' if (C.DEVICE == 'cuda' and big) else 'off'}")

    meta = {"model_id": model_id, "is_fallback": is_fallback,
            "total_params": base_params, "trainable_params": trainable}

    loader = DataLoader(SFTDataset(splits["train"], tok),
                        batch_size=batch_size, shuffle=True,
                        collate_fn=make_sft_collate(tok))
    val_loader = DataLoader(SFTDataset(splits["val"], tok),
                            batch_size=batch_size, shuffle=False,
                            collate_fn=make_sft_collate(tok))

    optimiser = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=C.P.decoder_lr)
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
    if C.DEVICE == "cuda" and big:
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
