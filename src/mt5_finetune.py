"""System 2: mT5-small (pretrained encoder-decoder) fine-tuned on the subset.

mT5 is trained with a plain seq2seq objective and a task prefix, which is how
the original T5/mT5 papers frame translation.  The training loop is written by
hand rather than using ``Seq2SeqTrainer`` so that the measured wall-clock time
covers exactly the optimisation work and nothing else, and so the notebook does
not break when the Trainer API changes.

Note on precision: mT5 is known to overflow in fp16, so it is always trained
and decoded in fp32 even when a GPU is available.
"""

from __future__ import annotations

import time

import torch
from torch.utils.data import DataLoader, Dataset

from src import config as C

PREFIX = f"translate {C.SRC_LANG_NAME} to {C.TGT_LANG_NAME}: "
ADAPTER_DIR = C.CHECKPOINT_DIR / "mt5_finetuned"


class Seq2SeqDataset(Dataset):
    def __init__(self, rows: list[dict], tok):
        self.rows = rows
        self.tok = tok

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        return self.rows[i]


def make_collate(tok):
    def collate(batch: list[dict]) -> dict:
        model_inputs = tok(
            [PREFIX + r["src"] for r in batch],
            max_length=C.P.max_source_len, truncation=True,
            padding=True, return_tensors="pt",
        )
        labels = tok(
            text_target=[r["tgt"] for r in batch],
            max_length=C.P.max_target_len, truncation=True,
            padding=True, return_tensors="pt",
        )["input_ids"]
        # -100 is ignored by the cross-entropy loss inside the model.
        labels[labels == tok.pad_token_id] = -100
        model_inputs["labels"] = labels
        return model_inputs

    return collate


def load_model():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(C.P.mt5_id, legacy=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        C.P.mt5_id, **{C.DTYPE_KEY: torch.float32})
    return model.to(C.DEVICE), tok


@torch.no_grad()
def _validate(model, loader) -> float:
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        batch = {k: v.to(C.DEVICE) for k, v in batch.items()}
        loss = model(**batch).loss
        total += loss.item() * batch["labels"].size(0)
        n += batch["labels"].size(0)
    return total / max(n, 1)


def finetune_mt5(splits: dict[str, list[dict]], force: bool = False) -> dict:
    """Fine-tune mT5-small and return the model, tokenizer and timings."""
    torch.manual_seed(C.SEED)
    C.ensure_dirs()

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    if ADAPTER_DIR.exists() and not force:
        tok = AutoTokenizer.from_pretrained(ADAPTER_DIR, legacy=False)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            ADAPTER_DIR, **{C.DTYPE_KEY: torch.float32}).to(C.DEVICE)
        meta_path = ADAPTER_DIR / "train_time.txt"
        train_time = float(meta_path.read_text()) if meta_path.exists() else float("nan")
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[mt5] loaded fine-tuned checkpoint ({train_time:.0f}s of training)")
        return {"model": model, "tokenizer": tok, "train_time_s": train_time,
                "total_params": n_params, "trainable_params": n_params, "history": []}

    model, tok = load_model()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[mt5] {C.P.mt5_id}: {n_params / 1e6:.1f}M parameters (all trainable)")

    collate = make_collate(tok)
    train_loader = DataLoader(Seq2SeqDataset(splits["train"], tok),
                              batch_size=C.P.mt5_batch_size, shuffle=True,
                              collate_fn=collate)
    val_loader = DataLoader(Seq2SeqDataset(splits["val"], tok),
                            batch_size=C.P.mt5_batch_size, shuffle=False,
                            collate_fn=collate)

    optimiser = torch.optim.AdamW(model.parameters(), lr=C.P.mt5_lr, weight_decay=0.01)
    total_steps = max(1, len(train_loader) * C.P.mt5_epochs)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=C.P.mt5_lr, total_steps=total_steps, pct_start=0.1)

    history = []
    t0 = time.perf_counter()
    for epoch in range(1, C.P.mt5_epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(C.DEVICE) for k, v in batch.items()}
            loss = model(**batch).loss
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            scheduler.step()
            running += loss.item()
            seen += 1
            if step % 100 == 0 or step == len(train_loader):
                print(f"[mt5] epoch {epoch} step {step}/{len(train_loader)} "
                      f"loss={running / seen:.4f}")
        val = _validate(model, val_loader)
        history.append({"epoch": epoch, "train_loss": running / max(seen, 1),
                        "val_loss": val})
        print(f"[mt5] epoch {epoch}/{C.P.mt5_epochs} "
              f"train_loss={running / max(seen, 1):.4f} val_loss={val:.4f}")
    train_time = time.perf_counter() - t0

    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ADAPTER_DIR)
    tok.save_pretrained(ADAPTER_DIR)
    (ADAPTER_DIR / "train_time.txt").write_text(f"{train_time:.3f}")
    print(f"[mt5] fine-tuned in {train_time:.0f}s")

    return {"model": model, "tokenizer": tok, "train_time_s": train_time,
            "total_params": n_params, "trainable_params": n_params, "history": history}


@torch.no_grad()
def translate(model, tok, sentences: list[str], batch_size: int | None = None,
              temperature: float = 0.0) -> list[str]:
    """Greedy by default; temperature > 0 switches to sampling."""
    model.eval()
    batch_size = batch_size or C.P.gen_batch_size
    out: list[str] = []
    for start in range(0, len(sentences), batch_size):
        chunk = sentences[start: start + batch_size]
        enc = tok([PREFIX + s for s in chunk], return_tensors="pt", padding=True,
                  truncation=True, max_length=C.P.max_source_len).to(C.DEVICE)
        gen_kwargs = dict(max_new_tokens=C.P.max_target_len, num_beams=1)
        if temperature and temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=float(temperature), top_p=1.0, top_k=0)
        else:
            gen_kwargs.update(do_sample=False)
        ids = model.generate(**enc, **gen_kwargs)
        out.extend(t.strip() for t in tok.batch_decode(ids, skip_special_tokens=True))
    return out
