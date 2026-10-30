"""System 1: a basic Transformer encoder-decoder trained from scratch.

Nothing here is pretrained. A joint byte-level BPE vocabulary is learned on the
training subset only, and the model is the textbook post-norm Transformer of
Vaswani et al. (2017) built on torch.nn.Transformer.
"""

from __future__ import annotations

import math
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src import config as C

PAD, BOS, EOS, UNK = 0, 1, 2, 3
SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]
TOKENIZER_PATH = C.CHECKPOINT_DIR / "bpe_tokenizer.json"
MODEL_PATH = C.CHECKPOINT_DIR / "scratch_transformer.pt"


# --------------------------------------------------------------------------- #
# Tokenizer
# --------------------------------------------------------------------------- #

def train_tokenizer(rows: list[dict], force: bool = False):
    """Train (or load) a joint English+Hindi byte-level BPE tokenizer."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    C.ensure_dirs()
    if TOKENIZER_PATH.exists() and not force:
        return Tokenizer.from_file(str(TOKENIZER_PATH))

    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=C.P.vocab_size,
        special_tokens=SPECIALS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )

    def corpus():
        for r in rows:
            yield r["src"]
            yield r["tgt"]

    tok.train_from_iterator(corpus(), trainer=trainer, length=2 * len(rows))
    tok.save(str(TOKENIZER_PATH))
    print(f"[scratch] trained joint BPE tokenizer, vocab={tok.get_vocab_size()}")
    return tok


def encode(tok, text: str, max_len: int, add_bos_eos: bool) -> list[int]:
    ids = tok.encode(text).ids
    budget = max_len - (2 if add_bos_eos else 0)
    ids = ids[:budget]
    return [BOS] + ids + [EOS] if add_bos_eos else ids


# --------------------------------------------------------------------------- #
# Data plumbing
# --------------------------------------------------------------------------- #

class PairDataset(Dataset):
    def __init__(self, rows: list[dict], tok):
        self.rows = rows
        self.tok = tok

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        r = self.rows[i]
        src = encode(self.tok, r["src"], C.P.max_source_len, add_bos_eos=True)
        tgt = encode(self.tok, r["tgt"], C.P.max_target_len, add_bos_eos=True)
        return torch.tensor(src), torch.tensor(tgt)


def collate(batch):
    srcs, tgts = zip(*batch)
    return _pad(srcs), _pad(tgts)


def _pad(seqs) -> torch.Tensor:
    width = max(len(s) for s in seqs)
    out = torch.full((len(seqs), width), PAD, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = s
    return out


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10_000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


class ScratchTransformer(nn.Module):
    """Standard encoder-decoder Transformer with a shared source/target vocabulary."""

    def __init__(self, vocab_size: int):
        super().__init__()
        p = C.P
        self.d_model = p.d_model
        self.embed = nn.Embedding(vocab_size, p.d_model, padding_idx=PAD)
        self.pos = PositionalEncoding(p.d_model, p.dropout)
        self.transformer = nn.Transformer(
            d_model=p.d_model,
            nhead=p.nhead,
            num_encoder_layers=p.num_layers,
            num_decoder_layers=p.num_layers,
            dim_feedforward=p.dim_feedforward,
            dropout=p.dropout,
            batch_first=True,
            norm_first=False,
        )
        self.generator = nn.Linear(p.d_model, vocab_size)
        # Weight tying: the output projection reuses the embedding matrix.
        self.generator.weight = self.embed.weight
        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _embed(self, ids: torch.Tensor) -> torch.Tensor:
        return self.pos(self.embed(ids) * math.sqrt(self.d_model))

    def encode(self, src: torch.Tensor):
        pad_mask = src == PAD
        memory = self.transformer.encoder(self._embed(src), src_key_padding_mask=pad_mask)
        return memory, pad_mask

    def decode(self, tgt_in: torch.Tensor, memory: torch.Tensor,
               memory_pad_mask: torch.Tensor) -> torch.Tensor:
        # Boolean rather than float, to match the boolean key-padding masks
        # (mixing mask dtypes is deprecated in torch >= 2.4).
        causal = torch.triu(
            torch.ones(tgt_in.size(1), tgt_in.size(1), dtype=torch.bool,
                       device=tgt_in.device),
            diagonal=1,
        )
        out = self.transformer.decoder(
            self._embed(tgt_in),
            memory,
            tgt_mask=causal,
            tgt_key_padding_mask=tgt_in == PAD,
            memory_key_padding_mask=memory_pad_mask,
        )
        return self.generator(out)

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        memory, pad_mask = self.encode(src)
        return self.decode(tgt_in, memory, pad_mask)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def _run_epoch(model, loader, criterion, optimiser=None, scheduler=None) -> float:
    train = optimiser is not None
    model.train(train)
    total, n_tokens = 0.0, 0
    for src, tgt in loader:
        src, tgt = src.to(C.DEVICE), tgt.to(C.DEVICE)
        tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
        with torch.set_grad_enabled(train):
            logits = model(src, tgt_in)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        if train:
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            scheduler.step()
        ntok = int((tgt_out != PAD).sum())
        total += loss.item() * ntok
        n_tokens += ntok
    return total / max(n_tokens, 1)


def train_scratch(splits: dict[str, list[dict]], force: bool = False) -> dict:
    """Train the from-scratch Transformer and return model, tokenizer and timings."""
    torch.manual_seed(C.SEED)
    tok = train_tokenizer(splits["train"], force=force)
    vocab = tok.get_vocab_size()

    model = ScratchTransformer(vocab).to(C.DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[scratch] parameters: {n_params / 1e6:.2f}M (vocab {vocab})")

    if MODEL_PATH.exists() and not force:
        state = torch.load(MODEL_PATH, map_location=C.DEVICE)
        if state.get("vocab") == vocab:
            model.load_state_dict(state["model"])
            print(f"[scratch] loaded checkpoint ({state['train_time_s']:.0f}s of training)")
            return {
                "model": model, "tokenizer": tok,
                "train_time_s": state["train_time_s"],
                "total_params": n_params, "trainable_params": n_params,
                "history": state.get("history", []),
            }

    train_loader = DataLoader(
        PairDataset(splits["train"], tok), batch_size=C.P.scratch_batch_size,
        shuffle=True, collate_fn=collate, drop_last=False,
    )
    val_loader = DataLoader(
        PairDataset(splits["val"], tok), batch_size=C.P.scratch_batch_size,
        shuffle=False, collate_fn=collate,
    )

    criterion = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.1)
    optimiser = torch.optim.AdamW(model.parameters(), lr=C.P.scratch_lr,
                                  betas=(0.9, 0.98), eps=1e-9, weight_decay=0.01)
    total_steps = max(1, len(train_loader) * C.P.scratch_epochs)
    warmup = max(1, min(2_000, total_steps // 10))

    def lr_lambda(step: int) -> float:
        step = max(step, 1)
        if step < warmup:
            return step / warmup
        return (warmup / step) ** 0.5

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda)

    best_val, best_state, history = float("inf"), None, []
    t0 = time.perf_counter()
    for epoch in range(1, C.P.scratch_epochs + 1):
        tr = _run_epoch(model, train_loader, criterion, optimiser, scheduler)
        va = _run_epoch(model, val_loader, criterion)
        history.append({"epoch": epoch, "train_loss": tr, "val_loss": va,
                        "val_ppl": math.exp(min(va, 20))})
        print(f"[scratch] epoch {epoch:>2}/{C.P.scratch_epochs} "
              f"train_loss={tr:.4f} val_loss={va:.4f} val_ppl={math.exp(min(va, 20)):.1f}")
        if va < best_val:
            best_val = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    train_time = time.perf_counter() - t0

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save({"model": model.state_dict(), "vocab": vocab,
                "train_time_s": train_time, "history": history}, MODEL_PATH)
    print(f"[scratch] trained in {train_time:.0f}s, best val_loss={best_val:.4f}")

    return {"model": model, "tokenizer": tok, "train_time_s": train_time,
            "total_params": n_params, "trainable_params": n_params, "history": history}


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #

@torch.no_grad()
def translate(model, tok, sentences: list[str], batch_size: int | None = None,
              temperature: float = 0.0) -> list[str]:
    """Batched decoding. temperature == 0 means greedy (the default)."""
    model.eval()
    batch_size = batch_size or C.P.gen_batch_size
    out: list[str] = []
    for start in range(0, len(sentences), batch_size):
        chunk = sentences[start: start + batch_size]
        src = _pad([torch.tensor(encode(tok, s, C.P.max_source_len, True)) for s in chunk])
        src = src.to(C.DEVICE)
        memory, pad_mask = model.encode(src)

        ys = torch.full((src.size(0), 1), BOS, dtype=torch.long, device=C.DEVICE)
        done = torch.zeros(src.size(0), dtype=torch.bool, device=C.DEVICE)
        for _ in range(C.P.max_target_len - 1):
            logits = model.decode(ys, memory, pad_mask)[:, -1, :]
            if temperature and temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                nxt = torch.multinomial(probs, 1).squeeze(-1)
            else:
                nxt = logits.argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, PAD), nxt)
            ys = torch.cat([ys, nxt.unsqueeze(1)], dim=1)
            done |= nxt == EOS
            if bool(done.all()):
                break

        for row in ys.tolist():
            ids = [i for i in row[1:] if i not in (PAD, BOS)]
            if EOS in ids:
                ids = ids[: ids.index(EOS)]
            out.append(tok.decode(ids).strip())
    return out
