import argparse
import math
import os
import time
import urllib.request
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from nua import Nua
from muon import Muon


def get_shakespeare_data() -> str:
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, "tinyshakespeare.txt")

    if not os.path.exists(file_path):
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        try:
            urllib.request.urlretrieve(url, file_path)
        except Exception:
            sample = """First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you know Caius Marcius is chief enemy to the people.

All:
We know't, we know't.

First Citizen:
Let us kill him, and we'll have corn at our own price.
Is't a verdict?

All:
No more talking on't; let it be done: away, away!

Second Citizen:
One word, good citizens.
""" * 500
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(sample)

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text


class CharDataset:
    def __init__(self, text: str, block_size: int = 128, val_ratio: float = 0.1):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        self.block_size = block_size

        data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        n = int((1 - val_ratio) * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]

    def get_batch(self, split: str, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.train_data if split == "train" else self.val_data
        ix = torch.randint(len(data) - self.block_size, (batch_size,))
        x = torch.stack([data[i : i + self.block_size] for i in ix]).to(device)
        y = torch.stack([data[i + 1 : i + 1 + self.block_size] for i in ix]).to(device)
        return x, y


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, max_seq_len: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, max_seq_len: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, max_seq_len)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim, bias=False),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class ShakespeareGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 192,
        num_layers: int = 6,
        num_heads: int = 6,
        max_seq_len: int = 128,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.tok_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, max_seq_len) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 0.8) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.max_seq_len else idx[:, -self.max_seq_len :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


@torch.no_grad()
def estimate_val_loss(model: nn.Module, dataset: CharDataset, batch_size: int, device: torch.device, eval_iters: int = 20) -> float:
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = dataset.get_batch("val", batch_size, device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def train_model(
    opt_name: str,
    dataset: CharDataset,
    device: torch.device,
    steps: int = 1000,
    lr: float = 2e-3,
    batch_size: int = 32,
    block_size: int = 128,
) -> Dict:
    print(f"\n{'='*70}\nTRAINING WITH OPTIMIZER: {opt_name}\n{'='*70}")
    torch.manual_seed(1337)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()

    model = ShakespeareGPT(
        vocab_size=dataset.vocab_size,
        embed_dim=192,
        num_layers=6,
        num_heads=6,
        max_seq_len=block_size,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    if opt_name.upper() == "ADAMW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    elif opt_name.upper() == "MUON":
        optimizer = Muon(model.parameters(), lr=lr, momentum=0.95, weight_decay=0.01)
    elif opt_name.upper() == "NUA":
        optimizer = Nua(model.parameters(), lr=lr, betas=(0.9, 0.999), alpha=0.5, weight_decay=0.01)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")

    train_losses = []
    val_losses = []
    step_times = []

    model.train()
    start_all = time.time()

    for step in range(1, steps + 1):
        t0 = time.time()
        x, y = dataset.get_batch("train", batch_size, device)

        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        dt = time.time() - t0
        step_times.append(dt)
        train_losses.append(loss.item())

        if step % 100 == 0 or step == steps:
            val_loss = estimate_val_loss(model, dataset, batch_size, device)
            val_losses.append((step, val_loss))
            vram_mb = (
                torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                if device.type == "cuda"
                else 0.0
            )
            print(
                f"[{opt_name:5s}] Step {step:4d}/{steps:4d} | "
                f"Train Loss: {loss.item():.4f} | "
                f"Val Loss: {val_loss:.4f} (PPL: {math.exp(min(val_loss, 20)):.2f}) | "
                f"Speed: {dt*1000:5.1f} ms/step | "
                f"Peak VRAM: {vram_mb:5.1f} MB"
            )

    total_time = time.time() - start_all
    avg_step_ms = (sum(step_times) / len(step_times)) * 1000
    final_val = estimate_val_loss(model, dataset, batch_size, device)
    peak_vram = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else 0.0
    )

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=150)[0].tolist()
    sample_text = "".join([dataset.itos[i] for i in generated])

    print(f"\n--- Generated Text Sample ({opt_name}) ---\n{sample_text}\n" + "-" * 40)

    return {
        "name": opt_name,
        "final_train_loss": train_losses[-1],
        "final_val_loss": final_val,
        "final_ppl": math.exp(min(final_val, 20)),
        "total_time": total_time,
        "avg_step_ms": avg_step_ms,
        "peak_vram_mb": peak_vram,
        "val_history": val_losses,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shakespeare NanoGPT Optimizer Pretraining")
    parser.add_argument("--optimizer", type=str, default="all", choices=["all", "adamw", "muon", "nua"])
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    try:
        device = torch.device(args.device)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                print("CUDA not available. Falling back to CPU.")
                device = torch.device("cpu")
            else:
                _ = torch.zeros(1, device=device)
    except Exception:
        device = torch.device("cpu")

    print(f"Running on: {device}")
    raw_text = get_shakespeare_data()
    dataset = CharDataset(raw_text, block_size=128)

    opts_to_run = ["AdamW", "Muon", "Nua"] if args.optimizer == "all" else [args.optimizer.upper()]

    all_results = []
    for opt in opts_to_run:
        res = train_model(
            opt_name=opt,
            dataset=dataset,
            device=device,
            steps=args.steps,
            lr=args.lr,
            batch_size=args.batch_size,
            block_size=128,
        )
        all_results.append(res)

    print("\n" + "=" * 80)
    print("FINAL SHAKESPEARE PRETRAINING SCORECARD")
    print("=" * 80)
    print(
        f"{'Optimizer':<10} | {'Train Loss':<12} | {'Val Loss':<10} | {'Val PPL':<10} | "
        f"{'Peak VRAM':<12} | {'Avg Step Time':<14}"
    )
    print("-" * 80)
    for r in all_results:
        print(
            f"{r['name']:<10} | {r['final_train_loss']:<12.4f} | {r['final_val_loss']:<10.4f} | "
            f"{r['final_ppl']:<10.2f} | {r['peak_vram_mb']:<10.1f} MB | {r['avg_step_ms']:<11.1f} ms"
        )
    print("=" * 80)
