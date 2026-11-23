import argparse
import math
import os
import time
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from nua import Nua
from muon import Muon


def run_experiment_synthetic(
    device: torch.device,
    dim: int = 128,
    num_outliers: int = 4,
    noise_std: float = 0.5,
    steps: int = 300,
    lr: float = 1e-2,
) -> Dict[str, List[float]]:
    print("\n" + "=" * 70)
    print(f"EXPERIMENT 1: Ill-Conditioned Landscape (dim={dim}, noise={noise_std})")
    print("=" * 70)

    torch.manual_seed(42)
    evals = torch.zeros(dim, device=device)
    evals[:num_outliers] = torch.linspace(50.0, 10.0, num_outliers, device=device)
    evals[num_outliers:] = torch.linspace(0.1, 0.01, dim - num_outliers, device=device)

    Q, _ = torch.linalg.qr(torch.randn(dim, dim, device=device))
    H = Q @ torch.diag(evals) @ Q.t()
    H_sqrt = Q @ torch.diag(torch.sqrt(evals)) @ Q.t()

    W_star = torch.randn(dim, dim, device=device)

    results = {}

    optimizers_to_test = [
        ("AdamW", lambda p: torch.optim.AdamW([p], lr=lr, weight_decay=0.0)),
        ("Muon",  lambda p: Muon([p], lr=lr, momentum=0.9, weight_decay=0.0)),
        ("Nua",   lambda p: Nua([p], lr=lr, betas=(0.9, 0.99), alpha=0.5, weight_decay=0.0)),
    ]

    for opt_name, opt_fn in optimizers_to_test:
        torch.manual_seed(1234)
        W = nn.Parameter(torch.randn(dim, dim, device=device))
        optimizer = opt_fn(W)

        loss_history = []
        dist_history = []

        start_time = time.time()
        for step in range(steps):
            optimizer.zero_grad()

            diff = W - W_star
            grad_exact = H @ diff
            noise = torch.randn_like(grad_exact) * noise_std
            W.grad = grad_exact + noise

            loss_true = 0.5 * torch.sum((H_sqrt @ diff) ** 2).item()
            dist_to_star = torch.norm(diff).item()

            loss_history.append(loss_true)
            dist_history.append(dist_to_star)

            optimizer.step()

        elapsed = time.time() - start_time
        final_loss = loss_history[-1]
        final_dist = dist_history[-1]

        results[opt_name] = loss_history
        print(f"[{opt_name:7s}] Final Loss: {final_loss:10.4f} | Distance to W*: {final_dist:8.4f} | Time: {elapsed:.3f}s")

    return results


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, max_seq_len: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len)
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


class MiniNanoGPT(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128, num_layers: int = 4, num_heads: int = 4, max_seq_len: int = 64):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, max_seq_len) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


def get_synthetic_text_batches(vocab_size: int = 65, batch_size: int = 16, seq_len: int = 64, num_batches: int = 200, device: torch.device = "cpu"):
    torch.manual_seed(42)
    batches = []
    for _ in range(num_batches):
        base_pattern = torch.randint(0, vocab_size, (batch_size, 8), device=device)
        repeated = base_pattern.repeat(1, (seq_len + 1) // 8 + 1)[:, : seq_len + 1]
        x = repeated[:, :-1].contiguous()
        y = repeated[:, 1:].contiguous()
        batches.append((x, y))
    return batches


def run_experiment_transformer(
    device: torch.device,
    steps: int = 300,
    lr: float = 2e-3,
) -> Dict[str, List[float]]:
    print("\n" + "=" * 70)
    print(f"EXPERIMENT 2: Mini-Transformer Language Model Pretraining ({steps} steps)")
    print("=" * 70)

    vocab_size = 65
    embed_dim = 128
    num_layers = 4
    num_heads = 4
    seq_len = 64
    batch_size = 16

    batches = get_synthetic_text_batches(vocab_size, batch_size, seq_len, num_batches=steps, device=device)

    optimizers = [
        ("AdamW", lambda m: torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)),
        ("Muon",  lambda m: Muon(m.parameters(), lr=lr, momentum=0.95, weight_decay=0.01)),
        ("Nua",   lambda m: Nua(m.parameters(), lr=lr, betas=(0.9, 0.999), alpha=0.5, weight_decay=0.01)),
    ]

    results = {}

    for opt_name, opt_builder in optimizers:
        torch.manual_seed(42)
        model = MiniNanoGPT(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            max_seq_len=seq_len,
        ).to(device)

        opt = opt_builder(model)

        loss_curve = []
        grad_norms = []

        start_time = time.time()
        for step, (x, y) in enumerate(batches):
            opt.zero_grad()
            _, loss = model(x, y)
            loss.backward()

            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            grad_norms.append(total_norm)

            opt.step()
            loss_curve.append(loss.item())

            if (step + 1) % 50 == 0 or step == steps - 1:
                print(f"[{opt_name:7s}] Step {step+1:3d}/{steps:3d} | Loss: {loss.item():.4f} | Grad Norm: {total_norm:6.2f}")

        elapsed = time.time() - start_time
        print(f"[{opt_name:7s}] Complete! Final Loss: {loss_curve[-1]:.4f} | Total Time: {elapsed:.2f}s ({elapsed/steps*1000:.1f} ms/step)")
        results[opt_name] = loss_curve

    return results


def print_ascii_summary(title: str, results: Dict[str, List[float]]):
    print("\n" + "=" * 70)
    print(f"SUMMARY: {title}")
    print("=" * 70)
    print(f"{'Optimizer':<10} | {'Initial Loss':<14} | {'Step 50':<12} | {'Step 100':<12} | {'Final Loss':<12}")
    print("-" * 70)
    for name, loss in results.items():
        init_l = f"{loss[0]:.4f}"
        s50_l = f"{loss[49]:.4f}" if len(loss) > 49 else "N/A"
        s100_l = f"{loss[99]:.4f}" if len(loss) > 99 else "N/A"
        final_l = f"{loss[-1]:.4f}"
        print(f"{name:<10} | {init_l:<14} | {s50_l:<12} | {s100_l:<12} | {final_l:<12}")
    print("-" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nua Benchmark Suite")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()

    try:
        device = torch.device(args.device)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                print("CUDA not available. Falling back to CPU.")
                device = torch.device("cpu")
            else:
                _ = torch.zeros(1, device=device)
    except Exception as e:
        print(f"Could not initialize CUDA ({e}). Falling back to CPU.")
        device = torch.device("cpu")

    print(f"Running benchmarks on: {device}")

    res_synthetic = run_experiment_synthetic(device, dim=128, steps=args.steps)
    print_ascii_summary("Synthetic Ill-Conditioned Power-Law Landscape", res_synthetic)

    res_transformer = run_experiment_transformer(device, steps=args.steps)
    print_ascii_summary("Mini-Transformer Pretraining Loss", res_transformer)
