import math
from typing import Callable, Iterable, Optional, Tuple
import torch
from torch.optim.optimizer import Optimizer


def standard_newton_schulz_5(G: torch.Tensor, steps: int = 5, eps: float = 1e-12) -> torch.Tensor:
    m, n = G.shape
    device = G.device
    dtype = G.dtype
    comp_dtype = torch.float32 if dtype == torch.float16 else dtype

    frob = torch.norm(G.to(comp_dtype)) + eps
    X = G.to(comp_dtype) / frob

    c_a = 15.0 / 8.0
    c_b = -10.0 / 8.0
    c_c = 3.0 / 8.0

    for _ in range(steps):
        A = torch.mm(X.t(), X)
        Poly = torch.mm(A, A)
        Poly.mul_(c_c)
        Poly.add_(A, alpha=c_b)
        Poly.diagonal().add_(c_a)
        X = torch.mm(X, Poly)

    return X.to(dtype)


class Muon(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        momentum: float = 0.95,
        betas: Tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.01,
        ns_steps: int = 5,
        eps: float = 1e-8,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            betas=betas,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
            eps=eps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            mom = group["momentum"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            ns_steps = group["ns_steps"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    if p.ndim != 2:
                        state["exp_avg_sq"] = torch.zeros_like(p)

                state["step"] += 1
                step = state["step"]
                exp_avg = state["exp_avg"]

                if weight_decay != 0:
                    p.mul_(1.0 - lr * weight_decay)

                if p.ndim == 2:
                    exp_avg.mul_(mom).add_(grad, alpha=1.0 - mom)
                    
                    m, n = p.shape
                    M_matrix = exp_avg
                    transposed = False
                    if m < n:
                        M_matrix = M_matrix.t()
                        transposed = True

                    update = standard_newton_schulz_5(M_matrix, steps=ns_steps, eps=eps)
                    if transposed:
                        update = update.t()

                    min_dim = min(m, n)
                    max_dim = max(m, n)
                    scale = math.sqrt(min_dim) * max(1.0, math.sqrt(max_dim / min_dim))
                    p.add_(update, alpha=-lr * scale)
                else:
                    exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                    bias_correction1 = 1.0 - beta1 ** step
                    bias_correction2 = 1.0 - beta2 ** step

                    step_size = lr / bias_correction1
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                    p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
