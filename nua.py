import math
from typing import Callable, Iterable, Optional, Tuple, Union
import torch
from torch.optim.optimizer import Optimizer


def shifted_newton_schulz_5(
    G: torch.Tensor,
    tau: Union[float, torch.Tensor],
    steps: int = 5,
    eps: float = 1e-12,
) -> torch.Tensor:
    m, n = G.shape
    device = G.device
    dtype = G.dtype
    comp_dtype = torch.float32 if dtype == torch.float16 else dtype

    frob_sq = torch.sum(G.to(comp_dtype) ** 2)
    tau_t = tau if isinstance(tau, torch.Tensor) else torch.tensor(tau, device=device, dtype=comp_dtype)
    tau_sq = tau_t ** 2
    rho = torch.sqrt(frob_sq + n * tau_sq + eps)

    X = G.to(comp_dtype) / rho
    mu_sq = tau_sq / (rho ** 2)

    c_a = 15.0 / 8.0
    c_b = -10.0 / 8.0
    c_c = 3.0 / 8.0

    for _ in range(steps):
        A = torch.mm(X.t(), X)
        A.diagonal().add_(mu_sq)
        
        Poly = torch.mm(A, A)
        Poly.mul_(c_c)
        Poly.add_(A, alpha=c_b)
        Poly.diagonal().add_(c_a)

        X = torch.mm(X, Poly)
        mu_sq = mu_sq * 0.5

    return X.to(dtype)


class Nua(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.01,
        alpha: float = 0.5,
        ns_steps: int = 5,
        eps: float = 1e-8,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1 parameter: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2 parameter: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            alpha=alpha,
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
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            alpha = group["alpha"]
            ns_steps = group["ns_steps"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Nua does not support sparse gradients.")

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if p.ndim == 2:
                        state["var_scalar"] = torch.zeros(1, device=p.device, dtype=torch.float32)
                    else:
                        state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                state["step"] += 1
                step = state["step"]
                exp_avg = state["exp_avg"]

                if weight_decay != 0:
                    p.mul_(1.0 - lr * weight_decay)

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)

                if p.ndim == 2:
                    m, n = p.shape
                    
                    diff_norm_sq = torch.sum((grad - exp_avg) ** 2)
                    state["var_scalar"].mul_(beta2).add_(diff_norm_sq, alpha=1.0 - beta2)
                    
                    var_unbiased = state["var_scalar"] / (1.0 - beta2 ** step)
                    sigma_noise = torch.sqrt(torch.clamp(var_unbiased / (m * n), min=eps))
                    
                    raw_tau = alpha * sigma_noise * (math.sqrt(m) + math.sqrt(n))
                    
                    m_frob = torch.norm(exp_avg.to(torch.float32))
                    min_dim = min(m, n)
                    max_dim = max(m, n)
                    m_scale = m_frob / math.sqrt(min_dim)
                    tau = torch.minimum(raw_tau.squeeze(), 0.8 * m_scale)

                    M_matrix = exp_avg
                    transposed = False
                    if m < n:
                        M_matrix = M_matrix.t()
                        transposed = True

                    update = shifted_newton_schulz_5(
                        M_matrix,
                        tau=tau,
                        steps=ns_steps,
                        eps=eps,
                    )

                    if transposed:
                        update = update.t()

                    scale = math.sqrt(min_dim) * max(1.0, math.sqrt(max_dim / min_dim))
                    p.add_(update, alpha=-lr * scale)

                else:
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                    bias_correction1 = 1.0 - beta1 ** step
                    bias_correction2 = 1.0 - beta2 ** step

                    step_size = lr / bias_correction1
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                    p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
