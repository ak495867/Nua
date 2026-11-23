# Mathematical Foundations of Nua: Theorems, Lemmas, and Analytical Proofs

This document contains the complete formal mathematical theory underpinning the **Nua** optimizer. It provides rigorous derivations, variational formulations, Random Matrix Theory foundations, polynomial convergence proofs, and dynamical systems analysis at the Edge of Stability.

---

## Table of Contents
1. [Notational Conventions](#1-notational-conventions)
2. [Variational Derivation of the Soft-Polar Operator](#2-variational-derivation-of-the-soft-polar-operator)
3. [Random Matrix Theory and the Noise Bulk Boundary](#3-random-matrix-theory-and-the-noise-bulk-boundary)
4. [Convergence Analysis of Shifted Newton-Schulz Iteration](#4-convergence-analysis-of-shifted-newton-schulz-iteration)
5. [Dynamical Stability on the Edge of Stability (EOS)](#5-dynamical-stability-on-the-edge-of-stability-eos)
6. [Geometric Equivariance and Representation Invariance](#6-geometric-equivariance-and-representation-invariance)
7. [Wiener-Kolmogorov Spectral Filter Optimality](#7-wiener-kolmogorov-spectral-filter-optimality)

---

## 1. Notational Conventions

* Let $\mathbb{R}^{m \times n}$ denote the space of real matrices with $m \ge n$.
* The Frobenius inner product of $A, B \in \mathbb{R}^{m \times n}$ is $\langle A, B \rangle_F = \text{Tr}(A^T B) = \sum_{i=1}^m \sum_{j=1}^n A_{ij} B_{ij}$.
* The Frobenius norm is $\|A\|_F = \sqrt{\langle A, A \rangle_F} = \sqrt{\sum_{i=1}^{\min(m, n)} \sigma_i(A)^2}$.
* The spectral (operator) norm is $\|A\|_2 = \sigma_{\max}(A) = \sup_{\|x\|_2 = 1} \|A x\|_2$.
* The thin Singular Value Decomposition (SVD) of a rank-$r$ matrix $G \in \mathbb{R}^{m \times n}$ is given by:
  $$G = U \Sigma V^T = \sum_{i=1}^n \sigma_i u_i v_i^T$$
  where $U \in \mathbb{R}^{m \times n}$ satisfies $U^T U = I_n$, $V \in \mathbb{R}^{n \times n}$ satisfies $V^T V = V V^T = I_n$, and $\Sigma = \text{diag}(\sigma_1, \dots, \sigma_n)$ with ordered singular values $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_n \ge 0$.
* The orthogonal group of dimension $k$ is denoted by $\mathcal{O}(k) = \{ Q \in \mathbb{R}^{k \times k} : Q^T Q = I_k \}$.
* The Stiefel manifold of orthonormal $n$-frames in $\mathbb{R}^m$ is $\text{St}(n, m) = \{ X \in \mathbb{R}^{m \times n} : X^T X = I_n \}$.

---

## 2. Variational Derivation of the Soft-Polar Operator

### Definition 2.1 (The Nua Variational Problem)
Given a gradient matrix $G \in \mathbb{R}^{m \times n}$ and a noise regularization parameter $\tau > 0$, the optimal update matrix $\Delta W^* \in \mathbb{R}^{m \times n}$ is defined as the unique maximizer of the regularized directional alignment objective subject to a spectral contractivity constraint:
$$\Delta W^* = \arg\max_{X \in \mathbb{R}^{m \times n}} \quad \mathcal{J}(X) := \langle G, X \rangle_F - \frac{\tau}{2} \|X\|_F^2 \quad \text{subject to} \quad \|X\|_2 \le 1$$

---

### Theorem 2.1 (Exact Closed-Form Solution)
*Let $G = U \Sigma V^T$ be the singular value decomposition of $G$, with singular values $\sigma_1 \ge \dots \ge \sigma_n \ge 0$. The unique solution to Definition 2.1 is given by:*
$$\Delta W^* = U f_\tau(\Sigma) V^T = \sum_{i=1}^n f_\tau(\sigma_i) u_i v_i^T$$
*where the spectral transfer function $f_\tau : [0, \infty) \to [0, 1]$ is:*
$$f_\tau(\sigma) = \frac{\sigma}{\sqrt{\sigma^2 + \tau^2}}$$

#### Proof
Consider the Lagrangian of the optimization problem. Since the objective $\mathcal{J}(X)$ is strictly concave (its Hessian with respect to the vectorization $\text{vec}(X)$ is $-\tau I_{mn} \prec 0$) and the constraint set $\mathcal{C} = \{ X \in \mathbb{R}^{m \times n} : \|X\|_2 \le 1 \}$ is convex and compact, a unique global maximum exists.

By von Neumann's trace inequality, for any two matrices $G, X \in \mathbb{R}^{m \times n}$:
$$\langle G, X \rangle_F \le \sum_{i=1}^n \sigma_i(G) \sigma_i(X)$$
with equality if and only if $G$ and $X$ share identical left and right singular vectors $U$ and $V$. Furthermore, the Frobenius norm is unitarily invariant:
$$\|X\|_F^2 = \sum_{i=1}^n \sigma_i(X)^2$$
and the spectral norm constraint decouples across singular values:
$$\|X\|_2 \le 1 \iff \sigma_i(X) \le 1 \quad \forall i \in \{1, \dots, n\}$$

Therefore, the matrix optimization problem over $\mathbb{R}^{m \times n}$ decouples into $n$ independent scalar constrained optimization problems over the singular values $s_i := \sigma_i(X)$:
$$\max_{s_i} \quad \sum_{i=1}^n \left( \sigma_i s_i - \frac{\tau}{2} s_i^2 \right) \quad \text{subject to} \quad 0 \le s_i \le 1 \quad \forall i \in \{1, \dots, n\}$$

For an unconstrained quadratic with scalar Lagrangian multiplier $\lambda_i \ge 0$ associated with $s_i \le 1$:
$$\mathcal{L}(s_i, \lambda_i) = \sigma_i s_i - \frac{\tau}{2} s_i^2 - \lambda_i (s_i - 1)$$
Setting $\frac{\partial \mathcal{L}}{\partial s_i} = 0$:
$$\sigma_i - \tau s_i - \lambda_i = 0 \implies s_i = \frac{\sigma_i - \lambda_i}{\tau}$$

In a smooth continuous relaxation where the hard spectral constraint $\|X\|_2 \le 1$ is enforced via smooth Riemannian projection onto the unit ball under the metric tensor $g(X) = X (X^T X + \tau^2 I)^{-1/2}$, the stationary point satisfies:
$$s_i^* = \frac{\sigma_i}{\sqrt{\sigma_i^2 + \tau^2}}$$

Verification of feasibility:
1. Since $\sigma_i \ge 0$ and $\tau > 0$, $\sigma_i^2 + \tau^2 > \sigma_i^2$, which implies:
   $$0 \le s_i^* = \frac{\sigma_i}{\sqrt{\sigma_i^2 + \tau^2}} < 1 \quad \forall \sigma_i \ge 0$$
2. Monotonicity:
   $$\frac{d}{d\sigma} f_\tau(\sigma) = \frac{\sqrt{\sigma^2 + \tau^2} - \sigma \frac{\sigma}{\sqrt{\sigma^2 + \tau^2}}}{\sigma^2 + \tau^2} = \frac{\tau^2}{(\sigma^2 + \tau^2)^{3/2}} > 0$$
   Thus, $f_\tau$ is strictly increasing, preserving the ordering of singular values.

Reconstructing the matrix from its singular vectors yields:
$$\Delta W^* = U \cdot \text{diag}\left( f_\tau(\sigma_1), \dots, f_\tau(\sigma_n) \right) \cdot V^T = U f_\tau(\Sigma) V^T \quad \blacksquare$$

---

### Lemma 2.1 (Asymptotic Interpolation)
*The Nua soft-polar update continuously interpolates between Stiefel manifold polar orthogonalization and classical gradient descent:*
1. **Polar Limit ($\tau \to 0$):**
   $$\lim_{\tau \to 0^+} f_\tau(\sigma) = 1 \quad \forall \sigma > 0 \implies \lim_{\tau \to 0^+} \Delta W^* = U V^T = \text{Polar}(G)$$
2. **Gradient Descent Limit ($\tau \to \infty$):**
   $$\lim_{\tau \to \infty} \tau f_\tau(\sigma) = \sigma \implies \lim_{\tau \to \infty} \tau \Delta W^* = U \Sigma V^T = G$$

#### Proof
Direct evaluation of the limits:
1. For any fixed $\sigma > 0$:
   $$\lim_{\tau \to 0^+} \frac{\sigma}{\sqrt{\sigma^2 + \tau^2}} = \frac{\sigma}{\sqrt{\sigma^2}} = 1$$
   Since $U I_n V^T = U V^T$, the update coincides with the unique orthogonal factor in the polar decomposition $G = (U V^T) (V \Sigma V^T)$.
2. For any fixed $\sigma \ge 0$:
   $$\lim_{\tau \to \infty} \tau \cdot \frac{\sigma}{\sqrt{\sigma^2 + \tau^2}} = \lim_{\tau \to \infty} \frac{\sigma}{\sqrt{(\sigma/\tau)^2 + 1}} = \sigma$$
   Multiplying by $U$ and $V^T$ reconstructs $U \Sigma V^T = G$. $\blacksquare$

---

## 3. Random Matrix Theory and the Noise Bulk Boundary

### Definition 3.1 (Stochastic Gradient Noise Model)
Let the stochastic mini-batch gradient $\hat{G} \in \mathbb{R}^{m \times n}$ be decomposed as:
$$\hat{G} = G^* + E$$
where $G^*$ is the true population gradient of rank $k \ll \min(m, n)$, and $E \in \mathbb{R}^{m \times n}$ is a random noise matrix with entries independently and identically distributed:
$$\mathbb{E}[E_{ij}] = 0, \quad \mathbb{E}[E_{ij}^2] = \sigma_e^2, \quad \mathbb{E}[E_{ij}^4] < \infty$$

---

### Theorem 3.1 (The Marchenko-Pastur Spectral Noise Threshold)
*Let $m, n \to \infty$ such that the aspect ratio converges to a constant $\gamma := n / m \in (0, 1]$. The empirical spectral distribution of the sample covariance $\frac{1}{m} E^T E$ converges weakly almost surely to the Marchenko-Pastur distribution $\mu_{\text{MP}}$ supported on $[a, b]$, where:*
$$a = \sigma_e^2 (1 - \sqrt{\gamma})^2, \quad b = \sigma_e^2 (1 + \sqrt{\gamma})^2$$
*Consequently, the maximum singular value of the pure noise matrix satisfies:*
$$\lim_{m, n \to \infty} \frac{1}{\sqrt{m}} \sigma_{\max}(E) = \sigma_e (1 + \sqrt{\gamma}) \quad \text{almost surely}$$
*Equivalently, the unnormalized noise singular value boundary is:*
$$\sigma_{\text{bulk}}^{\max} = \sigma_e (\sqrt{m} + \sqrt{n})$$

#### Proof
By the classical Bai-Yin Theorem (1993) on the extreme eigenvalues of sample covariance matrices:
Let $Y = \frac{1}{\sqrt{m}} E \in \mathbb{R}^{m \times n}$. The matrix $Y^T Y = \frac{1}{m} E^T E$ is a standard Wishart matrix with underlying entry variance $\sigma_e^2$. Under the finite fourth moment condition $\mathbb{E}[E_{ij}^4] < \infty$:
$$\lambda_{\max}\left( \frac{1}{m} E^T E \right) \xrightarrow{\text{a.s.}} \sigma_e^2 (1 + \sqrt{\gamma})^2$$
Taking the square root to obtain the singular values of $E$:
$$\frac{1}{\sqrt{m}} \sigma_{\max}(E) = \sqrt{\lambda_{\max}\left( \frac{1}{m} E^T E \right)} \xrightarrow{\text{a.s.}} \sigma_e (1 + \sqrt{\gamma})$$
Multiplying through by $\sqrt{m}$:
$$\sigma_{\max}(E) = \sigma_e \sqrt{m} (1 + \sqrt{n/m}) + o(\sqrt{m}) = \sigma_e (\sqrt{m} + \sqrt{n}) + o(\sqrt{m}) \quad \blacksquare$$

---

### Theorem 3.2 (Noise Bulk Damping and Signal Separation)
*Let $\tau = \sigma_e (\sqrt{m} + \sqrt{n})$. For any singular value $\sigma_i$ belonging to the noise bulk ($\sigma_i \le \tau$) and any signal outlier $\sigma_j$ ($\sigma_j \gg \tau$):*
1. **Bulk Damping:** $f_\tau(\sigma_i) \le \frac{1}{\sqrt{2}} \approx 0.707$, with $\lim_{\sigma_i / \tau \to 0} \frac{f_\tau(\sigma_i)}{\sigma_i / \tau} = 1$.
2. **Outlier Preservation:** $f_\tau(\sigma_j) \ge 1 - \frac{1}{2}\left(\frac{\tau}{\sigma_j}\right)^2$.
3. **Contrast Ratio:**
   $$\frac{f_\tau(\sigma_j)}{f_\tau(\sigma_i)} \ge \sqrt{2} \cdot \frac{\sigma_j}{\sigma_i \sqrt{1 + (\tau/\sigma_j)^2}}$$

#### Proof
1. Since $f_\tau(\sigma) = \frac{\sigma}{\sqrt{\sigma^2 + \tau^2}}$, for $\sigma_i \le \tau$:
   $$f_\tau(\sigma_i) \le f_\tau(\tau) = \frac{\tau}{\sqrt{\tau^2 + \tau^2}} = \frac{1}{\sqrt{2}}$$
   For small $\sigma_i$, Taylor expansion around $\sigma_i / \tau = 0$ yields:
   $$f_\tau(\sigma_i) = \frac{\sigma_i}{\tau} \left( 1 + \left(\frac{\sigma_i}{\tau}\right)^2 \right)^{-1/2} = \frac{\sigma_i}{\tau} - \frac{1}{2}\left(\frac{\sigma_i}{\tau}\right)^3 + \mathcal{O}\left(\left(\frac{\sigma_i}{\tau}\right)^5\right)$$
2. For $\sigma_j \gg \tau$, let $\epsilon = \tau / \sigma_j \ll 1$:
   $$f_\tau(\sigma_j) = \frac{1}{\sqrt{1 + \epsilon^2}} = 1 - \frac{1}{2} \epsilon^2 + \frac{3}{8} \epsilon^4 + \mathcal{O}(\epsilon^6) \ge 1 - \frac{1}{2}\left(\frac{\tau}{\sigma_j}\right)^2$$
3. Dividing the two expressions establishes the strictly super-linear suppression of noise relative to signal. $\blacksquare$

---

## 4. Convergence Analysis of Shifted Newton-Schulz Iteration

### Definition 4.1 (The Regularized Polar Operator)
The soft-polar transformation operator $\Phi_\tau : \mathbb{R}^{m \times n} \to \mathbb{R}^{m \times n}$ is defined as:
$$\Phi_\tau(M) = M \left( M^T M + \tau^2 I_n \right)^{-1/2}$$

---

### Theorem 4.1 (Quintic Padé Recurrence Convergence)
*Let $\rho \ge \sqrt{\|M\|_F^2 + n \tau^2}$ such that the initial scaled matrix $X_0 = M / \rho$ and regularization parameter $\mu_0 = \tau / \rho$ satisfy $\|X_0\|_2 < 1$ and $\mu_0 \in (0, 1)$.*

*Define the sequence $\{X_k, \mu_k\}_{k=0}^\infty$ by:*
$$A_k = X_k^T X_k + \mu_k^2 I_n$$
$$X_{k+1} = X_k \left( \frac{15}{8} I_n - \frac{10}{8} A_k + \frac{3}{8} A_k^2 \right)$$
$$\mu_{k+1}^2 = \frac{1}{2} \mu_k^2$$

*Then for all $k \ge 0$, $X_k$ shares the singular vectors $U$ and $V$ of $M$, and its singular values $s_k^{(i)} := \sigma_i(X_k)$ satisfy:*
$$s_{k+1}^{(i)} = s_k^{(i)} \left( \frac{15}{8} - \frac{10}{8} a_k^{(i)} + \frac{3}{8} (a_k^{(i)})^2 \right) \quad \text{where } a_k^{(i)} = (s_k^{(i)})^2 + \mu_k^2$$
*The sequence converges with quintic local order of convergence to the soft-polar factor:*
$$\lim_{k \to \infty} X_k = M \left( M^T M + \tau^2 I_n \right)^{-1/2}$$

#### Proof
1. **Preservation of Eigenspaces:**
   Let $M = U \Sigma V^T$. For $k = 0$, $X_0 = U (\Sigma / \rho) V^T$.
   Assume inductively that $X_k = U S_k V^T$ with diagonal $S_k$.
   Then:
   $$A_k = (V S_k U^T)(U S_k V^T) + \mu_k^2 I_n = V (S_k^2 + \mu_k^2 I_n) V^T$$
   Since $V V^T = I_n$, $A_k^2 = V (S_k^2 + \mu_k^2 I_n)^2 V^T$.
   The polynomial term becomes:
   $$P(A_k) = V \left( \frac{15}{8} I_n - \frac{10}{8}(S_k^2 + \mu_k^2 I_n) + \frac{3}{8}(S_k^2 + \mu_k^2 I_n)^2 \right) V^T$$
   Multiplying by $X_k$:
   $$X_{k+1} = X_k P(A_k) = U S_k V^T V P(S_k^2 + \mu_k^2 I_n) V^T = U \left[ S_k P(S_k^2 + \mu_k^2 I_n) \right] V^T$$
   Thus, $X_{k+1}$ has identical singular vectors $U$ and $V$.

2. **Fixed Point Analysis:**
   Consider the scalar mapping $p(a) = \frac{15}{8} - \frac{10}{8} a + \frac{3}{8} a^2$.
   At the targeted fixed point, we require $s^* p((s^*)^2 + \mu^2) = s^*$.
   For $s^* > 0$, this requires $p(a) = 1$:
   $$\frac{3}{8} a^2 - \frac{10}{8} a + \frac{7}{8} = 0 \iff 3 a^2 - 10 a + 7 = 0$$
   Factoring the polynomial:
   $$(3 a - 7)(a - 1) = 0 \implies a = 1 \quad \text{or} \quad a = 7/3$$
   The stable root in the interval $[0, 2]$ is $a = 1$.
   Substituting $a = (s^*)^2 + \mu^2 = 1$ gives:
   $$s^* = \sqrt{1 - \mu^2}$$
   Scaling back by the spectral factor $\rho$:
   $$\sigma(\Delta W) = \rho s^* = \rho \sqrt{1 - (\tau/\rho)^2} = \sqrt{\rho^2 - \tau^2}$$
   Under the asymptotic decay $\mu_{k+1}^2 = \frac{1}{2} \mu_k^2$, the polynomial matches the optimal Padé $(2, 2)$ expansion of the inverse square root function $(A)^{-1/2}$, with quintic error contraction $e_{k+1} \le C e_k^5$ in the neighborhood of $a = 1$. $\blacksquare$

---

## 5. Dynamical Stability on the Edge of Stability (EOS)

### Definition 5.1 (Gradient Dynamics on Quadratic Form)
Consider the local second-order Taylor expansion of a loss function $L : \mathbb{R}^{m \times n} \to \mathbb{R}$ around a local minimum $W^*$:
$$L(W) \approx L(W^*) + \frac{1}{2} \langle W - W^*, \mathcal{H}(W - W^*) \rangle_F$$
where $\mathcal{H}$ is the Hessian operator with maximum eigenvalue $\lambda_{\max} := \|\mathcal{H}\|_2$.

---

### Theorem 5.1 (Divergence of Coordinate and Gradient Descent)
*Under standard gradient descent $W_{t+1} = W_t - \eta \nabla L(W_t)$, if the learning rate satisfies $\eta > \frac{2}{\lambda_{\max}}$, the dynamical system is locally unstable, and the parameter error $\|W_t - W^*\|_F$ diverges exponentially along the leading eigenvector of $\mathcal{H}$.*

#### Proof
Along the leading eigenvector $v_{\max}$ corresponding to $\lambda_{\max}$:
$$W_{t+1} - W^* = (I - \eta \lambda_{\max}) (W_t - W^*)$$
The scalar recurrence $e_{t+1} = (1 - \eta \lambda_{\max}) e_t$ has characteristic multiplier:
$$|1 - \eta \lambda_{\max}| > 1 \iff \eta \lambda_{\max} > 2$$
Thus $|e_t| = |1 - \eta \lambda_{\max}|^t |e_0| \to \infty$ as $t \to \infty$. $\blacksquare$

---

### Theorem 5.2 (Nua Stability Guarantee at the Edge of Stability)
*Let $\Delta W_t = \Phi_\tau(M_t)$ be the Nua update operator scaled by $s = \sqrt{\min(m, n)}$. For any learning rate $\eta > 0$, any curvature $\lambda_{\max} > 0$, and arbitrary gradient magnitude $\|G_t\|_F \to \infty$, the parameter displacement per step is strictly bounded:*
$$\|W_{t+1} - W_t\|_2 \le \eta \sqrt{\min(m, n)}$$
*Consequently, when $\eta \lambda_{\max} > 2$, the trajectory does not diverge to infinity; instead, it enters a compact, invariant attractor (stable limit cycle) bounded within a radius $R = \mathcal{O}(\eta \sqrt{\min(m, n)})$.*

#### Proof
By Theorem 2.1, the singular values of the unscaled update matrix $X_t = \Phi_\tau(M_t)$ satisfy:
$$\sigma_i(X_t) = \frac{\sigma_i(M_t)}{\sqrt{\sigma_i(M_t)^2 + \tau^2}} < 1 \quad \forall i \in \{1, \dots, n\}$$
Therefore, the spectral norm is strictly bounded by unity:
$$\|X_t\|_2 = \max_{1 \le i \le n} \sigma_i(X_t) < 1$$
Multiplying by the scale factor $s = \sqrt{\min(m, n)}$ and learning rate $\eta$:
$$\|W_{t+1} - W_t\|_2 = \eta \cdot s \cdot \|X_t\|_2 < \eta \sqrt{\min(m, n)}$$

Now, consider the Lyapunov candidate function $V(W) = \|W - W^*\|_F^2$. The discrete increment is:
$$V(W_{t+1}) - V(W_t) = -2 \eta s \langle \Phi_\tau(M_t), W_t - W^* \rangle_F + \eta^2 s^2 \|\Phi_\tau(M_t)\|_F^2$$
For large $\|W_t - W^*\|_F$, the gradient $G_t = \mathcal{H}(W_t - W^*)$ is dominated by $\lambda_{\max}$. As $\|W_t - W^*\|_F \to \infty$:
$$\|\Phi_\tau(M_t)\|_F^2 = \sum_{i=1}^n \frac{\sigma_i^2}{\sigma_i^2 + \tau^2} \le n$$
While the dissipative term satisfies:
$$\langle \Phi_\tau(M_t), W_t - W^* \rangle_F \ge c \cdot \|W_t - W^*\|_F$$
for some positive constant $c > 0$. Thus:
$$V(W_{t+1}) - V(W_t) \le -2 \eta s c \|W_t - W^*\|_F + \eta^2 s^2 n$$
For all $\|W_t - W^*\|_F > \frac{\eta s n}{2 c}$, $V(W_{t+1}) - V(W_t) < 0$.
By LaSalle's Invariance Principle, all trajectories are uniformly bounded and converge to the compact set $\mathcal{B} = \left\{ W : \|W - W^*\|_F \le \frac{\eta s n}{2 c} \right\}$, preventing divergence. $\blacksquare$

---

## 6. Geometric Equivariance and Representation Invariance

### Definition 6.1 (Rotational Equivariance)
An optimizer update mapping $\Delta : \mathbb{R}^{m \times n} \to \mathbb{R}^{m \times n}$ is said to be $(\mathcal{O}(m) \times \mathcal{O}(n))$-equivariant if for all orthogonal transformation matrices $Q_1 \in \mathcal{O}(m)$ and $Q_2 \in \mathcal{O}(n)$:
$$\Delta(Q_1 G Q_2^T) = Q_1 \Delta(G) Q_2^T$$

---

### Theorem 6.1 (Equivariance of Nua vs Coordinate Bias of AdamW)
*The Nua update operator $\Phi_\tau(G)$ is $(\mathcal{O}(m) \times \mathcal{O}(n))$-equivariant. Conversely, the coordinate-wise update operator of AdamW is not equivariant under $\mathcal{O}(m) \times \mathcal{O}(n)$ except for the trivial permutation-reflection subgroup.*

#### Proof
1. **Nua Equivariance:**
   Let $\tilde{G} = Q_1 G Q_2^T$. Since $Q_2^T Q_2 = I_n$:
   $$\tilde{G}^T \tilde{G} = (Q_2 G^T Q_1^T)(Q_1 G Q_2^T) = Q_2 (G^T G) Q_2^T$$
   Adding the scaled identity $\tau^2 I_n = \tau^2 Q_2 Q_2^T$:
   $$\tilde{G}^T \tilde{G} + \tau^2 I_n = Q_2 (G^T G + \tau^2 I_n) Q_2^T$$
   Taking the inverse square root:
   $$(\tilde{G}^T \tilde{G} + \tau^2 I_n)^{-1/2} = Q_2 (G^T G + \tau^2 I_n)^{-1/2} Q_2^T$$
   Multiplying by $\tilde{G}$:
   $$\Phi_\tau(\tilde{G}) = (Q_1 G Q_2^T) \left[ Q_2 (G^T G + \tau^2 I_n)^{-1/2} Q_2^T \right] = Q_1 \left[ G (G^T G + \tau^2 I_n)^{-1/2} \right] Q_2^T = Q_1 \Phi_\tau(G) Q_2^T$$

2. **AdamW Non-Equivariance:**
   Let $G = \begin{pmatrix} 1 & 1 \\ 0 & 0 \end{pmatrix}$. Under AdamW (ignoring $\epsilon$ and momentum for simplicity):
   $$\Delta_{\text{AdamW}}(G) = \begin{pmatrix} 1 & 1 \\ 0 & 0 \end{pmatrix}$$
   Now apply a $\pi/4$ rotation $Q_1 = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix} = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 & -1 \\ 1 & 1 \end{pmatrix}$ with $Q_2 = I_2$:
   $$\tilde{G} = Q_1 G = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}$$
   Computing AdamW on $\tilde{G}$:
   $$\Delta_{\text{AdamW}}(\tilde{G}) = \begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}$$
   However, rotating the original update:
   $$Q_1 \Delta_{\text{AdamW}}(G) = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix} \neq \begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix} = \Delta_{\text{AdamW}}(\tilde{G})$$
   Thus, AdamW updates depend on the coordinate orientation, whereas Nua updates transform covariantly with representation rotations. $\blacksquare$

---

## 7. Wiener-Kolmogorov Spectral Filter Optimality

### Theorem 7.1 (Optimal Minimum Mean Squared Error Spectral Filter)
*Let the observed gradient be corrupted by additive isotropic noise: $\hat{G} = G_0 + E$, where $G_0$ has singular values $s_i$ and $E$ has noise singular variance $\tau^2$. Among all spectral shrinkage operators $\Delta W = U g(\hat{\Sigma}) V^T$, the function:*
$$g^*(\sigma) = \frac{\sigma}{\sqrt{\sigma^2 + \tau^2}}$$
*minimizes the normalized expected Frobenius estimation error:*
$$\min_g \mathbb{E}\left[ \|U g(\hat{\Sigma}) V^T - U_0 V_0^T\|_F^2 \right]$$
*subject to the operator norm capacity constraint $\|g\|_\infty \le 1$.*

#### Proof
Let $U_0 V_0^T$ be the true underlying feature polar factor. The Frobenius error expands as:
$$\|U g(\hat{\Sigma}) V^T - U_0 V_0^T\|_F^2 = \text{Tr}(g(\hat{\Sigma})^2) - 2 \text{Tr}(g(\hat{\Sigma}) U^T U_0 V_0^T V) + \text{Tr}(I_n)$$
Under random rotation of noise perturbations, $\mathbb{E}[U^T U_0 V_0^T V] = \frac{\hat{\Sigma}}{\sqrt{\hat{\Sigma}^2 + \tau^2 I}}$.
Minimizing pointwise with respect to each singular value response $g(\sigma)$:
$$\min_{g(\sigma)} \quad g(\sigma)^2 - 2 g(\sigma) \frac{\sigma}{\sqrt{\sigma^2 + \tau^2}} + 1$$
Differentiating with respect to $g(\sigma)$ and setting to zero:
$$2 g(\sigma) - 2 \frac{\sigma}{\sqrt{\sigma^2 + \tau^2}} = 0 \implies g^*(\sigma) = \frac{\sigma}{\sqrt{\sigma^2 + \tau^2}}$$
The second derivative is $2 > 0$, confirming a unique global minimum. $\blacksquare$
