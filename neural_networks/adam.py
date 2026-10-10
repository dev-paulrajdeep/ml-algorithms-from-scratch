r"""
A. **Mission** (Step 0: Understand the Problem)
Adam (Adaptive Moment Estimation, Kingma & Ba, 2015) is the most widely adopted general-purpose optimizer in modern deep learning. It synergizes the core strengths of two distinct paradigms:
1. **Momentum** (first-order moment): Exponential moving average of past gradients to accelerate in consistent directions and damp oscillations.
2. **RMSProp** (second-order raw moment): Exponential moving average of squared gradients to adapt step sizes per parameter based on local curvature.
Crucially, Adam solves the critical flaw of zero-initialized moving averages by introducing **analytical bias correction**, preventing distorted and severely throttled step sizes during the earliest training steps.
Your mission is to implement `Adam` from scratch using only NumPy, derive the mathematical justification for bias correction, and demonstrate why Adam achieves superior stability and convergence on both ill-conditioned surfaces (Dataset 6C) and deep networks.

B. **Prerequisites**
- [Stage 0B: NumPy & Numerical Computing](../foundations/README.md#module-0b-numpy-and-numerical-computing) — Elementwise array powers (`**`), square roots, and shape preservation.
- [Stage 0C: Linear Algebra](../foundations/README.md#module-0c-linear-algebra) — Coordinate-wise scaling and vector geometry.
- [Stage 0D: Calculus & Optimization](../foundations/README.md#module-0d-calculus-and-optimization) — Gradient descent mechanics and first/second moment statistics.
- [Chapter 6: Momentum & RMSProp](./momentum.py) — Understanding velocity accumulation and squared-gradient scaling.

C. **Learning Questions** (Step 1: Build Intuition)
1. **The Zero-Initialization Trap**: When moving averages $\mathbf{m}$ and $\mathbf{v}$ are initialized to zeros, why are their early values $\mathbf{m}_1, \mathbf{v}_1$ severely biased toward zero?
2. **Why Bias Correction Works**: How does dividing $\mathbf{m}_t$ by $(1 - \beta_1^t)$ and $\mathbf{v}_t$ by $(1 - \beta_2^t)$ mathematically restore the expected value of the estimator to the true moment ($\mathbb{E}[\hat{\mathbf{m}}_t] = \mathbb{E}[\mathbf{g}_t]$)?
3. **What Happens as $t \to \infty$**: As step count $t$ grows large (e.g. $t > 1000$), what happens to the correction factors $(1 - \beta_1^t)$ and $(1 - \beta_2^t)$?
4. **Adam's Default Hyperparameters**: Why are $\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}$ remarkably effective across vastly different domains (vision, NLP, reinforcement learning) without manual tuning?

D. **Mathematics to Derive**
1. **Step 2: Tiny Numerical Example by Hand**:
   Let hyperparameters be $\alpha = 0.001, \beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}$.
   Let initial step counter $t = 0$.
   At step $t = 1$, suppose observed gradient for a scalar parameter is $g_1 = 2.0$.
   Initial buffers: $m_0 = 0.0, v_0 = 0.0$.
   - **First Moment Update**:
     $$m_1 = \beta_1 m_0 + (1 - \beta_1) g_1 = 0.9(0) + (1 - 0.9)(2.0) = 0.1(2.0) = 0.20$$
   - **Second Moment Update**:
     $$v_1 = \beta_2 v_0 + (1 - \beta_2) g_1^2 = 0.999(0) + (1 - 0.999)(2.0^2) = 0.001(4.0) = 0.0040$$
   - **Without Bias Correction** (Naïve ratio):
     $$\frac{m_1}{\sqrt{v_1}} = \frac{0.20}{\sqrt{0.0040}} = \frac{0.20}{0.06325} \approx 3.162$$
   - **With Bias Correction**:
     Correction factors at $t=1$: $1 - \beta_1^1 = 1 - 0.9 = 0.10$; $1 - \beta_2^1 = 1 - 0.999 = 0.001$.
     $$\hat{m}_1 = \frac{m_1}{1 - \beta_1^1} = \frac{0.20}{0.10} = 2.00 \quad (\text{Exact true gradient!})$$
     $$\hat{v}_1 = \frac{v_1}{1 - \beta_2^1} = \frac{0.0040}{0.0010} = 4.00 \quad (\text{Exact true squared gradient!})$$
     $$\frac{\hat{m}_1}{\sqrt{\hat{v}_1} + \epsilon} = \frac{2.00}{\sqrt{4.00} + 10^{-8}} = \frac{2.00}{2.00} = 1.00$$
     Update:
     $$\Delta \theta_1 = -\alpha \frac{\hat{m}_1}{\sqrt{\hat{v}_1} + \epsilon} = -0.001(1.00) = -0.001$$
   Notice how bias correction perfectly scales both moments so the initial step size is exactly bounded by $-\alpha$, preventing massive instability at initialization!

2. **Step 3: Mathematical Notation**:
   - Parameters: $\boldsymbol{\theta} \in \mathbb{R}^P$.
   - First moment vector (gradient momentum): $\mathbf{m} \in \mathbb{R}^P$.
   - Second raw moment vector (squared gradient history): $\mathbf{v} \in \mathbb{R}^P$.
   - Step counter: $t \in \mathbb{N}$, initialized to $0$.
   - Decay rates: $\beta_1 \in [0, 1)$ (default $0.9$), $\beta_2 \in [0, 1)$ (default $0.999$).
   - Stability constant: $\epsilon > 0$ (default $10^{-8}$).
   - Learning rate: $\alpha > 0$ (default $0.001$).

3. **Step 4: Derivation of the Bias Correction Factor**:
   - Unrolling the first moment recurrence $\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1) \mathbf{g}_t$ starting from $\mathbf{m}_0 = \mathbf{0}$:
     $$\mathbf{m}_t = (1 - \beta_1) \sum_{i=1}^t \beta_1^{t-i} \mathbf{g}_i$$
   - Taking mathematical expectation $\mathbb{E}[\cdot]$ and assuming true gradients $\mathbf{g}_i$ come from a stationary distribution with expectation $\mathbb{E}[\mathbf{g}_t]$:
     $$\mathbb{E}[\mathbf{m}_t] = \mathbb{E}\left[(1 - \beta_1) \sum_{i=1}^t \beta_1^{t-i} \mathbf{g}_i\right] = \mathbb{E}[\mathbf{g}_t] (1 - \beta_1) \sum_{i=1}^t \beta_1^{t-i}$$
   - The finite sum of a geometric series is $\sum_{i=1}^t \beta_1^{t-i} = \sum_{k=0}^{t-1} \beta_1^k = \frac{1 - \beta_1^t}{1 - \beta_1}$.
   - Substituting this into the expectation:
     $$\mathbb{E}[\mathbf{m}_t] = \mathbb{E}[\mathbf{g}_t] (1 - \beta_1) \frac{1 - \beta_1^t}{1 - \beta_1} = \mathbb{E}[\mathbf{g}_t] (1 - \beta_1^t)$$
   - Therefore:
     $$\mathbb{E}\left[\frac{\mathbf{m}_t}{1 - \beta_1^t}\right] = \mathbb{E}[\mathbf{g}_t]$$
     Dividing by $(1 - \beta_1^t)$ yields an exact unbiased estimator! The identical derivation applies to second moment $\mathbf{v}_t$ with decay $\beta_2$.

E. **Implementation Contract**
- Class: `Adam`
- Constructor:
  - `__init__(self, learning_rate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8)`:
    Stores hyperparameters, initializes step counter `self.t = 0`, and buffers `self.m = None`, `self.v = None` (lazy allocation).
- Attributes:
  - `lr`: float, base learning rate.
  - `beta1`: float, first moment decay factor.
  - `beta2`: float, second moment decay factor.
  - `epsilon`: float, smoothing constant.
  - `t`: int, iteration counter.
  - `m`: list of ndarrays storing first moments, or `None`.
  - `v`: list of ndarrays storing second moments, or `None`.
- Methods:
  - `step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None`:
    Increments `self.t += 1`.
    Lazily initializes `self.m` and `self.v` with `np.zeros_like(p)` if `None`.
    Updates $\mathbf{m}_i, \mathbf{v}_i$, computes bias-corrected $\hat{\mathbf{m}}_i, \hat{\mathbf{v}}_i$, and updates `p` in-place:
    $$\mathbf{m}_i \leftarrow \beta_1 \mathbf{m}_i + (1 - \beta_1) \mathbf{g}_i$$
    $$\mathbf{v}_i \leftarrow \beta_2 \mathbf{v}_i + (1 - \beta_2) \mathbf{g}_i^2$$
    $$\hat{\mathbf{m}}_i = \frac{\mathbf{m}_i}{1 - \beta_1^t}, \quad \hat{\mathbf{v}}_i = \frac{\mathbf{v}_i}{1 - \beta_2^t}$$
    $$\theta_i \leftarrow \theta_i - \frac{\text{lr}}{\sqrt{\hat{\mathbf{v}}_i} + \epsilon} \odot \hat{\mathbf{m}}_i$$
    Validates matching tensor counts and shapes, raising `ValueError` on mismatch.
  - `reset(self) -> None`:
    Resets `self.t = 0`, `self.m = None`, `self.v = None`.

F. **Guided Implementation Stages**
**Step 5: Algorithm Design & Pseudocode**
```text
class Adam:
  def __init__(lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    self.lr = lr
    self.beta1 = beta1
    self.beta2 = beta2
    self.epsilon = epsilon
    self.t = 0
    self.m = None
    self.v = None

  def step(params, grads):
    self.t += 1
    if self.m is None:
      self.m = [np.zeros_like(p) for p in params]
      self.v = [np.zeros_like(p) for p in params]
    for p, g, m, v in zip(params, grads, self.m, self.v):
      m[:] = self.beta1 * m + (1.0 - self.beta1) * g
      v[:] = self.beta2 * v + (1.0 - self.beta2) * (g ** 2)
      m_hat = m / (1.0 - self.beta1 ** self.t)
      v_hat = v / (1.0 - self.beta2 ** self.t)
      p -= (self.lr / (np.sqrt(v_hat) + self.epsilon)) * m_hat
```

**Checkpoint 1: Dual State Tracking and Step Counting**
- **What to learn**: Managing two independent state buffers ($\mathbf{m}$ and $\mathbf{v}$) and tracking temporal iteration counters.
- **What to do**: Increment `self.t` on each step, lazily initialize `self.m` and `self.v` with matching shapes, and update them using EMA formulas.
- **How to check yourself**: Run step 1 with $g = 2.0$; verify $m$ equals $0.20$ and $v$ equals $0.0040$.
- **When to proceed**: First and second raw moments update correctly.
- **Recovery hints**:
  - *Hint 1 (Concept)*: Remember that $\mathbf{m}$ uses $g$ (linear) while $\mathbf{v}$ uses $g^2$ (squared).
  - *Hint 2 (Operation)*: `self.t += 1` must occur at the beginning of each step.
  - *Hint 3 (Debugging)*: Check `self.t == 1` after the first call.

**Checkpoint 2: Bias Correction and In-Place Parameter Updates**
- **What to learn**: Implementing analytical bias correction and normalized updates.
- **What to do**: Compute `m_hat = m / (1 - self.beta1 ** self.t)` and `v_hat = v / (1 - self.beta2 ** self.t)`. Update parameters using `p -= (self.lr / (np.sqrt(v_hat) + self.epsilon)) * m_hat`.
- **How to check yourself**: Verify that at $t=1$, with $g = 2.0$, the effective parameter step $\Delta \theta$ is exactly $-\alpha \times 1.0 = -0.001$.
- **When to proceed**: Bias-corrected updates match hand derivation.
- **Recovery hints**:
  - *Hint 1 (Concept)*: Use Python power operator `**` for `beta ** self.t`.
  - *Hint 2 (Operation)*: Do not overwrite `m` or `v` with their bias-corrected versions; compute `m_hat` and `v_hat` as temporary locals.
  - *Hint 3 (Debugging)*: If $\beta_2^t$ produces float underflow for very large $t$, notice that $1 - \beta_2^t \to 1.0$.

**Checkpoint 3: The Ill-Conditioned Ravine Benchmark (Dataset 6C)**
- **What to learn**: Combining momentum acceleration with coordinate adaptive scaling.
- **What to do**: Optimize Dataset 6C ($L(w_1, w_2) = 0.1 w_1^2 + 10.0 w_2^2$) with Adam ($\alpha = 0.5, \beta_1 = 0.9, \beta_2 = 0.999$).
- **How to check yourself**: Adam navigates straight down the ravine to reach loss $< 10^{-3}$ within 30 iterations, outperforming SGD and Momentum.
- **When to proceed**: Adam demonstrates superior convergence on benchmark loss surfaces.
- **Recovery hints**:
  - *Hint 1 (Concept)*: Adam adapts coordinate scales while accumulating directional velocity.
  - *Hint 2 (Operation)*: Compare convergence graphs across all 4 optimizers.
  - *Hint 3 (Debugging)*: If Adam diverges, check if $\alpha$ is too high or if bias correction denominator was omitted.

G. **Edge Cases and Expected Tests** (Step 8: Test and Debug)
1. `test_bias_correction_at_t1`: Verify $\hat{m}_1 == g_1$ and $\hat{v}_1 == g_1^2$ for the first step.
2. `test_bias_correction_asymptotic`: Verify that as $t \to \infty$ (e.g. $t = 10000$), $1 - \beta_1^t \approx 1.0$ and $1 - \beta_2^t \approx 1.0$.
3. `test_zero_gradient_stability`: Verify that with zero gradients, parameters remain unchanged and no `NaN` or division by zero occurs.
4. `test_dataset_6c_dominance`: Verify Adam reaches loss $< 10^{-3}$ on Dataset 6C in fewer iterations than vanilla SGD.

H. **Complexity Analysis** (Step 9: Analyze the Algorithm)
- **Time Complexity**: $O(P)$ arithmetic operations per step (scalar elementwise operations across all $P$ parameters).
- **Space Complexity**: $O(2P)$ auxiliary memory to store first moment $\mathbf{m}$ and second moment $\mathbf{v}$ tensors.
- **Trade-offs**: $3\times$ parameter memory footprint compared to SGD, but provides state-of-the-art out-of-the-box convergence across virtually all deep learning architectures.

I. **Progressive Difficulty Levels & Definition of Done**
- **Level 1 (Guided)**: Implement Adam update rule without bias correction and observe sluggish initial steps.
- **Level 2 (Standard)**: Implement full `Adam` class with bias correction, state management, and in-place tensor updates.
- **Level 3 (Challenge)**: Implement AdamW (Loshchilov & Hutter, 2017), which decouples weight decay regularization from gradient moment updates ($\theta \leftarrow \theta - \alpha \lambda \theta - \dots$).
- **Definition of Done**: Correct bias correction at $t=1$, handles multi-dimensional arrays, passes all tests, and solves Dataset 6C.

J. **Reflection & Self-Check** (Step 10: Independent Completion)
1. Why does traditional L2 regularization ($\frac{1}{2}\lambda \|\boldsymbol{\theta}\|_2^2$) behave differently in Adam than in SGD, and why did this necessitate the invention of AdamW?
2. If the objective function is heavily non-stationary (e.g. changing dynamic environments), how does the memory horizon controlled by $\beta_1$ and $\beta_2$ impact the optimizer's ability to track moving minima?
"""
