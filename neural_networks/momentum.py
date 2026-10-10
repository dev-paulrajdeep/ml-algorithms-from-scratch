r"""
A. **Mission** (Step 0: Understand the Problem)
Stochastic Gradient Descent struggles on loss surfaces where curvature is highly anisotropic—surfaces featuring steep canyon walls and a gently sloping floor (e.g., Dataset 6C). Vanilla SGD oscillates violently across the ravine while making glacial progress along the base.
Momentum solves this by drawing a physical analogy to a heavy marble rolling down a hill: the marble accumulates velocity in directions of persistent gradient force while cancelling out oscillatory forces.
Your mission is to implement the classic `Momentum` optimizer from scratch using only NumPy. You will derive how the velocity buffer functions as an exponentially weighted moving average, prove why it provides an effective $10\times$ acceleration along flat directions (for $\beta = 0.9$) while damping cross-canyon oscillations, and show that it dramatically outperforms vanilla SGD on Dataset 6C.

B. **Prerequisites**
- [Stage 0B: NumPy & Numerical Computing](../foundations/README.md#module-0b-numpy-and-numerical-computing) — Array copying, zeros initialization, and in-place vector arithmetic.
- [Stage 0C: Linear Algebra](../foundations/README.md#module-0c-linear-algebra) — Vector directions, velocity buffers, and geometric projections.
- [Stage 0D: Calculus & Optimization](../foundations/README.md#module-0d-calculus-and-optimization) — Gradient descent mechanics and condition numbers of Hessian matrices.
- [Chapter 6: SGD](./sgd.py) — Vanilla SGD implementation and understanding the ravine failure.

C. **Learning Questions** (Step 1: Build Intuition)
1. **The Physical Analogy**: How does introducing a mass/inertia parameter prevent the optimizer from getting stuck in shallow local optima or oscillating endlessly across steep valleys?
2. **Effective Terminal Velocity**: If the gradient is constant ($\mathbf{g}_t = \mathbf{g}$), what is the steady-state velocity $\mathbf{v}_{\infty}$ as $t \to \infty$? Why does $\beta = 0.9$ multiply the effective learning rate by a factor of $\frac{1}{1 - \beta} = 10$?
3. **Oscillation Damping**: When gradients alternate signs across successive iterations ($+g, -g, +g, -g$), how does the momentum buffer mathematically cancel out these orthogonal perturbations?
4. **Stateful Optimizers**: Why does Momentum require persistent state memory (velocity buffers) matching the exact dimensions of every model parameter?

D. **Mathematics to Derive**
1. **Step 2: Tiny Numerical Example by Hand**:
   Let learning rate $\alpha = 0.1$, momentum coefficient $\beta = 0.9$.
   Consider an oscillating coordinate where gradients alternate: $g_1 = +2.0, g_2 = -2.0, g_3 = +2.0$.
   Initial velocity $v_0 = 0.0$, initial parameter $\theta_0 = 10.0$.
   - **Step 1**:
     $v_1 = \beta v_0 + g_1 = 0.9(0.0) + 2.0 = 2.0$.
     Update: $\theta_1 = \theta_0 - \alpha v_1 = 10.0 - 0.1(2.0) = 9.80$.
   - **Step 2**:
     $v_2 = \beta v_1 + g_2 = 0.9(2.0) + (-2.0) = 1.8 - 2.0 = -0.20$.
     Update: $\theta_2 = \theta_1 - \alpha v_2 = 9.80 - 0.1(-0.20) = 9.82$.
   Notice: In step 2, vanilla SGD would have jumped by $-0.1(-2.0) = +0.20$, whereas Momentum dampens the jump to only $+0.02$—a $10\times$ reduction in oscillatory amplitude!
   Now consider a flat coordinate where the gradient is consistently $g = 1.0$:
   - $v_1 = 0.9(0) + 1.0 = 1.0 \implies \Delta\theta_1 = -0.1(1.0) = -0.10$.
   - $v_2 = 0.9(1.0) + 1.0 = 1.9 \implies \Delta\theta_2 = -0.1(1.9) = -0.19$.
   - $v_3 = 0.9(1.9) + 1.0 = 2.71 \implies \Delta\theta_3 = -0.1(2.71) = -0.271$.
   The optimizer accelerates smoothly down the slope!

2. **Step 3: Mathematical Notation**:
   - Parameters: $\boldsymbol{\theta} \in \mathbb{R}^P$.
   - Velocity buffer: $\mathbf{v} \in \mathbb{R}^P$, initialized to $\mathbf{0}$.
   - Gradient at step $t$: $\mathbf{g}_t = \nabla_{\boldsymbol{\theta}} L(\boldsymbol{\theta}^{(t)})$.
   - Momentum hyperparameter: $\beta \in [0, 1)$ (typically $0.9$).
   - Learning rate: $\alpha > 0$.

3. **Step 4: Derivation of the Effective Learning Rate**:
   - Update equations:
     $$\mathbf{v}_t = \beta \mathbf{v}_{t-1} + \mathbf{g}_t$$
     $$\boldsymbol{\theta}_{t} = \boldsymbol{\theta}_{t-1} - \alpha \mathbf{v}_t$$
   - Expanding $\mathbf{v}_t$ recursively assuming constant gradient $\mathbf{g}$:
     $$\mathbf{v}_t = \mathbf{g} + \beta \mathbf{g} + \beta^2 \mathbf{g} + \dots + \beta^{t-1} \mathbf{g} = \mathbf{g} \sum_{k=0}^{t-1} \beta^k$$
   - Applying the infinite geometric series sum $\sum_{k=0}^{\infty} \beta^k = \frac{1}{1 - \beta}$:
     $$\lim_{t \to \infty} \mathbf{v}_t = \frac{1}{1 - \beta} \mathbf{g}$$
   - The effective step size is therefore:
     $$\Delta \boldsymbol{\theta}_{\infty} = -\frac{\alpha}{1 - \beta} \mathbf{g}$$
     For $\beta = 0.9$, the effective step size is $10\alpha \mathbf{g}$!

E. **Implementation Contract**
- Class: `Momentum`
- Constructor:
  - `__init__(self, learning_rate: float = 0.01, beta: float = 0.9)`:
    Stores hyperparameters and initializes `self.velocities = None` (lazy allocation).
- Attributes:
  - `lr`: float, learning rate.
  - `beta`: float, momentum decay factor.
  - `velocities`: list of ndarrays storing momentum buffers, or `None` before first step.
- Methods:
  - `step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None`:
    Lazily initializes `self.velocities` to zeros matching each parameter's shape if `self.velocities is None`.
    Updates each velocity buffer and parameter in-place:
    $$\mathbf{v}_i \leftarrow \beta \mathbf{v}_i + \mathbf{g}_i$$
    $$\theta_i \leftarrow \theta_i - \text{lr} \cdot \mathbf{v}_i$$
    Validates that `len(params) == len(grads)` and shapes match, raising `ValueError` on mismatch.
  - `reset(self) -> None`:
    Resets `self.velocities = None` to clear accumulated momentum between independent training runs.

F. **Guided Implementation Stages**
**Step 5: Algorithm Design & Pseudocode**
```text
class Momentum:
  def __init__(lr=0.01, beta=0.9):
    self.lr = lr
    self.beta = beta
    self.velocities = None

  def step(params, grads):
    if self.velocities is None:
      self.velocities = [np.zeros_like(p) for p in params]
    for p, g, v in zip(params, grads, self.velocities):
      v[:] = self.beta * v + g
      p -= self.lr * v
```

**Checkpoint 1: Lazy Buffer Initialization and Shape Tracking**
- **What to learn**: Managing stateful optimizer buffers across multi-dimensional arrays without requiring explicit model registration.
- **What to do**: On the first call to `step`, inspect `params` and initialize `self.velocities = [np.zeros_like(p) for p in params]`.
- **How to check yourself**: Initialize two parameter tensors of shapes `(3, 4)` and `(4,)`. Call `step` once. Verify `self.velocities` contains two arrays matching shapes `(3, 4)` and `(4,)` filled with the initial gradient values.
- **When to proceed**: Lazy allocation handles arbitrary tensor shapes cleanly.
- **Recovery hints**:
  - *Hint 1 (Concept)*: `np.zeros_like(p)` creates a new zero array matching the exact shape and dtype of `p`.
  - *Hint 2 (Operation)*: Do not overwrite `self.velocities` on subsequent calls if it already exists!
  - *Hint 3 (Debugging)*: Check `len(self.velocities) == len(params)`.

**Checkpoint 2: Velocity Accumulation and In-Place Parameter Updates**
- **What to learn**: Combining momentum accumulation with parameter updates.
- **What to do**: Update `v` in-place using `v[:] = self.beta * v + g` and apply `p -= self.lr * v`.
- **How to check yourself**: Run 3 manual steps with constant gradient $g=1.0$ and verify that velocity values match $1.0, 1.9, 2.71$ exactly.
- **When to proceed**: Unit checks confirm numerical equivalence to hand calculations.
- **Recovery hints**:
  - *Hint 1 (Concept)*: Using `v[:] = ...` updates the existing array in the velocity list in-place.
  - *Hint 2 (Operation)*: `p -= self.lr * v` performs the in-place parameter update.
  - *Hint 3 (Debugging)*: Ensure you add `g`, not subtract `g`, when computing $\mathbf{v} = \beta \mathbf{v} + \mathbf{g}$.

**Checkpoint 3: Conquering the Ill-Conditioned Ravine (Dataset 6C)**
- **What to learn**: Demonstrating accelerated convergence over vanilla SGD.
- **What to do**: Run Momentum on Dataset 6C ($L(w_1, w_2) = 0.1 w_1^2 + 10.0 w_2^2$) starting from $[10.0, 1.0]$ with $\alpha = 0.05, \beta = 0.9$.
- **How to check yourself**: Compare the number of iterations required to reach loss $< 10^{-2}$ against vanilla SGD with the same learning rate. Momentum reaches the target in $< 40$ steps, while SGD requires $> 200$ steps!
- **When to proceed**: Momentum empirically proves its theoretical superiority on anisotropic ravines.
- **Recovery hints**:
  - *Hint 1 (Concept)*: The ratio of progress along $w_1$ is boosted by $10\times$.
  - *Hint 2 (Operation)*: Record loss history and compare convergence curves.
  - *Hint 3 (Debugging)*: If Momentum overshoots and explodes, ensure $\alpha < \frac{2(1+\beta)}{L_{\max}}$.

G. **Edge Cases and Expected Tests** (Step 8: Test and Debug)
1. `test_lazy_initialization`: Verify `velocities` is `None` before first step and initialized to matching zero arrays after first step.
2. `test_constant_gradient_acceleration`: With constant gradient $1.0$, verify velocity approaches $\frac{1}{1-\beta}$ as $t$ increases.
3. `test_alternating_gradient_damping`: With alternating gradients $+1, -1$, verify velocity magnitude remains $< 1.0$.
4. `test_dataset_6c_speedup`: Verify Momentum achieves loss $< 0.01$ on Dataset 6C in at least $3\times$ fewer iterations than vanilla SGD.

H. **Complexity Analysis** (Step 9: Analyze the Algorithm)
- **Time Complexity**: $O(P)$ arithmetic operations per step (identical asymptotic order to SGD).
- **Space Complexity**: $O(P)$ additional memory to store velocity buffers $\mathbf{v}$ for all $P$ parameters.
- **Trade-offs**: $2\times$ parameter memory footprint compared to SGD, but drastically accelerates convergence on ravines and plateaus.

I. **Progressive Difficulty Levels & Definition of Done**
- **Level 1 (Guided)**: Implement momentum velocity accumulation for a single parameter scalar.
- **Level 2 (Standard)**: Implement full `Momentum` class with lazy velocity allocation, in-place updates, and reset method.
- **Level 3 (Challenge)**: Implement Nesterov Accelerated Gradient (NAG), evaluating gradients at the lookahead position $\boldsymbol{\theta} - \beta \mathbf{v}$.
- **Definition of Done**: Solves Dataset 6C in $< 50$ iterations, validates shape matching, and passes all unit tests.

J. **Reflection & Self-Check** (Step 10: Independent Completion)
1. Why does setting $\beta = 0.999$ without adjusting the learning rate often cause the optimizer to overshoot the minimum and oscillate wildly like an underdamped harmonic oscillator?
2. How does the velocity buffer in Momentum differ mathematically from the exponential moving average used in RMSProp?
"""
