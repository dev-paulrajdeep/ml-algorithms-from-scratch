r"""
A. **Mission** (Step 0: Understand the Problem)
In deep neural networks and anisotropic loss surfaces (e.g., Dataset 6C), gradient magnitudes often vary by several orders of magnitude across different weights. A learning rate that is stable for steep parameters is far too slow for flat parameters, while a learning rate suitable for flat parameters causes explosive divergence along steep parameters.
AdaGrad (Duchi et al., 2011) attempted to fix this by dividing by the cumulative sum of historical squared gradients, but this monotonic accumulation causes the effective learning rate to decay to zero, prematurely halting training.
Geoffrey Hinton proposed **RMSProp** (Root Mean Square Propagation, 2012) to solve this: by replacing the infinite historical sum with an **exponentially decaying moving average of squared gradients**, RMSProp maintains an adaptive per-parameter learning rate that responds dynamically to current curvature.
Your mission is to implement `RMSProp` from scratch using only NumPy, derive how it equalizes step sizes across disparate gradient scales, and demonstrate how it navigates Dataset 6C without manual coordinate tuning.

B. **Prerequisites**
- [Stage 0B: NumPy & Numerical Computing](../foundations/README.md#module-0b-numpy-and-numerical-computing) — Elementwise squaring (`g**2`), square roots (`np.sqrt`), and small constant stabilization.
- [Stage 0C: Linear Algebra](../foundations/README.md#module-0c-linear-algebra) — Coordinate-wise scaling and Hadamard (elementwise) operations.
- [Stage 0D: Calculus & Optimization](../foundations/README.md#module-0d-calculus-and-optimization) — Second-moment scaling and adaptive step sizes.
- [Chapter 6: SGD](./sgd.py) — Understanding why a single scalar learning rate fails on ill-conditioned ravines.

C. **Learning Questions** (Step 1: Build Intuition)
1. **The AdaGrad Pitfall**: Why does accumulating all past squared gradients ($\mathbf{s} \leftarrow \mathbf{s} + \mathbf{g}^2$) guarantee that an optimizer will eventually freeze and stop learning?
2. **Exponential Forgetting**: How does the decay coefficient $\beta \in [0.9, 0.99]$ introduce an effective memory window of approximately $\frac{1}{1 - \beta}$ steps, allowing the optimizer to adapt when moving from a steep ravine to a flat plateau?
3. **Step Size Equalization**: If parameter $A$ has gradient $g_A = 100.0$ and parameter $B$ has gradient $g_B = 0.01$, what does RMSProp do to the effective updates $\Delta \theta_A$ and $\Delta \theta_B$?
4. **The Epsilon Safeguard**: Why is adding a tiny scalar $\epsilon \approx 10^{-8}$ inside the denominator square root strictly mandatory for numerical stability?

D. **Mathematics to Derive**
1. **Step 2: Tiny Numerical Example by Hand**:
   Let $\alpha = 0.1, \beta = 0.9, \epsilon = 10^{-8}$.
   Suppose we optimize two parameters: $\theta_1$ with steep gradient $g_1 = 10.0$, and $\theta_2$ with gentle gradient $g_2 = 0.1$.
   Initial second moments: $s_1 = 0.0, s_2 = 0.0$.
   - **Step 1 for steep coordinate $\theta_1$**:
     $$s_1^{(1)} = \beta s_1^{(0)} + (1 - \beta) g_1^2 = 0.9(0) + 0.1(10.0^2) = 0.1(100.0) = 10.0$$
     Effective scale: $\sqrt{s_1^{(1)}} + \epsilon = \sqrt{10.0} + 10^{-8} \approx 3.1623$.
     Update:
     $$\Delta \theta_1 = -\frac{\alpha}{\sqrt{s_1^{(1)}} + \epsilon} g_1 = -\frac{0.1}{3.1623}(10.0) \approx -0.3162$$
   - **Step 1 for gentle coordinate $\theta_2$**:
     $$s_2^{(1)} = \beta s_2^{(0)} + (1 - \beta) g_2^2 = 0.9(0) + 0.1(0.1^2) = 0.1(0.01) = 0.001$$
     Effective scale: $\sqrt{s_2^{(1)}} + \epsilon = \sqrt{0.001} + 10^{-8} \approx 0.03162$.
     Update:
     $$\Delta \theta_2 = -\frac{\alpha}{\sqrt{s_2^{(1)}} + \epsilon} g_2 = -\frac{0.1}{0.03162}(0.1) \approx -0.3162$$
   **Astonishing Result**: Although the original gradients differed by a factor of $100\times$ ($10.0$ vs. $0.1$), RMSProp normalized both step sizes to almost the exact same magnitude ($\approx -0.3162$)! The optimizer automatically navigates steep and shallow directions with uniform speed!

2. **Step 3: Mathematical Notation**:
   - Parameters: $\boldsymbol{\theta} \in \mathbb{R}^P$.
   - Second moment buffer (exponential moving average of squared gradients): $\mathbf{s} \in \mathbb{R}^P$.
   - Gradient at step $t$: $\mathbf{g}_t = \nabla_{\boldsymbol{\theta}} L(\boldsymbol{\theta}^{(t)})$.
   - Decay factor: $\beta \in [0, 1)$ (default $0.9$).
   - Smoothing term: $\epsilon > 0$ (default $10^{-8}$).
   - Base learning rate: $\alpha > 0$.

3. **Step 4: The RMSProp Update Equations**:
   - Update exponential moving average of squared gradients:
     $$\mathbf{s}_t = \beta \mathbf{s}_{t-1} + (1 - \beta) (\mathbf{g}_t \odot \mathbf{g}_t)$$
   - Parameter update:
     $$\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \frac{\alpha}{\sqrt{\mathbf{s}_t} + \epsilon} \odot \mathbf{g}_t$$
   - Notice that if gradient $\mathbf{g}_t$ is constant over many iterations:
     $$\mathbf{s}_{\infty} = \mathbf{g}^2 \implies \frac{\mathbf{g}}{\sqrt{\mathbf{s}_{\infty}}} = \frac{\mathbf{g}}{|\mathbf{g}|} = \operatorname{sign}(\mathbf{g})$$
     RMSProp behaves like sign gradient descent with step size $\alpha$, making it exceptionally immune to exploding or vanishing gradient scales!

E. **Implementation Contract**
- Class: `RMSProp`
- Constructor:
  - `__init__(self, learning_rate: float = 0.01, beta: float = 0.9, epsilon: float = 1e-8)`:
    Stores hyperparameters and sets `self.squared_avgs = None` (lazy allocation).
- Attributes:
  - `lr`: float, learning rate.
  - `beta`: float, EMA decay factor.
  - `epsilon`: float, numerical stability constant.
  - `squared_avgs`: list of ndarrays storing $\mathbf{s}$ buffers, or `None` prior to first step.
- Methods:
  - `step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None`:
    Lazily initializes `self.squared_avgs = [np.zeros_like(p) for p in params]` if `None`.
    Updates $\mathbf{s}$ and `params` in-place:
    $$\mathbf{s}_i \leftarrow \beta \mathbf{s}_i + (1 - \beta) \mathbf{g}_i^2$$
    $$\theta_i \leftarrow \theta_i - \frac{\text{lr}}{\sqrt{\mathbf{s}_i} + \epsilon} \odot \mathbf{g}_i$$
    Validates matching lengths and tensor shapes, raising `ValueError` on discrepancy.
  - `reset(self) -> None`:
    Clears `self.squared_avgs = None` to reset internal state.

F. **Guided Implementation Stages**
**Step 5: Algorithm Design & Pseudocode**
```text
class RMSProp:
  def __init__(lr=0.01, beta=0.9, epsilon=1e-8):
    self.lr = lr
    self.beta = beta
    self.epsilon = epsilon
    self.squared_avgs = None

  def step(params, grads):
    if self.squared_avgs is None:
      self.squared_avgs = [np.zeros_like(p) for p in params]
    for p, g, s in zip(params, grads, self.squared_avgs):
      s[:] = self.beta * s + (1.0 - self.beta) * (g ** 2)
      p -= (self.lr / (np.sqrt(s) + self.epsilon)) * g
```

**Checkpoint 1: Lazy Buffer Initialization and Second-Moment Tracking**
- **What to learn**: Maintaining elementwise second-moment buffers across arbitrary parameter shapes.
- **What to do**: Lazily initialize `self.squared_avgs` with `np.zeros_like(p)` and update `s[:] = self.beta * s + (1 - self.beta) * (g ** 2)`.
- **How to check yourself**: Pass $g = 2.0$ with $\beta = 0.9$. Verify $s$ equals $0.1(4.0) = 0.4$.
- **When to proceed**: Second-moment tracking runs cleanly and matches hand calculation.
- **Recovery hints**:
  - *Hint 1 (Concept)*: Always square gradients elementwise: `g ** 2` or `np.square(g)`.
  - *Hint 2 (Operation)*: Do not forget the `(1 - self.beta)` multiplier on the new squared gradient.
  - *Hint 3 (Debugging)*: Check `s.shape == p.shape`.

**Checkpoint 2: Adaptive Scaling and In-Place Parameter Updates**
- **What to learn**: Combining coordinate-wise division with epsilon stabilization.
- **What to do**: Compute adaptive step $\Delta \theta = -\frac{\alpha}{\sqrt{s} + \epsilon} \odot g$ and update parameter in-place.
- **How to check yourself**: Replicate the Step 2 hand calculation ($g_1=10.0, g_2=0.1$); verify updates match $\approx -0.3162$ on both coordinates.
- **When to proceed**: Scale normalization is verified across disparate gradient orders of magnitude.
- **Recovery hints**:
  - *Hint 1 (Concept)*: Add `epsilon` outside or inside the square root? The standard formulation is `np.sqrt(s) + epsilon`.
  - *Hint 2 (Operation)*: `p -= (self.lr / (np.sqrt(s) + self.epsilon)) * g`.
  - *Hint 3 (Debugging)*: If updates produce `NaN`, ensure `s` never contains negative values and `epsilon > 0`.

**Checkpoint 3: Conquering Dataset 6C Without Tuning**
- **What to learn**: Automatic adaptation to anisotropic curvature.
- **What to do**: Optimize Dataset 6C ($L(w_1, w_2) = 0.1 w_1^2 + 10.0 w_2^2$) with RMSProp ($\alpha = 0.1, \beta = 0.9$).
- **How to check yourself**: RMSProp advances smoothly down the canyon without the violent cross-axis oscillations seen in SGD, reaching loss $< 10^{-2}$ in $< 50$ iterations.
- **When to proceed**: Optimizer confirms autonomous coordinate adaptation.
- **Recovery hints**:
  - *Hint 1 (Concept)*: RMSProp automatically scales down the $w_2$ step size and scales up the $w_1$ step size.
  - *Hint 2 (Operation)*: Track trajectory coordinates over time.
  - *Hint 3 (Debugging)*: If learning is too slow, increase base learning rate $\alpha$.

G. **Edge Cases and Expected Tests** (Step 8: Test and Debug)
1. `test_lazy_initialization`: Verify `squared_avgs` is `None` prior to step and matches parameter shapes after step.
2. `test_zero_gradient_stability`: When $g=0$, verify denominator does not divide by zero due to $\epsilon$, leaving parameters unchanged.
3. `test_scale_invariance`: Verify that scaling a gradient by $1000\times$ does not scale the parameter step size by $1000\times$ (due to square root normalization).
4. `test_dataset_6c_convergence`: Verify RMSProp converges on Dataset 6C within 60 steps with $\alpha = 0.1$.

H. **Complexity Analysis** (Step 9: Analyze the Algorithm)
- **Time Complexity**: $O(P)$ elementwise operations per step (squaring, square root, division, addition).
- **Space Complexity**: $O(P)$ memory to store second moment buffers $\mathbf{s}$.
- **Trade-offs**: Solves gradient scale disparity and avoids AdaGrad's learning rate decay, but does not accumulate directional momentum.

I. **Progressive Difficulty Levels & Definition of Done**
- **Level 1 (Guided)**: Implement scalar RMSProp update rule on a 1D quadratic function.
- **Level 2 (Standard)**: Implement full `RMSProp` class with lazy initialization, shape checking, and in-place updates across tensor lists.
- **Level 3 (Challenge)**: Benchmark convergence speeds of Vanilla SGD vs. Momentum vs. RMSProp on Dataset 6C across multiple learning rates.
- **Definition of Done**: Equalizes step sizes across disparate scales, passes all numerical and edge case tests, and solves Dataset 6C.

J. **Reflection & Self-Check** (Step 10: Independent Completion)
1. Why does RMSProp divide by $\sqrt{\mathbf{s}}$ instead of $\mathbf{s}$? What would be the dimensional unit of the parameter update if we divided by $\mathbf{s}$ (the second moment) directly?
2. Why is RMSProp particularly popular for recurrent neural networks (RNNs) and reinforcement learning?
"""
