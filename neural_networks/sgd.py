r"""
A. **Mission** (Step 0: Understand the Problem)
Stochastic Gradient Descent (SGD) is the bedrock optimization algorithm of machine learning and modern deep learning. While standard Batch Gradient Descent computes gradients across the entire dataset $N$ before making a single parameter update, SGD approximates the true gradient using a single random sample (or a small mini-batch).
Your mission is to implement a modular, reusable `SGD` optimizer from scratch using only NumPy. You will analyze why stochastic gradient estimates are unbiased estimators of the true full-batch gradient, observe how noisy gradient updates help escape saddle points, and expose the classic limitation of vanilla SGD on ill-conditioned loss surfaces (Dataset 6C), motivating momentum and adaptive algorithms.

B. **Prerequisites**
- [Stage 0B: NumPy & Numerical Computing](../foundations/README.md#module-0b-numpy-and-numerical-computing) — In-place array operations (`-=`), broadcasting, and list handling.
- [Stage 0C: Linear Algebra](../foundations/README.md#module-0c-linear-algebra) — Vector dot products, Euclidean norms, and gradient vector geometry.
- [Stage 0D: Calculus & Optimization](../foundations/README.md#module-0d-calculus-and-optimization) — Gradients as vectors of steepest ascent, step size, and convergence vs. divergence.
- [Stage 0E: Probability & Statistics](../foundations/README.md#module-0e-probability-and-statistics) — Expected values and sample variance of estimators.

C. **Learning Questions** (Step 1: Build Intuition)
1. **The Unbiased Estimator**: Why is the gradient of a randomly sampled mini-batch an unbiased estimator of the full-batch gradient ($\mathbb{E}[\nabla_{\theta} L_i(\theta)] = \nabla_{\theta} L(\theta)$)?
2. **Batch vs. Mini-batch vs. Pure SGD**:
   - Full Batch ($B=N$): Exact gradient direction, but computationally prohibitive for large datasets.
   - Pure Stochastic ($B=1$): Extremely fast updates, but high variance causes noisy trajectories.
   - Mini-batch ($B \in [16, 128]$): Optimal compromise—maximizes vectorized hardware throughput while injecting just enough noise to escape saddle points.
3. **The Ravine Problem (Dataset 6C)**: Why does vanilla SGD struggle on anisotropic surfaces (where the curvature is steep in one direction and shallow in another)? Why does it oscillate violently across the valley walls while making glacial progress along the base?

D. **Mathematics to Derive**
1. **Step 2: Tiny Numerical Example by Hand**:
   Consider optimizing the 2D function:
   $$f(\theta_1, \theta_2) = \theta_1^2 + 2\theta_2^2$$
   True analytical gradient:
   $$\nabla f(\boldsymbol{\theta}) = \begin{bmatrix} \frac{\partial f}{\partial \theta_1} \\ \frac{\partial f}{\partial \theta_2} \end{bmatrix} = \begin{bmatrix} 2\theta_1 \\ 4\theta_2 \end{bmatrix}$$
   Let initial parameters be $\boldsymbol{\theta}^{(0)} = [3.0, 4.0]^T$, and learning rate $\alpha = 0.1$.
   - **Step 1**:
     $\nabla f(\boldsymbol{\theta}^{(0)}) = [2(3.0), 4(4.0)]^T = [6.0, 16.0]^T$.
     Update:
     $$\boldsymbol{\theta}^{(1)} = \boldsymbol{\theta}^{(0)} - \alpha \nabla f = \begin{bmatrix} 3.0 - 0.1(6.0) \\ 4.0 - 0.1(16.0) \end{bmatrix} = \begin{bmatrix} 2.4 \\ 2.4 \end{bmatrix}$$
     Loss change: $f(3, 4) = 9 + 32 = 41.0 \implies f(2.4, 2.4) = 5.76 + 2(5.76) = 17.28$ (Dramatic decrease!).
   - **Step 2**:
     $\nabla f(\boldsymbol{\theta}^{(1)}) = [2(2.4), 4(2.4)]^T = [4.8, 9.6]^T$.
     Update:
     $$\boldsymbol{\theta}^{(2)} = \begin{bmatrix} 2.4 - 0.1(4.8) \\ 2.4 - 0.1(9.6) \end{bmatrix} = \begin{bmatrix} 1.92 \\ 1.44 \end{bmatrix}$$
     Loss change: $f(1.92, 1.44) = 3.6864 + 2(2.0736) = 7.8336$.
   Notice: $\theta_2$ converges faster than $\theta_1$ because its partial derivative is steeper!

2. **Step 3: Mathematical Notation**:
   - Total dataset: $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N$.
   - Parameters: $\boldsymbol{\theta} \in \mathbb{R}^P$ (or a collection of weight matrices and bias vectors).
   - Full loss objective:
     $$L(\boldsymbol{\theta}) = \frac{1}{N} \sum_{i=1}^N \ell(\mathbf{x}_i, y_i; \boldsymbol{\theta})$$
   - Mini-batch $\mathcal{B} \subset \mathcal{D}$ of size $B = |\mathcal{B}|$:
     $$\mathbf{g}_{\mathcal{B}}(\boldsymbol{\theta}) = \frac{1}{B} \sum_{i \in \mathcal{B}} \nabla_{\boldsymbol{\theta}} \ell(\mathbf{x}_i, y_i; \boldsymbol{\theta})$$

3. **Step 4: The SGD Update Rule & Convergence Criterion**:
   - Parameter update:
     $$\boldsymbol{\theta}^{(t+1)} = \boldsymbol{\theta}^{(t)} - \alpha \mathbf{g}_{\mathcal{B}}(\boldsymbol{\theta}^{(t)})$$
   - Robbins-Monro conditions for convergence under stochastic noise:
     $$\sum_{t=1}^{\infty} \alpha_t = \infty \quad \text{and} \quad \sum_{t=1}^{\infty} \alpha_t^2 < \infty$$
     (The step size must be large enough to traverse any distance, but decrease fast enough to quench stochastic noise).

E. **Implementation Contract**
- Class: `SGD`
- Constructor:
  - `__init__(self, learning_rate: float = 0.01)`:
    Stores the scalar step size.
- Attributes:
  - `lr`: float, the learning rate.
- Methods:
  - `step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None`:
    Updates each parameter array in `params` **in-place** using its corresponding gradient in `grads`:
    $$\theta \leftarrow \theta - \text{lr} \cdot \mathbf{g}$$
    Must raise `ValueError` if `len(params) != len(grads)` or if any parameter array shape does not match its gradient shape.
  - `zero_grad(self, grads: list[np.ndarray]) -> None`:
    Optional convenience helper to zero out arrays in `grads` in-place.

F. **Guided Implementation Stages**
**Step 5: Algorithm Design & Pseudocode**
```text
class SGD:
  def __init__(lr=0.01):
    self.lr = lr

  def step(params, grads):
    if len(params) != len(grads):
      raise ValueError("Mismatched params and grads length")
    for p, g in zip(params, grads):
      if p.shape != g.shape:
        raise ValueError("Mismatched tensor shape")
      p -= self.lr * g  # in-place modification
```

**Checkpoint 1: Parameter Verification and In-Place Updates**
- **What to learn**: Managing lists of parameter arrays and performing strictly in-place array modifications.
- **What to do**: Loop over zipped `(params, grads)`, validate shapes, and execute `p -= self.lr * g`.
- **How to check yourself**: Initialize `p = np.array([1.0, 2.0])`, pass `[p]` and `[np.array([0.5, -0.5])]` with `lr = 0.1`. Verify that `p` is modified directly to `[0.95, 2.05]`.
- **When to proceed**: In-place modification works cleanly across arbitrary 1D and 2D arrays.
- **Recovery hints**:
  - *Hint 1 (Concept)*: Reassigning `p = p - lr * g` creates a new local variable and breaks in-place modification of the caller's parameter! Use `p -= lr * g` or `p[:] = p - lr * g`.
  - *Hint 2 (Operation)*: `for p, g in zip(params, grads): p -= self.lr * g`.
  - *Hint 3 (Debugging)*: Check `p is original_p` to verify the array identity is preserved.

**Checkpoint 2: Optimizing the Quadratic Bowl**
- **What to learn**: Tracking parameter trajectories on convex loss functions.
- **What to do**: Minimize $f(\theta_1, \theta_2) = \theta_1^2 + 2\theta_2^2$ starting from $[3.0, 4.0]$ for 50 steps with $\alpha = 0.1$.
- **How to check yourself**: Parameters converge to $\approx [0.0, 0.0]$ with loss $< 10^{-4}$.
- **When to proceed**: Optimizer cleanly solves isotropic and well-conditioned quadratics.
- **Recovery hints**:
  - *Hint 1 (Concept)*: Compute the exact analytical gradients $\nabla f = [2\theta_1, 4\theta_2]$ at each step.
  - *Hint 2 (Operation)*: Call `optimizer.step([theta], [grad])` inside your loop.
  - *Hint 3 (Debugging)*: If values explode to `NaN`, your learning rate is too large ($\alpha > 2 / L_{\max}$).

**Checkpoint 3: The Ill-Conditioned Ravine (Dataset 6C)**
- **What to learn**: Observing oscillations in high-curvature directions.
- **What to do**: Test SGD on Dataset 6C ($L(w_1, w_2) = 0.1 w_1^2 + 10.0 w_2^2$) starting from $[10.0, 1.0]$.
- **How to check yourself**: If $\alpha = 0.15$, $w_2$ oscillates wildly and diverges ($10.0 \times 2 \times 0.15 = 3.0 > 2$). If $\alpha$ is reduced to $0.05$ to stabilize $w_2$, progress along $w_1$ slows to a crawl ($0.1 \times 2 \times 0.05 = 0.01$).
- **When to proceed**: Demonstrably document the trade-off that motivates Momentum!
- **Recovery hints**:
  - *Hint 1 (Concept)*: The condition number $\kappa = \frac{\lambda_{\max}}{\lambda_{\min}} = \frac{20}{0.2} = 100$.
  - *Hint 2 (Operation)*: Plot or print $(w_1, w_2)$ coordinates across 100 iterations.
  - *Hint 3 (Debugging)*: This limitation is intrinsic to vanilla SGD; it cannot adapt step sizes independently per coordinate.

G. **Edge Cases and Expected Tests** (Step 8: Test and Debug)
1. `test_in_place_mutation`: Confirm that parameters in the caller's scope are updated in-place (same object ID).
2. `test_shape_mismatch_error`: Confirm that passing tensors of different shapes raises `ValueError`.
3. `test_quadratic_bowl_convergence`: Verify convergence to within $10^{-3}$ of origin on $f(\theta) = \|\theta\|_2^2$.
4. `test_zero_gradient_stability`: Confirm that when gradients are identically zero, parameters remain unchanged.

H. **Complexity Analysis** (Step 9: Analyze the Algorithm)
- **Time Complexity**: $O(P)$ arithmetic operations per step, where $P = \sum_l |\boldsymbol{\theta}_l|$ is total parameter count.
- **Space Complexity**: $O(1)$ auxiliary memory (zero state storage required).
- **Trade-offs**: Minimal memory footprint and simplest implementation, but susceptible to oscillations on anisotropic surfaces and slow traversal through plateaus.

I. **Progressive Difficulty Levels & Definition of Done**
- **Level 1 (Guided)**: Implement `step` for a single 1D parameter array.
- **Level 2 (Standard)**: Implement `step` supporting arbitrary lists of multi-dimensional tensors with rigorous shape validation.
- **Level 3 (Challenge)**: Build a mini-batch iterator that shuffles dataset indices every epoch and optimizes the 2-layer MLP on XOR.
- **Definition of Done**: Correct in-place tensor mutations, passes all shape validation tests, and optimizes convex loss surfaces.

J. **Reflection & Self-Check** (Step 10: Independent Completion)
1. Why does shuffling dataset samples before partitioning into mini-batches each epoch prevent systematic bias in the gradient estimates?
2. What happens to the convergence of SGD if the learning rate is kept constant rather than gradually decayed according to the Robbins-Monro conditions?
"""
