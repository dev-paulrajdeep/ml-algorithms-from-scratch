r"""
A. **Mission** (Step 0: Understand the Problem)
Lasso Regression (Least Absolute Shrinkage and Selection Operator) applies $L_1$ regularization to the mean squared error objective.
Real-world problem: In high-dimensional data (e.g., genomics with 20,000 genes or text with 50,000 words), most features are uninformative noise. Ordinary least squares and Ridge assign small non-zero weights to every single feature, resulting in dense, uninterpretable models.
Lasso drives irrelevant feature weights to **exactly zero**, performing automatic feature selection and producing compact, sparse models.

B. **Prerequisites**
- [Stage 0B: NumPy & Numerical Computing](../foundations/README.md#module-0b-numpy-and-numerical-computing) — Array masking and coordinate iteration.
- [Stage 0C: Linear Algebra](../foundations/README.md#module-0c-linear-algebra) — $L_1$ vector norm ($\|\mathbf{w}\|_1 = \sum |w_j|$).
- [Stage 0D: Calculus & Optimization](../foundations/README.md#module-0d-calculus-and-optimization) — Non-differentiability at zero and subgradient calculus.
- [Stage 0F: Machine Learning Core](../foundations/README.md#module-0f-machine-learning-core-concepts) — Feature selection and sparsity.
- [Ridge Regression](./ridge_regression.py) — Understanding $L_2$ regularization to contrast with $L_1$.

C. **Learning Questions** (Step 1: Build Intuition)
1. Why does the geometric constraint region of the $L_1$ norm (a diamond in 2D, octahedron in 3D) have sharp corners on coordinate axes? Why do loss contours hit these corners, setting weights to exactly zero?
2. Why is there NO closed-form analytical solution for Lasso like there is for Ridge?
3. What is the soft-thresholding operator, and how does coordinate descent solve non-differentiable $L_1$ optimization without oscillating?

D. **Mathematics to Derive**
1. **Step 2: Tiny Numerical Example by Hand**:
   Consider 1 feature with no noise: $x = [1, 2, 3]$, $y = [2, 4, 6]$ (true slope is $2.0$).
   - Standard OLS slope estimate is $\hat{w} = 2.0$.
   - Suppose the soft-thresholding operator is $S(\hat{w}, \lambda) = \text{sign}(\hat{w}) \max(0, |\hat{w}| - \lambda)$.
   - For penalty $\lambda = 0.5$: $S(2.0, 0.5) = +1 \cdot \max(0, 1.5) = 1.5$ (shrunk toward 0).
   - For penalty $\lambda = 2.5$: $S(2.0, 2.5) = +1 \cdot \max(0, -0.5) = 0.0$ (shrunk to EXACT ZERO).
2. **Step 3: Mathematical Notation**:
   - $N$: number of samples, $D$: number of features.
   - $\mathbf{X} \in \mathbb{R}^{N \times D}$, $\mathbf{y} \in \mathbb{R}^{N}$.
   - $\mathbf{w} \in \mathbb{R}^{D}$: weight vector.
   - $\alpha \ge 0$: $L_1$ penalty hyperparameter.
   - $\text{sign}(u)$: $+1$ if $u > 0$, $-1$ if $u < 0$, and $[-1, 1]$ if $u = 0$.
3. **Step 4: Derive the Coordinate Descent Update & Soft-Thresholding**:
   - Write the $L_1$-penalized objective function:
     $$J(\mathbf{w}, b) = \frac{1}{2N} \sum_{i=1}^N \left( \sum_{j=1}^D x_{ij} w_j + b - y_i \right)^2 + \alpha \sum_{j=1}^D |w_j|$$
   - In coordinate descent, we optimize one weight $w_k$ while holding all other weights $w_{j \neq k}$ fixed.
   - Define the partial residual for sample $i$ without feature $k$:
     $$r_{i}^{(k)} = y_i - b - \sum_{j \neq k} x_{ij} w_j$$
   - The scalar objective for $w_k$ becomes:
     $$J(w_k) = \frac{1}{2N} \sum_{i=1}^N (r_i^{(k)} - x_{ik} w_k)^2 + \alpha |w_k|$$
   - If features are normalized such that $\frac{1}{N} \sum_{i=1}^N x_{ik}^2 = 1$, setting the subgradient to 0 yields:
     $$\rho_k = \frac{1}{N} \sum_{i=1}^N x_{ik} r_i^{(k)}$$
   - The exact closed-form scalar minimum is given by the **Soft-Thresholding Operator**:
     $$w_k = S(\rho_k, \alpha) = \begin{cases} \rho_k - \alpha & \text{if } \rho_k > \alpha \\ 0 & \text{if } |\rho_k| \le \alpha \\ \rho_k + \alpha & \text{if } \rho_k < -\alpha \end{cases}$$

E. **Implementation Contract**
- Class: `LassoRegression`
- Methods:
  - `__init__(self, alpha: float = 1.0, max_iter: int = 1000, tol: float = 1e-4)`:
    Stores hyperparameters. Validates `alpha >= 0.0`.
  - `fit(self, X: np.ndarray, y: np.ndarray) -> self`:
    Fits weights using Coordinate Descent. Learns `weights_` of shape `(D,)` and `intercept_` float. Continues until max_iter or max weight change $< tol$.
  - `predict(self, X: np.ndarray) -> np.ndarray`:
    Computes $\hat{\mathbf{y}} = \mathbf{X}\mathbf{w} + b$. Returns shape `(N,)`.

F. **Guided Implementation Stages**
**Step 5: Design the Algorithm (Pseudocode)**
```text
fit(X, y):
  Compute mean_X = mean(X, 0), mean_y = mean(y)
  X_c = X - mean_X
  y_c = y - mean_y
  Compute column norms: norm_sq = sum(X_c**2, axis=0) / N

  Initialize weights = zeros(D)
  For iteration in range(max_iter):
    max_change = 0.0
    For k in range(D):
      # Compute partial residual: y_c - X_c @ weights + X_c[:, k] * weights[k]
      residual_k = y_c - (X_c @ weights - X_c[:, k] * weights[k])
      rho_k = (X_c[:, k] @ residual_k) / N
      new_w_k = soft_threshold(rho_k, alpha) / norm_sq[k]
      max_change = max(max_change, abs(new_w_k - weights[k]))
      weights[k] = new_w_k
    If max_change < tol:
      break
  weights_ = weights
  intercept_ = mean_y - mean_X @ weights_
```

**Checkpoint 1: Soft-Thresholding Operator**
- **What to learn**: Implementing the piecewise shrinkage function.
- **What to do**: Implement `_soft_threshold(rho, alpha)`.
- **How to check yourself**: `_soft_threshold(2.0, 0.5) == 1.5`, `_soft_threshold(0.3, 0.5) == 0.0`, `_soft_threshold(-2.0, 0.5) == -1.5`.
- **When to proceed**: Function satisfies all three piecewise regimes.
- **Recovery hints**:
  - *Hint 1 (Concept)*: Use `np.sign(rho) * max(0, abs(rho) - alpha)`.
  - *Hint 2 (Operation)*: Ensure scalar and array inputs are handled without syntax errors.
  - *Hint 3 (Debugging)*: If $\rho = 0$, the output must be exactly $0.0$.

**Checkpoint 2: Coordinate Descent Loop**
- **What to learn**: Cyclic coordinate optimization across all features.
- **What to do**: Loop over each feature column $k$, compute partial residual, and update $w_k$ in-place.
- **How to check yourself**: On a dataset with 2 informative features and 2 pure noise features, the noise features should become $0.0$ within a few iterations.
- **When to proceed**: The loop converges and decreases the objective monotonically.
- **Recovery hints**:
  - *Hint 1 (Concept)*: Do not recompute the full matrix product `X @ w` from scratch inside the inner loop; update residuals efficiently.
  - *Hint 2 (Operation)*: Remember to divide by the column variance `norm_sq[k]`.
  - *Hint 3 (Debugging)*: Ensure the intercept is computed from uncentered means after weights converge.

**Checkpoint 3: Sparsity Verification**
- **What to learn**: How increasing $\alpha$ induces exact zeros.
- **What to do**: Fit models with increasing $\alpha \in [0.01, 0.1, 1.0, 10.0]$ and count `np.count_nonzero(model.weights_)`.
- **How to check yourself**: The number of non-zero weights must be non-increasing with respect to $\alpha$.
- **When to proceed**: High $\alpha$ drives non-bias weights to exactly zero.
- **Recovery hints**:
  - *Hint 1 (Concept)*: If weights oscillate near zero without hitting exact zero, you may be using subgradient descent instead of coordinate descent.
  - *Hint 2 (Operation)*: Verify using `np.isclose(weights, 0.0)`.
  - *Hint 3 (Debugging)*: Ensure $\alpha$ is scaled appropriately relative to sample size $N$.

G. **Edge Cases and Expected Tests** (Step 8: Test and Debug)
1. `test_alpha_zero_matches_ols`: When $\alpha = 0$, coordinate descent converges to the ordinary least squares solution.
2. `test_feature_selection_sparsity`: On a dataset with 5 informative features and 15 noise features (random Gaussian columns), Lasso with moderate $\alpha$ sets all 15 noise features to exactly $0.0$.
3. `test_large_alpha_all_zeros`: When $\alpha$ is sufficiently large ($\alpha \ge \max_k |\frac{1}{N} \mathbf{x}_k^T \mathbf{y}|$), ALL feature weights must be exactly $0.0$, and the model must predict the mean $\bar{y}$.
4. `test_intercept_unregularized`: Verify that even with massive $\alpha$, `intercept_` remains equal to $\bar{y}$ and is never shrunken to zero.

H. **Complexity Analysis** (Step 9: Analyze the Algorithm)
- **Time Complexity**:
  - Per coordinate step: $O(N)$ time.
  - Per full sweep over $D$ features: $O(N \cdot D)$ time.
  - Total fitting time: $O(\text{iterations} \cdot N \cdot D)$.
  - Inference time: $O(N_{\text{test}} \cdot D)$ or $O(N_{\text{test}} \cdot \|\mathbf{w}\|_0)$ utilizing sparse dot products.
- **Space Complexity**: $O(D)$ auxiliary memory for weights and column norms.
- **Trade-offs**: Coordinate descent is remarkably fast for sparse models, but lacks a single-step closed-form solution.

I. **Progressive Difficulty Levels & Definition of Done**
- **Level 1 (Guided)**: Implement subgradient descent with learning rate and sign function on a toy 2-feature dataset.
- **Level 2 (Standard)**: Implement full cyclic Coordinate Descent with soft-thresholding, convergence tolerance, data centering, and unit tests.
- **Level 3 (Challenge)**: Implement pathwise coordinate descent (computing the entire regularization path for a decreasing sequence of $\alpha$ values, using warm starts from the previous $\alpha$ solution).
- **Definition of Done**: Demonstrates exact feature sparsity (zeros), passes all edge tests, and converges faster than subgradient descent.

J. **Reflection & Self-Check** (Step 10: Independent Completion)
1. Why does subgradient descent oscillate around zero rather than achieving exact sparsity, while coordinate descent hits exact zero easily?
2. When multiple features are highly correlated, how does Lasso choose between them, and why does this differ from Ridge?
"""
