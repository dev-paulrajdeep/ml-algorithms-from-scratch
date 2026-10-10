r"""
A. **Mission** (Step 0: Understand the Problem)
Ridge Regression applies $L_2$ Tikhonov regularization to the linear regression objective.
Real-world problem: When datasets possess correlated features (multicollinearity) or high feature counts relative to sample size ($D \approx N$ or $D > N$), ordinary least squares causes parameter variances to explode—tiny changes in training data produce wildly different weights with huge opposite-sign coefficients.
Ridge regression introduces an analytical weight penalty that shrinks parameters toward zero, dampens variance, and guarantees a unique, stable closed-form solution.

B. **Prerequisites**
- [Stage 0B: NumPy & Numerical Computing](../foundations/README.md#module-0b-numpy-and-numerical-computing) — Matrix inversion (`np.linalg.inv`), identity matrices (`np.eye`).
- [Stage 0C: Linear Algebra](../foundations/README.md#module-0c-linear-algebra) — Matrix invertibility, eigenvalues, full rank, and $L_2$ norms.
- [Stage 0D: Calculus & Optimization](../foundations/README.md#module-0d-calculus-and-optimization) — Vector derivatives of quadratic forms.
- [Stage 0F: Machine Learning Core](../foundations/README.md#module-0f-machine-learning-core-concepts) — Regularization, bias-variance tradeoff.
- [Polynomial Regression](./polynomial_regression.py) — Understanding the OLS objective before adding regularization penalties.

C. **Learning Questions** (Step 1: Build Intuition)
1. Why does penalizing the squared magnitude of weights ($\sum w_j^2$) prevent model overfitting?
2. Geometrically, why is the $L_2$ penalty constraint represented as a hypersphere (a circle in 2D)? How does this ensure weights are shrunken continuously rather than set to exact zeros?
3. Why must the intercept (bias term) NEVER be regularized? What happens to predictions if the intercept is penalized heavily toward zero?

D. **Mathematics to Derive**
1. **Step 2: Tiny Numerical Example by Hand**:
   Consider Dataset R3 with collinear features:
   $\mathbf{X} = \begin{bmatrix} 1 & 2 \\ 2 & 4 \end{bmatrix}$, $\mathbf{y} = \begin{bmatrix} 3 \\ 6 \end{bmatrix}$.
   - Compute $\mathbf{X}^T \mathbf{X} = \begin{bmatrix} 5 & 10 \\ 10 & 20 \end{bmatrix}$. Note that $\det(\mathbf{X}^T \mathbf{X}) = 5(20) - 10(10) = 0$. It is singular! OLS fails.
   - Now add $L_2$ penalty $\alpha = 1.0$: $\mathbf{X}^T \mathbf{X} + \alpha \mathbf{I} = \begin{bmatrix} 6 & 10 \\ 10 & 21 \end{bmatrix}$.
   - Hand-compute the determinant: $\det = 6(21) - 100 = 126 - 100 = 26 > 0$. The matrix is now strictly invertible!
2. **Step 3: Mathematical Notation**:
   - $N$: number of samples, $D$: number of features.
   - $\mathbf{X} \in \mathbb{R}^{N \times D}$: centered feature matrix (or augmented with bias column).
   - $\mathbf{y} \in \mathbb{R}^{N}$: target vector.
   - $\mathbf{w} \in \mathbb{R}^{D}$: weight vector.
   - $\alpha \ge 0$: regularization strength hyperparameter.
   - $\mathbf{I} \in \mathbb{R}^{D \times D}$: identity matrix.
3. **Step 4: Derive the Closed-Form Normal Equations**:
   - Write the penalized objective function:
     $$J(\mathbf{w}) = \frac{1}{2N} \|\mathbf{X}\mathbf{w} - \mathbf{y}\|_2^2 + \frac{\alpha}{2} \|\mathbf{w}\|_2^2$$
   - Expand the matrix quadratic terms:
     $$J(\mathbf{w}) = \frac{1}{2N} (\mathbf{w}^T \mathbf{X}^T \mathbf{X} \mathbf{w} - 2 \mathbf{y}^T \mathbf{X} \mathbf{w} + \mathbf{y}^T \mathbf{y}) + \frac{\alpha}{2} \mathbf{w}^T \mathbf{w}$$
   - Take the gradient with respect to $\mathbf{w}$ and set it to $\mathbf{0}$:
     $$\nabla_{\mathbf{w}} J = \frac{1}{N} (\mathbf{X}^T \mathbf{X} \mathbf{w} - \mathbf{X}^T \mathbf{y}) + \alpha \mathbf{w} = \mathbf{0}$$
     $$\left(\frac{1}{N} \mathbf{X}^T \mathbf{X} + \alpha \mathbf{I}\right) \mathbf{w} = \frac{1}{N} \mathbf{X}^T \mathbf{y}$$
     $$\mathbf{w} = (\mathbf{X}^T \mathbf{X} + N \alpha \mathbf{I})^{-1} \mathbf{X}^T \mathbf{y}$$
   - *Theorem*: Since $\mathbf{X}^T \mathbf{X}$ is positive semi-definite (all eigenvalues $\ge 0$), adding $\lambda \mathbf{I}$ shifts every eigenvalue by $\lambda > 0$. Therefore, all eigenvalues become strictly positive, guaranteeing invertibility!

E. **Implementation Contract**
- Class: `RidgeRegression`
- Methods:
  - `__init__(self, alpha: float = 1.0, fit_intercept: bool = True)`:
    Stores hyperparameters. Validates that `alpha >= 0.0`.
  - `fit(self, X: np.ndarray, y: np.ndarray) -> self`:
    Computes analytical weights via closed-form solution. Stores `weights_` of shape `(D,)` and `intercept_` as float.
  - `predict(self, X: np.ndarray) -> np.ndarray`:
    Computes $\hat{\mathbf{y}} = \mathbf{X}\mathbf{w} + b$. Returns shape `(N,)`.

F. **Guided Implementation Stages**
**Step 5: Design the Algorithm (Pseudocode)**
```text
fit(X, y):
  If fit_intercept:
    Compute mean_X = mean(X, axis=0)
    Compute mean_y = mean(y)
    Center data: X_c = X - mean_X, y_c = y - mean_y
  Else:
    X_c, y_c = X, y

  Compute A = X_c.T @ X_c + alpha * I_D
  Solve for w: w = np.linalg.solve(A, X_c.T @ y_c)
  weights_ = w

  If fit_intercept:
    intercept_ = mean_y - mean_X @ w
  Else:
    intercept_ = 0.0
```

**Checkpoint 1: Unpenalized Intercept Handling**
- **What to learn**: Why centering data cleanly separates the intercept from regularized slopes.
- **What to do**: Subtract feature means and target mean before forming scatter matrices.
- **How to check yourself**: Ensure `X_c.mean(axis=0)` is $\approx 0$ and `y_c.mean()` is $\approx 0$.
- **When to proceed**: Centering logic works without altering the original input array in-place.
- **Recovery hints**:
  - *Hint 1 (Concept)*: Centering projects data to the origin, where the regression line passes through $(0, 0)$.
  - *Hint 2 (Operation)*: Use `X - np.mean(X, axis=0)` with NumPy broadcasting.
  - *Hint 3 (Debugging)*: Never mutate `X` directly; create `X_c = X - ...`.

**Checkpoint 2: Regularized Matrix Equation**
- **What to learn**: Constructing $(\mathbf{X}^T \mathbf{X} + \alpha \mathbf{I})$ and solving for weights.
- **What to do**: Construct the identity matrix with `np.eye(D)` and solve using `np.linalg.solve`.
- **How to check yourself**: When `alpha = 0`, solutions match ordinary least squares exactly on non-singular data.
- **When to proceed**: The system solves reliably for both $\alpha = 0$ and $\alpha > 0$.
- **Recovery hints**:
  - *Hint 1 (Concept)*: Prefer `np.linalg.solve(A, b)` over `np.linalg.inv(A) @ b` for numerical stability.
  - *Hint 2 (Operation)*: `A = X_c.T @ X_c + alpha * np.eye(D)`.
  - *Hint 3 (Debugging)*: Ensure dimension of `np.eye` matches `X.shape[1]`.

**Checkpoint 3: Recovering the Intercept & Prediction**
- **What to learn**: Reconstructing the uncentered bias term $b = \bar{y} - \bar{\mathbf{x}}^T \mathbf{w}$.
- **What to do**: Set `self.intercept_ = mean_y - mean_X @ self.weights_`. In `predict`, compute `X @ self.weights_ + self.intercept_`.
- **How to check yourself**: On Dataset R1 with $\alpha = 0$, prediction matches $y = 2x$.
- **When to proceed**: Predictions on test sets yield correct dimensions and values.
- **Recovery hints**:
  - *Hint 1 (Concept)*: The intercept is the baseline prediction when all features are zero.
  - *Hint 2 (Operation)*: Matrix-vector product `mean_X @ w` yields a scalar.
  - *Hint 3 (Debugging)*: Check that `predict(X)` returns shape `(N,)` and not `(N, 1)`.

G. **Edge Cases and Expected Tests** (Step 8: Test and Debug)
1. `test_alpha_zero_matches_ols`: On standard non-collinear regression data, setting `alpha = 0.0` recovers ordinary least squares.
2. `test_extreme_alpha_shrinkage`: As `alpha -> 1e6`, all weights shrink toward 0.0, and predictions approach the training mean $\bar{y}$.
3. `test_collinear_features_stability`: On Dataset R3 (where $x_2 = 2 x_1$), OLS crashes or yields huge arbitrary weights, whereas Ridge with $\alpha = 1.0$ produces stable, finite, symmetric weights.
4. `test_single_feature_and_single_sample`: Verify execution on minimal array shapes without dimensional errors.

H. **Complexity Analysis** (Step 9: Analyze the Algorithm)
- **Time Complexity**:
  - Gram matrix computation $\mathbf{X}^T \mathbf{X}$: $O(N \cdot D^2)$ time.
  - Matrix solve / inversion: $O(D^3)$ time.
  - Total fitting time: $O(N D^2 + D^3)$.
  - Inference time: $O(N_{\text{test}} \cdot D)$.
- **Space Complexity**:
  - Auxiliary matrix storage: $O(D^2)$ for the Gram matrix and regularizer.
  - Model parameters: $O(D)$ for weights.
- **Trade-offs**: Closed-form Ridge is extremely fast when $D \le 1000$, but cubic $O(D^3)$ inversion becomes intractable when $D \ge 100,000$ (where iterative gradient descent is required).

I. **Progressive Difficulty Levels & Definition of Done**
- **Level 1 (Guided)**: Implement closed-form solution with data centering for 2D collinear data (Dataset R3).
- **Level 2 (Standard)**: Implement full `RidgeRegression` class using `np.linalg.solve`, supporting `fit_intercept`, input validation, and unit tests.
- **Level 3 (Challenge)**: Implement Ridge regression using Singular Value Decomposition (`np.linalg.svd`), expressing the shrinkage factors directly in terms of singular values $\frac{\sigma_i^2}{\sigma_i^2 + \alpha}$.
- **Definition of Done**: Passes all tests, guarantees invertibility on singular inputs, and correctly shrinks weights as $\alpha$ increases.

J. **Reflection & Self-Check** (Step 10: Independent Completion)
1. How does the condition number of $(\mathbf{X}^T \mathbf{X} + \alpha \mathbf{I})$ change as $\alpha$ increases from $0$ to large values?
2. If two features are identical ($x_1 = x_2$), how does Ridge distribute weights between them compared to Lasso?
"""
