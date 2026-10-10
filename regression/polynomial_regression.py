r"""
A. **Mission** (Step 0: Understand the Problem)
Polynomial Regression extends linear models to capture non-linear physical and mathematical relationships by mapping inputs into higher-degree polynomial feature spaces.
Real-world context: A projectile's height over time, temperature variations over seasons, or stopping distances vs. vehicle speed follow curved paths. A straight line underfits, but higher polynomial degrees risk wild oscillations.
Your objective is to build a polynomial transformer and linear solver from scratch to observe the bias-variance tradeoff firsthand.

B. **Prerequisites**
- [Stage 0B: NumPy & Numerical Computing](../foundations/README.md#module-0b-numpy-and-numerical-computing) — Column stacking, elementwise powers, and broadcasting.
- [Stage 0C: Linear Algebra](../foundations/README.md#module-0c-linear-algebra) — Matrix multiplication, Vandermonde matrices.
- [Stage 0D: Calculus & Optimization](../foundations/README.md#module-0d-calculus-and-optimization) — Gradient descent and loss minimization.
- [Stage 0F: Machine Learning Core](../foundations/README.md#module-0f-machine-learning-core-concepts) — Underfitting, overfitting, and the bias-variance tradeoff.
- [Linear Regression](./linear_regression.py) — Understanding gradient descent on raw scalar inputs.

C. **Learning Questions** (Step 1: Build Intuition)
1. Why is polynomial regression still considered a "linear model", even though its predictions form curves?
2. What happens to the training error as you increase the polynomial degree from $d=1$ to $d=15$ on a 5-point dataset? What happens to the test error on unseen points?
3. Why do features like $x^{10}$ quickly cause numerical overflow or explosion during gradient descent if inputs are not scaled?

D. **Mathematics to Derive**
1. **Step 2: Tiny Numerical Example by Hand**:
   Take Dataset R2: $x = [1.0, 2.0, 3.0]$, $y = [1.0, 4.0, 9.0]$.
   For degree $d = 2$, expand each scalar $x_i$ into the row vector $[x_i^1, x_i^2]$.
   - Hand-compute the feature row for $x = 2.0$: $[2.0, 4.0]$.
   - If weights are $\mathbf{w} = [0.0, 1.0]^T$ and bias is $b = 0.0$, hand-compute predictions for all three points. Verify that MSE $= 0.0$.
2. **Step 3: Mathematical Notation**:
   - $N$: number of samples.
   - $D_{\text{in}}$: number of input features (here $D_{\text{in}} = 1$ for scalar inputs).
   - $d$: polynomial degree.
   - $\mathbf{X} \in \mathbb{R}^{N \times 1}$: raw feature matrix.
   - $\mathbf{\Phi}(\mathbf{X}) \in \mathbb{R}^{N \times d}$: transformed polynomial feature matrix where column $j$ contains $x^j$.
   - $\mathbf{w} \in \mathbb{R}^{d}$: learned weight vector.
   - $b \in \mathbb{R}$: learned bias scalar.
   - $\hat{\mathbf{y}} = \mathbf{\Phi}(\mathbf{X})\mathbf{w} + b$: model predictions.
3. **Step 4: Derive the Mathematics**:
   - Write the Mean Squared Error objective:
     $$J(\mathbf{w}, b) = \frac{1}{2N} \sum_{i=1}^N (\hat{y}_i - y_i)^2 = \frac{1}{2N} \|\mathbf{\Phi}(\mathbf{X})\mathbf{w} + b\mathbf{1} - \mathbf{y}\|_2^2$$
   - Derive the partial derivatives using the multivariate chain rule:
     $$\nabla_{\mathbf{w}} J = \frac{1}{N} \mathbf{\Phi}(\mathbf{X})^T (\hat{\mathbf{y}} - \mathbf{y})$$
     $$\frac{\partial J}{\partial b} = \frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i)$$

E. **Implementation Contract**
- Class: `PolynomialRegression`
- Methods:
  - `__init__(self, degree: int = 2, learning_rate: float = 0.01, epochs: int = 1000)`:
    Stores hyperparameters. Validates that `degree >= 1`.
  - `transform(self, X: np.ndarray) -> np.ndarray`:
    Expands input matrix `X` of shape `(N, 1)` into polynomial features of shape `(N, degree)`.
  - `fit(self, X: np.ndarray, y: np.ndarray) -> self`:
    Transforms `X`, initializes `weights_` of shape `(degree,)` to zeros and `bias_` to 0.0. Runs batch gradient descent for `epochs`. Tracks `loss_history_`.
  - `predict(self, X: np.ndarray) -> np.ndarray`:
    Transforms `X` and computes $\hat{\mathbf{y}} = \mathbf{\Phi}(\mathbf{X})\mathbf{w} + b$. Returns 1D array of shape `(N,)`.

F. **Guided Implementation Stages**
**Step 5: Design the Algorithm (Pseudocode)**
```text
transform(X):
  For each power p from 1 to degree:
    compute column X ** p
  horizontally stack columns into Phi
  return Phi

fit(X, y):
  Phi = transform(X)
  weights = zeros(degree)
  bias = 0.0
  For epoch in range(epochs):
    y_hat = Phi @ weights + bias
    error = y_hat - y
    grad_w = (Phi.T @ error) / N
    grad_b = sum(error) / N
    weights -= learning_rate * grad_w
    bias -= learning_rate * grad_b
```

**Checkpoint 1: Polynomial Transformation (`transform`)**
- **What to learn**: Constructing polynomial feature matrices from raw vectors.
- **What to do**: Implement `transform(X)` using a list comprehension or `np.hstack([X**p for p in range(1, degree + 1)])`.
- **How to check yourself**: For `X = np.array([[2.0], [3.0]])` and `degree = 3`, verify output shape is `(2, 3)` and rows are `[[2, 4, 8], [3, 9, 27]]`.
- **When to proceed**: Transformed array shapes and values match hand-calculated powers exactly.
- **Recovery hints**:
  - *Hint 1 (Concept)*: Polynomial powers operate elementwise on array coordinates.
  - *Hint 2 (Operation)*: Use `np.column_stack` or `np.hstack` over a list of powered arrays.
  - *Hint 3 (Debugging)*: If `X.ndim == 1`, reshape it to `(N, 1)` before transforming.

**Checkpoint 2: Forward Pass & Scalar Residuals**
- **What to learn**: Matrix prediction on transformed feature matrices.
- **What to do**: Compute `y_hat = Phi @ weights + bias` and verify prediction shapes match `(N,)`.
- **How to check yourself**: On Dataset R2 with weights `[0, 1]` and bias `0`, predictions equal `[1, 4, 9]`.
- **When to proceed**: Predictions on known weights match ground truth with zero residual.
- **Recovery hints**:
  - *Hint 1 (Concept)*: The matrix multiplication `Phi @ weights` produces an `(N,)` vector when weights is 1D.
  - *Hint 2 (Operation)*: Avoid shape mismatches between `(N, 1)` and `(N,)` by flattening `y`.
  - *Hint 3 (Debugging)*: Use `assert y_hat.shape == y.shape` to catch broadcasting bugs early.

**Checkpoint 3: Gradient Descent Updates**
- **What to learn**: Vectorized gradient calculation for transformed features.
- **What to do**: Implement the training loop in `fit(X, y)`. Record MSE loss per epoch in `loss_history_`.
- **How to check yourself**: On Dataset R1 with `degree = 1`, loss should drop monotonically and weights should converge to $w_1 \approx 2.0, b \approx 0.0$.
- **When to proceed**: Loss decreases steadily on every epoch without NaN or divergence.
- **Recovery hints**:
  - *Hint 1 (Concept)*: The gradient formula is identical to multivariate linear regression on `Phi`.
  - *Hint 2 (Operation)*: `grad_w = (Phi.T @ error) / N`.
  - *Hint 3 (Debugging)*: If loss becomes `NaN` or `inf`, decrease `learning_rate` by a factor of 10.

G. **Edge Cases and Expected Tests** (Step 8: Test and Debug)
1. `test_degree_1_matches_ols`: When `degree = 1`, the learned weight and bias must match standard linear regression on linear data.
2. `test_quadratic_perfect_fit`: On $y = x^2$ (Dataset R2), a degree-2 model must reach MSE $< 10^{-3}$.
3. `test_overfitting_on_few_points`: Train degree 10 on 5 noisy points. Show that training loss approaches zero, while prediction between points oscillates wildly.
4. `test_input_shape_robustness`: Ensure the class accepts both 1D arrays `(N,)` and 2D arrays `(N, 1)`.

H. **Complexity Analysis** (Step 9: Analyze the Algorithm)
- **Time Complexity**:
  - Feature transformation: $O(N \cdot d)$ time.
  - Fit: $O(\text{epochs} \cdot N \cdot d)$ time.
  - Predict: $O(N_{\text{test}} \cdot d)$ time.
- **Space Complexity**:
  - Feature storage: $O(N \cdot d)$ auxiliary memory for `Phi`.
  - Model parameters: $O(d)$ for weights and bias.
- **Trade-offs**: As degree $d$ increases, expressiveness grows, but the condition number of $\mathbf{\Phi}^T \mathbf{\Phi}$ explodes exponentially, making gradient descent ill-conditioned.

I. **Progressive Difficulty Levels & Definition of Done**
- **Level 1 (Guided)**: Implement `transform` and `fit` specifically for `degree = 2` on Dataset R2.
- **Level 2 (Standard)**: Implement full class with arbitrary `degree`, proper parameter initialization, loss history tracking, and input shape validation.
- **Level 3 (Challenge)**: Add internal feature standardization (Z-score normalization of polynomial features) inside `fit` to prevent exploding gradients at degree $d \ge 5$ without changing the public API.
- **Definition of Done**: Passes all tests, correctly fits quadratic curves, and demonstrates overfitting at high degrees.

J. **Reflection & Self-Check** (Step 10: Independent Completion)
1. Why does high-degree polynomial regression exhibit Runge's phenomenon (wild oscillations near the endpoints of the data interval)?
2. How does standardizing features before computing powers differ from standardizing the powers after computing them?
"""
