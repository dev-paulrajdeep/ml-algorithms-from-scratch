r"""
A. **Mission** (Step 0: Understand the Problem)
Logistic Regression is a foundational discriminative model for binary classification.
Real-world problem: Predicting whether an email is spam ($y=1$) or ham ($y=0$), whether a loan applicant will default, or whether a tumor is malignant.
Instead of predicting an unbounded continuous number as in linear regression, Logistic Regression models the posterior probability $P(y=1|\mathbf{x}) \in [0, 1]$ using a logistic sigmoid transformation, optimized by minimizing Binary Cross-Entropy (Negative Log-Likelihood) via gradient descent.

B. **Prerequisites**
- [Stage 0B: NumPy & Numerical Computing](../foundations/README.md#module-0b-numpy-and-numerical-computing) — Array operations, broadcasting, and numerical clipping (`np.clip`).
- [Stage 0C: Linear Algebra](../foundations/README.md#module-0c-linear-algebra) — Dot products and hyperplanes ($\mathbf{w}^T \mathbf{x} + b = 0$).
- [Stage 0D: Calculus & Optimization](../foundations/README.md#module-0d-calculus-and-optimization) — Gradient descent and the multivariable chain rule.
- [Stage 0E: Probability & Statistics](../foundations/README.md#module-0e-probability-and-statistics) — Bernoulli distribution, likelihood, and log-likelihood.
- [Chapter 1: Regression](../regression/README.md) — Gradient descent mechanics and parameter updates.

C. **Learning Questions** (Step 1: Build Intuition)
1. Why does applying linear regression with Mean Squared Error directly to binary labels $\{0, 1\}$ fail catastrophically when outliers are present?
2. How does the sigmoid function squash any real score $z \in (-\infty, +\infty)$ into a valid probability $p \in (0, 1)$? What happens at $z = 0$?
3. Why does Binary Cross-Entropy penalize confident incorrect predictions (e.g., predicting $p=0.999$ when true $y=0$) infinitely more harshly than MSE?

D. **Mathematics to Derive**
1. **Step 2: Tiny Numerical Example by Hand**:
   Consider 2 samples in 1D: $x_1 = 1.0, y_1 = 0$ and $x_2 = 3.0, y_2 = 1$.
   Suppose current parameters are $w = 1.0, b = -2.0$.
   - Sample 1: $z_1 = 1(1) - 2 = -1.0$. Probability $p_1 = \sigma(-1.0) = \frac{1}{1 + e^1} \approx \frac{1}{1 + 2.718} \approx 0.269$.
   - Sample 2: $z_2 = 1(3) - 2 = +1.0$. Probability $p_2 = \sigma(+1.0) = \frac{1}{1 + e^{-1}} \approx 0.731$.
   - Errors: $e_1 = p_1 - y_1 = 0.269 - 0 = +0.269$; $e_2 = p_2 - y_2 = 0.731 - 1 = -0.269$.
   - Gradients:
     $\frac{\partial J}{\partial w} = \frac{1}{2} (e_1 x_1 + e_2 x_2) = \frac{1}{2}(0.269(1) - 0.269(3)) = \frac{-0.538}{2} = -0.269$.
     Since gradient is negative, $w$ will increase, making the decision boundary sharper!
2. **Step 3: Mathematical Notation**:
   - $\mathbf{X} \in \mathbb{R}^{N \times D}$, $\mathbf{y} \in \{0, 1\}^N$.
   - $\mathbf{w} \in \mathbb{R}^D$: weights, $b \in \mathbb{R}$: bias.
   - $z = \mathbf{w}^T \mathbf{x} + b$: raw log-odds (logit).
   - $\sigma(z) = \frac{1}{1 + e^{-z}}$: sigmoid squashing function.
   - $p = P(y=1|\mathbf{x}) = \sigma(z)$.
3. **Step 4: Derive the Cross-Entropy Gradient**:
   - Show that the derivative of the sigmoid is:
     $$\sigma'(z) = \sigma(z)(1 - \sigma(z))$$
   - Write the likelihood for independent Bernoulli observations:
     $$L(\mathbf{w}, b) = \prod_{i=1}^N p_i^{y_i} (1 - p_i)^{1 - y_i}$$
   - Take the negative log-likelihood (Binary Cross-Entropy Loss):
     $$J(\mathbf{w}, b) = -\frac{1}{N} \sum_{i=1}^N [y_i \ln(p_i) + (1 - y_i) \ln(1 - p_i)]$$
   - Apply the chain rule $\frac{\partial J}{\partial w_j} = \frac{\partial J}{\partial p} \frac{\partial p}{\partial z} \frac{\partial z}{\partial w_j}$:
     $$\nabla_{\mathbf{w}} J = \frac{1}{N} \mathbf{X}^T (\mathbf{p} - \mathbf{y})$$
     $$\frac{\partial J}{\partial b} = \frac{1}{N} \sum_{i=1}^N (p_i - y_i)$$
   - Notice the astonishing elegance: the gradient formula has the exact same algebraic form as linear regression, but with probabilities $\mathbf{p}$ replacing predictions $\hat{\mathbf{y}}$!

E. **Implementation Contract**
- Class: `LogisticRegression`
- Methods:
  - `__init__(self, learning_rate: float = 0.01, epochs: int = 1000)`:
    Stores hyperparameters.
  - `fit(self, X: np.ndarray, y: np.ndarray) -> self`:
    Initializes `weights_` of shape `(D,)` to zeros and `bias_` to 0.0. Runs batch gradient descent for `epochs`. Tracks `loss_history_`.
  - `predict_proba(self, X: np.ndarray) -> np.ndarray`:
    Computes probabilities $P(y=1|\mathbf{x}) = \sigma(\mathbf{X}\mathbf{w} + b)$. Returns 1D array of shape `(N,)` with values in $[0, 1]$.
  - `predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray`:
    Returns binary labels $0$ or $1$ based on `predict_proba(X) >= threshold`. Returns shape `(N,)` of type `int`.

F. **Guided Implementation Stages**
**Step 5: Design the Algorithm (Pseudocode)**
```text
sigmoid(z):
  clip z to [-500, 500] to prevent overflow
  return 1 / (1 + exp(-z))

fit(X, y):
  weights = zeros(D), bias = 0.0
  For epoch in range(epochs):
    z = X @ weights + bias
    p = sigmoid(z)
    loss = -mean(y * log(p + eps) + (1 - y) * log(1 - p + eps))
    dw = (X.T @ (p - y)) / N
    db = mean(p - y)
    weights -= learning_rate * dw
    bias -= learning_rate * db
```

**Checkpoint 1: Numerically Stable Sigmoid**
- **What to learn**: Preventing floating-point overflow (`RuntimeWarning: overflow in exp`).
- **What to do**: Implement `_sigmoid(z)`. Clip $z$ between $[-250, 250]$ or use piecewise formulation:
  $\sigma(z) = \frac{1}{1 + e^{-z}}$ for $z \ge 0$, and $\frac{e^z}{1 + e^z}$ for $z < 0$.
- **How to check yourself**: `_sigmoid(0.0) == 0.5`, `_sigmoid(100.0) == 1.0`, `_sigmoid(-100.0) == 0.0`.
- **When to proceed**: Function runs without warnings on extreme values like $\pm 1000$.
- **Recovery hints**:
  - *Hint 1 (Concept)*: $e^{709}$ is the largest representable 64-bit float in IEEE 754.
  - *Hint 2 (Operation)*: Use `np.clip(z, -250, 250)` before calling `np.exp(-z)`.
  - *Hint 3 (Debugging)*: Check scalar and array inputs.

**Checkpoint 2: Cross-Entropy Loss Calculation**
- **What to learn**: Safe computation of log probabilities.
- **What to do**: Compute loss $J = -\frac{1}{N} \sum [y \ln(p) + (1-y) \ln(1-p)]$.
- **How to check yourself**: If true labels match predictions perfectly ($p \to y$), loss approaches $0.0$.
- **When to proceed**: Loss is non-negative and finite for all training iterations.
- **Recovery hints**:
  - *Hint 1 (Concept)*: $\ln(0)$ is $-\infty$, producing `NaN` when multiplied by 0.
  - *Hint 2 (Operation)*: Clip probabilities using `p = np.clip(p, 1e-15, 1 - 1e-15)`.
  - *Hint 3 (Debugging)*: Use `np.mean` across samples.

**Checkpoint 3: Gradient Descent Updates**
- **What to learn**: Vectorized parameter optimization.
- **What to do**: Compute error vector $\mathbf{e} = \mathbf{p} - \mathbf{y}$ and update `self.weights_` and `self.bias_`.
- **How to check yourself**: On Dataset C1, training loss must strictly decrease over epochs.
- **When to proceed**: Model achieves $> 95\%$ accuracy on linearly separable data.
- **Recovery hints**:
  - *Hint 1 (Concept)*: Gradient uses `(p - y)`, not `(y - p)`. Watch the minus sign!
  - *Hint 2 (Operation)*: `dw = (X.T @ (p - y)) / N`, `db = np.mean(p - y)`.
  - *Hint 3 (Debugging)*: If loss explodes, reduce `learning_rate` to $0.01$ or $0.001$.

G. **Edge Cases and Expected Tests** (Step 8: Test and Debug)
1. `test_linearly_separable_convergence`: On Dataset C1, the model achieves 100% classification accuracy.
2. `test_probability_bounds`: For any input, all outputs of `predict_proba` must strictly lie in $[0.0, 1.0]$.
3. `test_decision_threshold_tuning`: Verify that lowering the threshold from $0.5$ to $0.2$ increases recall for class 1.
4. `test_perfectly_separable_weights_growth`: Notice that without regularization, weights grow very large on perfectly separable data as the model attempts to output probabilities of exactly $0$ and $1$.

H. **Complexity Analysis** (Step 9: Analyze the Algorithm)
- **Time Complexity**:
  - Per epoch: $O(N \cdot D)$ time for forward pass and gradient computation.
  - Total fitting: $O(\text{epochs} \cdot N \cdot D)$.
  - Inference: $O(N_{\text{test}} \cdot D)$.
- **Space Complexity**: $O(D)$ auxiliary memory for parameters $\mathbf{w}$ and gradient buffer.
- **Trade-offs**: Fast, convex, probabilistic, and interpretable, but limited strictly to linear decision boundaries without feature engineering.

I. **Progressive Difficulty Levels & Definition of Done**
- **Level 1 (Guided)**: Implement forward pass with explicit loop, compute sigmoid, and perform gradient updates on 1D Dataset C2.
- **Level 2 (Standard)**: Vectorized `LogisticRegression` class with stable sigmoid, threshold prediction, and unit tests.
- **Level 3 (Challenge)**: Add an $L_2$ regularization penalty to prevent weight divergence on perfectly separable data, and implement multiclass Softmax regression (One-vs-Rest or Multinomial).
- **Definition of Done**: Converges on Dataset C1, outputs valid probabilities, handles extreme logits without overflow, and loss decreases monotonically.

J. **Reflection & Self-Check** (Step 10: Independent Completion)
1. Why does Logistic Regression fail to solve the XOR problem (Dataset C3)?
2. How does the decision boundary orientation relate to the weight vector $\mathbf{w}$? (Hint: $\mathbf{w}$ is orthogonal to the boundary hyperplane).
"""
