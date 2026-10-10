r"""
A. **Mission** (Step 0: Understand the Problem)
A Linear Support Vector Machine (SVM) finds the optimal separating hyperplane that maximizes the geometric margin between two classes.
Real-world problem: High-dimensional binary classification such as bioinformatics (classifying cancer types from microarray gene expressions) or text categorization.
Many lines can separate two linearly separable clusters, but lines that pass very close to training points are fragile and overfit. The SVM finds the unique line that stays as far away as possible from the nearest data points of both classes (the "support vectors"), providing theoretical guarantees on generalization error.

B. **Prerequisites**
- [Stage 0B: NumPy & Numerical Computing](../foundations/README.md#module-0b-numpy-and-numerical-computing) — Vectorized conditional updates (`np.where`).
- [Stage 0C: Linear Algebra](../foundations/README.md#module-0c-linear-algebra) — Hyperplanes ($\mathbf{w}^T \mathbf{x} + b = 0$), orthogonal projection, and geometric distance to a plane.
- [Stage 0D: Calculus & Optimization](../foundations/README.md#module-0d-calculus-and-optimization) — Piecewise subgradient descent.
- [Stage 0F: Machine Learning Core](../foundations/README.md#module-0f-machine-learning-core-concepts) — Regularization parameter $\lambda$ / $C$.
- [Chapter 1: Regression](../regression/README.md) — Gradient descent mechanics.

C. **Learning Questions** (Step 1: Build Intuition)
1. What is the "geometric margin", and why does maximizing the margin minimize the VC-dimension (bound on test error)?
2. What are "support vectors"? Why does moving a training point that is far away from the decision boundary have zero effect on the learned SVM hyperplane?
3. How does Hinge Loss differ from Cross-Entropy Loss? Why does Hinge Loss ignore points that are classified correctly and lie outside the margin?

D. **Mathematics to Derive**
1. **Step 2: Tiny Numerical Example by Hand**:
   Consider 2 points in 1D: $x_1 = 1.0, y_1 = -1$ and $x_2 = 3.0, y_2 = +1$.
   The midpoint separating them is $x = 2.0$.
   Let candidate hyperplane be $w \cdot x + b = 0$ with $w = 1.0, b = -2.0$.
   - Margin for point 1: $y_1(w x_1 + b) = -1(1(1) - 2) = -1(-1) = +1.0 \ge 1$. (On the margin; hinge loss $= 0$).
   - Margin for point 2: $y_2(w x_2 + b) = +1(1(3) - 2) = +1(+1) = +1.0 \ge 1$. (On the margin; hinge loss $= 0$).
   - Total hinge loss is exactly $0.0$. Both points are support vectors!
2. **Step 3: Mathematical Notation**:
   - $\mathbf{X} \in \mathbb{R}^{N \times D}$, $\mathbf{y} \in \{-1, +1\}^N$ (Note the $\pm 1$ label convention!).
   - $\mathbf{w} \in \mathbb{R}^D$: normal vector to hyperplane, $b \in \mathbb{R}$: bias.
   - Distance from point $\mathbf{x}_i$ to hyperplane: $\gamma_i = \frac{y_i(\mathbf{w}^T \mathbf{x}_i + b)}{\|\mathbf{w}\|_2}$.
   - Total margin width: $\frac{2}{\|\mathbf{w}\|_2}$. Maximizing the margin is equivalent to minimizing $\frac{1}{2} \|\mathbf{w}\|_2^2$.
3. **Step 4: Derive the Hinge Loss & Subgradient Update**:
   - Write the Soft-Margin SVM unconstrained objective:
     $$J(\mathbf{w}, b) = \frac{\lambda}{2} \|\mathbf{w}\|_2^2 + \frac{1}{N} \sum_{i=1}^N \max(0, 1 - y_i(\mathbf{w}^T \mathbf{x}_i + b))$$
   - The Hinge Loss for sample $i$ is $L_i = \max(0, 1 - y_i(\mathbf{w}^T \mathbf{x}_i + b))$.
   - Differentiate $L_i$ with respect to $\mathbf{w}$ and $b$:
     $$\text{If } y_i(\mathbf{w}^T \mathbf{x}_i + b) \ge 1 \quad (\text{correctly classified outside margin}):$$
     $$\nabla_{\mathbf{w}} L_i = \mathbf{0}, \quad \frac{\partial L_i}{\partial b} = 0$$
     $$\text{If } y_i(\mathbf{w}^T \mathbf{x}_i + b) < 1 \quad (\text{margin violation or misclassified}):$$
     $$\nabla_{\mathbf{w}} L_i = -y_i \mathbf{x}_i, \quad \frac{\partial L_i}{\partial b} = -y_i$$
   - The full subgradient step for sample $i$:
     $$\mathbf{w} \leftarrow \mathbf{w} - \alpha \left( \lambda \mathbf{w} - \frac{1}{N} y_i \mathbf{x}_i \cdot \mathbb{I}(y_i z_i < 1) \right)$$
     $$b \leftarrow b + \alpha \frac{1}{N} y_i \cdot \mathbb{I}(y_i z_i < 1)$$

E. **Implementation Contract**
- Class: `LinearSVM`
- Methods:
  - `__init__(self, learning_rate: float = 0.001, lambda_param: float = 0.01, epochs: int = 1000)`:
    Stores hyperparameters.
  - `fit(self, X: np.ndarray, y: np.ndarray) -> self`:
    Validates that $y$ contains $\{-1, +1\}$ (or converts $\{0, 1\}$ to $\{-1, +1\}$). Initializes `weights_` to zeros `(D,)` and `bias_` to 0.0. Runs subgradient descent for `epochs`.
  - `predict(self, X: np.ndarray) -> np.ndarray`:
    Computes $\text{sign}(\mathbf{X}\mathbf{w} + b)$. Returns 1D array of shape `(N,)` with labels in $\{-1, +1\}$.

F. **Guided Implementation Stages**
**Step 5: Design the Algorithm (Pseudocode)**
```text
fit(X, y):
  y_signed = where(y <= 0, -1, 1)
  weights = zeros(D), bias = 0.0
  For epoch in range(epochs):
    margins = y_signed * (X @ weights + bias)
    violations = (margins < 1) # boolean mask of margin violations
    
    # Vectorized gradient computation
    dw = lambda_param * weights - (X.T @ (y_signed * violations)) / N
    db = -mean(y_signed * violations)
    
    weights -= learning_rate * dw
    bias -= learning_rate * db
```

**Checkpoint 1: Label Verification and Conversion**
- **What to learn**: Managing label conventions ($\pm 1$ vs $0/1$).
- **What to do**: In `fit`, check `np.unique(y)`. If labels are $\{0, 1\}$, map $0 \to -1$.
- **How to check yourself**: Ensure transformed labels are strictly $-1$ and $+1$.
- **When to proceed**: Class handles both $\{0, 1\}$ and $\{-1, +1\}$ input labels.
- **Recovery hints**:
  - *Hint 1 (Concept)*: Hinge loss relies on the product $y \cdot z$. If $y=0$, the loss would be $1 - 0 = 1$ unconditionally.
  - *Hint 2 (Operation)*: Use `np.where(y <= 0, -1, 1)`.
  - *Hint 3 (Debugging)*: Store the original label format if you want `predict` to return the user's original convention.

**Checkpoint 2: Margin Violation Masking**
- **What to learn**: Identifying which samples violate the margin condition.
- **What to do**: Compute $z = \mathbf{X}\mathbf{w} + b$ and boolean condition `violations = (y_signed * z < 1.0)`.
- **How to check yourself**: Points with margin $\ge 1.0$ have `violations == False`.
- **When to proceed**: Boolean mask correctly separates support vectors from safely classified points.
- **Recovery hints**:
  - *Hint 1 (Concept)*: A sample is safe ONLY if it is on the correct side AND distance $\ge 1$.
  - *Hint 2 (Operation)*: Array condition `y * z < 1` yields a boolean array of length $N$.
  - *Hint 3 (Debugging)*: Don't forget the bias term inside $z$.

**Checkpoint 3: Subgradient Parameter Updates**
- **What to learn**: Applying regularization and error gradients simultaneously.
- **What to do**: Update weights and bias per epoch.
- **How to check yourself**: On Dataset C1, weights converge to a separating hyperplane.
- **When to proceed**: Loss decreases and accuracy reaches 100% on linearly separable data.
- **Recovery hints**:
  - *Hint 1 (Concept)*: When there are zero violations, weights decay toward zero: $\mathbf{w} \leftarrow \mathbf{w}(1 - \alpha \lambda)$.
  - *Hint 2 (Operation)*: Compute full gradient using matrix multiplication `X.T @ (y * violations)`.
  - *Hint 3 (Debugging)*: If weights explode, decrease `learning_rate` or increase `lambda_param`.

G. **Edge Cases and Expected Tests** (Step 8: Test and Debug)
1. `test_linearly_separable_dataset_c1`: Linear SVM must achieve 100% classification accuracy on Dataset C1.
2. `test_margin_violation_gradient`: Verify that a point far outside the margin does NOT alter the gradient with respect to $\mathbf{w}$ (except for weight decay).
3. `test_lambda_regularization_effect`: Increasing $\lambda$ shrinks $\|\mathbf{w}\|_2$, widening the geometric margin $\frac{2}{\|\mathbf{w}\|_2}$ at the cost of allowing more margin violations.
4. `test_prediction_sign`: Predictions on points where $\mathbf{w}^T \mathbf{x} + b > 0$ return $+1$; points where $\mathbf{w}^T \mathbf{x} + b < 0$ return $-1$.

H. **Complexity Analysis** (Step 9: Analyze the Algorithm)
- **Time Complexity**:
  - Per epoch: $O(N \cdot D)$ time for forward pass, condition check, and matrix product.
  - Total fitting: $O(\text{epochs} \cdot N \cdot D)$.
  - Inference: $O(N_{\text{test}} \cdot D)$ time.
- **Space Complexity**: $O(D)$ auxiliary memory for weights vector $\mathbf{w}$.
- **Trade-offs**: Robust against outliers far from the boundary, but slower to converge on non-separable data compared to smooth loss functions like Logistic Regression.

I. **Progressive Difficulty Levels & Definition of Done**
- **Level 1 (Guided)**: Implement sample-by-sample loop over $N$ to compute hinge loss subgradients on 2D Dataset C1.
- **Level 2 (Standard)**: Fully vectorized `LinearSVM` class with condition masking, label adaptation, and unit tests.
- **Level 3 (Challenge)**: Implement mini-batch Pegasos (Primal Estimated sub-GrAdient SOlver for SVM) with learning rate decay $\alpha_t = \frac{1}{\lambda t}$.
- **Definition of Done**: Finds maximum margin separating hyperplane on Dataset C1, respects hinge loss conditions, and satisfies all edge tests.

J. **Reflection & Self-Check** (Step 10: Independent Completion)
1. How does the Support Vector Machine's focus on boundary points contrast with Logistic Regression's consideration of all points?
2. What happens to the SVM boundary when a point far away from the margin is moved even further away? (Answer: Nothing!).
"""
