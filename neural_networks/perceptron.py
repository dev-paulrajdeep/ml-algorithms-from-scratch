r"""
A. **Mission** (Step 0: Understand the Problem)
The Perceptron is the foundational atom of neural networks: a single artificial neuron that computes a weighted sum of inputs and applies a step threshold activation function.
Historical and practical significance: Introduced by Frank Rosenblatt in 1958, it was the first algorithmic model capable of learning weights directly from data. However, as famously proven by Minsky and Papert in 1969, a single perceptron can **only solve linearly separable problems**.
Your objective is to implement the classic Perceptron learning rule, prove that it successfully solves logical AND and OR gates, and demonstrate why it mathematically fails to solve the logical XOR problem (motivating multi-layer networks).

B. **Prerequisites**
- [Stage 0B: NumPy & Numerical Computing](../foundations/README.md#module-0b-numpy-and-numerical-computing) — Dot products and conditional thresholding (`np.where`).
- [Stage 0C: Linear Algebra](../foundations/README.md#module-0c-linear-algebra) — Dot products as projections and hyperplanes ($\mathbf{w}^T \mathbf{x} + b = 0$).
- [Stage 0D: Calculus & Optimization](../foundations/README.md#module-0d-calculus-and-optimization) — Parameter update mechanics and why non-differentiable step functions cannot use standard gradient descent.
- [Stage 0F: Machine Learning Core](../foundations/README.md#module-0f-machine-learning-core-concepts) — Binary classification framing.

C. **Learning Questions** (Step 1: Build Intuition)
1. Why does a step threshold function ($\text{step}(z) = 1$ if $z \ge 0$ else $0$) have a derivative of zero everywhere (except at $z=0$ where it is undefined)? Why does this make standard gradient descent impossible?
2. What does the Perceptron Convergence Theorem state? If data is linearly separable, does the learning rule guarantee convergence to a separating hyperplane in finite steps?
3. Geometrically, why is the logical XOR function impossible to separate with any single straight line?

D. **Mathematics to Derive**
1. **Step 2: Tiny Numerical Example by Hand**:
   Consider the logical AND gate (Dataset 6A):
   $\mathbf{x}_1 = [0, 0], y_1 = 0$; $\mathbf{x}_2 = [0, 1], y_2 = 0$; $\mathbf{x}_3 = [1, 0], y_3 = 0$; $\mathbf{x}_4 = [1, 1], y_4 = 1$.
   Start at $\mathbf{w} = [0.0, 0.0], b = 0.0$, $\alpha = 0.1$.
   - Sample 4 ($[1, 1], y=1$): $z = 0(1) + 0(1) + 0 = 0 \implies \hat{y} = 1$. Error $e = 1 - 1 = 0$ (No update).
   - Sample 2 ($[0, 1], y=0$): $z = 0 \implies \hat{y} = 1$. Error $e = 0 - 1 = -1$.
     Update: $\mathbf{w} \leftarrow \mathbf{w} + 0.1(-1)[0, 1] = [0.0, -0.1]$; $b \leftarrow b + 0.1(-1) = -0.1$.
   - After a few epochs, weights converge to $w_1 = 0.2, w_2 = 0.2, b = -0.3$.
     Check: $x=[1, 1] \implies 0.2(1) + 0.2(1) - 0.3 = +0.1 \ge 0 \implies \hat{y}=1$.
     Check: $x=[0, 1] \implies 0.2(0) + 0.2(1) - 0.3 = -0.1 < 0 \implies \hat{y}=0$. All correct!
2. **Step 3: Mathematical Notation**:
   - $\mathbf{X} \in \mathbb{R}^{N \times D}$, $\mathbf{y} \in \{0, 1\}^N$.
   - $\mathbf{w} \in \mathbb{R}^D$: weights, $b \in \mathbb{R}$: bias.
   - $z = \mathbf{w}^T \mathbf{x} + b$: pre-activation.
   - $\hat{y} = \Theta(z) = \begin{cases} 1 & \text{if } z \ge 0 \\ 0 & \text{if } z < 0 \end{cases}$: Heaviside step function.
3. **Step 4: Derive the Perceptron Learning Rule and the XOR Proof**:
   - The Perceptron update rule for sample $i$:
     $$e_i = y_i - \hat{y}_i$$
     $$\mathbf{w} \leftarrow \mathbf{w} + \alpha e_i \mathbf{x}_i$$
     $$b \leftarrow b + \alpha e_i$$
   - Notice: If prediction is correct ($e_i = 0$), parameters do not change.
   - If $y_i = 1$ but $\hat{y}_i = 0$ ($e_i = +1$), $\mathbf{w}$ moves toward $\mathbf{x}_i$, increasing $z$.
   - If $y_i = 0$ but $\hat{y}_i = 1$ ($e_i = -1$), $\mathbf{w}$ moves away from $\mathbf{x}_i$, decreasing $z$.
   - **Proof of XOR Non-Separability**:
     Suppose weights $w_1, w_2, b$ exist such that:
     1. $x=[0, 0] \implies b < 0$
     2. $x=[0, 1] \implies w_2 + b \ge 0$
     3. $x=[1, 0] \implies w_1 + b \ge 0$
     4. $x=[1, 1] \implies w_1 + w_2 + b < 0$
     Adding inequalities 2 and 3: $w_1 + w_2 + 2b \ge 0$.
     Substituting inequality 4 ($w_1 + w_2 < -b$): $(-b) + 2b > 0 \implies b > 0$.
     This contradicts inequality 1 ($b < 0$)! Therefore, no linear perceptron can ever solve XOR!

E. **Implementation Contract**
- Class: `Perceptron`
- Constructor:
  - `__init__(self, learning_rate: float = 0.1, epochs: int = 100)`:
    Stores hyperparameters.
- Attributes:
  - `weights_`: array of shape `(D,)`.
  - `bias_`: float.
  - `errors_history_`: list recording the count of misclassified samples per epoch.
- Methods:
  - `fit(self, X: np.ndarray, y: np.ndarray) -> self`:
    Initializes weights to zeros and trains using the Perceptron learning rule for `epochs` (or halts early if zero errors).
  - `predict(self, X: np.ndarray) -> np.ndarray`:
    Computes $\Theta(\mathbf{X}\mathbf{w} + b)$. Returns shape `(N,)` with integer labels in $\{0, 1\}$.

F. **Guided Implementation Stages**
**Step 5: Design the Algorithm (Pseudocode)**
```text
fit(X, y):
  weights = zeros(D), bias = 0.0
  For epoch in range(epochs):
    errors = 0
    For i in range(N):
      z = dot(X[i], weights) + bias
      y_hat = 1 if z >= 0 else 0
      delta = y[i] - y_hat
      if delta != 0:
        weights += learning_rate * delta * X[i]
        bias += learning_rate * delta
        errors += 1
    record errors
    if errors == 0:
      break # converged early!
```

**Checkpoint 1: Pre-Activation and Step Activation**
- **What to learn**: Forward computation of a single neuron.
- **What to do**: Implement $z = \mathbf{x}^T \mathbf{w} + b$ and apply step threshold $\hat{y} = 1$ if $z \ge 0$ else $0$.
- **How to check yourself**: With $\mathbf{w} = [1, 1], b = -1.5$, input $[1, 1]$ yields $1$, input $[0, 1]$ yields $0$.
- **When to proceed**: Forward prediction runs for single samples and 2D arrays.
- **Recovery hints**:
  - *Hint 1 (Concept)*: A single neuron divides space into two half-planes.
  - *Hint 2 (Operation)*: `return np.where(X @ self.weights_ + self.bias_ >= 0, 1, 0)`.
  - *Hint 3 (Debugging)*: Check scalar vs array thresholding.

**Checkpoint 2: The Online Learning Rule**
- **What to learn**: Error-driven weight updates per sample.
- **What to do**: Loop over samples in `X`, compute error $e = y_i - \hat{y}_i$, and update $\mathbf{w}$ and $b$.
- **How to check yourself**: On Dataset 6A (AND gate), error count decreases to 0 within 10 epochs.
- **When to proceed**: Perceptron achieves 100% accuracy on AND and OR gates.
- **Recovery hints**:
  - *Hint 1 (Concept)*: The update only occurs when a sample is misclassified ($e \neq 0$).
  - *Hint 2 (Operation)*: Ensure both `weights_` and `bias_` are updated simultaneously.
  - *Hint 3 (Debugging)*: If looping fails to converge, check the sign of the error term (`y - y_hat`).

**Checkpoint 3: Verifying the XOR Failure**
- **What to learn**: Recognizing non-linear separability.
- **What to do**: Fit the Perceptron on Dataset 6B (XOR).
- **How to check yourself**: Accuracy must stall at $50\%$ or $75\%$, and `errors_history_` will never reach zero.
- **When to proceed**: Demonstrably reproduces the classic XOR failure.
- **Recovery hints**:
  - *Hint 1 (Concept)*: The algorithm will cycle through parameter oscillations indefinitely.
  - *Hint 2 (Operation)*: Print final training accuracy on XOR: `assert accuracy <= 0.75`.
  - *Hint 3 (Debugging)*: This failure is mathematically expected and desired!

G. **Edge Cases and Expected Tests** (Step 8: Test and Debug)
1. `test_and_gate_convergence`: Must achieve 100% accuracy on logical AND gate within 20 epochs.
2. `test_or_gate_convergence`: Must achieve 100% accuracy on logical OR gate within 20 epochs.
3. `test_xor_gate_failure`: Must fail to achieve 100% accuracy on XOR (accuracy $\le 75\%$), confirming Minsky & Papert's theorem.
4. `test_early_stopping`: When all training samples are classified correctly, the training loop halts before `epochs` is exhausted.

H. **Complexity Analysis** (Step 9: Analyze the Algorithm)
- **Time Complexity**:
  - Per epoch: $O(N \cdot D)$ time.
  - Total fitting: $O(\text{epochs} \cdot N \cdot D)$.
  - Inference: $O(N_{\text{test}} \cdot D)$ time.
- **Space Complexity**: $O(D)$ memory for weight vector $\mathbf{w}$.
- **Trade-offs**: Extremely fast, but fundamentally limited to linearly separable datasets.

I. **Progressive Difficulty Levels & Definition of Done**
- **Level 1 (Guided)**: Implement the step activation and weight update loop for the 2D AND gate.
- **Level 2 (Standard)**: Complete `Perceptron` class with early stopping, error tracking, and validation on multiple logic gates.
- **Level 3 (Challenge)**: Visualize the decision boundary line $w_1 x_1 + w_2 x_2 + b = 0$ overlaid on the 2D scatter plot for AND, OR, and XOR, proving visually why XOR cannot be separated.
- **Definition of Done**: Solves AND and OR gates, proves failure on XOR, and passes all unit tests.

J. **Reflection & Self-Check** (Step 10: Independent Completion)
1. How did the XOR limitation lead directly to the "AI Winter" of the 1970s, and how did Rumelhart, Hinton, and Williams resolve it in 1986 with the Multi-Layer Perceptron?
2. Why can't we use standard gradient descent directly with the step activation function $\Theta(z)$?
"""
