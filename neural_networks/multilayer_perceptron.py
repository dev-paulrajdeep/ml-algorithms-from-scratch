r"""
A. **Mission** (Step 0: Understand the Problem)
The Multilayer Perceptron (MLP) is a feedforward artificial neural network that overcomes the fundamental limitation of single-layer perceptrons (Minsky & Papert, 1969). By introducing one or more hidden layers equipped with non-linear activation functions, an MLP acts as a **universal function approximator** capable of learning arbitrary non-linear decision boundaries.
Your mission is to construct a fully connected 2-layer MLP from scratch using only NumPy. You will derive and implement **Backpropagation** via the multivariable chain rule, verify analytical gradients against numerical finite differences, demonstrate why symmetry breaking (random weight initialization) is mathematically mandatory, and train your model to solve the classic non-linear XOR benchmark (Dataset 6B) with 100% accuracy.

B. **Prerequisites**
- [Stage 0B: NumPy & Numerical Computing](../foundations/README.md#module-0b-numpy-and-numerical-computing) — Matrix broadcasting, transpositions, elementwise products (`*`), and array slicing.
- [Stage 0C: Linear Algebra](../foundations/README.md#module-0c-linear-algebra) — Matrix multiplications ($\mathbf{Z} = \mathbf{X}\mathbf{W} + \mathbf{b}$), dimension alignment, and vector norms.
- [Stage 0D: Calculus & Optimization](../foundations/README.md#module-0d-calculus-and-optimization) — Partial derivatives, multivariable chain rule, and gradient descent.
- [Stage 0F: Machine Learning Core](../foundations/README.md#module-0f-machine-learning-core-concepts) — Binary cross-entropy loss, decision boundaries, and overfitting.
- [Chapter 6: Perceptron](./perceptron.py) — Understanding why a single linear threshold unit fails on XOR.

C. **Learning Questions** (Step 1: Build Intuition)
1. **The Representation Bottleneck**: Why can a composition of strictly linear layers ($\mathbf{W}_2(\mathbf{W}_1 \mathbf{X})$) never learn non-linear functions, regardless of depth? Why is a non-linear activation function between layers essential?
2. **Symmetry Preservation vs. Breaking**: If all weights in a hidden layer are initialized to exactly zero (or identical constants), what happens to the pre-activations, post-activations, and gradients of all hidden neurons during training? Why does zero initialization trap all neurons into computing the exact same feature forever?
3. **Credit Assignment via Backpropagation**: How does the chain rule allow the error measured at the output layer to flow backward through the network to assign responsibility (gradients) to earlier hidden weights?
4. **Gradient Checking**: Why should you never trust an analytical backpropagation implementation until you have verified it against a finite-difference numerical gradient check?

D. **Mathematics to Derive**
1. **Step 2: Tiny Numerical Example by Hand**:
   Consider a minimal 2-layer MLP for binary classification with $D=2$ inputs, $H=2$ hidden neurons, and $K=1$ output neuron.
   Activation function: Sigmoid $\sigma(z) = \frac{1}{1 + e^{-z}}$, with derivative $\sigma'(z) = \sigma(z)(1 - \sigma(z))$.
   Loss: Binary Cross-Entropy (BCE) for a single sample $(\mathbf{x}, y)$:
   $$L = -[y \ln(\hat{y}) + (1-y)\ln(1-\hat{y})]$$
   Let sample be $\mathbf{x} = [0, 1]^T, y = 1$.
   Suppose initial parameters are:
   $$\mathbf{W}_1 = \begin{bmatrix} 0.1 & 0.2 \\ 0.3 & 0.4 \end{bmatrix}, \quad \mathbf{b}_1 = \begin{bmatrix} 0.0 \\ 0.0 \end{bmatrix}, \quad \mathbf{W}_2 = \begin{bmatrix} 0.5 \\ 0.6 \end{bmatrix}, \quad b_2 = 0.0$$
   *Forward Pass*:
   - Hidden pre-activation: $\mathbf{z}_1 = \mathbf{W}_1^T \mathbf{x} + \mathbf{b}_1 = [0.1(0) + 0.3(1), 0.2(0) + 0.4(1)]^T = [0.3, 0.4]^T$.
   - Hidden activation: $\mathbf{a}_1 = [\sigma(0.3), \sigma(0.4)]^T \approx [0.5744, 0.5987]^T$.
   - Output pre-activation: $z_2 = \mathbf{W}_2^T \mathbf{a}_1 + b_2 = 0.5(0.5744) + 0.6(0.5987) + 0.0 = 0.2872 + 0.3592 = 0.6464$.
   - Output prediction: $\hat{y} = a_2 = \sigma(0.6464) \approx 0.6562$.
   - Loss: $L = -\ln(0.6562) \approx 0.4213$.
   *Backward Pass*:
   - Output error: $\delta_2 = \frac{\partial L}{\partial z_2} = a_2 - y = 0.6562 - 1.0 = -0.3438$.
   - Gradient wrt $\mathbf{W}_2$: $\frac{\partial L}{\partial \mathbf{W}_2} = \mathbf{a}_1 \delta_2 = [0.5744(-0.3438), 0.5987(-0.3438)]^T = [-0.1975, -0.2058]^T$.
   - Gradient wrt $b_2$: $\frac{\partial L}{\partial b_2} = \delta_2 = -0.3438$.
   - Propagate error to hidden layer:
     $$\boldsymbol{\delta}_1 = (\mathbf{W}_2 \delta_2) \odot \sigma'(\mathbf{z}_1)$$
     $\mathbf{W}_2 \delta_2 = [0.5(-0.3438), 0.6(-0.3438)]^T = [-0.1719, -0.2063]^T$.
     $\sigma'(\mathbf{z}_1) = [0.5744(1-0.5744), 0.5987(1-0.5987)]^T = [0.2445, 0.2403]^T$.
     $\boldsymbol{\delta}_1 = [-0.1719 \times 0.2445, -0.2063 \times 0.2403]^T = [-0.0420, -0.0496]^T$.
   - Gradient wrt $\mathbf{W}_1$: $\frac{\partial L}{\partial \mathbf{W}_1} = \mathbf{x} \boldsymbol{\delta}_1^T = \begin{bmatrix} 0 \\ 1 \end{bmatrix} [-0.0420, -0.0496] = \begin{bmatrix} 0.0 & 0.0 \\ -0.0420 & -0.0496 \end{bmatrix}$.
   - Gradient wrt $\mathbf{b}_1$: $\frac{\partial L}{\partial \mathbf{b}_1} = \boldsymbol{\delta}_1 = [-0.0420, -0.0496]^T$.
   Notice how every single step follows directly from scalar multivariable calculus!

2. **Step 3: Mathematical Notation (Batch Form)**:
   - Input batch: $\mathbf{X} \in \mathbb{R}^{N \times D}$.
   - Layer 1 weights & bias: $\mathbf{W}_1 \in \mathbb{R}^{D \times H}$, $\mathbf{b}_1 \in \mathbb{R}^{1 \times H}$.
   - Layer 1 pre-activations: $\mathbf{Z}_1 = \mathbf{X}\mathbf{W}_1 + \mathbf{b}_1 \in \mathbb{R}^{N \times H}$.
   - Layer 1 activations: $\mathbf{A}_1 = \sigma(\mathbf{Z}_1) \in \mathbb{R}^{N \times H}$.
   - Layer 2 weights & bias: $\mathbf{W}_2 \in \mathbb{R}^{H \times 1}$, $b_2 \in \mathbb{R}^{1 \times 1}$.
   - Layer 2 pre-activations: $\mathbf{Z}_2 = \mathbf{A}_1\mathbf{W}_2 + b_2 \in \mathbb{R}^{N \times 1}$.
   - Layer 2 activations: $\mathbf{A}_2 = \sigma(\mathbf{Z}_2) \in \mathbb{R}^{N \times 1}$.
   - Batch Loss: $J = -\frac{1}{N} \sum_{i=1}^N [y_i \ln(a_{2, i}) + (1-y_i)\ln(1-a_{2, i})]$.

3. **Step 4: Deriving the Batch Backpropagation Equations**:
   - For Sigmoid output + BCE loss, the pre-activation derivative simplifies cleanly:
     $$\boldsymbol{\Delta}_2 = \frac{\partial J}{\partial \mathbf{Z}_2} = \frac{1}{N} (\mathbf{A}_2 - \mathbf{Y}) \in \mathbb{R}^{N \times 1}$$
   - Gradients for Layer 2:
     $$\frac{\partial J}{\partial \mathbf{W}_2} = \mathbf{A}_1^T \boldsymbol{\Delta}_2 \in \mathbb{R}^{H \times 1}$$
     $$\frac{\partial J}{\partial b_2} = \sum_{i=1}^N \Delta_{2, i} = \mathbf{1}^T \boldsymbol{\Delta}_2 \in \mathbb{R}^{1 \times 1}$$
   - Error propagated to Layer 1:
     $$\boldsymbol{\Delta}_1 = (\boldsymbol{\Delta}_2 \mathbf{W}_2^T) \odot \sigma'(\mathbf{Z}_1) = (\boldsymbol{\Delta}_2 \mathbf{W}_2^T) \odot [\mathbf{A}_1 \odot (1 - \mathbf{A}_1)] \in \mathbb{R}^{N \times H}$$
   - Gradients for Layer 1:
     $$\frac{\partial J}{\partial \mathbf{W}_1} = \mathbf{X}^T \boldsymbol{\Delta}_1 \in \mathbb{R}^{D \times H}$$
     $$\frac{\partial J}{\partial \mathbf{b}_1} = \sum_{i=1}^N \boldsymbol{\Delta}_{1, i} \in \mathbb{R}^{1 \times H}$$
   - Parameter updates (learning rate $\alpha$):
     $$\mathbf{W}_l \leftarrow \mathbf{W}_l - \alpha \frac{\partial J}{\partial \mathbf{W}_l}, \quad \mathbf{b}_l \leftarrow \mathbf{b}_l - \alpha \frac{\partial J}{\partial \mathbf{b}_l}$$

4. **Symmetry Breaking Proof**:
   Suppose $\mathbf{W}_1 = \mathbf{0}, \mathbf{b}_1 = \mathbf{0}, \mathbf{W}_2 = \mathbf{0}, b_2 = 0$.
   For any input $\mathbf{x}$, $\mathbf{Z}_1 = \mathbf{0} \implies \mathbf{A}_1 = [\sigma(0), \dots, \sigma(0)] = [0.5, \dots, 0.5]$.
   Every hidden neuron produces the exact same activation!
   In backward pass, $\boldsymbol{\Delta}_1 = (\boldsymbol{\Delta}_2 \mathbf{W}_2^T) \odot \sigma'(\mathbf{Z}_1) = \mathbf{0}$.
   Even if $\mathbf{W}_2$ is non-zero but uniform across columns, every column of $\frac{\partial J}{\partial \mathbf{W}_1}$ will be identical. All hidden neurons receive identical gradient updates, remaining permanently cloned.
   **Remedy**: Initialize weights with small Gaussian noise: $\mathbf{W} \sim \mathcal{N}(0, \sigma^2)$ (e.g. He or Xavier initialization).

E. **Implementation Contract**
- Class: `MultilayerPerceptron`
- Constructor:
  - `__init__(self, hidden_dim: int = 4, learning_rate: float = 0.1, epochs: int = 1000, random_state: int = 42)`:
    Stores hyperparameters and initializes random state.
- Attributes:
  - `W1_`: ndarray of shape `(input_dim, hidden_dim)`.
  - `b1_`: ndarray of shape `(1, hidden_dim)`.
  - `W2_`: ndarray of shape `(hidden_dim, 1)`.
  - `b2_`: ndarray of shape `(1, 1)`.
  - `loss_history_`: list of floats recording BCE loss per epoch.
- Methods:
  - `forward(self, X: np.ndarray) -> tuple[np.ndarray, dict]`:
    Computes forward pass. Returns prediction $\mathbf{A}_2 \in \mathbb{R}^{N \times 1}$ and `cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}`.
  - `backward(self, X: np.ndarray, y: np.ndarray, cache: dict) -> dict`:
    Computes analytical gradients using stored activations. Returns dictionary `{"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}`.
  - `fit(self, X: np.ndarray, y: np.ndarray) -> self`:
    Initializes weights breaking symmetry, iterates `epochs` performing forward, loss recording, backward, and parameter updates.
  - `predict_proba(self, X: np.ndarray) -> np.ndarray`:
    Computes and returns class probabilities $\hat{y} \in [0, 1]$ of shape `(N, 1)`.
  - `predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray`:
    Returns binary class predictions $\{0, 1\}$ of shape `(N,)`.
  - `gradient_check(self, X: np.ndarray, y: np.ndarray, epsilon: float = 1e-7) -> float`:
    Computes relative difference between numerical finite-difference gradients and analytical backprop gradients. Must return relative error $< 10^{-5}$.

F. **Guided Implementation Stages**
**Step 5: Algorithm Design & Pseudocode**
```text
fit(X, y):
  D = X.shape[1]
  W1 = randn(D, H) * sqrt(2/D), b1 = zeros(1, H)
  W2 = randn(H, 1) * sqrt(2/H), b2 = zeros(1, 1)
  For epoch in range(epochs):
    A2, cache = forward(X)
    loss = binary_cross_entropy(y, A2)
    grads = backward(X, y, cache)
    W1 -= lr * grads["dW1"]
    b1 -= lr * grads["db1"]
    W2 -= lr * grads["dW2"]
    b2 -= lr * grads["db2"]
```

**Checkpoint 1: Initialization and Forward Caching**
- **What to learn**: Symmetry breaking and caching intermediate computational graph tensors.
- **What to do**: Implement Xavier/He initialization for $\mathbf{W}_1, \mathbf{W}_2$, zero initialization for $\mathbf{b}_1, b_2$, and the forward pass storing $\mathbf{Z}_1, \mathbf{A}_1, \mathbf{Z}_2, \mathbf{A}_2$.
- **How to check yourself**: Verify that $\mathbf{A}_2$ produces outputs strictly in $(0, 1)$ with correct batch dimensions $(N, 1)$.
- **When to proceed**: Forward pass executes without shape errors for arbitrary $(N, D)$.
- **Recovery hints**:
  - *Hint 1 (Concept)*: Always cache pre-activations $\mathbf{Z}$ and post-activations $\mathbf{A}$ during the forward pass so they are available for the backward pass.
  - *Hint 2 (Operation)*: Use `1.0 / (1.0 + np.exp(-np.clip(Z, -500, 500)))` to prevent numerical overflow in the sigmoid function.
  - *Hint 3 (Debugging)*: Check matrix multiplication shapes: `(N, D) @ (D, H) -> (N, H)`, adding `(1, H)` via broadcasting.

**Checkpoint 2: Backpropagation and Finite-Difference Gradient Checking**
- **What to learn**: The multivariate chain rule and numerical gradient verification.
- **What to do**: Implement `backward` returning exact analytical gradients, and implement `gradient_check` using two-sided differences:
  $$\frac{\partial J}{\partial \theta_j} \approx \frac{J(\theta_j + \epsilon) - J(\theta_j - \epsilon)}{2\epsilon}$$
- **How to check yourself**: Run `gradient_check(X_toy, y_toy)`. The relative error must satisfy:
  $$\frac{\|\mathbf{g}_{\text{num}} - \mathbf{g}_{\text{anal}}\|_2}{\|\mathbf{g}_{\text{num}}\|_2 + \|\mathbf{g}_{\text{anal}}\|_2} < 10^{-5}$$
- **When to proceed**: Gradient check passes reliably across all parameters ($\mathbf{W}_1, \mathbf{b}_1, \mathbf{W}_2, b_2$).
- **Recovery hints**:
  - *Hint 1 (Concept)*: The output error $\Delta_2 = \frac{1}{N}(\mathbf{A}_2 - \mathbf{Y})$ already includes the derivative of sigmoid multiplied by cross-entropy loss.
  - *Hint 2 (Operation)*: For hidden layer bias gradient `db1`, sum along axis 0: `np.sum(delta1, axis=0, keepdims=True)`.
  - *Hint 3 (Debugging)*: If gradient checking fails with error $\approx 10^{-1}$, check whether you divided by $N$ in both numerical loss and analytical gradients.

**Checkpoint 3: Solving the XOR Problem (Dataset 6B)**
- **What to learn**: Non-linear feature representation and decision boundaries.
- **What to do**: Fit the MLP on Dataset 6B ($\mathbf{X} = [[0, 0], [0, 1], [1, 0], [1, 1]], \mathbf{y} = [0, 1, 1, 0]$) with $H=2$ or $H=4$.
- **How to check yourself**: Model reaches loss $< 0.05$ and achieves 100% training accuracy ($[0, 1, 1, 0]$).
- **When to proceed**: MLP reliably solves XOR from multiple random seeds.
- **Recovery hints**:
  - *Hint 1 (Concept)*: Two hidden neurons compute two lines that carve out the XOR diagonal.
  - *Hint 2 (Operation)*: If loss plateaus at $\approx 0.693$ ($\ln 2$), increase learning rate (e.g. $0.5$ or $1.0$) or train for more epochs (e.g. 5,000).
  - *Hint 3 (Debugging)*: If loss oscillates wildly, reduce learning rate.

G. **Edge Cases and Expected Tests** (Step 8: Test and Debug)
1. `test_gradient_checking`: Verify analytical gradients against two-sided numerical differences with relative tolerance $< 10^{-5}$.
2. `test_symmetry_preservation_failure`: When weights are initialized to identically zero, verify that all hidden neurons compute identical updates and the network fails to solve XOR.
3. `test_xor_convergence`: With random initialization, verify 100% accuracy on Dataset 6B within 5,000 epochs.
4. `test_batch_invariance`: Ensure forward and backward passes produce identical gradient values regardless of single-sample vs. mini-batch matrix layout.

H. **Complexity Analysis** (Step 9: Analyze the Algorithm)
- **Time Complexity**:
  - Forward pass: $O(N \cdot D \cdot H + N \cdot H \cdot 1)$ operations.
  - Backward pass: $O(N \cdot H \cdot 1 + N \cdot D \cdot H)$ operations.
  - Total per epoch: $O(N \cdot H \cdot (D + 1))$.
  - Inference: $O(N_{\text{test}} \cdot H \cdot (D + 1))$.
- **Space Complexity**:
  - Parameter storage: $O(D \cdot H + H)$ weights and biases.
  - Activation cache: $O(N \cdot (D + H + 1))$ memory per training batch.
- **Trade-offs**: Hidden layers unlock universal approximation, but non-convex loss surfaces introduce local minima and sensitivity to weight initialization.

I. **Progressive Difficulty Levels & Definition of Done**
- **Level 1 (Guided)**: Implement forward pass with sigmoid activations and evaluate BCE loss on a fixed batch.
- **Level 2 (Standard)**: Implement full backpropagation, parameter updates, and train to 100% accuracy on Dataset 6B (XOR).
- **Level 3 (Challenge)**: Implement numerical gradient checking and demonstrate empirical symmetry preservation when weights are initialized to zero.
- **Definition of Done**: Passes numerical gradient checking ($< 10^{-5}$ relative error), achieves 100% accuracy on XOR, and passes all unit tests.

J. **Reflection & Self-Check** (Step 10: Independent Completion)
1. Why does the combination of Sigmoid output and Binary Cross-Entropy loss produce such a simple error term ($\hat{y} - y$), whereas Sigmoid with Mean Squared Error produces a vanishing gradient term $(\hat{y} - y)\hat{y}(1-\hat{y})$?
2. If we increase the number of hidden layers to 10 using Sigmoid activations, what happens to the gradients in the earliest layers? (Hint: The Vanishing Gradient Problem).
"""
