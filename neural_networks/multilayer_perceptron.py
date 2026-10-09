"""
A. **Mission**
Build a Multilayer Perceptron (MLP) from scratch, deriving backpropagation via the chain rule across 2 layers.

B. **Prerequisites**
- [Stage 0C: Linear Algebra](../foundations/README.md#module-0c-linear-algebra): Matrix transformations.
- [Stage 0D: Calculus & Optimization](../foundations/README.md#module-0d-calculus-and-optimization): Chain rule, gradients.

C. **Learning Questions**
1. Why does random init BREAK symmetry while zero init PRESERVES symmetry?
2. How does the chain rule allow error to propagate backwards?

D. **Mathematics to Derive**
1. Derivation of backprop gradients for weights and biases across 2 layers.

E. **Implementation Contract**
- Class: `MLP`
- `__init__(self, layer_sizes)`
- `forward(self, X)`
- `backward(self, y_true)`
- `update(self, lr)`
- `predict(self, X)`, `train(self, X, y, epochs, lr)`

F. **Guided Implementation Stages**
- **Stage 0: Initialization**
  - **What to learn**: Symmetry breaking.
  - **What to do**: Initialize weights randomly, biases to zero.
  - **How to check yourself**: Print weights to ensure they are small and random.
  - **When to proceed**: When weights are correctly shaped matrices.
  - **Recovery hints 1/2/3**: Use np.random.randn; scale by 0.01; check layer sizes.
- **Stage 1: Forward Pass**
  - **What to learn**: Activations and caching.
  - **What to do**: Compute layer outputs and cache pre-activations.
  - **How to check yourself**: Output should be in range [0, 1] if sigmoid.
  - **When to proceed**: When caching works.
  - **Recovery hints 1/2/3**: np.dot(X, W) + b; check shapes; store in self.cache.
- **Stage 2: Backward Pass**
  - **What to learn**: Chain rule in code.
  - **What to do**: Compute dW and db.
  - **How to check yourself**: Perform finite-difference gradient check.
  - **When to proceed**: When analytical and numerical gradients match.
  - **Recovery hints 1/2/3**: Transpose correctly; check activation derivative; sum over batch.

G. **Edge Cases and Expected Tests**
- Finite-difference gradient check with relative error < 1e-5.
- Symmetry preservation check with zero weights (fails to learn properly).

H. **Complexity Analysis**
- Time: O(epochs * N * (L1*L2 + L2*L3)). Space: O(batch_size * hidden_size).

I. **Progressive Difficulty Levels & Definition of Done**
- Level 1 Guided: Forward pass only.
- Level 2 Standard: Backprop and train on Dataset 6B (XOR).
- Level 3 Challenge: Gradient checking.
- DoD: Solves XOR completely.

J. **Reflection**
Why is backprop better than perturbing weights randomly to find lower loss?
"""
