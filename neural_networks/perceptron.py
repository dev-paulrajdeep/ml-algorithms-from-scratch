"""
A. **Mission**
Implement a single artificial neuron with step and sigmoid activations to understand linear decision boundaries and their limitations.

B. **Prerequisites**
- [Stage 0C: Linear Algebra](../foundations/README.md#module-0c-linear-algebra): Dot products.
- [Stage 0D: Calculus & Optimization](../foundations/README.md#module-0d-calculus-and-optimization): Gradients.

C. **Learning Questions**
1. Why does a step function make gradient descent impossible?
2. Why does the perceptron fail on Dataset 6B (XOR)?

D. **Mathematics to Derive**
1. Derive the weight update rule for sigmoid activation using binary cross-entropy.

E. **Implementation Contract**
- Class: `Perceptron`
- Constructor: `__init__(self, activation='step')`
- Methods: `fit(self, X, y, lr=0.01, epochs=100)`, `predict(self, X)`

F. **Guided Implementation Stages**
- **Stage 0: Initialization**
  - **What to learn**: Weight initialization.
  - **What to do**: Set weights to zero.
  - **How to check yourself**: Are weights shape (n_features,)?
  - **When to proceed**: When weights are correct shape.
  - **Recovery hints 1/2/3**: Use np.zeros; check shapes; ensure bias is 0.
- **Stage 1: Forward Pass**
  - **What to learn**: Weighted sum and activation.
  - **What to do**: Implement dot product and step/sigmoid.
  - **How to check yourself**: Does predict output 0/1 or probabilities?
  - **When to proceed**: When forward pass runs.
  - **Recovery hints 1/2/3**: np.dot; threshold at 0; check sigmoid formula.
- **Stage 2: Training Loop**
  - **What to learn**: Iterative updates.
  - **What to do**: Update weights per error.
  - **How to check yourself**: Does loss decrease on Dataset 6A?
  - **When to proceed**: When 100% accuracy on AND gate.
  - **Recovery hints 1/2/3**: Check lr; check error sign; loop epochs.

G. **Edge Cases and Expected Tests**
- Verify symmetry preservation check with zero weights (all updates act the same if multiple neurons).
- Fails Dataset 6B (XOR).
- Solves Dataset 6A (AND/OR).

H. **Complexity Analysis**
- Time: O(epochs * N * D). Space: O(D).

I. **Progressive Difficulty Levels & Definition of Done**
- Level 1 Guided: Step activation on AND gate.
- Level 2 Standard: Sigmoid activation.
- Level 3 Challenge: Visualize decision boundary.
- DoD: 100% on Dataset 6A, fails 6B.

J. **Reflection**
How did this limitation inspire multi-layer networks?
"""
