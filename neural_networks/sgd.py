"""
A. **Mission**
Stochastic Gradient Descent (SGD) is the foundational optimisation algorithm for training neural networks. Instead of computing gradients over the entire dataset (which is slow for large datasets), SGD approximates the true gradient by computing it on small random subsets (mini-batches) or even single examples. This exercise implements the core mechanism for updating model parameters.

B. **Prerequisites**
- Calculus: Gradients (Hard).
- Multilayer Perceptron exercise (Helpful).

C. **Learning Questions**
1. How does the variance in the gradient estimates of SGD affect the convergence path compared to Batch Gradient Descent?
2. What is the role of the learning rate, and how do you choose a good one?
3. Why is mini-batch SGD preferred in practice over both pure SGD (batch size 1) and full batch gradient descent?

D. **Mathematics to Derive**
1. Write down the update rule for Vanilla SGD.
2. Formulate the expectation of the stochastic gradient and show it is an unbiased estimator of the full gradient.

E. **Implementation Contract**
- Class: `SGD`
- Constructor: `__init__(self, lr=0.01)`
  - `lr`: Learning rate (float).
- Methods:
  - `step(self, params, grads)`: Update parameters.
    - `params`: List of NumPy arrays representing model weights/biases.
    - `grads`: List of NumPy arrays representing the gradients for each parameter.

F. **Guided Implementation Stages**
1. **Initialisation**: Store the learning rate.
2. **Update Rule**: For each parameter and corresponding gradient, apply the rule: `param = param - lr * grad`.
3. **In-place Modification**: Ensure the update modifies the parameters appropriately (or returns the new parameters, depending on your design).

G. **Edge Cases and Expected Tests**
- Test with a simple synthetic parameter and gradient to ensure the math is correct.
- Test with lists of differently shaped arrays (mimicking layers of a neural network).
- Ensure zero gradients result in no change to the parameters.

H. **Complexity Analysis**
- Time/Space: What is the overhead of the SGD update step relative to the backpropagation step?

I. **Definition of Done**
- The optimiser correctly applies the vanilla SGD update rule.
- It can handle multiple parameter arrays of varying shapes simultaneously.

J. **Reflection**
1. What are the limitations of Vanilla SGD when navigating a loss landscape with narrow valleys (high curvature in some directions, low in others)?
"""
