# Chapter 6: Neural Networks

## Chapter Objectives
In this chapter, you will transition from traditional machine learning models to the foundations of deep learning. Objectives include understanding artificial neurons, forward propagation, activation functions, backpropagation via the chain rule, numerical gradient checking, and the evolution of optimization algorithms.

## Prerequisites & Stage 0 Links
**Hard Prerequisites:**
- [Stage 0C: Linear Algebra](../foundations/README.md#module-0c-linear-algebra): Matrix transformations, dot products.
- [Stage 0D: Calculus & Optimization](../foundations/README.md#module-0d-calculus-and-optimization): Chain rule, gradients, gradient descent.

## Recommended Study Sequence
The exercises must be completed in this strict sequence:
1. `perceptron.py`
2. `multilayer_perceptron.py`
3. `sgd.py`
4. `momentum.py`
5. `rmsprop.py`
6. `adam.py`

## Chapter 6 Toy Datasets
- **Dataset 6A (Logic gates - AND / OR):** Linearly separable, solvable by a single Perceptron.
- **Dataset 6B (XOR problem):** `X = [[0, 0], [0, 1], [1, 0], [1, 1]]`, `y = [0, 1, 1, 0]`. Single perceptron fails; 2-layer MLP solves it!
- **Dataset 6C (Ill-conditioned quadratic loss surface):** Illustrates why Momentum and RMSProp accelerate where vanilla SGD struggles.

## Granular Progression
Weighted sums -> Perceptron -> Step vs Sigmoid activation -> Differentiable neurons -> Stacking into layers -> Forward pass -> Loss -> Chain rule across 2 layers -> Backprop -> Numerical gradient checking -> Weight initialization (Zero init PRESERVES symmetry; Random init BREAKS symmetry) -> Optimizers evolution (SGD -> Momentum -> RMSProp -> Adam).

## Scope
Fully-connected networks only. NO CNNs, RNNs, transformers.

## Exercise Index
1. [Perceptron](./perceptron.py)
2. [Multilayer Perceptron (MLP)](./multilayer_perceptron.py)
3. [Stochastic Gradient Descent (SGD)](./sgd.py)
4. [SGD with Momentum](./momentum.py)
5. [RMSProp](./rmsprop.py)
6. [Adam](./adam.py)

## Chapter Math Learning Goals
- Master the **Chain Rule** for deriving backpropagation algorithms across layers.
- Understand how optimizers build upon each other (moments, running averages).

## Cross-Chapter Conceptual Questions
- How does the sigmoid activation here relate to logistic regression in Chapter 2?
- Why is backpropagation fundamentally just an application of the chain rule from Stage 0D?

## Completion Checklist
- [ ] Understand linear vs non-linear separability (XOR).
- [ ] Implement and gradient-check a 2-layer MLP.
- [ ] Train MLP on XOR successfully.
- [ ] Implement SGD, Momentum, RMSProp, Adam.

## Personal Notes
*Use this space for your own derivations and thoughts.*
