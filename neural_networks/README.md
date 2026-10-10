# Chapter 6: Neural Networks

Welcome to Chapter 6! In this chapter, you will transition from classical machine learning models to the foundational building blocks of deep learning: artificial neural networks. You will build differentiable computational graphs from scratch using only matrix algebra, derive and verify backpropagation via the multivariate chain rule, and implement first-order modern adaptive optimizers.

---

## Chapter Objectives
- Understand artificial neurons: weighted sums ($z = \mathbf{w}^T \mathbf{x} + b$) and non-linear activations.
- Implement the Perceptron and prove why it cannot learn non-linearly separable patterns like the XOR gate.
- Understand the need for differentiable activation functions (Sigmoid, ReLU) over non-differentiable step functions.
- Build a Multi-Layer Perceptron (MLP) with dense layers and forward propagation.
- Derive and implement **Backpropagation** using the chain rule to compute gradients across multiple layers.
- Verify analytical gradients using numerical **finite-difference gradient checking**.
- Understand weight initialization: why zero initialization **preserves symmetry** (neurons compute identical updates), and why small random initialization **breaks symmetry**.
- Implement and compare the modern family of first-order optimizers: **SGD**, **Momentum**, **RMSProp**, and **Adam**.

---

## Prerequisites
Before tackling this chapter, ensure you have completed:
- **[Stage 0C: Linear Algebra](../foundations/README.md#module-0c-linear-algebra)** — Matrix transformations ($\mathbf{Z} = \mathbf{X}\mathbf{W} + \mathbf{b}$), dimensions and transpose identities.
- **[Stage 0D: Calculus & Optimization](../foundations/README.md#module-0d-calculus-and-optimization)** — The chain rule, partial derivatives, and gradient vectors.
- **[Chapter 1: Regression](../regression/README.md)** — Gradient descent mechanics and loss tracking.
- **[Chapter 2: Classification](../classification/README.md)** — The Sigmoid function and Binary Cross-Entropy loss.

> [!NOTE]
> **Scope & Boundaries**: This chapter focuses strictly on fully-connected multi-layer feedforward networks. Advanced architectures like Convolutional Neural Networks (CNNs), Recurrent Neural Networks (RNNs), and Transformers are outside the curriculum scope.

---

## 🪜 The Layered Deep Learning Progression

To make deep learning completely beginner-proof, progress through these sequential stages:

```text
1. Weighted Sums (z = w · x + b)
     │
     ▼
2. Single Perceptron (Step function threshold)
     │
     ▼
3. Non-Linearity Limit (Perceptron fails on XOR!)
     │
     ▼
4. Differentiable Neuron (Sigmoid activation σ(z))
     │
     ▼
5. Layer Stacking (Input → Hidden → Output)
     │
     ▼
6. Forward Propagation (Caching activations Z and A)
     │
     ▼
7. Loss Computation (Binary Cross-Entropy)
     │
     ▼
8. The Chain Rule Across 2 Layers (Backpropagation)
     │
     ▼
9. Numerical Gradient Checking (Finite-difference verification)
     │
     ▼
10. Symmetry Breaking (Random weight initialization)
     │
     ▼
11. Optimizer Evolution (SGD ➔ Momentum ➔ RMSProp ➔ Adam)
```

---

## Recommended Study Sequence

The exercises in this chapter follow a strict dependency order:
1. **[`perceptron.py`](./perceptron.py)** — Single artificial neuron with step activation. Proves the XOR limitation.
2. **[`multilayer_perceptron.py`](./multilayer_perceptron.py)** — Dense 2-layer network with forward pass, backpropagation, and gradient checking. Solves the XOR problem!
3. **[`sgd.py`](./sgd.py)** — Stochastic and mini-batch gradient descent optimizer.
4. **[`momentum.py`](./momentum.py)** — Polyak momentum and velocity accumulation to dampen oscillations.
5. **[`rmsprop.py`](./rmsprop.py)** — Root Mean Square Propagation adapting per-parameter learning rates via squared gradient history.
6. **[`adam.py`](./adam.py)** — Adaptive Moment Estimation combining first moments (Momentum) and second moments (RMSProp) with initialization bias correction.

---

## Chapter 6 Toy Datasets

- **Dataset 6A (Logic Gates: AND / OR)**:
  - $\mathbf{X} = [[0, 0], [0, 1], [1, 0], [1, 1]]$
  - $\mathbf{y}_{\text{AND}} = [0, 0, 0, 1]$, $\mathbf{y}_{\text{OR}} = [0, 1, 1, 1]$
  - Hand check: Linearly separable. Solvable by a single Perceptron in fewer than 10 epochs.

- **Dataset 6B (The XOR Problem)**:
  - $\mathbf{X} = [[0, 0], [0, 1], [1, 0], [1, 1]]$
  - $\mathbf{y}_{\text{XOR}} = [0, 1, 1, 0]$
  - Hand check: Non-linearly separable. The single Perceptron **fails completely** (accuracy $\le 75\%$). The 2-layer MLP with 2 hidden neurons solves it with $100\%$ accuracy.

- **Dataset 6C (Ill-Conditioned Quadratic Ravine)**:
  - $L(w_1, w_2) = 0.1 w_1^2 + 10.0 w_2^2$.
  - Hand check: The gradient along $w_2$ is $100\times$ larger than along $w_1$. Vanilla SGD oscillates wildly back and forth across the ravine while barely moving toward the minimum. Momentum and Adam accelerate smoothly down the canyon!

---

## Completion Checklist
- [ ] Implement and test `perceptron.py` (solves AND/OR, fails XOR).
- [ ] Implement `multilayer_perceptron.py` with forward pass and backprop.
- [ ] Perform numerical gradient checking with relative error $< 10^{-5}$.
- [ ] Train MLP on Dataset 6B (XOR) to $100\%$ accuracy.
- [ ] Verify symmetry preservation with zero weights vs. symmetry breaking with random weights.
- [ ] Implement all four optimizers (`sgd.py`, `momentum.py`, `rmsprop.py`, `adam.py`).
- [ ] Compare optimizers on Dataset 6C (quadratic ravine).

---

## Cross-Chapter Conceptual Questions
1. **From Logistic Regression to MLP**: How is a single output neuron with sigmoid activation mathematically identical to the Logistic Regression model from Chapter 2?
2. **Symmetry Preservation**: Why does initializing all weights to zero in an MLP cause all hidden neurons in a layer to compute the exact same activation and gradient forever?
3. **Adaptive Learning Rates**: Why does Adam perform well across a wide variety of architectures without requiring fine-grained per-parameter tuning?

---

## Personal Notes
*(Use this space to record your derivations, observations, and insights.)*
