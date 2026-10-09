# Chapter 6: Neural Networks

> **Hard Prerequisites:**
> - **Calculus:** partial derivatives and the chain rule (for backpropagation)
> - **Linear algebra:** matrix multiplication, dot products, vector operations (for forward propagation)
> - **Gradient descent:** iterative parameter updates and learning rate (Chapter 1)
>
> **Recommended context:** Chapter 2 (classification, sigmoid, cross-entropy loss), but not required.

> **Scope:** Forward propagation, activation functions, backpropagation, and gradient
> descent optimisers for fully-connected networks only. No CNN or RNN placeholders —
> master the MLP completely first.

---

## Learning Objectives

After completing this chapter, you should be able to:

- Explain what a single artificial neuron computes and why activation functions are essential
- Implement forward propagation and backpropagation from scratch using matrix operations
- Explain the vanishing gradient problem and how activation choice affects it
- Implement and compare optimisers beyond vanilla SGD

## Questions to Answer in Your Own Words

1. What does a single artificial neuron compute and what role does the activation function play?
2. Why can't a network of linear layers (no activations) learn non-linear functions, no matter how deep?
3. Walk through backpropagation step by step for a two-layer network using the chain rule.
4. What is the vanishing gradient problem? Which activation functions suffer from it and which mitigate it?
5. How does momentum modify the gradient update? How does Adam extend momentum with adaptive learning rates?

## Algorithms to Implement from Scratch

- [ ] Single Perceptron (with step and sigmoid activations)
- [ ] Multi-Layer Perceptron (MLP) with backpropagation
- [ ] Optimisers: SGD → SGD with Momentum → RMSProp → Adam

## Mathematical Derivations to Complete

- Derive the backpropagation update rules for a two-layer network from first principles (using the chain rule)
- Derive the Adam update rule: first moment estimate, second moment estimate, bias correction
- Show that the derivative of sigmoid is σ(x)(1 − σ(x))

## Edge Cases & Tests to Consider

- Zero weight initialisation: all weights set to 0 preserves symmetry — every neuron in a layer computes the same gradient and learns the same function. Explain why this prevents the network from learning distinct features.
- Verify your backpropagation gradients numerically using finite differences (gradient checking)
- XOR problem: demonstrate that a single perceptron cannot solve XOR; show that a two-layer MLP can
- Exploding gradients: what happens with very large initial weights?

## Completion Criteria

You are done with this chapter when you can:

- [ ] Train an MLP to solve XOR from scratch
- [ ] Derive backpropagation for a two-layer network without notes
- [ ] Implement Adam and show it converges faster than vanilla SGD on a test problem
- [ ] Explain the vanishing gradient problem and why ReLU helps
- [ ] Verify your gradients with finite-difference gradient checking (relative error < 1e-5)

## Notes

_Space for your own observations as you work through this chapter._
