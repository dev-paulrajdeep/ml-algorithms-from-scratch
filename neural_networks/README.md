# Chapter 6: Neural Networks

## Chapter Objectives
In this chapter, you will transition from traditional machine learning models to the foundations of deep learning. You will build artificial neurons, connect them into multilayer architectures, and train them using backpropagation. Finally, you will explore and implement the evolution of optimization algorithms used to train these networks efficiently.

## Prerequisites
**Hard Prerequisites:**
- **Calculus:** Chain rule and partial derivatives (essential for backpropagation).
- **Linear Algebra:** Matrix multiplication, dot products, transposes.
- **Gradient Descent:** Concepts from Chapter 1.

**Recommended Prerequisites:**
- **Chapter 2:** Classification concepts, the sigmoid function, and cross-entropy loss.

## Recommended Study Sequence
The exercises in this chapter are designed to be completed in a strict sequence, building up from a single neuron to a fully optimized neural network:

1. [perceptron.py](./perceptron.py): The single artificial neuron.
2. [multilayer_perceptron.py](./multilayer_perceptron.py): Stacking neurons and learning backpropagation.
3. [sgd.py](./sgd.py): The fundamental optimizer.
4. [momentum.py](./momentum.py): Accelerating SGD.
5. [rmsprop.py](./rmsprop.py): Adapting learning rates per parameter.
6. [adam.py](./adam.py): The culmination of optimization techniques.

## Exercise Index

### 1. [Perceptron](./perceptron.py)
Implement a single artificial neuron with step and sigmoid activations. Explore linear decision boundaries and demonstrate geometrically why a single layer cannot solve the XOR problem.

### 2. [Multilayer Perceptron (MLP)](./multilayer_perceptron.py)
The centerpiece of this chapter. Implement forward propagation, non-linear activations (ReLU, tanh, sigmoid), and derive backpropagation from first principles using the chain rule. You will implement numerical gradient checking and observe how random initialization breaks symmetry, allowing the network to solve XOR.

### 3. [Stochastic Gradient Descent (SGD)](./sgd.py)
Implement the base optimization algorithm. Understand the mechanics of parameter updates and the role of the learning rate.

### 4. [SGD with Momentum](./momentum.py)
Building on SGD, implement velocity accumulation. Learn how momentum dampens oscillations and accelerates convergence through narrow valleys in the loss landscape.

### 5. [RMSProp](./rmsprop.py)
Building on the limitations of a global learning rate in SGD/Momentum, implement RMSProp to adaptively scale learning rates for individual parameters based on historical gradient magnitudes.

### 6. [Adam](./adam.py)
Combine the heuristics of Momentum (first moment) and RMSProp (second moment) with bias correction to build Adam, one of the most widely used optimizers in deep learning.

## Conceptual Journey
- **Building the Network:** You start with a single neuron (`perceptron.py`) and encounter its linear limitations. You then stack these neurons (`multilayer_perceptron.py`), which theoretically enables universal function approximation, provided you can train it (via backpropagation).
- **Optimizing the Network:** Standard gradient descent is slow. `sgd.py` introduces stochasticity. `momentum.py` adds direction-awareness. `rmsprop.py` adds magnitude-awareness. Finally, `adam.py` synthesizes direction and magnitude awareness into a robust optimizer.

## Chapter Math Learning Goals
- Master the **Chain Rule** for deriving backpropagation algorithms.
- Derive analytical gradients for loss functions with respect to network weights.
- Understand the mathematical update rules and physical analogies of advanced optimizers (velocity, moving averages, bias correction).

## Completion Checklist
- [ ] Implement and test the single Perceptron (verify XOR failure).
- [ ] Derive backpropagation gradients manually for a 2-layer network.
- [ ] Implement the MLP and verify gradients using finite differences.
- [ ] Train the MLP to perfectly solve the XOR problem.
- [ ] Implement SGD and verify parameter updates.
- [ ] Implement Momentum and observe acceleration.
- [ ] Implement RMSProp and observe adaptive scaling.
- [ ] Implement Adam, complete with bias correction, and compare its convergence to SGD.

## Personal Notes
*Use this space to track your progress, note difficult derivations, or write down questions to revisit.*
