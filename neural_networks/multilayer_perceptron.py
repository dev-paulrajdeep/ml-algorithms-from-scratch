"""
A. **Mission**
The Multilayer Perceptron (MLP) extends the single neuron model into a powerful universal function approximator. By stacking multiple dense layers with non-linear activation functions, the MLP can learn complex, non-linear mappings. This exercise is the centrepiece of the chapter, introducing forward propagation and backpropagation (the workhorse of modern deep learning) from first principles.

B. **Prerequisites**
- Calculus: Chain rule and partial derivatives (Hard).
- Linear Algebra: Matrix multiplication, transposes (Hard).
- Gradient Descent principles (Hard).
- Chapter 2: Cross-entropy loss (Helpful).

C. **Learning Questions**
1. Why are non-linear activation functions essential? What would happen if we only used linear activations?
2. Why does initialising all weights to zero preserve symmetry? How does this impact the network's ability to learn, and why is random initialisation necessary to break this symmetry?
3. How does the chain rule enable backpropagation through multiple layers?
4. What is the vanishing gradient problem, and how do different activation functions (e.g., ReLU vs. sigmoid) mitigate or exacerbate it?

D. **Mathematics to Derive**
1. Derive the derivative of the sigmoid, ReLU, and tanh activation functions.
2. For a simple 2-layer MLP, derive the gradients of the loss with respect to the output layer weights and the hidden layer weights using the chain rule.
3. Formulate the matrix operations for computing the gradients for a batch of samples.

E. **Implementation Contract**
- Class: `MLP`
- Constructor: `__init__(self, layer_sizes)`
  - `layer_sizes`: List of integers representing the number of neurons in each layer (e.g., [2, 4, 1]).
- Properties:
  - `weights`: List of weight matrices. `weights[l]` has shape `(n_l, n_{l+1})`.
  - `biases`: List of bias vectors. `biases[l]` has shape `(1, n_{l+1})`.
- Methods:
  - `forward(self, X)`: Perform forward propagation. Store intermediate activations for backprop.
  - `backward(self, y_true)`: Compute gradients using backpropagation.
  - `update(self, lr)`: Update weights and biases using computed gradients.
  - `predict(self, X)`: Return network predictions.
  - `train(self, X, y, epochs, lr)`: Execute the training loop.

F. **Guided Implementation Stages**
1. **Initialisation**: Initialise `weights` with small random values to break symmetry. Do not use zero initialisation for weights, as it preserves symmetry (all neurons would compute the same gradient, learning the exact same function). Initialise `biases` to zero.
2. **Activation Functions**: Implement sigmoid, ReLU, and tanh, along with their derivatives.
3. **Forward Propagation**: Compute linear combinations and apply activations layer by layer. Cache pre-activation and activation values.
4. **Loss Computation**: Implement Mean Squared Error or Cross-Entropy loss.
5. **Backpropagation**: Start from the output loss gradient and propagate backwards using the chain rule to compute gradients for all weights and biases.
6. **Numerical Gradient Checking**: Implement finite differences to verify that your analytical backprop gradients are correct.
7. **Training Loop**: Combine forward pass, loss computation, backward pass, and parameter updates.

G. **Edge Cases and Expected Tests**
- Test that random initialisation successfully breaks symmetry (weights diverge during training).
- Gradient checking: Ensure analytical gradients match numerical gradients within a small tolerance (e.g., 1e-5).
- XOR Problem: Train a [2, 2, 1] or [2, 4, 1] MLP on the XOR dataset and achieve 100% accuracy.
- Test shapes of forward and backward passes.

H. **Complexity Analysis**
- Time Complexity: What is the computational cost of one forward and backward pass for a network with L layers?
- Space Complexity: How much memory is required to store the activations needed for backpropagation?

I. **Definition of Done**
- Backpropagation is implemented correctly and verified via numerical gradient checking.
- The network successfully learns the XOR dataset (a key milestone demonstrating non-linear capability).
- Weight initialisation correctly breaks symmetry.
- No external libraries besides NumPy are used.

J. **Reflection**
1. How tedious was manual backpropagation derivation compared to using autograd tools?
2. What happens if you forget to cache intermediate values during the forward pass?
"""
