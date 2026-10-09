"""
A. **Mission**
The Perceptron is a fundamental building block of neural networks—a single artificial neuron. It takes inputs, computes a weighted sum, and passes it through an activation function to make a binary decision. Originally using a step function, modern variations use a sigmoid function. This algorithm exists to demonstrate the simplest form of linear classification and supervised learning, forming the basis for more complex networks.

B. **Prerequisites**
- Linear Algebra: Dot products and vector manipulation (Hard).
- Calculus: Basic derivatives (Helpful).
- Chapter 2: Classification and the sigmoid function (Helpful).

C. **Learning Questions**
1. How does the choice of activation function (step vs. sigmoid) affect the learning rule?
2. Why is a single perceptron fundamentally unable to solve the XOR problem?
3. Geometrically, what does the perceptron's decision boundary represent in feature space?
4. How does the learning rate influence the convergence of the perceptron?

D. **Mathematics to Derive**
1. Derive the weight update rule for a perceptron using a step activation function.
2. Derive the gradient of the loss with respect to the weights for a sigmoid activation function (assuming binary cross-entropy loss).
3. Prove geometrically or algebraically why XOR cannot be separated by a single hyperplane.

E. **Implementation Contract**
- Class: `Perceptron`
- Constructor: `__init__(self, activation='step')`
  - `activation`: String, either 'step' or 'sigmoid'.
- Methods:
  - `fit(self, X, y, learning_rate=0.01, epochs=100)`: Train the perceptron.
    - `X`: NumPy array of shape (n_samples, n_features).
    - `y`: NumPy array of shape (n_samples,).
  - `predict(self, X)`: Return predicted labels.
    - `X`: NumPy array of shape (n_samples, n_features).
    - Returns: NumPy array of shape (n_samples,).

F. **Guided Implementation Stages**
1. **Initialisation**: Set weights and bias to small random values or zeros. (Note: For a single neuron, zero initialisation is fine, but consider the implications).
2. **Forward Pass**: Implement the weighted sum of inputs plus bias.
3. **Activation**: Apply the step or sigmoid function based on the parameter.
4. **Learning Rule (Step)**: If the prediction is wrong, update weights: `w = w + lr * (y - y_hat) * x`.
5. **Learning Rule (Sigmoid)**: Update weights using gradient descent on the loss.
6. **Training Loop**: Iterate over the dataset for the specified number of epochs, updating weights.

G. **Edge Cases and Expected Tests**
- Test with linearly separable data (e.g., AND, OR gates); ensure 100% accuracy.
- Test with non-linearly separable data (e.g., XOR gate); verify it fails to converge to 100%.
- Ensure correct behaviour when all inputs are zero.
- Test both 'step' and 'sigmoid' activation modes.

H. **Complexity Analysis**
- Time Complexity: What is the time complexity of a single training epoch? How does it scale with features and samples?
- Space Complexity: What is the memory footprint of the Perceptron model?

I. **Definition of Done**
- The Perceptron successfully learns linearly separable datasets.
- The failure on the XOR dataset is clearly reproducible and demonstrated.
- Both step and sigmoid activations are fully functional and tested.
- No imports other than NumPy are used.

J. **Reflection**
1. What did you learn about linear separability?
2. How does the limitation of the single perceptron motivate the need for multilayer networks?
"""
