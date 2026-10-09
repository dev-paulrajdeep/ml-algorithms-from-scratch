"""
A. **Mission**
Logistic Regression is a foundational linear model for binary (and multi-class) classification. Despite its name, it estimates the probability that a given instance belongs to a particular class by mapping a linear combination of features through the sigmoid (logistic) function. It solves the problem of binary prediction with well-calibrated probabilities and provides interpretable feature weights.

B. **Prerequisites**
- Hard prerequisites: Gradient descent, loss functions, linear algebra (dot products, matrix operations), and single-variable calculus (derivatives).
- Recommended: Familiarity with odds and log-odds, Bernoulli distributions, and vectorization in NumPy.

C. **Learning Questions**
1. Why is Mean Squared Error (MSE) problematic when optimizing a logistic regression model with sigmoid activation? (Consider non-convexity).
2. What is the relationship between the logit (log-odds) and the linear combination of inputs w^T x + b?
3. How does the choice of classification threshold (default 0.5) affect precision versus recall?
4. What happens to the learned weights when features are perfectly linearly separable without regularization?
5. Under what data distributions or assumptions does Logistic Regression produce optimal decision boundaries?

D. **Mathematics to Derive**
1. Derive the sigmoid activation function sigma(z) = 1 / (1 + exp(-z)) and demonstrate that its derivative is sigma'(z) = sigma(z) * (1 - sigma(z)).
2. Derive the Binary Cross-Entropy (negative log-likelihood) loss function starting from the Bernoulli likelihood of N independent samples.
3. Derive the analytical gradient of the Binary Cross-Entropy loss with respect to weights w and bias b. Observe its striking algebraic similarity to the gradient in linear regression.

E. **Implementation Contract**
- Class: `LogisticRegression`
- API:
  - `__init__(self, lr=0.01, epochs=1000, threshold=0.5)`
  - `fit(self, X, y)`: Train the model using batch gradient descent.
    - `X`: NumPy ndarray of shape `(n_samples, n_features)`.
    - `y`: NumPy ndarray of binary labels `{0, 1}` of shape `(n_samples,)`.
    - Returns `self`.
  - `predict_proba(self, X)`: Compute predicted probabilities for the positive class (1).
    - `X`: NumPy ndarray of shape `(n_samples, n_features)`.
    - Returns ndarray of shape `(n_samples,)` with values in `[0.0, 1.0]`.
  - `predict(self, X)`: Predict binary class labels `{0, 1}` based on `self.threshold`.
    - Returns ndarray of shape `(n_samples,)`.
- Attributes stored upon fitting:
  - `weights`: ndarray of shape `(n_features,)`.
  - `bias`: float scalar.
  - `loss_history`: list of scalar cross-entropy loss values recorded across epochs.
- Numerical library policy: Use NumPy for standard vectorized array operations. Implement sigmoid, cross-entropy, and gradient updates manually from scratch without scikit-learn.

F. **Guided Implementation Stages**
1. **Model Initialization**: Set hyperparameters (`lr`, `epochs`, `threshold`).
2. **Sigmoid & Numerical Stability**: Implement a numerically stable sigmoid helper: clamp z or use `np.where(z >= 0, ...)` to avoid exponential overflow when computing `exp(-z)`.
3. **Loss Computation**: Implement binary cross-entropy loss with an epsilon clip (e.g. `1e-15`) to prevent `log(0)`.
4. **Gradient Descent Loop**:
   - In `fit`, initialize weights to zeros or small random numbers, and bias to 0.0.
   - For each epoch:
     - Compute linear logits: `z = X @ w + b`.
     - Compute probabilities: `y_hat = sigmoid(z)`.
     - Calculate loss and append to `loss_history`.
     - Calculate gradients: `dw = (1 / n_samples) * (X.T @ (y_hat - y))`, `db = (1 / n_samples) * np.sum(y_hat - y)`.
     - Update parameters: `w -= lr * dw`, `b -= lr * db`.
5. **Inference**:
   - In `predict_proba(X)`, return `sigmoid(X @ w + b)`.
   - In `predict(X)`, return `(predict_proba(X) >= self.threshold).astype(int)`.

G. **Edge Cases and Expected Tests**
1. `test_linearly_separable`: Verify that on two cleanly separated 2D Gaussian blobs, classification accuracy reaches 100%.
2. `test_probability_bounds`: Ensure all outputs of `predict_proba` strictly lie within `[0.0, 1.0]`, even for large positive/negative feature magnitudes.
3. `test_loss_monotonic_decrease`: Check that `loss_history` decreases monotonically when using a suitably small learning rate.
4. `test_single_sample_inference`: Verify that `predict` and `predict_proba` handle a single input sample of shape `(1, n_features)` correctly.
5. `test_decision_boundary`: For balanced data, check that inputs where `w^T x + b == 0` yield probability 0.5.

H. **Complexity Analysis**
1. What is the time complexity per epoch of batch gradient descent as a function of `n_samples` and `n_features`?
2. What is the space complexity for storing parameters and intermediate activation states?
3. How does this compare with analytical solutions or second-order optimization methods (e.g., Newton-Raphson / IRLS)?

I. **Definition of Done**
- All conceptual questions and mathematical derivations are completed by hand.
- `LogisticRegression` is implemented according to the contract using only NumPy for array operations.
- The model trains and converges on synthetic binary classification datasets.
- Handled edge cases (numerical overflow in sigmoid, clipping in log loss).
- Gradients verified and loss documented to decrease over training epochs.

J. **Reflection**
1. How does gradient descent for logistic regression compare to the linear regression update rule you studied in Chapter 1?
2. Why is logistic regression considered a linear classifier even though the sigmoid function is non-linear?
3. How would you extend this binary formulation to multi-class classification (e.g., One-vs-Rest vs Softmax/Multinomial)?
"""
