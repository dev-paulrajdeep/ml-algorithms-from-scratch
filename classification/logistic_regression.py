"""
A. **Mission**
Logistic Regression is a fundamental binary classification algorithm. It predicts the probability that an instance belongs to the positive class.

B. **Prerequisites**
- [Stage 0D: Calculus & Optimization](../foundations/README.md#module-0d-calculus-and-optimization)
- Understanding of log-loss (cross-entropy loss)

C. **Learning Questions**
1. How does the sigmoid function squash real-valued outputs into probabilities?
2. Why do we use cross-entropy loss instead of mean squared error for classification?

D. **Mathematics to Derive**
1. Derive the gradient of the log-loss (cross-entropy) with respect to the weights.
2. Formulate the weight update step.

E. **Implementation Contract**
- Class: `LogisticRegression`
- `__init__(self, learning_rate=0.01, num_iterations=1000)`
- `fit(self, X, y)`: Train the model. `X` is (n_samples, n_features), `y` is (n_samples,) with 0/1 labels.
- `predict_proba(self, X)`: Return probability estimates of the positive class. Returns (n_samples,).
- `predict(self, X, threshold=0.5)`: Predict class labels. Returns (n_samples,).
- Allow NumPy for array operations.

F. **Guided Implementation Stages**
Step 1: The Sigmoid Function
- **What to learn**: How to map any real number to (0, 1).
- **What to do**: Implement `_sigmoid(z)`.
- **How to check yourself**: `_sigmoid(0)` should be 0.5. `_sigmoid(100)` ~ 1.
- **When to proceed**: When sigmoid handles arrays correctly.
- **Recovery hints 1/2/3**: Check `np.exp`, watch for overflow, clip values if needed.

Step 2: Initialization
- **What to learn**: Starting points for optimization.
- **What to do**: Initialize `self.weights` and `self.bias` to zeros.
- **How to check yourself**: Shape matches `X.shape[1]`.
- **When to proceed**: Arrays are initialized.
- **Recovery hints 1/2/3**: Remember bias is a scalar. Zero initialization works here.

Step 3: Forward Pass & Gradient Computation
- **What to learn**: Log-loss gradient.
- **What to do**: In a loop, compute `z`, apply sigmoid, compute gradients `dw` and `db`.
- **How to check yourself**: `dw` shape matches weights.
- **When to proceed**: Gradients are computed without explicit loops over samples.
- **Recovery hints 1/2/3**: `dw = (1/n) * X.T @ (y_pred - y)`.

Step 4: Parameter Update
- **What to learn**: Gradient descent step.
- **What to do**: Update weights and bias.
- **How to check yourself**: Loss decreases over iterations.
- **When to proceed**: Model converges on simple datasets.
- **Recovery hints 1/2/3**: Check learning rate scale.

G. **Edge Cases and Expected Tests**
1. `test_linearly_separable`: Should perfectly classify Dataset 2A.
2. `test_zero_initialization`: Model should converge even with zero-initialized weights.

H. **Complexity Analysis**
1. What is the time complexity of the `fit` method?

I. **Progressive Difficulty Levels & Definition of Done**
- **Level 1 (Guided)**: Implement using the explicit stages.
- **Level 2 (Standard)**: Write the class from scratch based on the API.
- **Level 3 (Challenge)**: Add L2 regularization.
- **Definition of Done**: All tests pass, loss monotonically decreases.

J. **Reflection**
1. What happens if the data is perfectly linearly separable (in terms of weight magnitudes)?
"""
