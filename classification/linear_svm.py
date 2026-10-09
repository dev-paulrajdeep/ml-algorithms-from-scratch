"""
A. **Mission**
The Linear Support Vector Machine (SVM) aims to find the optimal hyperplane that separates classes with the maximum margin. It is a powerful discriminative classifier. This exercise focuses on solving the unconstrained optimization problem using the hinge loss formulation via gradient descent.

B. **Prerequisites**
- Vector geometry (dot products, hyperplanes, margins)
- Optimization via gradient descent
- Understanding of hinge loss

C. **Learning Questions**
1. What is a margin, and why is maximizing it desirable?
2. What role do support vectors play in defining the decision boundary?
3. How does the regularization parameter 'C' (or lambda) balance margin size and classification errors?

D. **Mathematics to Derive**
1. Write the formulation for hinge loss: max(0, 1 - y_i * (w^T x_i - b)).
2. Derive the gradient of the objective function (hinge loss + L2 regularization penalty) with respect to weights and bias.

E. **Implementation Contract**
- Class: `LinearSVM`
- `__init__(self, learning_rate=0.001, lambda_param=0.01, num_iterations=1000)`
- `fit(self, X, y)`: Train the SVM. `X` is (n_samples, n_features), `y` is (n_samples,) with -1/+1 labels.
- `predict(self, X)`: Predict class labels (-1 or 1). Returns (n_samples,).
- Allow NumPy for array operations.

F. **Guided Implementation Stages**
1. Ensure labels are in the set {-1, 1}.
2. Initialize weights and bias to zeros.
3. In a loop for `num_iterations`:
   a. Check the margin condition for each sample: y_i * (w^T x_i - b) >= 1.
   b. If the condition is met, the gradient only comes from the regularization term.
   c. If not met, the gradient comes from both regularization and hinge loss.
   d. Update weights and bias accordingly based on the aggregated gradients.

G. **Edge Cases and Expected Tests**
1. `test_label_conversion`: Validate behavior if labels are passed as 0/1 instead of -1/1 (fail early or convert).
2. `test_margin_violation`: Ensure points inside the margin contribute to the gradient.
3. `test_linear_separability`: Should find a separating hyperplane on simple linearly separable data.

H. **Complexity Analysis**
1. What is the time complexity of training?
2. How sparse is the solution in terms of which data points actually dictate the final weights?

I. **Definition of Done**
- All tests pass.
- Model minimizes the hinge loss objective using gradient descent.
- Implementation correctly handles the piecewise derivative of the hinge loss.

J. **Reflection**
1. Why might we prefer SVM over Logistic Regression for certain datasets?
2. How does the linear formulation limit the model, and how might you solve non-linear problems?
"""
