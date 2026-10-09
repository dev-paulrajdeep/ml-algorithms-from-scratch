"""
A. Mission
Polynomial regression extends linear regression to model non-linear relationships by creating new features that are polynomial powers of the original input. It is useful when the true relationship curves, but it still falls under the umbrella of linear models because it remains linear in its parameters (the weights).

B. Prerequisites
- Familiarity with basic linear regression and gradient descent.
- Understanding of mean squared error (MSE) loss.
- Matrix multiplication basics.

C. Learning Questions
1. Why is polynomial regression still considered a "linear" model?
2. What happens to the training error and testing error as you increase the polynomial degree excessively?
3. How does this model illustrate the bias-variance tradeoff?
4. What role does feature scaling play before creating polynomial features, especially for high degrees?

D. Mathematics to Derive
1. Write the hypothesis function for a polynomial of degree d.
2. Given an input vector x, write out the expanded feature vector.
3. Derive the gradient of the MSE loss with respect to the weights for the expanded feature matrix.

E. Implementation Contract
Create a class `PolynomialRegression` with the following methods:
- `__init__(self, degree: int, learning_rate: float = 0.01, epochs: int = 1000)`
- `fit(self, X: np.ndarray, y: np.ndarray) -> None`: Computes weights via gradient descent. NumPy allowed for array ops.
- `predict(self, X: np.ndarray) -> np.ndarray`: Returns predictions.
- `_transform_features(self, X: np.ndarray) -> np.ndarray`: Private method to map X to [1, X, X^2, ..., X^degree].

F. Guided Implementation Stages
1. Implement the feature expansion step: given an array X, generate powers up to `degree` and concatenate them into a new matrix, along with an intercept column.
2. Initialize weights array to zeros.
3. Implement the forward pass (dot product of expanded X and weights).
4. Compute the error and the gradient of the MSE loss.
5. Update weights in a loop for `epochs` iterations using `learning_rate`.

G. Edge Cases and Expected Tests
1. Test with degree=1: should match the behavior of basic linear regression.
2. Test with a toy non-linear dataset (e.g., a simple parabola): verify the fit captures the curve better than degree=1.
3. High degree overfitting: write a test with sparse data and high degree, verifying that training loss is near 0 but predictions between points vary wildly.

H. Complexity Analysis
1. What is the time complexity of the feature transformation step?
2. What is the time complexity of each gradient descent step relative to the number of samples N and degree d?
3. How does space complexity scale with the polynomial degree?

I. Definition of Done
The class is fully implemented and passes all expected tests. It can fit a non-linear target and correctly apply the learned weights to new inputs via feature expansion.

J. Reflection
1. Was gradient descent stable with higher degrees?
2. Did you encounter numerical overflow, and if so, how might standardization of features help?
"""
