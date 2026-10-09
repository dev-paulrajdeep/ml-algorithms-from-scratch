"""
Polynomial Regression Curriculum Guide

A. Step 0: Understand the problem
Polynomial curve fitting allows us to model non-linear relationships using linear regression techniques by transforming the input features. Think of modeling the trajectory of a thrown ball (a parabola) rather than a straight line.

B. Step 1: Build intuition
Plot a curve through 5 points that look like a U-shape. A straight line will pass through the middle, missing the points and yielding high error. By squaring the input feature, we create a new feature space where a linear model can fit a quadratic curve.

C. Step 2: Tiny numerical example
Data: x = [1, 2, 3], y = [1, 4, 9], degree = 2.
Features for x=2 become [1, 2, 4] (bias, x^1, x^2).
A model with weights [0, 0, 1] perfectly predicts the data.

D. Step 3: Mathematical notation
Let x be a scalar input. A polynomial of degree d is:
y = w_0 + w_1 x + w_2 x^2 + ... + w_d x^d
We can create a Vandermonde matrix X where each column is x^j for j=0 to d.

E. Step 4: Derive the math
Once features are transformed, the objective is standard MSE:
J(w) = (1/2N) ||Xw - y||^2
The gradient is: dJ/dw = (1/N) X^T (Xw - y).

F. Step 5: Design algorithm in plain English and pseudocode
1. Define a `transform` method that takes input X and returns a new matrix with polynomial features up to `degree`.
2. In `fit`, transform X, then use gradient descent (or closed form) to find weights.
3. In `predict`, transform X, then compute Xw.

G. Step 6: Implement simplest version (Level 1 - Guided)
- Degree 2 only.
- 3 data points.
- Hardcode the transformation for degree 2.

H. Step 7: Generalize (Level 2 - Standard)
- Implement `PolynomialRegression` class with `__init__(degree, learning_rate, epochs)`.
- Write a general `transform(X)` for arbitrary degree.
- Implement `fit` and `predict`.

I. Step 8: Test and debug
- Test with degree 1 on linear data; it should match linear regression.
- Test with degree 2 on quadratic data.
- Watch out for exploding gradients with high degrees!

J. Step 9: Analyze (Level 3 - Challenge)
- As degree increases, values like x^10 become huge. Feature scaling (standardization) before or after transformation is critical.
- Discuss numerical stability and condition number of the Vandermonde matrix.

Implementation Contract:
class PolynomialRegression:
    def __init__(self, degree=2, learning_rate=0.01, epochs=1000): ...
    def transform(self, X): ...
    def fit(self, X, y): ...
    def predict(self, X): ...

Edge Cases to Check:
- degree = 0 (predicts the mean of y).
- negative degree (raise ValueError).

Mastery Gate Hints:
- Concept Pointer: Think of polynomial regression as linear regression on transformed features.
- Narrower: Ensure your `transform` method properly adds the bias term (x^0).
- Debugging: If loss goes to NaN, your higher degree features are causing overflow. Scale your inputs!
"""
