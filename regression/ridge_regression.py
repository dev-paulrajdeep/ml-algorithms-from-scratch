"""
Ridge Regression (L2 Regularization) Curriculum Guide

A. Step 0: Understand the problem
When features are highly correlated (multicollinearity) or we have more features than samples, ordinary least squares (OLS) can overfit, resulting in wildly large weights. Ridge regression adds a penalty to constrain the weights.

B. Step 1: Build intuition
Imagine balancing a scale. OLS only cares about minimizing prediction error, even if it means putting 1000kg on one side and -999kg on the other. Ridge adds a cost to the weight itself, encouraging the model to use smaller, more balanced weights.

C. Step 2: Tiny numerical example
With 2 identical features x1=x2 and output y, OLS could pick weights [1000, -999]. Ridge with penalty alpha=1 will strongly prefer weights like [0.5, 0.5] because 0.5^2 + 0.5^2 (0.5) is much smaller than 1000^2 + (-999)^2.

D. Step 3: Mathematical notation
Objective function: J(w) = MSE + alpha * ||w||_2^2
Note: ||w||_2^2 is the squared L2 norm (sum of squared weights). alpha (α) is the regularization strength.

E. Step 4: Derive the math
The Ridge objective is: J(w) = (Xw - y)^T (Xw - y) + alpha * w^T w
Taking derivative and setting to zero yields the closed-form solution:
w = (X^T X + alpha * I)^{-1} X^T y
The addition of alpha * I makes the matrix strictly positive definite, guaranteeing invertibility!
CRITICAL: We do not penalize the bias term, so the first diagonal element of I should be 0.

F. Step 5: Design algorithm in plain English and pseudocode
1. Prepend a column of 1s to X for the bias term.
2. Create an identity matrix I of size (features + 1).
3. Set I[0, 0] = 0 to avoid penalizing the bias.
4. Compute w = inverse(X^T X + alpha * I) @ X^T y.

G. Step 6: Implement simplest version (Level 1 - Guided)
- Implement for 1D data with a predefined alpha.
- Ensure the bias trick is applied correctly.

H. Step 7: Generalize (Level 2 - Standard)
- Create `RidgeRegression` class.
- Implement the closed-form solution for arbitrary dimensions.

I. Step 8: Test and debug
- Test with alpha=0 (should match OLS).
- Test with large alpha (weights should shrink towards zero).
- Compare predictions against sklearn's Ridge.

J. Step 9: Analyze (Level 3 - Challenge)
- Analyze the computational complexity of the matrix inversion O(d^3).
- Implement gradient descent for Ridge as an alternative to closed-form.

Implementation Contract:
class RidgeRegression:
    def __init__(self, alpha=1.0): ...
    def fit(self, X, y): ...
    def predict(self, X): ...

Mastery Gate Hints:
- Concept Pointer: Ridge pulls weights to zero, but rarely exactly zero.
- Narrower: Make sure you don't regularize the intercept.
- Debugging: If predictions are uniformly too low, you might be shrinking the bias term.
"""
