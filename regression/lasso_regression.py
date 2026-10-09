"""
Lasso Regression (L1 Regularization) Curriculum Guide

A. Step 0: Understand the problem
Sometimes we have many features, but only a few are actually important. We want a model that performs feature selection automatically by forcing the weights of irrelevant features to exactly zero. Lasso (Least Absolute Shrinkage and Selection Operator) does this.

B. Step 1: Build intuition
While Ridge (L2) acts like a rubber band pulling weights toward zero, Lasso (L1) acts like a diamond-shaped constraint. The corners of this diamond lie exactly on the axes, making it highly probable for the optimization path to hit a corner, setting some weights to exactly zero.

C. Step 2: Tiny numerical example
Features: x1 (useful), x2 (random noise), x3 (random noise).
Lasso might yield weights [2.5, 0.0, 0.0], effectively selecting only x1 and discarding x2 and x3.

D. Step 3: Mathematical notation
Objective: J(w) = MSE + alpha * ||w||_1
||w||_1 is the sum of absolute values of weights.
Because the absolute value function is not differentiable at zero, we use subgradients or coordinate descent.

E. Step 4: Derive the math
For coordinate descent, we update one weight at a time while holding others fixed.
The update rule uses the soft-thresholding operator:
S(p, lambda) = sign(p) * max(0, |p| - lambda)
Where p is the OLS update for a single coordinate.

F. Step 5: Design algorithm in plain English and pseudocode
We will use Gradient Descent with subgradients for simplicity in this curriculum, though coordinate descent is more common in practice.
Subgradient of |w| is sign(w) (and we can choose 0 if w=0).
Update rule: w = w - learning_rate * (gradient_of_MSE + alpha * sign(w)).
Remember: Do not penalize or shrink the bias term!

G. Step 6: Implement simplest version (Level 1 - Guided)
- Single feature + bias.
- Implement subgradient descent.

H. Step 7: Generalize (Level 2 - Standard)
- Implement `LassoRegression` class.
- Fit method with loop over epochs, updating weights using the subgradient.

I. Step 8: Test and debug
- Ensure irrelevant features have their weights driven to zero (or very close to it, depending on learning rate).
- Test with alpha=0 to ensure it matches standard linear regression.

J. Step 9: Analyze (Level 3 - Challenge)
- Subgradient descent for Lasso can oscillate around zero without reaching it exactly due to the fixed learning rate.
- Implement coordinate descent instead of subgradient descent for true sparsity.

Implementation Contract:
class LassoRegression:
    def __init__(self, alpha=1.0, learning_rate=0.01, epochs=1000): ...
    def fit(self, X, y): ...
    def predict(self, X): ...

Mastery Gate Hints:
- Concept Pointer: L1 regularization encourages sparsity (exact zeros).
- Narrower: Use `np.sign()` for the subgradient of the absolute value.
- Debugging: If weights bounce around zero and never settle, your learning rate is too high or you need to implement a soft-thresholding step.
"""
