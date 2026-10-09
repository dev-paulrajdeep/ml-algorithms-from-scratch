"""
A. Mission
Ridge Regression (L2 regularization) adds a penalty proportional to the square of the magnitude of the weights to the MSE loss. It prevents overfitting, stabilizes estimates when features are highly correlated (multicollinearity), and ensures a unique mathematical solution.

B. Prerequisites
- Solid understanding of linear regression.
- L2 norm definition.
- Matrix calculus for closed-form derivation.

C. Learning Questions
1. Why does adding a penalty on weights prevent overfitting?
2. How does the alpha parameter control the strength of the penalty? What happens when alpha=0 or alpha approaches infinity?
3. Why does L2 regularization shrink weights toward zero but rarely exactly to zero?
4. Consider the geometry of the L2 constraint region (a circle/sphere); how does this shape lead to non-zero weights?

D. Mathematics to Derive
1. Write the full objective function (MSE + L2 penalty).
2. Derive the closed-form normal equation for Ridge Regression. Ensure you handle the intercept term correctly so it isn't penalized.
3. Prove that the matrix (X^T X + alpha I) is always invertible for alpha > 0.

E. Implementation Contract
Create a class `RidgeRegression` with the following methods:
- `__init__(self, alpha: float)`
- `fit(self, X: np.ndarray, y: np.ndarray) -> None`: Computes weights using the closed-form solution. NumPy allowed for array ops.
- `predict(self, X: np.ndarray) -> np.ndarray`: Returns predictions.

F. Guided Implementation Stages
1. Add a column of ones to X for the bias term.
2. Construct the identity matrix of appropriate size.
3. Modify the first element of the identity matrix to 0, ensuring the bias weight is not penalized.
4. Compute (X^T X + alpha * I).
5. Solve for weights using the pseudo-inverse or linear solver (X^T y).

G. Edge Cases and Expected Tests
1. Test with alpha=0: should match ordinary least squares.
2. Test with perfectly collinear features: standard OLS fails, Ridge should output stable weights.
3. Test with a very large alpha: weights (excluding bias) should be extremely close to 0.

H. Complexity Analysis
1. What is the time complexity of computing the closed-form solution?
2. How does the number of features affect the computational cost of matrix inversion?

I. Definition of Done
The class correctly computes the closed-form Ridge weights, handles the unpenalized intercept properly, and gracefully fits datasets with multicollinearity.

J. Reflection
1. Under what circumstances would you choose gradient descent over the closed-form solution for Ridge?
2. How sensitive is the final model to the scale of the input features?
"""
