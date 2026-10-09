"""
A. Mission
Lasso Regression (L1 regularization) adds a penalty proportional to the absolute value of the weights. Unlike Ridge, Lasso tends to push some weights to exactly zero, effectively performing automatic feature selection and producing sparse models.

B. Prerequisites
- Understanding of linear regression and loss functions.
- L1 norm definition.
- Familiarity with the concept of subgradients (since absolute value is not differentiable at zero).

C. Learning Questions
1. Geometrically, why does the L1 constraint region (a diamond in 2D) cause the loss contours to often hit corners, resulting in exactly zero weights?
2. How does Lasso compare to Ridge in the presence of highly correlated features?
3. Why is there no closed-form solution for Lasso?

D. Mathematics to Derive
1. Write the full objective function (MSE + L1 penalty).
2. Derive the coordinate descent update rule for a single weight, or the subgradient of the L1 penalty.
3. Formulate the soft-thresholding operator used in the update step if using coordinate descent.

E. Implementation Contract
Create a class `LassoRegression` with the following methods:
- `__init__(self, alpha: float, learning_rate: float = 0.01, epochs: int = 1000)`
- `fit(self, X: np.ndarray, y: np.ndarray) -> None`: Computes weights using subgradient descent (or coordinate descent). NumPy allowed for array ops.
- `predict(self, X: np.ndarray) -> np.ndarray`: Returns predictions.

F. Guided Implementation Stages
1. Initialize weights to zero and augment X with a bias column.
2. Set up the training loop for `epochs`.
3. In the forward pass, calculate current predictions and the error.
4. Compute the gradient of the MSE part.
5. Compute the subgradient of the L1 penalty (using sign(weights), handling 0 appropriately). Ensure the bias term is not penalized.
6. Update weights by moving opposite the sum of both gradients.

G. Edge Cases and Expected Tests
1. Test with alpha=0: should approximate OLS.
2. Test with dataset containing irrelevant features: verify that Lasso sets the weights of irrelevant features to exactly 0 (or within a tight tolerance).
3. Test with large alpha: all non-bias weights should become 0.

H. Complexity Analysis
1. What is the time complexity of an epoch of subgradient descent?
2. How does the sparsity of weights affect potential prediction time speedups?

I. Definition of Done
The model successfully identifies and nullifies useless features in a dataset by driving their weights to 0, demonstrating the feature selection property of L1 regularization.

J. Reflection
1. Did you notice oscillation around 0 for weights due to the subgradient approach? How could coordinate descent fix this?
2. In practice, when would you prefer Lasso over Ridge?
"""
