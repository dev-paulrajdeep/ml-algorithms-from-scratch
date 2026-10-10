# Chapter 1: Regression

Welcome to Chapter 1! Regression is the foundational starting point for supervised machine learning. In this chapter, you will learn how to map continuous input features to a continuous numerical target variable, optimize parameters through gradient descent and analytical normal equations, and control model complexity through regularization.

---

## Chapter Objectives
- Understand the core mechanics of mapping input features to continuous targets ($f(\mathbf{x}) \approx y$).
- Learn how loss functions, specifically Mean Squared Error (MSE), quantify prediction error.
- Derive and implement first-order iterative optimization (gradient descent) for parameter learning.
- Derive and solve the closed-form Normal Equations using linear algebra.
- Explore the bias-variance tradeoff through polynomial feature expansion.
- Differentiate between unregularized, $L_2$-regularized (Ridge), and $L_1$-regularized (Lasso) models.

---

## Prerequisites
Before tackling Chapter 1, ensure you have completed or tested out of the following Stage 0 modules:
- **[Stage 0B: NumPy & Numerical Computing](../foundations/README.md#module-0b-numpy-and-numerical-computing)** — Array shapes, 2D matrix indexing, axis reductions (`mean(axis=0)`), broadcasting rules, and avoiding outer-product traps.
- **[Stage 0C: Linear Algebra](../foundations/README.md#module-0c-linear-algebra)** — Vectors, dot products as weighted sums, matrix multiplication ($\mathbf{X}\mathbf{w}$), transposes, and matrix invertibility.
- **[Stage 0D: Calculus & Optimization](../foundations/README.md#module-0d-calculus-and-optimization)** — Derivatives as slopes, partial derivatives, gradient vectors, and scalar gradient descent step-by-step.
- **[Stage 0F: Machine Learning Core](../foundations/README.md#module-0f-machine-learning-core-concepts)** — Features, labels, parameters, predictions, train/test splits, and baseline models.

---

## 🪜 The Chapter 1 Learning Ladder (Gentle 20-Stage Progression)

To make the jump from basic arithmetic to a complete machine learning model completely beginner-proof, work through these twenty sequential checkpoints before writing an object-oriented estimator:

1. **Manual Prediction for One Point**: Compute $\hat{y} = w \cdot x + b$ for $x = 3$ when $w = 2.0$ and $b = 1.0$. ($\hat{y} = 2(3) + 1 = 7$).
2. **Interpret Weight & Bias**: Recognize that $w$ governs the line's slope (sensitivity of $y$ to $x$) and $b$ is the intercept (output when $x = 0$).
3. **Calculate a Single Residual**: For observed true target $y = 6$ and prediction $\hat{y} = 7$, compute residual $e = \hat{y} - y = +1$.
4. **Compute Squared Error**: Calculate individual loss $L_i = \frac{1}{2}(\hat{y}_i - y_i)^2 = \frac{1}{2}(1)^2 = 0.5$. Understand why squaring penalizes large errors more heavily than small ones.
5. **Compute Mean Squared Error (MSE)**: For 3 points, calculate the average squared error: $\text{MSE} = \frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i)^2$.
6. **Reason About Line Fitting**: Visualize or sketch why rotating or shifting a line reduces total distance to data points.
7. **Effect of Adjusting Weight**: If predictions are consistently too steep for high $x$, observe that decreasing $w$ reduces error.
8. **Effect of Adjusting Bias**: If predictions are parallel to the data but shifted upward, observe that decreasing $b$ shifts the entire line down.
9. **The Objective of Learning**: Formalize the goal: find $(w, b)$ that minimizes $J(w, b) = \frac{1}{2N} \sum_{i=1}^N (w x_i + b - y_i)^2$.
10. **Derive the Scalar Derivatives**:
    $$\frac{\partial J}{\partial w} = \frac{1}{N} \sum_{i=1}^N (w x_i + b - y_i) x_i, \quad \frac{\partial J}{\partial b} = \frac{1}{N} \sum_{i=1}^N (w x_i + b - y_i)$$
11. **Negative Gradient Direction**: Understand why the update moves opposite the derivative: if $\frac{\partial J}{\partial w} > 0$, increasing $w$ increases loss, so we must decrease $w$.
12. **One Hand-Calculated GD Step**: On Dataset R1, starting at $w=0, b=0$ with learning rate $\alpha = 0.01$, compute the new parameters after 1 epoch by hand.
13. **Repeat for Several Iterations**: Observe how predictions move closer to true $y$ with each update.
14. **Track Training Loss**: Verify that loss decreases monotonically across epochs. If it oscillates or explodes, diagnose the learning rate.
15. **Generalize to Multiple Features**: Extend the scalar formula to $D$ features: $\hat{y} = w_1 x_1 + w_2 x_2 + \dots + w_D x_D + b$.
16. **Matrix & Vector Notation**: Express predictions compactly as $\hat{\mathbf{y}} = \mathbf{X}\mathbf{w} + b \mathbf{1}$, and the gradient as $\nabla_{\mathbf{w}} J = \frac{1}{N} \mathbf{X}^T (\hat{\mathbf{y}} - \mathbf{y})$.
17. **Vectorized NumPy Implementation**: Replace loops over features with matrix multiplication `X @ w`.
18. **Compare with Known Ground Truth**: Verify your vectorized code on Dataset R1 against the known mathematical minimum ($w=2.0, b=0.0$).
19. **Test Edge Cases & Shapes**: Verify behavior on zero inputs, single sample inputs ($N=1$), and mismatched dimensions.
20. **Analyze Complexity & Convergence**: Evaluate time complexity per epoch ($O(N \cdot D)$) and memory footprint ($O(D)$ parameters).

---

## Recommended Study Sequence

Follow this order through Chapter 1:
1. **[`linear_regression.py`](./linear_regression.py)** *(Reference Implementation)* — Inspect how a raw loop trains $w$ and $b$ using gradient descent on scalar data. Run it, observe loss decay per 500 epochs.
2. **[`log_transformed_exp_regression.py`](./log_transformed_exp_regression.py)** *(Reference Implementation)* — Observe how transforming targets into log-space allows a linear model to fit exponential relationships ($y = 2^x$).
3. **[`polynomial_regression.py`](./polynomial_regression.py)** *(Your First Implementation)* — Build a class that expands 1D features into polynomial powers, illustrating the bias-variance tradeoff.
4. **[`ridge_regression.py`](./ridge_regression.py)** *(L2 Regularization)* — Add an analytical $L_2$ penalty $(\mathbf{X}^T \mathbf{X} + \alpha \mathbf{I})^{-1} \mathbf{X}^T \mathbf{y}$ to solve multicollinearity and shrink weights.
5. **[`lasso_regression.py`](./lasso_regression.py)** *(L1 Regularization)* — Implement $L_1$ regularization via subgradient descent or coordinate descent with soft-thresholding to produce exact feature sparsity.

> [!IMPORTANT]
> **Prerequisite Bridge to Regularization**: Before attempting Ridge or Lasso, ensure you thoroughly understand why ordinary least squares can overfit when features are correlated or noisy. Regularization modifies the objective by adding a penalty term $P(\mathbf{w})$:
> $$\text{Total Loss} = \text{Data Misfit (MSE)} + \alpha \cdot \text{Model Complexity Penalty}$$

---

## Chapter 1 Toy Datasets for Hand Calculations

Use these tiny datasets to verify derivations before running code:

- **Dataset R1 (1D Linear)**:
  - $\mathbf{x} = [1, 2, 3, 4, 5]$
  - $\mathbf{y} = [2, 4, 6, 8, 10]$
  - Optimal solution: $w = 2.0$, $b = 0.0$, MSE $= 0.0$.
  - Hand check: At $w_0 = 0, b_0 = 0$, prediction $\hat{y} = [0, 0, 0, 0, 0]$, residuals $\hat{y} - y = [-2, -4, -6, -8, -10]$.

- **Dataset R2 (Quadratic Curve)**:
  - $\mathbf{x} = [1, 2, 3]$
  - $\mathbf{y} = [1, 4, 9]$
  - Transformed features for degree 2 (with bias): $\mathbf{X}_{\text{poly}} = \begin{bmatrix} 1 & 1 & 1 \\ 1 & 2 & 4 \\ 1 & 3 & 9 \end{bmatrix}$.
  - Optimal weights: $[w_0, w_1, w_2] = [0, 0, 1.0]$.

- **Dataset R3 (Collinear Features for Ridge)**:
  - $\mathbf{X} = \begin{bmatrix} 1 & 2 \\ 2 & 4 \\ 3 & 6 \end{bmatrix}$, $\mathbf{y} = [3, 6, 9]$
  - Feature 2 is exactly $2 \times$ Feature 1. $\mathbf{X}^T \mathbf{X}$ is singular (non-invertible). OLS fails, but Ridge $(\mathbf{X}^T \mathbf{X} + \alpha \mathbf{I})^{-1}$ succeeds!

---

## Completion Checklist
- [ ] Traced through all 20 stages of the Chapter 1 Learning Ladder.
- [ ] Ran and analyzed `linear_regression.py` and `log_transformed_exp_regression.py`.
- [ ] Completed `polynomial_regression.py` (passed Level 1, 2, and 3).
- [ ] Understood why high polynomial degrees cause oscillating weights and overfitting.
- [ ] Completed `ridge_regression.py` using closed-form Normal Equations.
- [ ] Verified that Ridge handles collinear features (Dataset R3) without numerical failure.
- [ ] Completed `lasso_regression.py` using subgradient or coordinate descent.
- [ ] Confirmed that Lasso drives irrelevant feature weights to exactly zero.

---

## Cross-Chapter Conceptual Questions
1. **Connection to Neural Networks**: How does the gradient descent update rule in linear regression compare to the weight updates in a single artificial neuron (Chapter 6)?
2. **Ridge vs Lasso**: Why does the spherical geometry of $L_2$ shrinkage never push weights to exactly zero, while the diamond geometry of $L_1$ shrinkage does?
3. **Unpenalized Intercept**: Why is it standard practice never to regularize the bias term $b$? What would happen to predictions if $b$ were heavily penalized toward zero?

---

## Personal Notes
*(Use this space to record your derivations, observations, and insights.)*
