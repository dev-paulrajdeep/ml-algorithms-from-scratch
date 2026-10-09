# Chapter 1: Regression

## Chapter Objectives
Regression is the foundation of supervised machine learning. In this chapter, you will:
- Understand the core mechanics of fitting a model to continuous data.
- Learn the concept of loss functions, specifically Mean Squared Error (MSE).
- Apply gradient descent to optimize parameters iteratively.
- Implement closed-form analytical solutions and iterative numerical solutions.
- Understand the bias-variance tradeoff through polynomial regression.
- Master regularization techniques (L1/Lasso and L2/Ridge) to prevent overfitting.

## Prerequisites
Before tackling this chapter, ensure you are comfortable with the following foundations:
- [Module 0B: NumPy and Numerical Computing](../foundations/README.md#module-0b-numpy-and-numerical-computing) (Vectorization, Matrix operations)
- [Module 0C: Calculus and Optimization](../foundations/README.md#module-0c-calculus-and-optimization) (Derivatives, Gradients, Loss functions)
- [Module 0D: Linear Algebra](../foundations/README.md#module-0d-linear-algebra) (Matrix multiplication, Transpose, Inverse)
- [Module 0F: Debugging Numerical Algorithms](../foundations/README.md#module-0f-debugging-numerical-algorithms) (Handling NaNs, numerical stability)

## Chapter 1 Learning Ladder (The Gentle Progression)
Before implementing a full model, complete these conceptual stages:
1. Predict output for single input manually
2. Understand weight and bias meaning
3. Calculate a residual for one observation
4. Calculate squared error
5. Calculate MSE over tiny dataset
6. Reason about fitting a line to data
7. How changing weight changes predictions
8. How changing bias changes predictions
9. Idea of minimizing a loss function
10. Derive derivative of simple scalar loss
11. Connect derivative to update in opposite direction
12. Perform one GD update by hand
13. Repeat for several iterations
14. Track loss and inspect training
15. Generalize to multiple features
16. Express using vector/matrix notation
17. Introduce vectorized implementation
18. Compare against hand-computable cases
19. Test edge cases
20. Analyze convergence and complexity

## Recommended Study Sequence
1. `linear_regression.py` (Reference implementation - Do not modify)
2. `log_transformed_exp_regression.py` (Reference implementation - Do not modify)
3. `polynomial_regression.py` (Your first implementation task)
4. `ridge_regression.py` (L2 Regularization)
5. `lasso_regression.py` (L1 Regularization)

**Important Prerequisite check:** Before attempting Ridge and Lasso, ensure you deeply understand the Ordinary Least Squares (OLS) objective function from polynomial regression and the meaning of a penalty term.

## Toy Datasets for Hand Calculations
- **Dataset R1 (Linear):** `xs = [1, 2, 3, 4, 5]`, `ys = [2, 4, 6, 8, 10]`
- **Dataset R2 (Quadratic):** `xs = [1, 2, 3]`, `ys = [1, 4, 9]`

## Completion Checklist
- [ ] Understand the 20-step Learning Ladder.
- [ ] Review `linear_regression.py` and understand vectorized GD.
- [ ] Implement `polynomial_regression.py` passing all 3 difficulty levels.
- [ ] Understand OLS matrix formulation and invertibility requirements.
- [ ] Implement `ridge_regression.py` using closed-form analytical solution.
- [ ] Compare Ridge behavior to unregularized regression.
- [ ] Understand subgradients and coordinate descent for L1 penalty.
- [ ] Implement `lasso_regression.py` and verify feature sparsity.

## Cross-Chapter Conceptual Questions
1. How does gradient descent for linear regression compare to the closed-form solution in terms of computational complexity as the number of features grows?
2. Why is it crucial not to penalize the bias term in Ridge and Lasso regression?
3. How does polynomial feature expansion relate to the kernel trick in Support Vector Machines?

## Personal Notes
> *Use this space to record your epiphanies, stuck points, and reflections as you work through the chapter.*
