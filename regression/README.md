# Chapter 1: Regression

## Chapter Objectives
- Understand the foundational mechanics of linear and non-linear regression models.
- Learn how to map input features to a continuous target variable.
- Differentiate between unregularized, L1, and L2 regularized approaches.
- Implement closed-form and iterative optimization techniques.

## Prerequisites
- None. This is the entry point to the ML curriculum.

## Recommended Study Sequence
1. `linear_regression.py` (Completed)
2. `log_transformed_exp_regression.py` (Completed)
3. `polynomial_regression.py` (Exercise)
4. `ridge_regression.py` (Exercise)
5. `lasso_regression.py` (Exercise)

## Exercise Index
- [linear_regression.py](./linear_regression.py) - Bare script with hardcoded xs/ys, gradient descent, prints loss per epoch (Completed Implementation)
- [log_transformed_exp_regression.py](./log_transformed_exp_regression.py) - Bare script, log-transforms exponential data, fits linear model in log-space, does inference (Completed Implementation)
- [polynomial_regression.py](./polynomial_regression.py) - Exercise: Extend linear regression to polynomial features, exploring bias-variance tradeoff.
- [ridge_regression.py](./ridge_regression.py) - Exercise: Implement L2 regularization via closed-form solution.
- [lasso_regression.py](./lasso_regression.py) - Exercise: Implement L1 regularization using subgradient/coordinate descent for feature selection.

## Cross-Chapter Conceptual Questions
- How does the concept of gradient descent here apply to the optimization of Neural Networks?
- Why might regularization techniques like Ridge and Lasso be necessary when building complex models like SVMs or deep networks?

## Chapter-Level Math Learning Goals
- Understand and derive the gradient of Mean Squared Error.
- Manipulate matrix operations to derive the closed-form normal equations.
- Explore the geometry of L1 and L2 penalty regions.

## Completion Checklist
- [ ] Read through and understand `linear_regression.py`.
- [ ] Run and analyze `log_transformed_exp_regression.py`.
- [ ] Implement and test `polynomial_regression.py`.
- [ ] Implement and test `ridge_regression.py`.
- [ ] Implement and test `lasso_regression.py`.

## Personal Notes
*(Use this space to jot down insights, difficult concepts, or reminders)*
