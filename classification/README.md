# Chapter 2: Classification

## Objectives
This chapter introduces the fundamental concepts of classification. By completing these exercises, you will understand:
- How to define and interpret decision boundaries.
- The difference between discriminative models and generative/probabilistic classifiers.
- The mechanics of distance-based and margin-based classification methods.

## Prerequisites
- **Hard Requirements**: Gradient descent, loss functions (from Chapter 1), linear algebra, and basic calculus.
- **Recommended**: Familiarity with vectorization in NumPy.

## Recommended Study Sequence
For the best learning experience, proceed in the following order:
1. **[Logistic Regression](./logistic_regression.py)**: Start by extending linear regression concepts to classification using log-loss.
2. **[K-Nearest Neighbors (KNN)](./knn.py)**: Shift to a non-parametric, distance-based approach to build intuition on instance-based learning.
3. **[Gaussian Naive Bayes](./gaussian_naive_bayes.py)**: Explore a probabilistic, generative approach using Bayes' theorem.
4. **[Linear SVM](./linear_svm.py)**: Conclude with margin maximization and hinge loss optimization.

## Exercise Index
- [`logistic_regression.py`](./logistic_regression.py): Binary classification using sigmoid and cross-entropy loss. Focuses on gradient descent optimization.
- [`knn.py`](./knn.py): K-Nearest Neighbors implementation. Covers distance metrics, the bias-variance tradeoff, and lazy learning.
- [`gaussian_naive_bayes.py`](./gaussian_naive_bayes.py): Generative probabilistic model applying Bayes' theorem with Gaussian class conditionals.
- [`linear_svm.py`](./linear_svm.py): Linear Support Vector Machine using hinge loss and gradient descent for margin maximization.

## Cross-Chapter Conceptual Questions
- How do discriminative classifiers (Logistic Regression, SVM) differ from generative classifiers (Naive Bayes) in terms of what they learn?
- When would you choose a lazy learner like KNN over an eager learner like Logistic Regression?
- How does the concept of loss functions evolve from Mean Squared Error in Chapter 1 to Cross-Entropy and Hinge Loss in this chapter?

## How Algorithms Relate
- **Logistic Regression vs Linear SVM**: Both learn a linear decision boundary but use different objective functions (log-loss vs hinge loss). SVM maximizes the margin, while Logistic Regression models probabilities.
- **KNN vs Naive Bayes**: Both can model non-linear boundaries (in their native forms), but KNN relies on local distance metrics while Naive Bayes relies on global statistical distributions under strong independence assumptions.

## Chapter Math Learning Goals
- Derive the gradient of cross-entropy (log-loss) with respect to weights.
- Understand the mathematical formulation of Euclidean distance.
- Formulate Bayes' theorem for classification and derive the log-likelihood for Gaussian conditionals.
- Formulate the hinge loss objective and derive its subgradients.

## Completion Checklist
- [ ] Implement Logistic Regression and pass all tests.
- [ ] Implement K-Nearest Neighbors and pass all tests.
- [ ] Implement Gaussian Naive Bayes and pass all tests.
- [ ] Implement Linear SVM and pass all tests.
- [ ] Answer all reflection questions in the docstrings.

## Personal Notes
> Use this space to record your insights, common bugs encountered, or ideas for further exploration.
