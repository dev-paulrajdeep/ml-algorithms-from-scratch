# Chapter 2: Classification

## Objectives
This chapter introduces the fundamental concepts of classification. By completing these exercises, you will understand:
- How to model discrete labels instead of continuous values.
- How score functions define decision boundaries.
- The difference between discriminative models and generative/probabilistic classifiers.
- The mechanics of distance-based vs margin-based classification methods.

## Prerequisites
- **[Stage 0D: Calculus & Optimization](../foundations/README.md#module-0d-calculus-and-optimization)**: Gradients, partial derivatives, optimization.
- **[Stage 0E: Probability & Statistics](../foundations/README.md#module-0e-probability-and-statistics)**: Bayes' theorem, distributions.
- **[Stage 0F: ML Core](../foundations/README.md#module-0f-machine-learning-core-concepts)**: Train/test split, overfitting.

## Recommended Study Sequence
For the best learning experience, proceed in the following order:
1. **[Logistic Regression](./logistic_regression.py)**
2. **[K-Nearest Neighbors (KNN)](./knn.py)**
3. **[Gaussian Naive Bayes](./gaussian_naive_bayes.py)**
4. **[Linear SVM](./linear_svm.py)**

## Chapter 2 Toy Datasets
These small, intuitive datasets will help you verify your implementations manually before scaling up:
- **Dataset 2A (1D binary separation)**: `X = [[1], [2], [3], [6], [7], [8]]`, `y = [0, 0, 0, 1, 1, 1]` (boundary at x=4.5)
- **Dataset 2B (2D Gaussian blobs)**: Class 0 centered at (1, 1), Class 1 centered at (4, 4)
- **Dataset 2C (Non-linearly separable / Concentric circles or XOR)**: illustrates linear boundary failures

## Pedagogical Progression
- **Before Logistic Regression**: Understand binary labels and how a raw score $z = w^Tx + b$ can be squashed into a probability range $[0,1]$ via the sigmoid function. Build intuition for cross-entropy loss before deriving its gradients.
- **Before KNN**: Understand distance metrics (Euclidean, Manhattan), the concept of voting among neighbors, and how the curse of dimensionality impacts distance in high-dimensional spaces.
- **Before Naive Bayes**: Review the concepts of prior, likelihood, Bayes' theorem, and the critical conditional independence assumption that makes this method "naive".
- **Before Linear SVM**: Understand geometric margins, the concept of support vectors, and how hinge loss provides a robust alternative to 0-1 loss for margin maximization.

## Exercise Index
- [`logistic_regression.py`](./logistic_regression.py): Binary classification using sigmoid and cross-entropy loss.
- [`knn.py`](./knn.py): K-Nearest Neighbors implementation and distance metrics.
- [`gaussian_naive_bayes.py`](./gaussian_naive_bayes.py): Generative probabilistic model applying Bayes' theorem.
- [`linear_svm.py`](./linear_svm.py): Linear SVM using hinge loss and subgradient descent.

## Cross-Chapter Conceptual Questions
- How do discriminative classifiers (Logistic Regression, SVM) differ from generative classifiers (Naive Bayes) in terms of what they learn?
- When would you choose a lazy learner like KNN over an eager learner like Logistic Regression?

## Completion Checklist
- [ ] Read through all pedagogical progression notes.
- [ ] Implement Logistic Regression and pass all tests.
- [ ] Implement K-Nearest Neighbors and pass all tests.
- [ ] Implement Gaussian Naive Bayes and pass all tests.
- [ ] Implement Linear SVM and pass all tests.
- [ ] Answer all reflection questions in the docstrings.

## Personal Notes
> Use this space to record your insights, common bugs encountered, or ideas for further exploration.
