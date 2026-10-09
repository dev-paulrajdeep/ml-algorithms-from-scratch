# Chapter 3: Trees and Ensembles

Welcome to Chapter 3! In this chapter, we transition from linear models to non-parametric, non-linear models based on recursive partitioning of the feature space. You will implement the foundational decision tree algorithm and then build two of the most powerful and widely-used ensemble methods in machine learning: Random Forests and Gradient Boosting Machines.

## Chapter Objectives
- Understand and implement recursive partitioning of feature spaces.
- Master different splitting criteria: Gini impurity, entropy, and variance reduction.
- Grasp the concepts of ensemble learning, specifically bagging and boosting.
- See how individual weak learners can be combined into highly accurate, robust models.

## Prerequisites
- **Hard Prerequisites**:
  - Supervised learning fundamentals (from Chapters 1 & 2).
  - Understanding of loss functions (MSE, Log Loss).
  - Bias-variance tradeoff and overfitting concepts.
- **Recommended**:
  - Basic probability and statistics.
  - Familiarity with recursive functions.

## Recommended Study Sequence
The algorithms in this chapter build directly upon one another. You must complete them in this order:
1. **Decision Tree** (The base learner)
2. **Random Forest** (Parallel ensemble of trees via bagging)
3. **Gradient Boosting** (Sequential ensemble of trees via boosting)

## Exercise Index
- [`decision_tree.py`](./decision_tree.py): Implement the CART algorithm for both classification and regression. You will build a recursive tree structure that splits data to maximize purity.
- [`random_forest.py`](./random_forest.py): Combine multiple decision trees using bootstrap aggregation (bagging) and feature subsampling to reduce variance and combat overfitting.
- [`gradient_boosting.py`](./gradient_boosting.py): Implement a sequential ensemble where each new tree fits the negative gradient (residuals) of the previous stage's predictions, essentially performing gradient descent in function space.

## Conceptual Questions
- **Trees vs. Linear Models**: Under what conditions would a simple linear regression model outperform a deep decision tree? Conversely, when is a decision tree vastly superior?
- **Ensemble Connections**: How does the architecture of a Random Forest address the weaknesses of a single Decision Tree? How does Gradient Boosting take a different approach to improving upon a single tree?
- **From Single to Boosted Trees**: How do the three algorithms build on each other? Consider how a single tree forms the base, bagged trees create a parallel ensemble to fix variance, and boosted trees create a sequential ensemble to fix bias.
- **Base Learners**: Why do Random Forests typically use deep, unpruned trees as base learners, while Gradient Boosting typically uses shallow trees (weak learners)?

## Chapter Math Learning Goals
- Formulate and calculate **Gini Impurity** and **Entropy**.
- Calculate **Information Gain** (Impurity Decrease) for a given split.
- Understand how **Variance Reduction** serves as the splitting criterion for regression trees.
- Derive the **Gradient Boosting update rule** by finding the negative gradient of the Mean Squared Error (and optionally, Log Loss).

## Completion Checklist
- [ ] Implement DecisionTree (Classification with Gini/Entropy)
- [ ] Implement DecisionTree (Regression with Variance Reduction)
- [ ] Implement RandomForest (Bagging and Feature Subsampling)
- [ ] Implement GradientBoosting (Sequential residual fitting)
- [ ] Answer all mathematical derivation questions in the docstrings.

## Personal Notes
*(Use this space to record your insights, gotchas, or reminders as you work through the chapter)*
