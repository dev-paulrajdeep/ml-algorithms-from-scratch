# Chapter 3: Trees and Ensembles

Welcome to Chapter 3! In this chapter, we transition from linear models to non-parametric, non-linear models based on recursive partitioning of the feature space.

## Chapter Objectives
- Understand and implement recursive partitioning of feature spaces.
- Master different splitting criteria: Gini impurity, entropy, and variance reduction.
- Differentiate bagging vs boosting.

## Prerequisites
- **Hard Prerequisites**:
  - [Stage 0F: ML Core](../foundations/README.md#module-0f-machine-learning-core-concepts) (overfitting, bias-variance tradeoff).
  - [Stage 0E: Probability & Statistics](../foundations/README.md#module-0e-probability-and-statistics) (variance reduction).
- **Recommended**:
  - Basic probability and statistics.
  - Familiarity with recursive functions.

## Recommended Study Sequence
The algorithms in this chapter build directly upon one another. You must complete them in this order:
1. **Decision Tree** (The base learner)
2. **Random Forest** (Parallel ensemble of trees via bagging)
3. **Gradient Boosting** (Sequential ensemble of trees via boosting)

## Chapter 3 Toy Datasets
Use these tiny datasets to verify your implementations:
- **Dataset 3A (1D single split dataset)**: `X = [[1], [2], [5], [6]]`, `y = [0, 0, 1, 1]` (split threshold at x=3.5 gives zero impurity).
- **Dataset 3B (2D axis-aligned step function)**: illustrates why orthogonal splits work well on axis-parallel boundaries.
- **Dataset 3C (Noisy polynomial)**: demonstrates single tree overfitting vs Random Forest variance reduction.

## Granular Progression
- **Start with manual split evaluation**: given 4 points, test split at threshold t and compute Gini of left and right subsets.
- **CART criteria**: CLEARLY DISTINGUISH classification (Gini impurity primary, entropy alternative) from regression (variance reduction / MSE reduction).
- **Ensembles**: Bagging (bootstrap samples + parallel aggregation to reduce variance) vs Boosting (sequential fitting of residuals / negative gradients to reduce bias).

## Exercise Index
- [`decision_tree.py`](./decision_tree.py): Implement the CART algorithm for both classification and regression.
- [`random_forest.py`](./random_forest.py): Combine multiple decision trees using bootstrap aggregation.
- [`gradient_boosting.py`](./gradient_boosting.py): Implement a sequential ensemble fitting residuals.

## Conceptual Questions
- Under what conditions would a simple linear regression model outperform a deep decision tree?
- How does the architecture of a Random Forest address the weaknesses of a single Decision Tree?

## Chapter Math Learning Goals
- Formulate and calculate **Gini Impurity** and **Entropy**.
- Calculate **Information Gain** (Impurity Decrease) for a given split.
- Understand how **Variance Reduction** serves as the splitting criterion for regression trees.
- Derive the **Gradient Boosting update rule**.

## Completion Checklist
- [ ] Implement DecisionTree (Classification with Gini/Entropy)
- [ ] Implement DecisionTree (Regression with Variance Reduction)
- [ ] Implement RandomForest
- [ ] Implement GradientBoosting
- [ ] Answer all mathematical derivation questions

## Personal Notes
*(Use this space to record your insights, gotchas, or reminders)*
