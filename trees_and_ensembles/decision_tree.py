"""
A. **Mission**
The Decision Tree is a non-parametric supervised learning method. It recursively partitions the feature space into pure regions. Highly interpretable but prone to overfitting unless regularized.

B. **Prerequisites**
- [Stage 0F: ML Core](../foundations/README.md#module-0f-machine-learning-core-concepts) (overfitting)
- [Stage 0E: Probability & Statistics](../foundations/README.md#module-0e-probability-and-statistics) (variance reduction)

C. **Learning Questions**
1. Why does a deep decision tree tend to overfit?
2. How does Gini vs Entropy affect tree structure?
3. Why variance reduction for regression?

D. **Mathematics to Derive**
1. Write the formula for Gini Impurity for proportions $p_k$.
2. Write the formula for Entropy.
3. Write the variance reduction formula.

E. **Implementation Contract**
- Implement `class DecisionTree`.
- Methods:
  - `__init__(self, task='classification', criterion='gini', max_depth=None, min_samples_leaf=1)`
  - `fit(self, X, y)`: Build tree.
  - `predict(self, X)`: Return predictions.

F. **Guided Implementation Stages**
- **Step 0**: Implement impurity functions (Gini, Entropy, Variance).
  - *What to learn*: Calculating node purity.
  - *What to do*: Implement `_gini(y)`, `_entropy(y)`, `_variance(y)`.
  - *How to check yourself*: Try with homogeneous and perfectly mixed arrays.
  - *When to proceed*: When tests pass.
  - *Recovery hints 1*: Check probability calculations.
  - *Recovery hints 2*: For variance, test against np.var.
  - *Recovery hints 3*: Handle empty array edge cases.
- **Step 1**: Calculate Information Gain.
  - *What to learn*: Evaluating splits.
  - *What to do*: Implement impurity decrease.
  - *How to check yourself*: Test with Dataset 3A.
  - *When to proceed*: Gain is highest for perfect split.
  - *Recovery hints 1*: Ensure weighted average is used for children.
- **Step 2**: Find best split.
  - *What to learn*: Iterating over features/thresholds.
  - *What to do*: Implement search for maximum information gain.
  - *How to check yourself*: Dataset 3A should find threshold 3.5.
  - *When to proceed*: Correct split found.
  - *Recovery hints 1*: Check for unique values across features.
- **Step 3**: Recursive Tree Building.
  - *What to learn*: Tree traversal and recursion limits.
  - *What to do*: Implement `_build_tree`.
  - *How to check yourself*: Tree halts at `max_depth`.
  - *When to proceed*: Correct tree built for Dataset 3B.
  - *Recovery hints 1*: Verify stopping conditions.
- **Step 4**: Prediction.
  - *What to learn*: Traversing the fitted tree.
  - *What to do*: Implement `predict`.
  - *How to check yourself*: Try predicting training set.
  - *When to proceed*: Works for both task types.
  - *Recovery hints 1*: Check leaf output (majority/mean).

G. **Edge Cases and Expected Tests**
- Pure node directly forms a leaf.
- Unsplittable data.
- Small `min_samples_leaf` constraints.

H. **Complexity Analysis**
1. Time complexity of finding best split?
2. Space complexity of tree?

I. **Progressive Difficulty Levels & Definition of Done**
- Level 1 Guided: Complete classification with Gini.
- Level 2 Standard: Add regression with Variance Reduction.
- Level 3 Challenge: Add Entropy, optimize split search.
- Done: Passes Dataset 3A/3B/3C tests.

J. **Reflection**
- Why do trees create orthogonal boundaries?
"""
