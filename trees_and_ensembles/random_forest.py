"""
A. **Mission**
Random Forest is a parallel ensemble combining multiple decision trees using bootstrap aggregating (bagging) and feature subsampling to reduce variance.

B. **Prerequisites**
- [Stage 0F: ML Core](../foundations/README.md#module-0f-machine-learning-core-concepts) (bias-variance tradeoff)
- Decision Tree implementation

C. **Learning Questions**
1. Why does Random Forest outperform a single Tree?
2. How does bagging reduce variance?
3. Why feature subsampling?

D. **Mathematics to Derive**
1. Why averaging $M$ independent models reduces variance by $1/M$.
2. Out-of-Bag probability limit as $N \to \infty$.

E. **Implementation Contract**
- Implement `class RandomForest`.
- Methods:
  - `__init__(self, task='classification', n_trees=100, max_depth=None, max_features='sqrt')`
  - `fit(self, X, y)`: Build forest.
  - `predict(self, X)`: Return aggregated predictions.

F. **Guided Implementation Stages**
- **Step 0**: Modify Decision Tree.
  - *What to learn*: Feature subsampling.
  - *What to do*: Allow `max_features` in DecisionTree split search.
  - *How to check yourself*: Tree only evaluates subset of features.
  - *When to proceed*: Subset logic works.
  - *Recovery hints 1*: Check random choice without replacement.
- **Step 1**: Bootstrapping.
  - *What to learn*: Sampling with replacement.
  - *What to do*: Implement bootstrap generator.
  - *How to check yourself*: Generated size matches original, has duplicates.
  - *When to proceed*: Generator works.
  - *Recovery hints 1*: Use `np.random.choice`.
- **Step 2**: Forest Construction.
  - *What to learn*: Building the ensemble.
  - *What to do*: Implement loop over `n_trees`, fit trees on bootstrap samples.
  - *How to check yourself*: Forest stores multiple different trees.
  - *When to proceed*: All trees fit successfully.
  - *Recovery hints 1*: Ensure random seed varies if explicitly set.
- **Step 3**: Aggregation.
  - *What to learn*: Voting and averaging.
  - *What to do*: Implement `predict` by aggregating tree predictions.
  - *How to check yourself*: Test on Dataset 3C (should be smoother than single tree).
  - *When to proceed*: Works for both task types.
  - *Recovery hints 1*: Use `mode` for classification, `mean` for regression.

G. **Edge Cases and Expected Tests**
- `n_trees=1` behaves similarly to single tree.
- Constant target prediction.

H. **Complexity Analysis**
1. Training time complexity vs single tree?
2. Memory footprint?

I. **Progressive Difficulty Levels & Definition of Done**
- Level 1 Guided: Implement classification forest.
- Level 2 Standard: Implement regression and feature subsampling.
- Level 3 Challenge: Calculate Out-of-Bag (OOB) error.
- Done: Reduces overfitting on Dataset 3C.

J. **Reflection**
- Interpretability vs Performance.
"""
