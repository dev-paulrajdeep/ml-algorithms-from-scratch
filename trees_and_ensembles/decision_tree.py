"""
A. **Mission**
The Decision Tree is a non-parametric supervised learning method used for both classification and regression. The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features. It recursively partitions the feature space into pure (or uniform) regions. Trees are highly interpretable but prone to overfitting unless regularized (e.g., via depth limits).

B. **Prerequisites**
- Understanding of variance and mean squared error (MSE) for regression.
- Understanding of probability distributions.
- Helpful: Information theory concepts (entropy).

C. **Learning Questions**
1. Why does a deep decision tree tend to overfit the training data?
2. How does the choice of splitting criterion (Gini vs. Entropy) affect the structure of the resulting tree?
3. In a regression tree, why do we use variance reduction as the splitting criterion instead of accuracy?
4. What happens if a feature is perfectly correlated with the target variable?
5. CART algorithms are greedy. What does this mean in the context of decision trees, and what are the implications for finding the globally optimal tree?

D. **Mathematics to Derive**
1. Write down the formula for Gini Impurity for a set with $K$ classes having proportions $p_1, \dots, p_K$. Show that it is maximized when all classes are equally represented and minimized when the set is pure.
2. Write down the formula for Entropy for a set with $K$ classes. How does its shape compare to Gini Impurity?
3. Define the Information Gain (or Impurity Decrease) when splitting a parent node into left and right child nodes. Use a general impurity measure $I$.
4. For regression, define the impurity of a node as the variance of the targets within that node. Show how the impurity decrease relates to minimizing the sum of squared errors in the children.

E. **Implementation Contract**
- Implement `class DecisionTree`.
- Methods:
  - `__init__(self, task='classification', criterion='gini', max_depth=None, min_samples_leaf=1)`
    - `task`: Either 'classification' or 'regression'.
    - `criterion`: 'gini' or 'entropy' for classification. Ignored for regression.
    - `max_depth`: Maximum depth of the tree.
    - `min_samples_leaf`: Minimum number of samples required to be at a leaf node.
  - `fit(self, X, y)`: Build the tree using recursive partitioning.
  - `predict(self, X)`: Return predictions for samples in X.
- You may use NumPy for array operations (e.g., finding best splits, calculating impurities); however, the tree building and partitioning algorithm itself must be manually implemented.

F. **Guided Implementation Stages**
1. **Node Class**: Create a helper class or dict to represent a node in the tree. It needs to store: feature index to split on, threshold value, left child, right child, and the predicted value (if it's a leaf).
2. **Impurity Functions**: Implement `_gini(y)`, `_entropy(y)`, and `_variance(y)` helper functions that take an array of targets and return the scalar impurity.
3. **Information Gain**: Implement a function to calculate the impurity decrease for a given split. Make sure to weight the child impurities by the fraction of samples that fall into each child.
4. **Best Split Search**: Implement a function that iterates over all features and all possible unique values (or midpoints) of each feature to find the split that maximizes information gain.
5. **Recursive Build**: Implement the recursive `_build_tree` method. Remember to check stopping criteria: `max_depth` reached, `min_samples_leaf` constraint, node is pure (all targets same), or no features provide positive information gain. If stopping, create a leaf node. Leaf predictions should be the majority class for classification, and the mean value for regression.
6. **Prediction**: Implement the `predict` logic that traverses the tree for each sample in X, routing left or right based on the stored feature thresholds until a leaf is reached.

G. **Edge Cases and Expected Tests**
- *Pure Node*: A dataset where all `y` are the same should immediately result in a single leaf node.
- *Unsplittable Data*: Data where all features are identical but targets differ. The tree should create a leaf with the majority class or mean target.
- *Depth Limits*: Ensure that a tree restricted to `max_depth=1` creates exactly one root node and two leaf children (a stump).
- *Single Sample*: `min_samples_leaf=2` should prevent a split that would leave a child with only 1 sample.

H. **Complexity Analysis**
1. What is the time complexity of building the tree at a single node with $N$ samples and $F$ features? (Assume checking all unique values).
2. What is the overall time complexity of `fit(X, y)` for a balanced tree?
3. What is the time complexity of `predict(X)` for a single sample?
4. What is the space complexity of storing the trained tree model?

I. **Definition of Done**
- DecisionTree class successfully fits a binary classification dataset and predicts accurately.
- DecisionTree class successfully fits a continuous target dataset and predicts accurately (regression).
- Tree respects `max_depth` and `min_samples_leaf` constraints.
- `criterion` argument correctly toggles between Gini and Entropy for classification.
- `task` correctly switches between classification and regression modes and criteria.

J. **Reflection**
- Why do decision trees create orthogonal (axis-aligned) decision boundaries?
- How sensitive is the tree structure to small changes in the training data?
"""
