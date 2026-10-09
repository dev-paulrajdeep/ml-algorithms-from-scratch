"""
A. **Mission**
Random Forest is an ensemble learning method that constructs a multitude of decision trees at training time and outputs the mode of the classes (classification) or mean prediction (regression) of the individual trees. It combats the tendency of single decision trees to overfit their training set. By using bagging (bootstrap aggregating) and random subspace methods (feature subsampling), Random Forests reduce variance while maintaining a low bias.

B. **Prerequisites**
- Solid understanding of Decision Trees (CART).
- Concept of bias-variance tradeoff.
- Statistical concept of bootstrapping.

C. **Learning Questions**
1. What is the primary reason that a Random Forest typically outperforms a single Decision Tree?
2. How does bootstrap sampling (bagging) help reduce the variance of the model without significantly increasing bias?
3. Why do we subsample features at each split instead of just bagging the data? (What happens if there is one incredibly dominant feature across all bootstrap samples?)
4. A Random Forest has high complexity. Does adding more trees increase the risk of overfitting? Why or why not?

D. **Mathematics to Derive**
1. Show mathematically why averaging the predictions of $M$ perfectly independent models reduces the variance of the ensemble prediction by a factor of $1/M$.
2. If the models are not perfectly independent but have an average correlation $\rho$, what is the variance of the ensemble? How does feature subsampling affect $\rho$?
3. In a bootstrap sample of size $N$ drawn from a dataset of size $N$, what is the probability that a specific sample is *never* selected (Out-Of-Bag)? Calculate the limit as $N \to \infty$.

E. **Implementation Contract**
- Implement `class RandomForest`.
- Methods:
  - `__init__(self, task='classification', n_trees=100, max_depth=None, max_features='sqrt')`
    - `task`: 'classification' or 'regression'.
    - `n_trees`: Number of trees in the forest.
    - `max_depth`: Passed to the underlying decision trees.
    - `max_features`: Number of features to consider when looking for the best split. Can be an integer, float (fraction), or 'sqrt' (typical for classification).
  - `fit(self, X, y)`: Build the forest of trees.
  - `predict(self, X)`: Return aggregated predictions.
- You MUST use your own `DecisionTree` implementation from `decision_tree.py`. You may need to adapt it slightly to handle `max_features` at each split, or implement that logic here if you structure your code to allow it.

F. **Guided Implementation Stages**
1. **Tree Modification**: Ensure your base `DecisionTree` can accept a `max_features` parameter and, during the best split search, randomly selects a subset of features to evaluate.
2. **Bootstrapping**: Implement a function to generate a bootstrap sample from `X` and `y` (sample with replacement, same size as original data).
3. **Forest Construction**: In `fit(X, y)`, loop `n_trees` times. In each iteration, generate a bootstrap sample, initialize a `DecisionTree` with the specified hyper-parameters (including `max_features`), and fit it to the bootstrap sample. Store the fitted tree.
4. **Ensemble Prediction**: In `predict(X)`, gather predictions from all trees in the forest for each sample.
5. **Aggregation**: For classification, implement a majority voting mechanism to determine the final class for each sample. For regression, average the predictions across all trees.

G. **Edge Cases and Expected Tests**
- *Single Tree Forest*: A Random Forest with `n_trees=1` and `max_features=None` (or all features) should perform similarly to a single standard Decision Tree.
- *Feature Subsampling*: Ensure that `max_features` correctly limits the number of features evaluated at *every* node split, not just at the root.
- *Constant Prediction*: If all `y` values are identical, the forest should predict that constant value.

H. **Complexity Analysis**
1. If training a single Decision Tree takes time $O(T)$, what is the time complexity of training a Random Forest with $M$ trees? Can it be parallelized?
2. What is the memory footprint of a Random Forest compared to a single Decision Tree?
3. How does `max_features < total_features` affect the training time per tree?

I. **Definition of Done**
- RandomForest class correctly initializes and stores `n_trees` decision trees.
- `fit` properly bootstraps data for each tree.
- `predict` correctly aggregates (majority vote for classification, mean for regression).
- Feature subsampling is correctly applied during tree building.

J. **Reflection**
- Why might Random Forest be less interpretable than a single Decision Tree? How might you estimate feature importance in a Random Forest?
- Under what circumstances might a single Decision Tree outperform a Random Forest?
"""
