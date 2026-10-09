"""
A. **Mission**
K-Nearest Neighbors (KNN) is a non-parametric, lazy learning algorithm used for classification. It serves as a strong baseline and intuition-builder for instance-based learning.

B. **Prerequisites**
- [Stage 0D: Calculus & Optimization](../foundations/README.md#module-0d-calculus-and-optimization) (Distance metrics)
- [Stage 0F: ML Core](../foundations/README.md#module-0f-machine-learning-core-concepts) (Bias-variance tradeoff)

C. **Learning Questions**
1. How does the choice of K affect the bias and variance of the model?
2. Why is it called a "lazy learner"?

D. **Mathematics to Derive**
1. Formulate the equation for Euclidean distance between two vectors.

E. **Implementation Contract**
- Class: `KNearestNeighbors`
- `__init__(self, k=3)`
- `fit(self, X, y)`: Store the training data. `X` is (n_samples, n_features), `y` is (n_samples,).
- `predict(self, X)`: Predict class labels for `X`. Returns (n_samples,).
- Allow NumPy for array operations.

F. **Guided Implementation Stages**
Step 1: Storing Data
- **What to learn**: Lazy learning concept.
- **What to do**: Store `X` and `y` in `fit`.
- **How to check yourself**: Attributes are set.
- **When to proceed**: Immediately.
- **Recovery hints 1/2/3**: No computation needed here.

Step 2: Distance Calculation
- **What to learn**: Vectorized distance metrics.
- **What to do**: Implement Euclidean distance between a point and all training points.
- **How to check yourself**: Distance to self is 0.
- **When to proceed**: Returns correct distances.
- **Recovery hints 1/2/3**: Use `np.linalg.norm` or manual calculation.

Step 3: Finding Neighbors & Voting
- **What to learn**: Sorting and mode selection.
- **What to do**: Use `np.argsort` to find top `k` indices, get labels, find most common.
- **How to check yourself**: Check with small array.
- **When to proceed**: Correctly predicts 1 point.
- **Recovery hints 1/2/3**: `np.bincount` and `np.argmax`.

Step 4: Full Prediction
- **What to learn**: Applying to all test samples.
- **What to do**: Loop over `X` in `predict` or fully vectorize.
- **How to check yourself**: Shape of output matches `X.shape[0]`.
- **When to proceed**: Model passes basic tests.
- **Recovery hints 1/2/3**: Start with list comprehension, optimize later.

G. **Edge Cases and Expected Tests**
1. `test_k_equals_1`: With k=1, the model perfectly memorizes the training set.
2. `test_tie_breaking`: Consistent tie-breaking when voting is split.

H. **Complexity Analysis**
1. What is the time complexity of a single prediction?

I. **Progressive Difficulty Levels & Definition of Done**
- **Level 1 (Guided)**: Follow stages.
- **Level 2 (Standard)**: Implement from API contract.
- **Level 3 (Challenge)**: Fully vectorize the pairwise distance computation across all test points simultaneously.
- **Definition of Done**: All tests pass, handles ties.

J. **Reflection**
1. Why is feature normalization crucial for KNN?
"""
