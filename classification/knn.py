"""
A. **Mission**
K-Nearest Neighbors (KNN) is a non-parametric, lazy learning algorithm used for classification (and regression). It classifies a new data point based on the majority class of its 'K' closest neighbors in the training set. It serves as a strong baseline and intuition-builder for instance-based learning.

B. **Prerequisites**
- Distance metrics (specifically Euclidean distance)
- Concept of bias-variance tradeoff
- Understanding of the curse of dimensionality

C. **Learning Questions**
1. How does the choice of K affect the bias and variance of the model?
2. Why is it called a "lazy learner" and what are the implications for inference time?
3. How does feature scaling impact the distance calculations?

D. **Mathematics to Derive**
1. Formulate the equation for Euclidean distance between two vectors.
2. Formulate the prediction rule as a mode operation over the K nearest neighbors.

E. **Implementation Contract**
- Class: `KNearestNeighbors`
- `__init__(self, k=3)`
- `fit(self, X, y)`: Store the training data. `X` is (n_samples, n_features), `y` is (n_samples,).
- `predict(self, X)`: Predict class labels for `X`. Returns (n_samples,).
- Allow NumPy for array operations.

F. **Guided Implementation Stages**
1. In `fit`, simply store `X` and `y` (no training phase).
2. In `predict`, for each row in `X`:
   a. Compute the distance between this row and all stored `X_train` rows.
   b. Identify the indices of the `k` smallest distances.
   c. Extract the corresponding `y_train` labels.
   d. Determine the most frequent label (majority vote).

G. **Edge Cases and Expected Tests**
1. `test_k_equals_1`: With k=1, the model should perfectly memorize the training set.
2. `test_tie_breaking`: Define and test a consistent tie-breaking mechanism when voting is split.
3. `test_different_k`: Show decision boundary smoothing as k increases.

H. **Complexity Analysis**
1. What is the space complexity of this algorithm?
2. What is the time complexity of a single prediction? How does it scale with training data size?

I. **Definition of Done**
- All tests pass.
- `predict` works on multiple samples simultaneously.
- Implementation handles ties gracefully.

J. **Reflection**
1. Why is feature normalization crucial for KNN?
2. What are the practical limitations of using KNN on very large datasets?
"""
