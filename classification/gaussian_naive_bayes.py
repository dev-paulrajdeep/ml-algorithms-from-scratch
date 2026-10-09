"""
A. **Mission**
Gaussian Naive Bayes is a probabilistic classifier based on Bayes' theorem with a strong independence assumption.

B. **Prerequisites**
- [Stage 0E: Probability & Statistics](../foundations/README.md#module-0e-probability-and-statistics)
- MLE intuition

C. **Learning Questions**
1. What is the "naive" assumption?
2. Why compute probabilities in log-space?

D. **Mathematics to Derive**
1. Write the log-likelihood formulation for numerical stability.

E. **Implementation Contract**
- Class: `GaussianNaiveBayes`
- `fit(self, X, y)`: Estimate mean, variance, and prior. `X` is (n_samples, n_features), `y` is (n_samples,).
- `predict_proba(self, X)`: Return posterior probabilities.
- `predict(self, X)`: Predict class labels.
- Allow NumPy.

F. **Guided Implementation Stages**
Step 1: Parameter Estimation
- **What to learn**: Maximum Likelihood Estimation for Gaussians.
- **What to do**: For each class, compute mean, variance, and prior.
- **How to check yourself**: Prior sums to 1.
- **When to proceed**: Parameters are stored in dictionaries or arrays.
- **Recovery hints 1/2/3**: Use boolean indexing for class filtering.

Step 2: Log-Density Computation
- **What to learn**: Log of Gaussian PDF.
- **What to do**: Implement a method to compute log density given mean and variance.
- **How to check yourself**: Ensure no log(0) or division by 0.
- **When to proceed**: Returns expected negative values for densities.
- **Recovery hints 1/2/3**: Add a small epsilon to variance.

Step 3: Posterior Computation
- **What to learn**: Combining log prior and log likelihoods.
- **What to do**: For each class, sum log prior and log densities of features.
- **How to check yourself**: Resulting log posteriors reflect class dominance.
- **When to proceed**: Can predict labels correctly using argmax.
- **Recovery hints 1/2/3**: Naive assumption means we sum log probabilities of individual features.

G. **Edge Cases and Expected Tests**
1. `test_zero_variance`: Add epsilon to variance to prevent division by zero.
2. `test_binary_and_multiclass`: Handle both seamlessly.

H. **Complexity Analysis**
1. How does the space complexity compare to KNN?

I. **Progressive Difficulty Levels & Definition of Done**
- **Level 1 (Guided)**: Follow steps explicitly.
- **Level 2 (Standard)**: Direct implementation.
- **Level 3 (Challenge)**: Implement `predict_proba` using the log-sum-exp trick.
- **Definition of Done**: Tests pass, epsilon used for numerical stability.

J. **Reflection**
1. How does the model perform when features are highly correlated?
"""
