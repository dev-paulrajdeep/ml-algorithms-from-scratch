"""
A. **Mission**
Gaussian Naive Bayes is a probabilistic classifier based on applying Bayes' theorem with a strong (naive) independence assumption between features. It assumes continuous features follow a Gaussian (normal) distribution. It is particularly useful for text classification (with other variants) and as a fast, robust baseline.

B. **Prerequisites**
- Probability theory (Bayes' theorem, conditional probability)
- Gaussian (Normal) distribution properties (mean, variance, probability density function)
- Maximum Likelihood Estimation (MLE) intuition

C. **Learning Questions**
1. What is the "naive" assumption, and why is it made?
2. Why do we compute probabilities in log-space rather than multiplying them directly?
3. How does this algorithm differ fundamentally from discriminative models like Logistic Regression?

D. **Mathematics to Derive**
1. Write down Bayes' theorem for classification: P(y|X) \\propto P(y) * \\prod P(x_i|y).
2. Write the formula for the Gaussian probability density function (PDF).
3. Derive the log-likelihood formulation for numerical stability.

E. **Implementation Contract**
- Class: `GaussianNaiveBayes`
- `fit(self, X, y)`: Estimate mean, variance, and prior probability per class via MLE. `X` is (n_samples, n_features), `y` is (n_samples,).
- `predict_proba(self, X)`: Return posterior probabilities. Returns (n_samples, n_classes).
- `predict(self, X)`: Predict class labels. Returns (n_samples,).
- Allow NumPy for array operations.

F. **Guided Implementation Stages**
1. Identify unique classes in `y`.
2. For each class, compute and store:
   a. The prior probability P(y).
   b. The mean of each feature.
   c. The variance of each feature.
3. In prediction, for each sample:
   a. Calculate the log prior for each class.
   b. Calculate the log of the Gaussian PDF for each feature value given the class parameters.
   c. Sum the log probabilities and return the class with the maximum posterior.

G. **Edge Cases and Expected Tests**
1. `test_zero_variance`: Add a small epsilon to variance to prevent division by zero if a feature is constant in a class.
2. `test_prior_distribution`: Ensure prior probabilities sum to 1.
3. `test_binary_and_multiclass`: Model should seamlessly handle both binary and multi-class target variables.

H. **Complexity Analysis**
1. What is the time complexity of the `fit` method?
2. How does the space complexity compare to KNN?

I. **Definition of Done**
- All tests pass.
- Log probabilities are used to prevent numerical underflow.
- Handles division by zero gracefully using a small epsilon in variance.

J. **Reflection**
1. How does the model perform when features are highly correlated?
2. When might you choose Naive Bayes over a more complex model?
"""
