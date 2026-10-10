r"""
A. **Mission** (Step 0: Understand the Problem)
Gaussian Naive Bayes is a generative probabilistic classifier that applies Bayes' theorem with the "naive" assumption of conditional feature independence given the class label.
Real-world problem: Text classification, medical diagnosis with continuous measurements, or real-time sensor analysis where data is noisy, training samples are scarce, and rapid non-iterative training is required.
Unlike Logistic Regression, Naive Bayes requires **no gradient descent**. It directly estimates class priors, means, and variances in a single closed-form pass, using Gaussian probability densities to compute posterior class probabilities.

B. **Prerequisites**
- [Stage 0B: NumPy & Numerical Computing](../foundations/README.md#module-0b-numpy-and-numerical-computing) — Boolean masking (`X[y == c]`), axis reductions, and log-space arithmetic.
- [Stage 0E: Probability & Statistics](../foundations/README.md#module-0e-probability-and-statistics) — Bayes' theorem, sample mean and variance, Gaussian PDF, and log-likelihood.
- [Stage 0F: Machine Learning Core](../foundations/README.md#module-0f-machine-learning-core-concepts) — Generative vs. discriminative modeling.

C. **Learning Questions** (Step 1: Build Intuition)
1. What does the "naive" conditional independence assumption actually state? If two features are strongly correlated (e.g., shoe size and height), how does this violate the assumption, and what happens to the predicted probabilities?
2. Why do we compute class posteriors using **log-probabilities** ($\sum \ln P(x_j|y)$) instead of multiplying raw probabilities ($\prod P(x_j|y)$)?
3. What is a "generative classifier"? How does modeling $P(\mathbf{x}|y) P(y)$ differ from directly learning the boundary $P(y|\mathbf{x})$?

D. **Mathematics to Derive**
1. **Step 2: Tiny Numerical Example by Hand**:
   Consider 1 feature $x$ and 2 classes with 2 samples each:
   Class 0: $x = [1.0, 3.0] \implies \mu_0 = 2.0, \sigma_0^2 = 1.0, P(y=0) = 0.5$.
   Class 1: $x = [7.0, 9.0] \implies \mu_1 = 8.0, \sigma_1^2 = 1.0, P(y=1) = 0.5$.
   Given a test query $x_{\text{test}} = 2.5$:
   - Distance to $\mu_0$ is $|2.5 - 2| = 0.5$; distance to $\mu_1$ is $|2.5 - 8| = 5.5$.
   - The Gaussian exponent for Class 0 is $-\frac{(2.5 - 2)^2}{2(1)} = -0.125$.
   - The Gaussian exponent for Class 1 is $-\frac{(2.5 - 8)^2}{2(1)} = -15.125$.
   - Since exponents are in log-space, Class 0 has vastly higher posterior log-probability: $\hat{y} = 0$.
2. **Step 3: Mathematical Notation**:
   - $N$: number of samples, $D$: number of features, $C$: number of unique classes.
   - $\mu_{jc}$: mean of feature $j$ for class $c$.
   - $\sigma_{jc}^2$: variance of feature $j$ for class $c$.
   - $\pi_c = P(y = c)$: prior probability of class $c$ ($N_c / N$).
   - $\epsilon$: variance smoothing jitter added to the diagonal to prevent division by zero.
3. **Step 4: Derive the Log-Posterior Formulation**:
   - Write Bayes' theorem:
     $$P(y = c | \mathbf{x}) = \frac{P(y = c) P(\mathbf{x} | y = c)}{P(\mathbf{x})} \propto P(y = c) \prod_{j=1}^D P(x_j | y = c)$$
   - Substitute the 1D Gaussian probability density function for each feature:
     $$P(x_j | y = c) = \frac{1}{\sqrt{2\pi \sigma_{jc}^2}} \exp\left( -\frac{(x_j - \mu_{jc})^2}{2\sigma_{jc}^2} \right)$$
   - Take the natural logarithm:
     $$\ln P(y = c | \mathbf{x}) \propto \ln \pi_c + \sum_{j=1}^D \left[ -\frac{1}{2} \ln(2\pi \sigma_{jc}^2) - \frac{(x_j - \mu_{jc})^2}{2\sigma_{jc}^2} \right]$$
   - Prediction rule:
     $$\hat{y} = \arg\max_{c \in \{0, \dots, C-1\}} \left( \ln \pi_c - \frac{1}{2} \sum_{j=1}^D \ln(2\pi \sigma_{jc}^2) - \frac{1}{2} \sum_{j=1}^D \frac{(x_j - \mu_{jc})^2}{\sigma_{jc}^2} \right)$$

E. **Implementation Contract**
- Class: `GaussianNaiveBayes`
- Methods:
  - `__init__(self, var_smoothing: float = 1e-9)`:
    Stores variance smoothing epsilon.
  - `fit(self, X: np.ndarray, y: np.ndarray) -> self`:
    Identifies unique classes. Computes and stores:
    - `classes_`: array of unique class labels, shape `(C,)`.
    - `class_prior_`: prior probability for each class, shape `(C,)`.
    - `theta_`: mean of each feature per class, shape `(C, D)`.
    - `var_`: variance of each feature per class (plus `var_smoothing`), shape `(C, D)`.
  - `predict_log_proba(self, X: np.ndarray) -> np.ndarray`:
    Computes joint log-likelihoods for each class, shape `(N, C)`.
  - `predict(self, X: np.ndarray) -> np.ndarray`:
    Returns `classes_[np.argmax(predict_log_proba(X), axis=1)]`. Returns shape `(N,)`.

F. **Guided Implementation Stages**
**Step 5: Design the Algorithm (Pseudocode)**
```text
fit(X, y):
  For each class c in unique(y):
    X_c = X[y == c]
    priors[c] = len(X_c) / N
    means[c] = mean(X_c, axis=0)
    vars[c] = var(X_c, axis=0) + var_smoothing

predict_log_proba(X):
  For each class c:
    log_prior = log(priors[c])
    log_likelihood = -0.5 * sum(log(2 * pi * vars[c]) + ((X - means[c])**2) / vars[c], axis=1)
    joint_log_prob[:, c] = log_prior + log_likelihood
  return joint_log_prob
```

**Checkpoint 1: Parameter Estimation in `fit`**
- **What to learn**: Extracting sufficient statistics per class using boolean masking.
- **What to do**: Loop over unique classes in `y`, compute class means and variances.
- **How to check yourself**: Ensure `theta_.shape == (C, D)` and `class_prior_.sum() == 1.0`.
- **When to proceed**: Class statistics match `np.mean` and `np.var` on isolated classes.
- **Recovery hints**:
  - *Hint 1 (Concept)*: Sufficient statistics for Gaussians are entirely captured by sample mean and sample variance.
  - *Hint 2 (Operation)*: Use boolean indexing `X_c = X[y == c]`.
  - *Hint 3 (Debugging)*: Add `var_smoothing * max_variance` to prevent zero variance if a feature is constant.

**Checkpoint 2: Gaussian Log-Likelihood Computation**
- **What to learn**: Safe computation of log Gaussian density without underflow.
- **What to do**: Implement log density formula: $-0.5 \cdot \ln(2\pi \sigma^2) - \frac{(x - \mu)^2}{2\sigma^2}$.
- **How to check yourself**: Density is maximized when $x = \mu$, where exponent evaluates to $0.0$.
- **When to proceed**: Densities are strictly finite negative numbers for query points.
- **Recovery hints**:
  - *Hint 1 (Concept)*: The log of a Gaussian is a quadratic penalty on distance from the mean.
  - *Hint 2 (Operation)*: Use `np.log(2.0 * np.pi * var)` and broadcasting across samples.
  - *Hint 3 (Debugging)*: Check `assert not np.isnan(log_likelihood).any()`.

**Checkpoint 3: Posterior Integration & Argmax**
- **What to learn**: Adding log-prior and taking argmax.
- **What to do**: Sum log-prior and log-likelihoods per class, and predict class with largest total.
- **How to check yourself**: On Dataset C1, predictions match class labels with 100% accuracy.
- **When to proceed**: Classifier correctly predicts binary and multi-class test samples.
- **Recovery hints**:
  - *Hint 1 (Concept)*: $\arg\max_c [\ln P(y=c) + \sum \ln P(x_j|y=c)]$ is mathematically identical to argmax of posterior.
  - *Hint 2 (Operation)*: `return self.classes_[np.argmax(joint_log, axis=1)]`.
  - *Hint 3 (Debugging)*: Remember to sum across features (`axis=1`) and argmax across classes (`axis=1`).

G. **Edge Cases and Expected Tests** (Step 8: Test and Debug)
1. `test_zero_variance_feature`: When a feature is identical across all samples in a class, variance is 0. Verify that `var_smoothing` prevents division by zero.
2. `test_well_separated_gaussians`: Generate two 2D clusters centered at $(0, 0)$ and $(10, 10)$ with variance 1.0. Model must achieve 100% accuracy.
3. `test_imbalanced_priors`: Test on a dataset with 90% Class 0 and 10% Class 1 where features overlap heavily. Show that the model predicts Class 0 due to the prior $\ln \pi_c$.
4. `test_multiclass_support`: Verify seamless execution on 3-class and 5-class datasets.

H. **Complexity Analysis** (Step 9: Analyze the Algorithm)
- **Time Complexity**:
  - Training: $O(N \cdot D)$ time (a single pass through the dataset; no gradient descent loops!).
  - Inference: $O(M \cdot D \cdot C)$ for $M$ test queries, $D$ features, and $C$ classes.
- **Space Complexity**:
  - Parameters: $O(C \cdot D)$ auxiliary memory for `theta_` and `var_`.
- **Trade-offs**: Blazingly fast training and low memory, but poor calibration when features are strongly correlated.

I. **Progressive Difficulty Levels & Definition of Done**
- **Level 1 (Guided)**: Implement 1D, 2-class Gaussian density by hand on Dataset C2.
- **Level 2 (Standard)**: Implement full vectorized `GaussianNaiveBayes` class with multi-class support and variance smoothing.
- **Level 3 (Challenge)**: Implement `predict_proba` returning true normalized probabilities $[0, 1]$ using the **log-sum-exp trick**:
  $$P(y=c|\mathbf{x}) = \frac{\exp(\ell_c)}{\sum_k \exp(\ell_k)} = \frac{\exp(\ell_c - \max_k \ell_k)}{\sum_k \exp(\ell_k - \max_k \ell_k)}$$
- **Definition of Done**: Correctly computes Gaussian statistics, handles zero-variance features, works for multi-class inputs, and operates strictly in log-space.

J. **Reflection & Self-Check** (Step 10: Independent Completion)
1. Why does the naive independence assumption cause Naive Bayes to output overconfident probabilities (probabilities pushed artificially close to 0 and 1)?
2. If two features are identical copies of each other ($x_1 = x_2$), how does Naive Bayes react compared to Logistic Regression?
"""
