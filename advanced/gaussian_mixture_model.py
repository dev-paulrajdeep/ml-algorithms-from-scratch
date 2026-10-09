"""
A. **Mission**
Gaussian Mixture Models (GMMs) are probabilistic models that assume all the data points are generated from a mixture of a finite number of Gaussian distributions with unknown parameters. They can be thought of as generalizing k-means clustering to incorporate information about the covariance structure of the data as well as the centers of the latent Gaussians. GMMs are optimized via Expectation-Maximization (EM), finding a local maximum of the log-likelihood or Evidence Lower Bound (ELBO).

B. **Prerequisites**
- Probability theory (Bayes' theorem, conditional probability, joint distributions).
- Multivariate Gaussian distributions (mean vectors, covariance matrices).
- Maximum Likelihood Estimation (MLE).
- Latent-variable model concepts.
- Understanding of Expectation, Variance, and Jensen's inequality.

C. **Learning Questions**
1. How does the Expectation-Maximization algorithm avoid computing the exact posterior over all possible latent variable assignments?
2. What role do the "responsibilities" play in the M-step, and how do they differ from hard assignments in k-means?
3. Why does the ELBO (Evidence Lower Bound) guarantee that the log-likelihood monotonically increases with each EM step?
4. Under what specific conditions does a GMM perfectly reduce to the k-means algorithm?

D. **Mathematics to Derive**
1. Derive the E-step: Express the posterior probability (responsibility) $\gamma(z_{nk}) = P(z_n=k | x_n, \theta)$ in terms of the mixing coefficients $\pi_k$ and Gaussian densities $\mathcal{N}(x_n | \mu_k, \Sigma_k)$.
2. Derive the M-step updates for $\mu_k, \Sigma_k, \pi_k$ by maximizing the expected complete-data log-likelihood.
3. Write down the expression for the log-likelihood of the observed data $X$ given the parameters.
4. Show that maximizing ELBO leads to maximizing the log marginal likelihood.

E. **Implementation Contract**
- Class name: `GaussianMixtureModel`
- Initialization parameters: `n_components` (int), `max_iters` (int), `tol` (float).
- Attributes: `means_` (ndarray of shape `(n_components, n_features)`), `covariances_` (ndarray of shape `(n_components, n_features, n_features)`), `weights_` (ndarray of shape `(n_components,)`), `log_likelihood_history_` (list of floats).
- `fit(X)`: Runs the EM algorithm to estimate model parameters.
- `predict_proba(X)`: Returns the responsibility (soft assignment) of each component for each sample in X, shape `(n_samples, n_components)`.
- `predict(X)`: Returns the index of the component with the highest responsibility for each sample, shape `(n_samples,)`.
- Note: Use NumPy for numerical operations, but the EM loop, density calculations, and updates must be manually implemented.

F. **Guided Implementation Stages**
1. **Initialization**: Initialize `weights_` to uniform probabilities, `means_` randomly or by sampling from data, and `covariances_` to spherical matrices.
2. **Log-Density Calculation**: Implement a helper to safely compute the log probability density of a multivariate Gaussian. Consider using the log-sum-exp trick to avoid underflow.
3. **E-step**: Compute the log responsibilities and responsibilities using Bayes' theorem and the log-sum-exp trick.
4. **M-step**: Update `weights_`, `means_`, and `covariances_` using the responsibilities. Add a small value to the diagonal of covariances to ensure they remain positive semi-definite.
5. **Convergence Check**: Compute the log-likelihood of the data and check if the change from the previous iteration is below `tol`. Terminate if so or if `max_iters` is reached.

G. **Edge Cases and Expected Tests**
1. Write a test where clusters are highly overlapping. Check that soft assignments correctly reflect the uncertainty.
2. Test on completely spherical, well-separated data and compare the cluster centers to k-means.
3. Write a test where covariance matrices might collapse to singular matrices. Ensure your implementation adds a regularization term (jitter) to the diagonal to prevent numerical errors.
4. Verify that `log_likelihood_history_` is monotonically increasing.

H. **Complexity Analysis**
1. What is the time complexity of a single E-step for $N$ samples, $K$ components, and $D$ dimensions?
2. What is the space complexity of storing the responsibilities and covariance matrices?
3. How does the convergence rate of EM compare to gradient descent methods on similar non-convex objectives?

I. **Definition of Done**
- `GaussianMixtureModel` successfully fits 2D synthetic data generated from multiple Gaussians.
- The `predict_proba` method correctly normalizes to 1 across the component axis.
- The `log_likelihood_history_` is strictly non-decreasing across iterations.
- Code handles numerical instability (e.g. log-sum-exp trick, covariance regularization).

J. **Reflection**
1. Why is the log-sum-exp trick critical in the E-step of GMMs? What would happen if you worked directly with probabilities?
2. How sensitive is the GMM to initialization, and how might you improve it (e.g., using k-means to initialize)?
"""
