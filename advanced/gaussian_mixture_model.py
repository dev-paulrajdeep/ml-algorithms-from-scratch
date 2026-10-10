r"""
A. **Mission** (Step 0: Understand the Problem)
A Gaussian Mixture Model (GMM) is a probabilistic generative model that represents data as being produced by a linear superposition of $K$ distinct multivariate Gaussian distributions. While K-Means (Chapter 4) performs hard, spherical clustering, GMM performs **soft clustering with uncertainty**, allowing each cluster to have its own elliptical shape, orientation, and spread governed by a full covariance matrix $\boldsymbol{\Sigma}_k$.
Because cluster assignments are hidden (latent), the marginal likelihood cannot be maximized in closed form due to the summation inside the logarithm. To solve this, we employ the **Expectation-Maximization (EM)** algorithm—an iterative coordinate ascent procedure on the Evidence Lower Bound (ELBO) that guarantees monotonic improvement in data log-likelihood.
Your mission is to implement a robust `GaussianMixtureModel` from scratch using only NumPy. You will derive both the E-step and M-step, implement the Log-Sum-Exp trick for numerical stability, add diagonal covariance regularization (jitter) to prevent singular collapses, and prove empirically that data log-likelihood is monotonically non-decreasing.

B. **Prerequisites**
- [Stage 0B: NumPy & Numerical Computing](../foundations/README.md#module-0b-numpy-and-numerical-computing) — 3D array broadcasting, matrix inversion (`np.linalg.inv`), and determinants (`np.linalg.slogdet`).
- [Stage 0C: Linear Algebra](../foundations/README.md#module-0c-linear-algebra) — Covariance matrices, positive semi-definiteness, quadratic forms $(\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})$, and regularization via identity matrices.
- [Stage 0D: Calculus & Optimization](../foundations/README.md#module-0d-calculus-and-optimization) — Constrained optimization with Lagrange multipliers (ensuring $\sum \pi_k = 1$) and Jensen's inequality.
- [Stage 0E: Probability & Statistics](../foundations/README.md#module-0e-probability-and-statistics) — Multivariate Gaussian probability density, Bayes' theorem, likelihood vs. log-likelihood, and latent variables.
- [Chapter 4: K-Means](../clustering/kmeans.py) — Understanding cluster initialization and hard vs. soft assignments.

C. **Learning Questions** (Step 1: Build Intuition)
1. **The Insoluble Log-Sum**: Why can't we find closed-form MLE solutions for GMM parameters simply by setting the gradient of $\ln P(\mathbf{X} \mid \boldsymbol{\theta})$ to zero? Why does the sum over $K$ inside the logarithm prevent decoupling the parameters?
2. **Soft Responsibilities**: What is the physical meaning of the responsibility $\gamma_{nk} = P(z_n = k \mid \mathbf{x}_n, \boldsymbol{\theta})$? Why does a sample positioned halfway between two cluster centers receive $\gamma_{n1} \approx 0.5, \gamma_{n2} \approx 0.5$?
3. **The Singular Collapse Pathology**: What happens if a Gaussian component's center $\boldsymbol{\mu}_k$ coincides exactly with a single data point $\mathbf{x}_n$ and its variance shrinks to zero ($\sigma_k^2 \to 0$)? Why does the likelihood explode to infinity, and how does diagonal jitter ($\boldsymbol{\Sigma}_k + \epsilon \mathbf{I}$) prevent this catastrophic singularity?
4. **K-Means as a Limiting Case**: Under what mathematical conditions on $\boldsymbol{\Sigma}_k$ does a GMM collapse into the hard K-Means algorithm?

D. **Mathematics to Derive**
1. **Step 2: Tiny Numerical Example by Hand**:
   Consider 1D data with 3 points: $x_1 = -1.0, x_2 = 0.0, x_3 = +1.0$.
   Let $K=2$ components with equal initial mixture weights $\pi_1 = 0.5, \pi_2 = 0.5$.
   Initial parameters: $\mu_1 = -1.0, \sigma_1^2 = 1.0$ and $\mu_2 = +1.0, \sigma_2^2 = 1.0$.
   1D Gaussian PDF: $\mathcal{N}(x \mid \mu, \sigma^2) = \frac{1}{\sqrt{2\pi \sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$.
   Note $\frac{1}{\sqrt{2\pi}} \approx 0.3989$.
   - **For point $x_2 = 0.0$**:
     - Density under Component 1: $\mathcal{N}(0 \mid -1, 1) = 0.3989 e^{-1/2} = 0.3989(0.6065) \approx 0.2420$.
     - Density under Component 2: $\mathcal{N}(0 \mid +1, 1) = 0.3989 e^{-1/2} \approx 0.2420$.
     - Responsibility $\gamma_{21} = \frac{0.5(0.2420)}{0.5(0.2420) + 0.5(0.2420)} = 0.50$, $\gamma_{22} = 0.50$ (Perfect symmetry!).
   - **For point $x_1 = -1.0$**:
     - Density under 1: $\mathcal{N}(-1 \mid -1, 1) = 0.3989 e^0 = 0.3989$.
     - Density under 2: $\mathcal{N}(-1 \mid +1, 1) = 0.3989 e^{-4/2} = 0.3989(0.1353) \approx 0.0540$.
     - Numerator 1: $0.5(0.3989) \approx 0.19945$. Numerator 2: $0.5(0.0540) \approx 0.0270$.
     - Denominator: $0.19945 + 0.0270 = 0.22645$.
     - Responsibility $\gamma_{11} = \frac{0.19945}{0.22645} \approx 0.8808$, $\gamma_{12} \approx 0.1192$.
   - By symmetry, for $x_3 = +1.0$: $\gamma_{31} \approx 0.1192, \gamma_{32} \approx 0.8808$.
   - **M-Step Parameter Updates**:
     - Effective sample count for component 1: $N_1 = \gamma_{11} + \gamma_{21} + \gamma_{31} = 0.8808 + 0.5000 + 0.1192 = 1.50$.
     - Effective sample count for component 2: $N_2 = 1.50$.
     - New mixture weights: $\pi_1 = \frac{1.5}{3.0} = 0.50, \pi_2 = 0.50$.
     - New mean $\mu_1 = \frac{0.8808(-1.0) + 0.50(0.0) + 0.1192(1.0)}{1.50} = \frac{-0.7616}{1.50} \approx -0.5077$.
     - By symmetry, $\mu_2 \approx +0.5077$.
   The means smoothly contract toward the observed cluster data!

2. **Step 3: Mathematical Notation**:
   - Dataset: $\mathbf{X} \in \mathbb{R}^{N \times D}$.
   - Number of components: $K$.
   - Latent cluster indicator vector for sample $n$: $\mathbf{z}_n \in \{0, 1\}^K$ such that $\sum_{k=1}^K z_{nk} = 1$.
   - Mixture weights: $\boldsymbol{\pi} = [\pi_1, \dots, \pi_K]^T$, where $\pi_k \ge 0$ and $\sum_{k=1}^K \pi_k = 1$.
   - Component means: $\boldsymbol{\mu}_k \in \mathbb{R}^D$.
   - Component covariance matrices: $\boldsymbol{\Sigma}_k \in \mathbb{R}^{D \times D}$ (symmetric, positive semi-definite).
   - Responsibilities: $\boldsymbol{\Gamma} \in \mathbb{R}^{N \times K}$, where $\gamma_{nk} = P(z_{nk} = 1 \mid \mathbf{x}_n, \boldsymbol{\theta})$.

3. **Step 4: Deriving the E-Step, M-Step, and ELBO Monotonicity**:
   - **Multivariate Gaussian PDF**:
     $$\mathcal{N}(\mathbf{x}_n \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) = \frac{1}{(2\pi)^{D/2} |\boldsymbol{\Sigma}_k|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x}_n - \boldsymbol{\mu}_k)^T \boldsymbol{\Sigma}_k^{-1} (\mathbf{x}_n - \boldsymbol{\mu}_k)\right)$$
   - **E-Step (Posterior Responsibility)**:
     By Bayes' theorem:
     $$\gamma_{nk} = P(z_{nk} = 1 \mid \mathbf{x}_n) = \frac{\pi_k \mathcal{N}(\mathbf{x}_n \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_{j=1}^K \pi_j \mathcal{N}(\mathbf{x}_n \mid \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)}$$
     To compute this stably, work in log-space: $\ln \text{num}_{nk} = \ln \pi_k + \ln \mathcal{N}(\mathbf{x}_n \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$, and use the Log-Sum-Exp trick to normalize.
   - **M-Step (Parameter Updates)**:
     Define effective cluster mass: $N_k = \sum_{n=1}^N \gamma_{nk}$.
     $$\pi_k = \frac{N_k}{N}$$
     $$\boldsymbol{\mu}_k = \frac{1}{N_k} \sum_{n=1}^N \gamma_{nk} \mathbf{x}_n$$
     $$\boldsymbol{\Sigma}_k = \frac{1}{N_k} \sum_{n=1}^N \gamma_{nk} (\mathbf{x}_n - \boldsymbol{\mu}_k)(\mathbf{x}_n - \boldsymbol{\mu}_k)^T + \epsilon \mathbf{I}$$
   - **Proof of Monotonicity via Jensen's Inequality**:
     For any distribution $q(\mathbf{Z})$ over latent variables:
     $$\ln P(\mathbf{X} \mid \boldsymbol{\theta}) = \ln \sum_{\mathbf{Z}} q(\mathbf{Z}) \frac{P(\mathbf{X}, \mathbf{Z} \mid \boldsymbol{\theta})}{q(\mathbf{Z})} \ge \sum_{\mathbf{Z}} q(\mathbf{Z}) \ln \frac{P(\mathbf{X}, \mathbf{Z} \mid \boldsymbol{\theta})}{q(\mathbf{Z})} \equiv \mathcal{L}(q, \boldsymbol{\theta})$$
     $\mathcal{L}(q, \boldsymbol{\theta})$ is the Evidence Lower Bound (ELBO).
     In the E-step, setting $q(\mathbf{Z}) = P(\mathbf{Z} \mid \mathbf{X}, \boldsymbol{\theta}^{(t)})$ makes the KL divergence zero and equates the bound: $\mathcal{L}(q, \boldsymbol{\theta}^{(t)}) = \ln P(\mathbf{X} \mid \boldsymbol{\theta}^{(t)})$.
     In the M-step, maximizing $\mathcal{L}$ with respect to $\boldsymbol{\theta}$ ensures:
     $$\ln P(\mathbf{X} \mid \boldsymbol{\theta}^{(t+1)}) \ge \mathcal{L}(q, \boldsymbol{\theta}^{(t+1)}) \ge \mathcal{L}(q, \boldsymbol{\theta}^{(t)}) = \ln P(\mathbf{X} \mid \boldsymbol{\theta}^{(t)})$$
     Hence, the log-likelihood is guaranteed to never decrease!

E. **Implementation Contract**
- Class: `GaussianMixtureModel`
- Constructor:
  - `__init__(self, n_components: int = 2, max_iters: int = 100, tol: float = 1e-4, reg_covar: float = 1e-6, random_state: int = 42)`:
    Stores hyperparameters and seeds the RNG.
- Attributes:
  - `weights_`: ndarray of shape `(K,)`, mixture weights summing to 1.
  - `means_`: ndarray of shape `(K, D)`, component mean vectors.
  - `covariances_`: ndarray of shape `(K, D, D)`, regularized covariance matrices.
  - `log_likelihood_history_`: list of floats, total data log-likelihood after each iteration.
  - `converged_`: bool, indicates if convergence criterion was satisfied.
- Methods:
  - `fit(self, X: np.ndarray) -> self`:
    Initializes parameters (e.g. uniform weights, random points for means, identity covariances) and executes EM loop until `|ll_{t} - ll_{t-1}| < tol` or `max_iters` reached.
  - `predict_proba(self, X: np.ndarray) -> np.ndarray`:
    Computes and returns soft responsibilities $\boldsymbol{\Gamma}$ of shape `(N, K)`, where each row sums to 1.0.
  - `predict(self, X: np.ndarray) -> np.ndarray`:
    Returns hard cluster assignments $\operatorname{argmax}_k \gamma_{nk}$ of shape `(N,)`.
  - `score(self, X: np.ndarray) -> float`:
    Returns the average log-likelihood per sample: $\frac{1}{N} \sum_{n=1}^N \ln P(\mathbf{x}_n \mid \boldsymbol{\theta})$.

F. **Guided Implementation Stages**
**Step 5: Algorithm Design & Pseudocode**
```text
fit(X):
  Initialize weights = uniform(K), means = random_choice(X, K), covars = [eye(D) for _ in K]
  For iteration in range(max_iters):
    # E-step:
    log_resp = log_weights + log_multivariate_gaussian_pdf(X, means, covars)
    log_prob_X = logsumexp(log_resp, axis=1) # shape (N,)
    resp = exp(log_resp - log_prob_X[:, None]) # shape (N, K)
    
    # Check convergence:
    current_ll = sum(log_prob_X)
    record current_ll
    if |current_ll - prev_ll| < tol:
      break
      
    # M-step:
    N_k = sum(resp, axis=0) # shape (K,)
    weights = N_k / N
    For k in range(K):
      means[k] = sum(resp[:, k, None] * X, axis=0) / N_k[k]
      diff = X - means[k]
      covars[k] = (resp[:, k, None] * diff).T @ diff / N_k[k] + reg_covar * eye(D)
```

**Checkpoint 1: Multivariate Gaussian Log-PDF with Log-Sum-Exp**
- **What to learn**: Safe computation of log densities using Cholesky or pseudo-inverse and preventing float underflow.
- **What to do**: Implement a helper `_log_gaussian_pdf(X, mean, cov)` using `np.linalg.slogdet` and quadratic Mahalanobis form. Implement `_logsumexp(a, axis=1)`.
- **How to check yourself**: Compute PDF for 1D standard normal at $x=0$. PDF must equal $\frac{1}{\sqrt{2\pi}} \approx 0.3989$ ($\ln \text{PDF} \approx -0.9189$). Log-Sum-Exp on `[1000, 1000]` must return $1000 + \ln(2) \approx 1000.693$ without overflow!
- **When to proceed**: Log-PDF and Log-Sum-Exp handle extreme values stably.
- **Recovery hints**:
  - *Hint 1 (Concept)*: Quadratic form: $-\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})$.
  - *Hint 2 (Operation)*: Use `np.linalg.solve(cov, diff.T)` or `scipy.linalg.cho_solve` equivalent.
  - *Hint 3 (Debugging)*: Check `sign` from `sign, logdet = np.linalg.slogdet(cov)`. If `sign <= 0`, increase `reg_covar`.

**Checkpoint 2: The E-Step (Responsibilities)**
- **What to learn**: Calculating normalized posterior probabilities across components.
- **What to do**: Combine `np.log(weights)` and Gaussian log densities, apply Log-Sum-Exp normalization, and exponentiate to obtain `resp`.
- **How to check yourself**: Assert `np.allclose(np.sum(resp, axis=1), 1.0)`.
- **When to proceed**: Responsibilities strictly sum to 1.0 across all samples.
- **Recovery hints**:
  - *Hint 1 (Concept)*: Subtract the row-wise Log-Sum-Exp before exponentiating: `np.exp(log_resp - log_prob_X[:, None])`.
  - *Hint 2 (Operation)*: Clip probabilities if tiny: `np.clip(resp, 1e-15, 1.0)`.
  - *Hint 3 (Debugging)*: Check shape of `log_prob_X[:, None]` broadcasting against `log_resp` of shape `(N, K)`.

**Checkpoint 3: The M-Step and Monotonicity Guarantee**
- **What to learn**: Updating means, regularized covariances, and mixture weights, and verifying that log-likelihood never drops.
- **What to do**: Implement parameter updates and record total log-likelihood in `self.log_likelihood_history_`.
- **How to check yourself**: On Dataset 7A, verify that for every iteration $t$, `log_likelihood_history_[t] >= log_likelihood_history_[t-1] - 1e-8`.
- **When to proceed**: Strict monotonicity is confirmed across all iterations.
- **Recovery hints**:
  - *Hint 1 (Concept)*: If log-likelihood drops, there is a bug in the E-step, M-step, or regularization.
  - *Hint 2 (Operation)*: Ensure `reg_covar * np.eye(D)` is added to every covariance matrix.
  - *Hint 3 (Debugging)*: Check that `means` uses the newly computed `resp`, and `covars` uses the newly updated `means`!

G. **Edge Cases and Expected Tests** (Step 8: Test and Debug)
1. `test_log_likelihood_monotonicity`: Verify that `log_likelihood_history_` is monotonically non-decreasing on synthetic multi-modal data.
2. `test_responsibility_normalization`: Verify that `predict_proba(X)` sums to 1.0 along `axis=1` for every sample.
3. `test_singular_covariance_regularization`: On collinear 2D data (where one eigenvalue is 0), verify that `reg_covar` prevents determinant collapse or `LinAlgError`.
4. `test_cluster_recovery`: On two widely separated Gaussian blobs ($\|\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2\| > 6\sigma$), verify accuracy $> 98\%$ against ground-truth components.

H. **Complexity Analysis** (Step 9: Analyze the Algorithm)
- **Time Complexity**:
  - E-step: $O(N \cdot K \cdot D^3)$ to invert $K$ covariance matrices and compute Mahalanobis distances.
  - M-step: $O(N \cdot K \cdot D^2)$ to accumulate outer products.
  - Total per iteration: $O(N \cdot K \cdot D^2 + K \cdot D^3)$.
  - Total for $I$ iterations: $O(I \cdot (N \cdot K \cdot D^2 + K \cdot D^3))$.
- **Space Complexity**:
  - Parameters: $O(K \cdot D^2)$ for covariances, $O(K \cdot D)$ for means.
  - Responsibilities: $O(N \cdot K)$ memory per iteration.
- **Trade-offs**: Captures rich multi-modal geometries and soft uncertainty, but non-convex ELBO is prone to local optima, requiring good initialization.

I. **Progressive Difficulty Levels & Definition of Done**
- **Level 1 (Guided)**: Implement 1D GMM with fixed variance ($\sigma^2=1$) and derive E-step and M-step for means and weights.
- **Level 2 (Standard)**: Complete general multivariate `GaussianMixtureModel` with full covariance matrices, diagonal regularization, and Log-Sum-Exp trick.
- **Level 3 (Challenge)**: Implement K-Means initialization (`means_init='kmeans'`) to jumpstart EM near high-density centers and compare convergence speed against random initialization.
- **Definition of Done**: Demonstrates monotonic log-likelihood increase, produces soft responsibilities summing to 1.0, and passes all edge cases.

J. **Reflection & Self-Check** (Step 10: Independent Completion)
1. Why does maximizing the expected complete-data log-likelihood in the M-step also increase the true marginal log-likelihood $\ln P(\mathbf{X})$?
2. What is the difference between diagonal covariance GMM ($\boldsymbol{\Sigma}_k = \operatorname{diag}(\sigma_{k1}^2, \dots)$), spherical GMM ($\boldsymbol{\Sigma}_k = \sigma_k^2 \mathbf{I}$), and full covariance GMM in terms of parameter count and geometric expressiveness?
"""
