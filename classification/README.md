# Chapter 2: Classification

Welcome to Chapter 2! In this chapter, we transition from predicting continuous numerical quantities to predicting **discrete categorical class labels**. You will explore how models construct decision boundaries, output probabilities, maximize geometric margins, and leverage Bayesian probability.

---

## Chapter Objectives
- Understand how classification differs from regression (discrete outputs vs. continuous targets).
- Learn how linear score functions $z = \mathbf{w}^T \mathbf{x} + b$ define decision boundaries in feature space.
- Bridge regression to probabilistic classification using the sigmoid squashing function.
- Understand and derive the Binary Cross-Entropy loss from Maximum Likelihood Estimation.
- Contrast **discriminative classifiers** (Logistic Regression, SVM) with **generative classifiers** (Naive Bayes).
- Contrast **eager parametric learners** (Logistic Regression, SVM) with **lazy non-parametric learners** (K-Nearest Neighbors).
- Understand the geometry of maximum-margin hyperplanes and hinge loss optimization.

---

## Prerequisites
Before tackling this chapter, ensure you have completed:
- **[Chapter 1: Regression](../regression/README.md)** — Gradient descent, MSE loss, parameter updates.
- **[Stage 0C: Linear Algebra](../foundations/README.md#module-0c-linear-algebra)** — Hyperplanes ($\mathbf{w}^T \mathbf{x} + b = 0$), dot products as projections, Euclidean distance.
- **[Stage 0D: Calculus & Optimization](../foundations/README.md#module-0d-calculus-and-optimization)** — Partial derivatives, gradients, the chain rule.
- **[Stage 0E: Probability & Statistics](../foundations/README.md#module-0e-probability-and-statistics)** — Conditional probability, Bayes' theorem, Gaussian distributions (critical for Naive Bayes and Logistic Regression).
- **[Stage 0F: Machine Learning Core](../foundations/README.md#module-0f-machine-learning-core-concepts)** — Classification framing, accuracy, confusion matrices, underfitting and overfitting.

---

## 🧭 Pedagogical Progression & Intuition Bridges

To prevent conceptual jumps, absorb these intuitions before tackling the code:

### 1. Before Logistic Regression: The Journey from Scores to Probabilities
- **Binary Labels**: Targets are discrete categories $y \in \{0, 1\}$.
- **Raw Scores**: A linear combination $z = \mathbf{w}^T \mathbf{x} + b$ produces any real number in $(-\infty, +\infty)$.
- **Sigmoid Squashing**: We cannot interpret $z = 14.2$ or $z = -3.5$ as a probability. The logistic sigmoid $\sigma(z) = \frac{1}{1 + e^{-z}}$ smoothly compresses $(-\infty, +\infty)$ into the valid probability range $(0, 1)$.
- **Cross-Entropy Intuition**: If true label $y = 1$, we want prediction $p \approx 1$. If the model predicts $p = 0.01$, the error should be catastrophic ($-\log(0.01) \approx 4.6$). If the model predicts $p = 0.99$, error is nearly zero ($-\log(0.99) \approx 0.01$). This yields Binary Cross-Entropy:
  $$L = -[y \log(p) + (1 - y) \log(1 - p)]$$

### 2. Before K-Nearest Neighbors: Instance-Based Voting
- KNN makes **no assumptions** about underlying data distributions (non-parametric).
- It stores the entire training dataset ("lazy learning"). At inference time, it measures Euclidean distances to all points, selects the $K$ closest points, and takes a majority vote.
- **Feature Scaling**: Because distance is computed across coordinates $\sqrt{\sum (x_i - z_i)^2}$, a feature with a large scale (e.g., income in dollars) will completely dominate a feature with a small scale (e.g., age in years). Standardization is mandatory!

### 3. Before Gaussian Naive Bayes: Generative Likelihoods
- Rather than directly modeling $P(y|\mathbf{x})$, Naive Bayes models the data-generation process $P(\mathbf{x}|y)$ and uses Bayes' theorem:
  $$P(y=c|\mathbf{x}) \propto P(\mathbf{x}|y=c) P(y=c)$$
- The "Naive" assumption states that conditioned on the class label $y$, all features $x_1, \dots, x_D$ are **mutually independent**:
  $$P(\mathbf{x}|y=c) = \prod_{j=1}^D P(x_j|y=c)$$
- For continuous features, each $P(x_j|y=c)$ is modeled as a 1D Gaussian bell curve $\mathcal{N}(\mu_{jc}, \sigma_{jc}^2)$.

### 4. Before Linear Support Vector Machines: Geometric Margin
- Many hyperplanes can separate two classes. Which one is best?
- The SVM chooses the unique hyperplane that maximizes the **geometric margin**—the distance between the separating boundary and the nearest training points (the "support vectors").
- Instead of probabilities, SVM uses **Hinge Loss** $\max(0, 1 - y_i(\mathbf{w}^T \mathbf{x}_i + b))$ where labels are $y \in \{-1, +1\}$. Correct predictions outside the margin incur zero loss!

---

## Recommended Study Sequence

1. **[`knn.py`](./knn.py)** — Start here. Instance-based learning requires only Euclidean distance and sorting; no calculus or gradient descent needed.
2. **[`logistic_regression.py`](./logistic_regression.py)** — Bridges linear regression to classification using sigmoid and cross-entropy.
3. **[`gaussian_naive_bayes.py`](./gaussian_naive_bayes.py)** — Generative modeling applying Bayes' theorem and class-conditional statistics.
4. **[`linear_svm.py`](./linear_svm.py)** — Maximum margin classification using hinge loss and subgradient descent.

---

## Chapter 2 Toy Datasets

- **Dataset C1 (2D Linearly Separable)**:
  - $\mathbf{X} = [[0.1, 0.2], [0.2, 0.1], [0.8, 0.9], [0.9, 0.8]]$
  - $\mathbf{y} = [0, 0, 1, 1]$
  - Hand check: Points near $(0, 0)$ are class 0; points near $(1, 1)$ are class 1. The separating line is $x_1 + x_2 = 1.0$.

- **Dataset C2 (1D Threshold Separation)**:
  - $\mathbf{X} = [[1.0], [2.0], [3.0], [6.0], [7.0], [8.0]]$
  - $\mathbf{y} = [0, 0, 0, 1, 1, 1]$
  - Hand check: Perfect decision threshold is at $x = 4.5$.

- **Dataset C3 (The XOR Problem / Non-Separable)**:
  - $\mathbf{X} = [[0, 0], [0, 1], [1, 0], [1, 1]]$
  - $\mathbf{y} = [0, 1, 1, 0]$
  - Hand check: Linear classifiers (Logistic Regression, Linear SVM) CANNOT separate this dataset, but KNN with $k=1$ perfectly fits it.

---

## Completion Checklist
- [ ] Implement and test `knn.py` (Level 1, 2, and 3).
- [ ] Implement and test `logistic_regression.py` with stable sigmoid.
- [ ] Implement and test `gaussian_naive_bayes.py` with log-space calculations.
- [ ] Implement and test `linear_svm.py` with hinge loss and subgradient updates.
- [ ] Compare decision boundaries of Logistic Regression vs Linear SVM on Dataset C1.

---

## Cross-Chapter Conceptual Questions
1. **Discriminative vs Generative**: Why can Gaussian Naive Bayes handle missing features during inference more naturally than Logistic Regression?
2. **Loss Landscape Comparison**: Plot Hinge Loss vs Binary Cross-Entropy vs 0-1 Loss. How does Hinge Loss behave for points that are correctly classified and far from the decision boundary?
3. **Lazy vs Eager Inference**: Why does KNN have $O(1)$ training time but $O(N \cdot D)$ inference time, while Logistic Regression has $O(\text{epochs} \cdot N \cdot D)$ training time but $O(D)$ inference time?

---

## Personal Notes
*(Use this space to record your thoughts, notes, and reflections.)*
