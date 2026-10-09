# ML Algorithms From Scratch 🧠

A structured, self-guided machine learning apprenticeship. This repository is designed as a question book where you learn machine learning by deriving mathematics, designing algorithms, implementing them from scratch, and verifying them with rigorous test cases.

---

## 🗺️ Curriculum Roadmap

The table below outlines the recommended progression across the seven core chapters, clearly distinguishing between **Hard Prerequisites** (the fundamental mathematical and conceptual concepts required) and the **Recommended Order**.

| # | Chapter | Hard Prerequisites | Recommended After | Status |
|---|---|---|---|:---:|
| 1 | [**regression/**](regression/) | Single-variable calculus, basic vectors | — | 🟡 In Progress |
| 2 | [**classification/**](classification/) | Gradient descent, loss functions, linear algebra | Chapter 1 | ⬜ Not Started |
| 3 | [**trees_and_ensembles/**](trees_and_ensembles/) | Supervised learning foundations, bias-variance tradeoff | Chapters 1 & 2 | ⬜ Not Started |
| 4 | [**clustering/**](clustering/) | Vector distance metrics, iterative coordinate updates | Chapter 1 | ⬜ Not Started |
| 5 | [**dimensionality_reduction/**](dimensionality_reduction/) | Matrix operations, eigendecomposition, covariance & variance | Chapter 1 | ⬜ Not Started |
| 6 | [**neural_networks/**](neural_networks/) | Multivariable calculus (chain rule), linear algebra, gradient descent | Chapters 1 & 2 | ⬜ Not Started |
| 7 | [**advanced/**](advanced/) | Probability theory, expectation/variance, MLE, latent-variable concepts | Chapters 1, 4, 6 | ⬜ Not Started |

> **Prerequisite Note**: Chapters 4 (Clustering), 5 (Dimensionality Reduction), and 6 (Neural Networks) have independent mathematical foundations and can be tackled in any order once their respective hard prerequisites are met. Chapter 7 requires probabilistic maturity and benefits from prior exposure to clustering (K-Means) and iterative optimization.

---

## 📚 Complete Exercise Index

### [Chapter 1: Regression](regression/)
*Continuous target prediction, gradient descent optimization, and shrinkage methods.*
- [`linear_regression.py`](regression/linear_regression.py) — Single-variable linear regression via batch gradient descent *(Existing Reference Implementation)*
- [`log_transformed_exp_regression.py`](regression/log_transformed_exp_regression.py) — Exponential target regression via logarithmic linear projection *(Existing Reference Implementation)*
- [`polynomial_regression.py`](regression/polynomial_regression.py) — Polynomial feature expansion and higher-degree curve fitting *(Exercise)*
- [`ridge_regression.py`](regression/ridge_regression.py) — $L_2$ Tikhonov regularization and normal equations *(Exercise)*
- [`lasso_regression.py`](regression/lasso_regression.py) — $L_1$ regularization, subgradients, and coordinate descent for sparsity *(Exercise)*

### [Chapter 2: Classification](classification/)
*Discrete decision boundaries, generative vs. discriminative classifiers, and margin optimization.*
- [`logistic_regression.py`](classification/logistic_regression.py) — Binary classification, sigmoid activations, and log-loss gradient descent *(Exercise)*
- [`knn.py`](classification/knn.py) — Non-parametric instance-based lazy learning with distance metrics *(Exercise)*
- [`gaussian_naive_bayes.py`](classification/gaussian_naive_bayes.py) — Generative classification via Bayes' theorem and Gaussian conditionals *(Exercise)*
- [`linear_svm.py`](classification/linear_svm.py) — Maximum-margin hyperplanes via hinge loss and subgradient descent *(Exercise)*

### [Chapter 3: Trees and Ensembles](trees_and_ensembles/)
*Non-linear recursive space partitioning, bagging variance reduction, and boosting.*
- [`decision_tree.py`](trees_and_ensembles/decision_tree.py) — CART algorithm supporting Gini/entropy classification and variance-reduction regression *(Exercise)*
- [`random_forest.py`](trees_and_ensembles/random_forest.py) — Bagging ensemble with bootstrap sampling and feature sub-selection *(Exercise)*
- [`gradient_boosting.py`](trees_and_ensembles/gradient_boosting.py) — Sequential residual fitting via gradient descent in function space *(Exercise)*

### [Chapter 4: Clustering](clustering/)
*Unsupervised pattern discovery without ground truth labels.*
- [`kmeans.py`](clustering/kmeans.py) — Lloyd's alternating minimization, WCSS inertia, and K-Means++ initialization *(Exercise)*
- [`dbscan.py`](clustering/dbscan.py) — Density-based clustering with core points, $\epsilon$-neighborhoods, and noise handling *(Exercise)*
- [`agglomerative_clustering.py`](clustering/agglomerative_clustering.py) — Hierarchical agglomerative clustering with single, complete, and average linkage *(Exercise)*

### [Chapter 5: Dimensionality Reduction](dimensionality_reduction/)
*Feature projection, variance preservation, and supervised discriminant projection.*
- [`pca.py`](dimensionality_reduction/pca.py) — Principal Component Analysis via covariance eigendecomposition and context-dependent standardization *(Exercise)*
- [`lda.py`](dimensionality_reduction/lda.py) — Supervised Linear Discriminant Analysis maximizing between-to-within class scatter *(Exercise)*

### [Chapter 6: Neural Networks](neural_networks/)
*Differentiable computational graphs, backpropagation, and first-order optimization algorithms.*
- [`perceptron.py`](neural_networks/perceptron.py) — Single artificial neuron, linear boundaries, and the XOR limitation *(Exercise)*
- [`multilayer_perceptron.py`](neural_networks/multilayer_perceptron.py) — Multilayer dense network, backpropagation via chain rule, symmetry preservation/breaking, and finite-difference gradient checking *(Exercise)*
- [`sgd.py`](neural_networks/sgd.py) — Stochastic and mini-batch gradient descent *(Exercise)*
- [`momentum.py`](neural_networks/momentum.py) — Polyak momentum and velocity accumulation *(Exercise)*
- [`rmsprop.py`](neural_networks/rmsprop.py) — Moving average of squared gradients and adaptive coordinate scaling *(Exercise)*
- [`adam.py`](neural_networks/adam.py) — Adaptive Moment Estimation combining first/second moments with initialization bias correction *(Exercise)*

### [Chapter 7: Advanced Probabilistic Models](advanced/)
*Latent variable modeling, the Expectation-Maximization algorithm, and Markov sequence inference.*
- [`gaussian_mixture_model.py`](advanced/gaussian_mixture_model.py) — Soft-clustering density estimation via Gaussian EM *(Exercise)*
- [`hmm_viterbi.py`](advanced/hmm_viterbi.py) — Hidden Markov Model dynamic programming trellis decoding in log-space *(Exercise)*
- [`hmm_forward_backward.py`](advanced/hmm_forward_backward.py) — Forward-Backward inference and Baum-Welch parameter learning *(Exercise)*

---

## 🛠️ How to Use This Question Book

### 1. The Core Design Rule
Every unfinished `.py` exercise file in this repository contains **exclusively a single comprehensive module-level docstring**. There are no functions, classes, imports, or boilerplate code outside that docstring. The docstring is the assignment; the empty file beneath it is your workspace.

### 2. The 7-Step Apprenticeship Workflow
For each algorithm:
1. **Study the Questions**: Read sections `A` (Mission), `B` (Prerequisites), and `C` (Learning Questions) in the module docstring. Answer the conceptual questions in your own words before writing code.
2. **Derive the Math by Hand**: Complete section `D` (Mathematics to Derive) with pen and paper. Derive loss functions, gradients, recurrence relations, or update equations from first principles.
3. **Review the Contract**: Inspect section `E` (Implementation Contract) for required class signatures, input/output tensor shapes, and state attributes.
4. **Implement from Scratch**: Follow section `F` (Guided Implementation Stages). Use **NumPy only for vectorized array operations**; never import pre-built estimators from `scikit-learn` or deep learning frameworks.
5. **Pass the Edge Cases**: Write tests as described in section `G` (Edge Cases and Expected Tests). Verify numerical stability, gradient correctness, and boundary behavior.
6. **Analyze Complexity**: Answer section `H` (Complexity Analysis) regarding time and space scaling.
7. **Reflect & Complete**: Answer section `J` (Reflection) and check off the item in the chapter's `README.md` completion checklist.
