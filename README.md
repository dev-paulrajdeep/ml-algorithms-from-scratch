# ML Algorithms From Scratch 🧠

> *"What I cannot create, I do not understand." — Richard Feynman*  
> Learn machine learning from the ground up: derive every equation by hand, design algorithms from first principles, implement each model from scratch using only Python and NumPy, and empirically verify your work against baselines.

---

## 🧭 The Learning Architecture

This repository is structured as a **self-guided apprenticeship and question book**:
- **Stage 0: Foundations** serves as the mathematical and computational on-ramp before Chapter 1, ensuring programming readiness, vectorization intuition, linear algebra, multivariable calculus, probability, and core ML discipline.
- **Seven Core ML Chapters** guide you through the classical algorithms of machine learning in a carefully sequenced pedagogical order.
- **Docstring-Only Exercise Files**: Every unfinished algorithm `.py` file contains **exclusively a single comprehensive assignment docstring**. There are no pre-written solutions, hidden boilerplate, or imports outside the docstring. The docstring is your interactive tutor and specification; the empty space beneath it is your workspace.
- **Reference Implementations**: Working baseline implementations in early chapters demonstrate how to write clean, minimal from-scratch ML code.

---

## 🚦 Progress Tracking & Mastery Levels

Track your journey through the curriculum using these seven statuses:

| Status | Symbol | Meaning |
| :--- | :---: | :--- |
| **Not Started** | ⬜ | You have not yet begun reading the docstring or deriving the math. |
| **Learning** | 📘 | Reading intuition, working through tiny numerical examples, studying mathematical derivations. |
| **Needs Review** | 🟧 | Encountered an error, failing test, or confusing mathematical step. Revisit Stage 0 foundations. |
| **Ready to Implement** | 🟨 | Handwritten derivations complete, pseudocode drafted, and API contracts understood. |
| **Implemented** | 🟩 | Code written from scratch in the `.py` file and runs on simple toy data. |
| **Tested** | 🧪 | All edge cases, numerical stability checks, and shape tests pass without errors. |
| **Mastered** | ⭐ | Level 3 challenges completed, complexity analyzed, reflection answered, and you can derive the algorithm from memory. |

> [!IMPORTANT]
> **Implemented $\neq$ Mastered.** Writing code that executes without error is only one milestone. True mastery requires verifying mathematical properties, testing boundary conditions, and understanding algorithmic trade-offs.

---

## 🧱 Stage 0: Foundations Before Machine Learning

Before tackling Chapter 1, complete or test out of **[Stage 0: Foundations Before Machine Learning](foundations/README.md)**. This is not an eighth algorithm chapter; it is the prerequisite bridge.

- **[Stage 0A: Programming Readiness](foundations/README.md#module-0a-programming-readiness)** — Python control flow, pure functions, error handling, tracebacks, and the minimal estimator class pattern (`__init__`, `fit`, `predict`).
- **[Stage 0B: NumPy & Numerical Computing](foundations/README.md#module-0b-numpy-and-numerical-computing)** — Array shapes, indexing, axis semantics (`axis=0` vs `axis=1`), broadcasting rules, vectorization vs loops, numerical stability, and reproducible RNG.
- **[Stage 0C: Linear Algebra](foundations/README.md#module-0c-linear-algebra)** — Vectors as coordinates, dot products as projections, matrix transformations, norms ($L_1, L_2$), distances, rank, invertibility, and eigenvectors/eigenvalues.
- **[Stage 0D: Calculus & Optimization](foundations/README.md#module-0d-calculus-and-optimization)** — Functions as input-output mappings, derivatives as slopes, partial derivatives, gradients as steepest ascent, chain rule, and scalar gradient descent step-by-step.
- **[Stage 0E: Probability & Statistics](foundations/README.md#module-0e-probability-and-statistics)** — Mean, variance, covariance, Bayes' theorem, Gaussian distributions, likelihood, log-likelihood, conditional independence, and latent variables.
- **[Stage 0F: Machine Learning Core](foundations/README.md#module-0f-machine-learning-core-concepts)** — Features, labels, parameters, predictions, supervised vs unsupervised, train/val/test splits, data leakage, baseline models, and the bias-variance tradeoff.
- **[The Experimental Workflow](foundations/README.md#the-experimental-workflow)** — The 10-step scientific loop and the distinction between debugging, testing, and experimentation.
- **[Stage 0 Exit Self-Assessment](foundations/README.md#stage-0-exit-self-assessment)** — Diagnostic conceptual questions, hand calculations, and practice tasks to confirm readiness.

---

## 🗺️ Curriculum Progression

### 1. Recommended Learning Order (Beginner Route)

For learners following the guided apprenticeship, follow this linear progression:

```text
               ┌──────────────────────────────────────────────────┐
               │         Stage 0: Mathematical & Programming      │
               │                     Foundations                  │
               └─────────────────────────┬────────────────────────┘
                                         ▼
               ┌──────────────────────────────────────────────────┐
               │ Chapter 1: Regression (Continuous Predictions,   │
               │            Gradients, MSE, Regularization)       │
               └─────────────────────────┬────────────────────────┘
                                         ▼
               ┌──────────────────────────────────────────────────┐
               │ Chapter 2: Classification (Decision Boundaries,  │
               │            Probabilities, Margins, Bayes)        │
               └─────────────────────────┬────────────────────────┘
                                         ▼
               ┌──────────────────────────────────────────────────┐
               │ Chapter 3: Trees & Ensembles (Recursive Space    │
               │            Partitioning, Bagging, Boosting)      │
               └─────────────────────────┬────────────────────────┘
                                         ▼
               ┌──────────────────────────────────────────────────┐
               │ Chapter 4: Clustering (Unsupervised Grouping,    │
               │            Centroids, Density, Hierarchies)      │
               └─────────────────────────┬────────────────────────┘
                                         ▼
               ┌──────────────────────────────────────────────────┐
               │ Chapter 5: Dimensionality Reduction (Variance    │
               │            Preservation, Projections, LDA)       │
               └─────────────────────────┬────────────────────────┘
                                         ▼
               ┌──────────────────────────────────────────────────┐
               │ Chapter 6: Neural Networks (Layered Graphs,      │
               │            Backpropagation, Modern Optimizers)   │
               └─────────────────────────┬────────────────────────┘
                                         ▼
               ┌──────────────────────────────────────────────────┐
               │ Chapter 7: Advanced Probabilistic Models         │
               │            (Latent Variables, EM Algorithm, HMM) │
               └──────────────────────────────────────────────────┘
```

### 2. Hard Prerequisite Graph

If you wish to explore chapters non-linearly, observe their **genuine mathematical dependencies**:

| Chapter | Hard Prerequisites | Recommended Context |
| :--- | :--- | :--- |
| **[Chapter 1: Regression](regression/)** | Stage 0B (NumPy), Stage 0C (Vectors), Stage 0D (Derivatives), Stage 0F (Core ML) | Stage 0A (Classes) |
| **[Chapter 2: Classification](classification/)** | Chapter 1 (Gradient Descent), Stage 0D, Stage 0E (for Naive Bayes), Stage 0C (for SVM) | Chapter 1 (Loss Functions) |
| **[Chapter 3: Trees & Ensembles](trees_and_ensembles/)** | Chapter 1 & 2 (Losses, Overfitting), Stage 0E (Variance & Entropy) | Chapters 1–2 |
| **[Chapter 4: Clustering](clustering/)** | Stage 0C (Euclidean Distance & Norms), Stage 0B (NumPy Arrays) *(Note: Dim reduction is NOT required)* | Chapter 1 |
| **[Chapter 5: Dimensionality Reduction](dimensionality_reduction/)** | Stage 0C (Eigenvectors, Covariance Matrices), Stage 0E (Variance) | Chapter 4 (Visualization) |
| **[Chapter 6: Neural Networks](neural_networks/)** | Stage 0C (Matrix Multiplication), Stage 0D (Chain Rule & Gradients), Chapter 1 (GD) | Chapter 2 (Sigmoid & Cross-Entropy) |
| **[Chapter 7: Advanced Models](advanced/)** | Stage 0E (Bayes, Expectation, Likelihood, Latent Variables), Chapter 4 (K-Means) | Chapter 6 (Optimization) |

---

## 📚 Master Chapter & Exercise Index

### [Chapter 1: Regression](regression/)
*Continuous target prediction, gradient descent optimization, and regularized shrinkage.*
- [`linear_regression.py`](regression/linear_regression.py) — Single-variable linear regression via batch gradient descent *(Reference Implementation)* `[🟩]`
- [`log_transformed_exp_regression.py`](regression/log_transformed_exp_regression.py) — Log-transformed exponential regression with manual inference *(Reference Implementation)* `[🟩]`
- [`polynomial_regression.py`](regression/polynomial_regression.py) — Feature transformations, Vandermonde matrices, and the bias-variance tradeoff `[⬜]`
- [`ridge_regression.py`](regression/ridge_regression.py) — $L_2$ regularization, analytical normal equations, and weight shrinkage `[⬜]`
- [`lasso_regression.py`](regression/lasso_regression.py) — $L_1$ regularization, subgradients, coordinate descent, and sparse feature selection `[⬜]`

### [Chapter 2: Classification](classification/)
*Discrete decision boundaries, probabilistic classifiers, and margin optimization.*
- [`knn.py`](classification/knn.py) — Non-parametric instance-based lazy learning with distance metrics `[⬜]`
- [`logistic_regression.py`](classification/logistic_regression.py) — Sigmoid squashing, binary cross-entropy loss, and probabilistic decision boundaries `[⬜]`
- [`gaussian_naive_bayes.py`](classification/gaussian_naive_bayes.py) — Generative classification via Bayes' theorem and Gaussian class conditionals `[⬜]`
- [`linear_svm.py`](classification/linear_svm.py) — Maximum-margin hyperplanes via hinge loss and subgradient descent `[⬜]`

### [Chapter 3: Trees and Ensembles](trees_and_ensembles/)
*Non-linear recursive space partitioning, bagging variance reduction, and boosting.*
- [`decision_tree.py`](trees_and_ensembles/decision_tree.py) — CART supporting Gini/entropy classification and variance-reduction regression `[⬜]`
- [`random_forest.py`](trees_and_ensembles/random_forest.py) — Bagging ensemble with bootstrap aggregation and feature sub-selection `[⬜]`
- [`gradient_boosting.py`](trees_and_ensembles/gradient_boosting.py) — Sequential residual fitting via gradient descent in function space `[⬜]`

### [Chapter 4: Clustering](clustering/)
*Unsupervised pattern discovery without ground truth labels.*
- [`kmeans.py`](clustering/kmeans.py) — Lloyd's alternating minimization, WCSS inertia, and K-Means++ initialization `[⬜]`
- [`dbscan.py`](clustering/dbscan.py) — Density-based clustering with core points, $\epsilon$-neighborhoods, and noise handling `[⬜]`
- [`agglomerative_clustering.py`](clustering/agglomerative_clustering.py) — Hierarchical agglomerative clustering with single, complete, and average linkage `[⬜]`

### [Chapter 5: Dimensionality Reduction](dimensionality_reduction/)
*Feature projection, variance preservation, and supervised discriminant projection.*
- [`pca.py`](dimensionality_reduction/pca.py) — Principal Component Analysis via covariance eigendecomposition and context-dependent standardization `[⬜]`
- [`lda.py`](dimensionality_reduction/lda.py) — Supervised Linear Discriminant Analysis maximizing between-to-within class scatter `[⬜]`

### [Chapter 6: Neural Networks](neural_networks/)
*Differentiable computational graphs, backpropagation, and first-order optimization algorithms.*
- [`perceptron.py`](neural_networks/perceptron.py) — Single artificial neuron, linear boundaries, and the XOR limitation `[⬜]`
- [`multilayer_perceptron.py`](neural_networks/multilayer_perceptron.py) — Multilayer dense network, backpropagation via chain rule, symmetry breaking, and gradient checking `[⬜]`
- [`sgd.py`](neural_networks/sgd.py) — Stochastic and mini-batch gradient descent `[⬜]`
- [`momentum.py`](neural_networks/momentum.py) — Polyak momentum and velocity accumulation `[⬜]`
- [`rmsprop.py`](neural_networks/rmsprop.py) — Moving average of squared gradients and adaptive coordinate scaling `[⬜]`
- [`adam.py`](neural_networks/adam.py) — Adaptive Moment Estimation combining first/second moments with initialization bias correction `[⬜]`

### [Chapter 7: Advanced Probabilistic Models](advanced/)
*Latent variable modeling, the Expectation-Maximization algorithm, and Markov sequence inference.*
- [`gaussian_mixture_model.py`](advanced/gaussian_mixture_model.py) — Soft-clustering density estimation via Gaussian EM `[⬜]`
- [`hmm_viterbi.py`](advanced/hmm_viterbi.py) — Hidden Markov Model dynamic programming trellis decoding in log-space `[⬜]`
- [`hmm_forward_backward.py`](advanced/hmm_forward_backward.py) — Forward-Backward inference and Baum-Welch parameter learning `[⬜]`

---

## 🔬 The 10-Step Apprenticeship Workflow

When approaching any exercise in this repository, follow this disciplined method:

1. **Step 0 — Understand the Problem**: Read the mission. Understand what real-world question the algorithm solves and what inputs/outputs are expected.
2. **Step 1 — Build Intuition**: Walk through the conceptual questions. What should happen visually or logically?
3. **Step 2 — Work Through a Tiny Numerical Example**: Calculate a small toy example with pencil and paper.
4. **Step 3 — Introduce Mathematical Notation**: Define symbols ($N, D, \mathbf{w}, b, \mathbf{X}, \hat{\mathbf{y}}, L$). Connect notation to the numerical example.
5. **Step 4 — Derive the Mathematics**: Derive gradients, update formulas, or objective functions step by step.
6. **Step 5 — Design the Algorithm**: Write plain English steps and pseudocode before touching Python.
7. **Step 6 — Implement the Simplest Correct Version (Level 1 - Guided)**: Write the scalar or loop-based logic for the smallest toy dataset.
8. **Step 7 — Generalize the Implementation (Level 2 - Standard)**: Vectorize with NumPy, handle $(N, D)$ shapes, and conform strictly to the class contract.
9. **Step 8 — Test and Debug**: Write tests for boundary conditions, extreme values, numerical stability, and degenerate shapes.
10. **Step 9 — Analyze and Reflect (Level 3 - Challenge)**: Measure scaling, compute big-O complexity, answer reflection prompts, and mark the exercise **Mastered** ⭐.

---

## 🧪 Toy Dataset Collection

These hand-computable synthetic datasets are shared across exercises to verify calculations before scaling up:

### Dataset R1: Tiny Linear Regression (1D)
- **Data**: $\mathbf{x} = [1, 2, 3, 4, 5]$, $\mathbf{y} = [2, 4, 6, 8, 10]$
- **Target Relationship**: $y = 2x + 0$ (perfect correlation, $R^2 = 1.0$)
- **Purpose**: Verify OLS gradient descent. The ideal weight is exactly $w=2.0$, bias $b=0.0$, MSE $= 0.0$.
- **Used In**: [`linear_regression.py`](regression/linear_regression.py), [`polynomial_regression.py`](regression/polynomial_regression.py), [`ridge_regression.py`](regression/ridge_regression.py)

### Dataset C1: 2-Feature Binary Classification
- **Data**: $\mathbf{X} = [[0.1, 0.2], [0.2, 0.1], [0.8, 0.9], [0.9, 0.8]]$, $\mathbf{y} = [0, 0, 1, 1]$
- **Purpose**: Linearly separable classes at opposite corners of the 2D unit square. Easy to compute decision boundary and distances by hand.
- **Used In**: [`logistic_regression.py`](classification/logistic_regression.py), [`knn.py`](classification/knn.py), [`linear_svm.py`](classification/linear_svm.py)

### Dataset T1: Single Split Decision Tree
- **Data**: $\mathbf{X} = [[1, 0], [1, 1], [0, 0], [0, 1]]$, $\mathbf{y} = [1, 1, 0, 0]$
- **Purpose**: Feature 0 perfectly splits the data ($x_0 > 0.5 \implies y=1$). A CART tree should find this split with 0 impurity in a single step.
- **Used In**: [`decision_tree.py`](trees_and_ensembles/decision_tree.py), [`random_forest.py`](trees_and_ensembles/random_forest.py)

### Dataset K1: Two Separated 2D Clusters
- **Data**: $\mathbf{X} = [[0, 0], [0, 1], [1, 0], [10, 10], [10, 11], [11, 10]]$
- **Purpose**: Two clusters with large separation gap ($[0,0]$ vs $[10,10]$). Centroids and assignments can be computed in seconds on paper.
- **Used In**: [`kmeans.py`](clustering/kmeans.py), [`dbscan.py`](clustering/dbscan.py), [`agglomerative_clustering.py`](clustering/agglomerative_clustering.py)

### Dataset P1: Perfectly Correlated Features
- **Data**: $\mathbf{X} = [[1, 2], [2, 4], [3, 6], [4, 8]]$
- **Purpose**: Feature 1 is exactly $2 \times$ Feature 0. The first principal component captures 100% of the variance; the second eigenvalue is exactly 0.
- **Used In**: [`pca.py`](dimensionality_reduction/pca.py), [`lda.py`](dimensionality_reduction/lda.py)

### Dataset X1: The XOR Problem
- **Data**: $\mathbf{X} = [[0, 0], [0, 1], [1, 0], [1, 1]]$, $\mathbf{y} = [0, 1, 1, 0]$
- **Purpose**: The canonical non-linearly separable problem. Demonstrates why a single perceptron fails and why a hidden layer in an MLP is necessary.
- **Used In**: [`perceptron.py`](neural_networks/perceptron.py), [`multilayer_perceptron.py`](neural_networks/multilayer_perceptron.py)

### Dataset H1: Short Observation Sequence
- **Data**: States = $\{S_1, S_2\}$, Observations = $\{O_1, O_2\}$. Sequence: $[O_1, O_1, O_2, O_2]$
- **Purpose**: Hand-trace the dynamic programming trellis for Viterbi decoding and Forward-Backward variables without computer assistance.
- **Used In**: [`hmm_viterbi.py`](advanced/hmm_viterbi.py), [`hmm_forward_backward.py`](advanced/hmm_forward_backward.py)

---

## 📂 Repository Structure

```text
ml-algorithms-from-scratch/
├── README.md                          # Main curriculum guide (you are here)
├── LICENSE                            # MIT License
├── .gitignore                         # Git ignore rules
├── foundations/                       # Stage 0: Foundations Before Machine Learning
│   └── README.md                      # Modules 0A through 0F, Workflow, and Self-Assessment
├── regression/                        # Chapter 1: Regression Models
│   ├── README.md                      # Chapter guide, 20-step learning ladder, checklists
│   ├── linear_regression.py           # ✅ Reference Implementation (scalar GD)
│   ├── log_transformed_exp_regression.py # ✅ Reference Implementation (log-linearization)
│   ├── polynomial_regression.py       # Exercise (docstring specification)
│   ├── ridge_regression.py            # Exercise (docstring specification)
│   └── lasso_regression.py            # Exercise (docstring specification)
├── classification/                    # Chapter 2: Classification Models
│   ├── README.md                      # Chapter guide, intuition bridges, checklists
│   ├── knn.py                         # Exercise (docstring specification)
│   ├── logistic_regression.py         # Exercise (docstring specification)
│   ├── gaussian_naive_bayes.py        # Exercise (docstring specification)
│   └── linear_svm.py                  # Exercise (docstring specification)
├── trees_and_ensembles/               # Chapter 3: Trees and Ensembles
│   ├── README.md                      # Chapter guide, impurity criteria, checklists
│   ├── decision_tree.py               # Exercise (docstring specification)
│   ├── random_forest.py               # Exercise (docstring specification)
│   └── gradient_boosting.py           # Exercise (docstring specification)
├── clustering/                        # Chapter 4: Clustering
│   ├── README.md                      # Chapter guide, unsupervised framing, checklists
│   ├── kmeans.py                      # Exercise (docstring specification)
│   ├── dbscan.py                      # Exercise (docstring specification)
│   └── agglomerative_clustering.py    # Exercise (docstring specification)
├── dimensionality_reduction/          # Chapter 5: Dimensionality Reduction
│   ├── README.md                      # Chapter guide, projection intuition, checklists
│   ├── pca.py                         # Exercise (docstring specification)
│   └── lda.py                         # Exercise (docstring specification)
├── neural_networks/                   # Chapter 6: Neural Networks
│   ├── README.md                      # Chapter guide, backprop mechanics, checklists
│   ├── perceptron.py                  # Exercise (docstring specification)
│   ├── multilayer_perceptron.py       # Exercise (docstring specification)
│   ├── sgd.py                         # Exercise (docstring specification)
│   ├── momentum.py                    # Exercise (docstring specification)
│   ├── rmsprop.py                     # Exercise (docstring specification)
│   └── adam.py                        # Exercise (docstring specification)
└── advanced/                          # Chapter 7: Advanced Probabilistic Models
    ├── README.md                      # Chapter guide, latent variables, checklists
    ├── gaussian_mixture_model.py      # Exercise (docstring specification)
    ├── hmm_viterbi.py                 # Exercise (docstring specification)
    └── hmm_forward_backward.py        # Exercise (docstring specification)
```

---

## ⚖️ License

This project is licensed under the [MIT License](LICENSE).
