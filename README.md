# ML Algorithms From Scratch 🧠

> *"What I cannot create, I do not understand." — Richard Feynman*

Welcome to the **ML Algorithms From Scratch** curriculum. This repository provides a self-directed, rigorous learning path to mastering machine learning algorithms by deriving their mathematics and implementing them from scratch in Python, using only `NumPy`.

The unique format of this repository relies on **docstring-only exercises**. The Python files provide detailed, guided learning material inside module-level docstrings. Your task is to read the guide, hand-compute the steps on toy datasets, and finally write the Python implementation below the docstring.

---

## Progress Tracking & Mastery Levels

As you progress through the curriculum, track your status for each algorithm. 
**Note:** *Implemented* does not mean *Mastered*. True mastery requires analyzing complexity, passing all tests, and being able to re-derive the algorithm from memory.

| Status | Symbol | Meaning |
| :--- | :---: | :--- |
| **Not Started** | ⬜ | Haven't begun reading or working on this module. |
| **Learning** | 📘 | Reading docstrings, understanding concepts, and hand-computing. |
| **Needs Review** | 🟧 | Encountered an error or mathematical confusion; need to revisit foundations. |
| **Ready to Implement** | 🟨 | Derivations complete, pseudocode written, ready to code. |
| **Implemented** | 🟩 | Code is written and runs without syntax errors. |
| **Tested** | 🧪 | All edge cases and assertions pass successfully. |
| **Mastered** | ⭐ | Level 3 challenges completed, time/space complexity analyzed, can re-derive from memory. |

---

## Stage 0: Foundations

Before writing ML algorithms, ensure your mathematical and programming foundations are solid. See [Foundations README](foundations/README.md) for details.

- **[0A: Linear Algebra Essentials](foundations/0a_linear_algebra.md)** - Vectors, matrices, dot products, matrix multiplication, transpose, inverses.
- **[0B: Calculus for Optimization](foundations/0b_calculus.md)** - Partial derivatives, gradients, chain rule, Jacobians.
- **[0C: Probability & Statistics](foundations/0c_probability.md)** - Expected value, variance, distributions, Bayes' theorem, maximum likelihood.
- **[0D: NumPy Mastery](foundations/0d_numpy.md)** - Vectorization, broadcasting, advanced indexing, avoiding `for` loops.
- **[0E: The Experimental Workflow](foundations/0e_workflow.md)** - How to test algorithms, track metrics, and debug effectively.
- **[0F: Exit Self-Assessment](foundations/0f_assessment.md)** - A test to determine if you are ready for Chapter 1.

---

## Curriculum Progression

### Recommended Learning Order (Beginner Route)

```text
[Stage 0: Foundations] 
          │
          ▼
[Chapter 1: Regression] ───────┐
          │                    │
          ▼                    ▼
[Chapter 2: Classification]    │
          │                    │
          ▼                    │
[Chapter 3: Trees/Ensembles]   │
          │                    │
          ▼                    ▼
[Chapter 4: Clustering] ◄──[Chapter 5: Dimensionality Reduction]
          │
          ▼
[Chapter 6: Neural Networks]
          │
          ▼
[Chapter 7: Advanced Topics]
```

### Hard Prerequisite Graph

| Chapter | Hard Prerequisites | Recommended Context |
| :--- | :--- | :--- |
| **Ch 1: Regression** | Stage 0 (0A, 0B, 0D) | Basic geometry of lines and planes. |
| **Ch 2: Classification** | Chapter 1, Stage 0 (0C) | Probability theory, Sigmoid function. |
| **Ch 3: Trees** | Stage 0 (0C) | Entropy, Gini Impurity, Recursion. |
| **Ch 4: Clustering** | Stage 0 (0A, 0D) | Distance metrics, basic optimization. |
| **Ch 5: Dim Reduction** | Stage 0 (0A, 0C) | Eigenvectors, eigenvalues, covariance matrices. |
| **Ch 6: Neural Networks** | Chapters 1 & 2, Stage 0 (0B) | Multivariable calculus, Chain Rule, Matrix calculus. |
| **Ch 7: Advanced** | Chapters 3, 4, 6 | Depends heavily on the specific algorithm. |

---

## Master Chapter & Exercise Index

### [Chapter 1: Regression Models](ch01_regression/README.md)
*Predicting continuous numerical values.*
- [🟩] `linear_regression.py` - Ordinary Least Squares (OLS) via Gradient Descent & Normal Equation.
- [⬜] `ridge_regression.py` - L2 Regularized linear regression.
- [⬜] `lasso_regression.py` - L1 Regularized linear regression with coordinate descent.
- [⬜] `polynomial_regression.py` - Non-linear regression using polynomial feature expansion.
- [🟩] `log_transformed_exp_regression.py` - Modeling exponential growth via log-linearization.

### [Chapter 2: Classification Models](ch02_classification/README.md)
*Predicting discrete classes and probabilities.*
- [⬜] `logistic_regression.py` - Binary classification using the sigmoid function and cross-entropy.
- [⬜] `knn_classifier.py` - K-Nearest Neighbors using Euclidean distance and voting.
- [⬜] `naive_bayes.py` - Gaussian Naive Bayes using Bayes' theorem and prior/posterior probabilities.
- [⬜] `svm_linear.py` - Support Vector Machine with linear kernel using hinge loss.

### [Chapter 3: Trees & Ensembles](ch03_trees/README.md)
*Non-parametric models relying on recursive partitioning.*
- [⬜] `decision_tree.py` - CART algorithm using Gini/Entropy for classification.
- [⬜] `random_forest.py` - Ensemble of decision trees with bagging and feature subsetting.
- [⬜] `gradient_boosting.py` - Sequential tree building to minimize residual errors.

### [Chapter 4: Clustering & Unsupervised](ch04_clustering/README.md)
*Finding hidden structures in unlabeled data.*
- [⬜] `kmeans.py` - K-Means clustering using Lloyd's algorithm.
- [⬜] `gaussian_mixture.py` - GMM using the Expectation-Maximization (EM) algorithm.
- [⬜] `dbscan.py` - Density-Based Spatial Clustering of Applications with Noise.

### [Chapter 5: Dimensionality Reduction](ch05_dim_reduction/README.md)
*Compressing feature spaces while preserving variance.*
- [⬜] `pca.py` - Principal Component Analysis via Eigendecomposition and SVD.
- [⬜] `tsne.py` - t-Distributed Stochastic Neighbor Embedding.

### [Chapter 6: Neural Networks (From Scratch)](ch06_neural_networks/README.md)
*Deep learning foundations using raw matrix operations.*
- [⬜] `perceptron.py` - Single-layer linear threshold unit.
- [⬜] `mlp_forward.py` - Multi-Layer Perceptron forward propagation.
- [⬜] `mlp_backprop.py` - Backpropagation and weight updates.
- [⬜] `optimizers.py` - SGD with Momentum, RMSProp, and Adam.

### [Chapter 7: Advanced Algorithms & Time Series](ch07_advanced/README.md)
*Specialized algorithms for sequences, states, and reinforcement.*
- [⬜] `hmm.py` - Hidden Markov Models (Forward-Backward, Viterbi).
- [⬜] `markov_chain.py` - State transitions and steady-state probabilities.

---

## The 10-Step Apprenticeship Workflow

To truly learn an algorithm, follow this workflow for every exercise file:

1. **Read the Context**: Understand the problem the algorithm solves.
2. **Review the Math**: Study the loss function, gradients, and update rules.
3. **Trace the Toy Example**: Work through the hand-computable dataset with pen and paper.
4. **Draft Pseudocode**: Write out the exact loop and operations in plain English.
5. **Implement Level 1 (Guided)**: Write the core algorithm focusing purely on correctness.
6. **Test Level 1**: Verify against the toy dataset output.
7. **Implement Level 2 (Standard)**: Add vectorization, hyperparameter handling, and error checking.
8. **Test Edge Cases**: Feed the model bad data (e.g., all zeros, collinear features) and handle exceptions.
9. **Refactor & Optimize (Challenge)**: Optimize matrix operations (e.g., replace loops with `NumPy` broadcasting).
10. **Reflect**: Answer the conceptual questions in the docstring to solidify your understanding.

---

## Toy Dataset Collection

These hand-computable datasets are designed to be traced on paper. Use them to debug your implementations before scaling up.

### Dataset R1: Tiny Linear Regression
- **Data**: `X = [[1], [2], [3], [4], [5]]`, `y = [2, 4, 6, 8, 10]`
- **Purpose**: Verify OLS gradient descent. The ideal weight is exactly 2.0 with 0 bias.
- **Used In**: `linear_regression.py`, `ridge_regression.py`

### Dataset C1: 2-Feature Binary Classification
- **Data**: `X = [[0.1, 0.2], [0.2, 0.1], [0.9, 0.8], [0.8, 0.9]]`, `y = [0, 0, 1, 1]`
- **Purpose**: Linearly separable classes at opposite corners of the unit square. 
- **Used In**: `logistic_regression.py`, `svm_linear.py`

### Dataset T1: Single Split Decision Tree
- **Data**: `X = [[1, 0], [1, 1], [0, 0], [0, 1]]`, `y = [1, 1, 0, 0]`
- **Purpose**: The first feature perfectly divides the classes. The tree should split on `feature_0` immediately.
- **Used In**: `decision_tree.py`

### Dataset K1: Two Separated Clusters
- **Data**: `X = [[0, 0], [0, 1], [1, 0], [10, 10], [10, 11], [11, 10]]`
- **Purpose**: Two distinct clusters (around `[0,0]` and `[10,10]`). Easy to verify center assignment.
- **Used In**: `kmeans.py`, `gaussian_mixture.py`

### Dataset P1: Perfectly Correlated Features
- **Data**: `X = [[1, 2], [2, 4], [3, 6], [4, 8]]`
- **Purpose**: Feature 2 is exactly 2x Feature 1. One principal component explains 100% of the variance.
- **Used In**: `pca.py`

### Dataset X1: The XOR Problem
- **Data**: `X = [[0, 0], [0, 1], [1, 0], [1, 1]]`, `y = [0, 1, 1, 0]`
- **Purpose**: A classic non-linear problem. Linear models will fail; MLPs will succeed.
- **Used In**: `mlp_forward.py`, `mlp_backprop.py`

### Dataset H1: Tiny Markov Sequence
- **Data**: States = `[Rain, Sun]`, Observations = `[Walk, Shop, Clean]`. Sequence: `[Walk, Walk, Clean, Shop, Clean]`
- **Purpose**: Test Viterbi decoding or Forward-Backward probabilities on a tiny 5-step sequence.
- **Used In**: `hmm.py`

---

## Repository Structure

```text
ml-algorithms-from-scratch/
├── README.md                      # Curriculum guide (you are here)
├── foundations/                   # Math & NumPy prerequisites
│   ├── README.md
│   ├── 0a_linear_algebra.md
│   └── ...
├── ch01_regression/               # Chapter 1
│   ├── linear_regression.py
│   └── ...
├── ch02_classification/
│   └── ...
├── ch03_trees/
│   └── ...
├── ch04_clustering/
│   └── ...
├── ch05_dim_reduction/
│   └── ...
├── ch06_neural_networks/
│   └── ...
├── ch07_advanced/
│   └── ...
└── LICENSE                        # MIT License
```

## License

This project is licensed under the [MIT License](LICENSE).
