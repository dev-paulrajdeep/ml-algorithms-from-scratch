"""
A. Mission
K-Means aims to partition $N$ observations into $K$ sets (clusters) by minimizing the within-cluster sum of squares (WCSS), also known as inertia. It iteratively refines assignments and centroid positions.

B. Prerequisites
- [Stage 0C: Linear Algebra](../foundations/README.md#module-0c-linear-algebra): Norms and distance metrics.
- [Stage 0F: ML Core](../foundations/README.md#module-0f-machine-learning-core-concepts): Unsupervised framing.

C. Learning Questions
1. Why does K-Means converge, and why does WCSS never increase?
2. How does the initial choice of centroids affect the final clustering?
3. What is the intuition behind K-Means++ initialization?
4. What happens on non-convex datasets like Dataset 4B?

D. Mathematics to Derive
1. Formulate the total objective function (Inertia/WCSS).
2. Prove that the assignment step and update step never increase WCSS.
3. Show that for a fixed cluster assignment, the mean minimizes the sum of squared Euclidean distances to the points in the cluster.

E. Implementation Contract
- Class name: `KMeans`
- Parameters: 
  - `k` (int)
  - `init` (str): 'random' or 'k-means++'
  - `max_iter` (int)
  - `tol` (float)
- Public Methods:
  - `fit(X)`
  - `predict(X)`
- Attributes:
  - `centroids_` (shape `(k, D)`)
  - `labels_` (shape `(N,)`)
  - `inertia_` (float)

F. Guided Implementation Stages
**Checkpoint 1: Initialization Strategy**
- *What to learn*: Establishing the initial cluster centers.
- *What to do*: Implement 'random' picking `k` data points.
- *How to check yourself*: Ensure the output is an array of shape `(k, D)`.
- *When to proceed*: Random init is working.
- *Recovery hints*:
  - Hint 1: Use `np.random.choice` with `replace=False`.
  - Hint 2: Index `X` with the chosen indices.
  - Hint 3: Keep it simple before adding K-Means++.

**Checkpoint 2: Assignment and Update Loop**
- *What to learn*: The iterative refinement of Lloyd's algorithm.
- *What to do*: Implement the assignment of points to the closest centroid, and update centroids to the mean of assigned points. Stop if centroids shift less than `tol`.
- *How to check yourself*: Track the WCSS; it should decrease or remain constant each step.
- *When to proceed*: Test on Dataset 4A; it should flawlessly separate the two clusters.
- *Recovery hints*:
  - Hint 1: Use `np.argmin` over distances to find assignments.
  - Hint 2: Compute new means using `np.mean(X[labels == i], axis=0)`.
  - Hint 3: Watch out for integer division or shape mismatches.

G. Edge Cases and Expected Tests
1. **Empty Clusters**: Re-initialize empty clusters to the furthest data point.
2. **Single Point Clusters**: Handle correctly without crashing.
3. **Identical Points**: Should partition them arbitrarily if needed or group them together.

H. Complexity Analysis
1. Time complexity per iteration? Space complexity?
2. Time complexity of K-Means++ initialization?

I. Progressive Difficulty Levels & Definition of Done
- Level 1 Guided: Implement basic K-Means with random init and test on Dataset 4A.
- Level 2 Standard: Add K-Means++ and empty cluster handling.
- Level 3 Challenge: Prove convergence mathematically and track inertia monotonically.

J. Reflection
1. Why might feature scaling (e.g., standardizing) drastically change the output of K-Means?
2. What role does $K$ play, and how would you automate its selection?
"""
