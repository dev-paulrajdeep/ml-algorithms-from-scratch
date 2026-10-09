"""
A. Mission
K-Means clustering aims to partition $N$ observations into $K$ sets (clusters) so as to minimize the within-cluster sum of squares (WCSS), also known as inertia. This is a foundational centroid-based clustering algorithm useful for discovering spherical, linearly separable groups in unlabeled data. It iteratively refines cluster assignments and centroid positions.

B. Prerequisites
- Vector distances (e.g., Euclidean distance).
- Concept of an objective function and iterative optimization.
- (Helpful but not strictly required) Basic probability for K-Means++ initialization.

C. Learning Questions
1. Why does K-Means converge, and why does the objective function (WCSS) never increase during the assignment or update steps?
2. How does the initial choice of centroids affect the final clustering? Why might random initialization be problematic?
3. What is the intuition behind K-Means++ initialization, and how does it improve upon random initialization?
4. What happens when K-Means is applied to clusters of varying densities or non-convex shapes (like concentric circles)?

D. Mathematics to Derive
1. Let $C_k$ be the set of points in the $k$-th cluster, and $\mu_k$ be its centroid. Show that for a fixed assignment $C_k$, setting $\mu_k = \frac{1}{|C_k|} \sum_{x \in C_k} x$ minimizes the sum of squared distances $\sum_{x \in C_k} ||x - \mu_k||^2$.
2. Formulate the total objective function (Inertia/WCSS).

E. Implementation Contract
- Class name: `KMeans`
- Parameters: 
  - `k` (int): Number of clusters.
  - `init` (str): Initialization method, either 'random' or 'k-means++'.
  - `max_iter` (int): Maximum number of iterations.
  - `tol` (float): Tolerance to declare convergence (when centroids shift by less than this amount).
- Public Methods:
  - `fit(X)`: Computes the cluster centroids on data `X` (shape `(N, D)`).
  - `predict(X)`: Returns the cluster labels for `X` based on nearest centroids. Shape `(N,)`.
- Attributes:
  - `centroids_`: Array of shape `(k, D)` containing final cluster centers.
  - `labels_`: Array of shape `(N,)` containing cluster assignments for the training data.
  - `inertia_`: Final WCSS (float).

F. Guided Implementation Stages
1. **Distance Calculation**: Write a helper to compute pairwise squared Euclidean distances between a set of points and a set of centroids.
2. **Initialization**: Implement 'random' (select `k` random points from `X` as initial centroids).
3. **K-Means++ Initialization**: Implement the probabilistic initialization: pick the first centroid randomly, then iteratively choose the next centroid from remaining points with probability proportional to their squared distance to the nearest existing centroid.
4. **Assignment Step**: Given centroids, assign each point to the nearest centroid.
5. **Update Step**: Given assignments, compute the new centroids as the mean of the assigned points.
6. **Main Loop**: Combine initialization, assignment, and update into the `fit` method. Stop when iterations hit `max_iter` or centroid movement is below `tol`. Calculate and store `inertia_` at the end.

G. Edge Cases and Expected Tests
1. **Empty Clusters**: What happens if a cluster loses all its points during the assignment step? Ensure your code gracefully handles this (e.g., by re-initializing the empty cluster's centroid to the furthest data point).
2. **Deterministic Behavior**: Verify that a very low `tol` or `max_iter=0` behaves correctly.
3. **K-Means++ advantage**: Test on a dataset where random initialization often falls into poor local optima, and verify K-Means++ performs consistently better.

H. Complexity Analysis
1. What is the time complexity of a single iteration of Lloyd's algorithm?
2. What is the space complexity to store the assignments and distances?
3. How does the time complexity of K-Means++ initialization scale with $N$, $K$, and $D$?

I. Definition of Done
- Algorithm correctly clusters cleanly separated blobs.
- Both 'random' and 'k-means++' initialization strategies are implemented and functional.
- The `inertia_` monotonically decreases across iterations in tests.
- Code avoids external clustering libraries (NumPy allowed).

J. Reflection
1. If you wanted to automatically determine the best $k$, how might you use the `inertia_` attribute across multiple runs?
2. Since K-Means relies heavily on Euclidean distance, how important is feature scaling before clustering?
"""
