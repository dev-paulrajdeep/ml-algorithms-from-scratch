"""
A. Mission
Agglomerative Clustering is a bottom-up hierarchical clustering method. It treats each data point as a singleton cluster at the outset and then successively merges (or agglomerates) pairs of clusters until all clusters have been merged into a single big cluster containing all data points. This approach produces a dendrogram (a tree-like structure) that can be cut at any level to yield different numbers of clusters without needing to specify $K$ in advance.

B. Prerequisites
- Pairwise distance matrices.
- Understanding of different linkage criteria (single, complete, average) to measure distance between sets of points.

C. Learning Questions
1. How do single linkage, complete linkage, and average linkage differ?
2. What is the "chaining effect" in single linkage, and why can it be problematic?
3. How does the choice of linkage criterion affect the shapes of the clusters found?
4. Why might you prefer a hierarchical method over K-Means for exploratory data analysis?

D. Mathematics to Derive
1. Write out the formula for the distance between two clusters $A$ and $B$ under Single Linkage: $\min_{a \in A, b \in B} d(a, b)$.
2. Write out the formulas for Complete Linkage and Average Linkage.
3. Observe how the distance matrix updates when clusters $A$ and $B$ are merged into a new cluster $A \cup B$.

E. Implementation Contract
- Class name: `AgglomerativeClustering`
- Parameters:
  - `n_clusters` (int): The number of clusters to find (level to cut the dendrogram).
  - `linkage` (str): The linkage criterion to use ('single', 'complete', or 'average').
- Public Methods:
  - `fit(X)`: Fits the hierarchical clustering on data `X` (shape `(N, D)`).
- Attributes:
  - `labels_`: Array of shape `(N,)` containing cluster labels for each point.
  - `dendrogram_`: A list or array recording the merge history (which two clusters merged at what distance, at each step).

F. Guided Implementation Stages
1. **Initial Distance Matrix**: Compute the pairwise Euclidean distance matrix for all $N$ data points. Treat each point as a cluster index $0 \dots N-1$.
2. **Merge Loop**: Create a loop that runs $N-1$ times.
3. **Find Closest Clusters**: At each step, find the two clusters with the minimum distance according to the current distance matrix.
4. **Record Merge**: Log the merge in `dendrogram_`. Create a new cluster index (starting from $N$).
5. **Update Distances**: Compute the distance between the newly formed cluster and all other remaining clusters using the specified `linkage` criterion. Remove the two merged clusters from consideration.
6. **Label Extraction**: After the tree is fully built (or built up to `n_clusters`), trace back from the root to extract the cluster labels for `n_clusters`.

G. Edge Cases and Expected Tests
1. **Ties in Distance**: How does your implementation handle multiple pairs having the exact same minimum distance?
2. **Two Points**: Test on an $N=2$ dataset. It should perform exactly one merge.
3. **Linkage Differences**: Test on an elongated, slightly curved dataset. Verify that single linkage chains them together, whereas complete linkage tends to create more compact, spherical clusters.

H. Complexity Analysis
1. What is the space complexity to store the distance matrix?
2. What is the time complexity of the naive agglomerative clustering algorithm?
3. How can the update step be optimized to avoid recomputing distances from scratch?

I. Definition of Done
- The algorithm successfully builds a full merge history.
- Can extract labels for a specific `n_clusters`.
- Implements single, complete, and average linkage correctly.
- Code implemented purely in Python/NumPy.

J. Reflection
1. Why is Agglomerative Clustering generally not suitable for very large datasets ($N > 100,000$)?
2. If you needed to update the clustering with a single new data point, would you have to rebuild the entire dendrogram?
"""
