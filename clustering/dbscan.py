"""
A. Mission
DBSCAN (Density-Based Spatial Clustering of Applications with Noise) groups together points that are closely packed together (points with many nearby neighbors), marking as outliers points that lie alone in low-density regions. Unlike K-Means, it does not require the number of clusters to be specified apriori and can find arbitrarily shaped, non-convex clusters.

B. Prerequisites
- Euclidean distance and spatial neighborhoods.
- Graph traversal concepts (BFS/DFS) for cluster expansion.

C. Learning Questions
1. How does DBSCAN define a "core point", a "border point", and a "noise point"?
2. What is density-reachability, and how does it form the basis of a cluster?
3. Why does DBSCAN excel at finding non-convex shapes where K-Means fails?
4. How do the parameters $\epsilon$ (epsilon) and `min_samples` affect the number and size of clusters found?

D. Mathematics to Derive
1. Formulate the definition of the $\epsilon$-neighborhood of a point $p$: $N_\epsilon(p) = \{q \in D | \text{dist}(p, q) \le \epsilon\}$.
2. Prove that if $p$ is a core point and $q$ is density-reachable from $p$, then $p$ and $q$ belong to the same cluster.

E. Implementation Contract
- Class name: `DBSCAN`
- Parameters:
  - `eps` (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
  - `min_samples` (int): The number of samples in a neighborhood for a point to be considered a core point (includes the point itself).
- Public Methods:
  - `fit(X)`: Performs DBSCAN clustering from features `X` (shape `(N, D)`).
- Attributes:
  - `labels_`: Array of shape `(N,)` containing cluster labels for each point. Noisy samples are given the label -1.

F. Guided Implementation Stages
1. **Neighborhood Query**: Implement a function to find all points within $\epsilon$ distance of a given point.
2. **Point Classification**: Iterate over all points. For each unvisited point, retrieve its $\epsilon$-neighborhood.
3. **Noise or Core**: If the neighborhood size is less than `min_samples`, label the point as noise (-1) temporarily. Otherwise, start a new cluster.
4. **Cluster Expansion**: Use a queue (BFS) or stack (DFS) to process the neighborhood of the newly found core point. For each neighbor, if it was labeled noise, change its label to the current cluster. If it's unvisited, label it with the current cluster and check its neighborhood. If its neighborhood has at least `min_samples`, add these neighbors to the queue/stack to expand the cluster.
5. **Iteration**: Continue until all points in the dataset have been visited.

G. Edge Cases and Expected Tests
1. **All Noise**: Set a very small `eps` and verify all points are labeled -1.
2. **Single Cluster**: Set a huge `eps` and verify all points belong to cluster 0.
3. **Non-Convex Shapes**: Test on the "moons" or "circles" dataset where DBSCAN should perfectly separate the distinct continuous structures.

H. Complexity Analysis
1. What is the naive time complexity of computing all neighborhoods without spatial indexing?
2. How would the time complexity change if you used a KD-Tree or Ball-Tree for neighborhood queries?
3. What is the space complexity of your implementation?

I. Definition of Done
- Correctly identifies noise points and core points.
- Successfully clusters non-convex datasets.
- Implemented purely in Python/NumPy (no spatial libraries required, naive $O(N^2)$ distance computation is acceptable).

J. Reflection
1. What are the limitations of DBSCAN when dealing with datasets that have clusters of varying densities?
2. In what situations might border points flip-flop between two different clusters depending on the order of data processing?
"""
