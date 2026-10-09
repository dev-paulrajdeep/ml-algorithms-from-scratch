# Chapter 4: Clustering

> **Hard Prerequisites:** Distance metrics (Euclidean distance) and iterative optimisation (Chapter 1).
>
> **Recommended context:** Chapter 2 (supervised vs unsupervised distinction), but not required.

---

## Learning Objectives

After completing this chapter, you should be able to:

- Explain unsupervised learning: discovering structure in data without labels
- Implement and compare centroid-based, density-based, and hierarchical clustering
- Evaluate clusters when there is no ground truth
- Reason about the trade-offs: when K-Means fails, when DBSCAN excels, and vice versa

## Questions to Answer in Your Own Words

1. How do you evaluate a clustering result when there are no ground-truth labels?
2. What is the elbow method and why is it a heuristic rather than an exact answer?
3. How does DBSCAN define a cluster? What are core points, border points, and noise points?
4. What is a dendrogram and how do you read the number of clusters from one?
5. When does K-Means fail to find the "true" clusters? Give a concrete geometric example.
6. Why is K-Means sensitive to initialisation? What does K-Means++ do about it?

## Algorithms to Implement from Scratch

- [ ] K-Means
- [ ] DBSCAN
- [ ] Hierarchical / Agglomerative Clustering

## Mathematical Derivations to Complete

- Derive the K-Means objective function (minimise within-cluster sum of squares)
- Show that the K-Means alternating assignment-update procedure never increases the objective (convergence proof)
- Define the DBSCAN ε-neighbourhood formally and describe the cluster expansion algorithm

## Edge Cases & Tests to Consider

- K-Means on non-convex clusters (e.g. concentric rings) — what happens and why?
- DBSCAN sensitivity to ε and min_samples — explore how results change
- K-Means convergence: does it always reach the global minimum? (No — explain why.)
- What happens when K-Means is initialised with all centroids at the same point?

## Completion Criteria

You are done with this chapter when you can:

- [ ] Run K-Means on 2D data and visualise cluster assignments
- [ ] Explain why K-Means fails on non-convex shapes
- [ ] Run DBSCAN on the same data and show it succeeds where K-Means fails
- [ ] Produce a dendrogram and choose K from it
- [ ] Explain the K-Means convergence proof without notes

## Notes

_Space for your own observations as you work through this chapter._
