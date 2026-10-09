# Chapter 3: Trees and Ensembles

> **Hard Prerequisites:** Supervised learning (regression and classification), loss functions, and the concept of overfitting (Chapters 1–2).

---

## Learning Objectives

After completing this chapter, you should be able to:

- Explain how decision trees partition feature space using recursive splits
- Distinguish splitting criteria: Gini impurity for classification, variance reduction for regression
- Explain why single trees overfit and how pruning and depth limits help
- Explain ensemble methods: how bagging reduces variance and boosting reduces bias

## Questions to Answer in Your Own Words

1. How does a decision tree choose which feature to split on and at what threshold?
2. What is information gain? How do Gini impurity and entropy differ as splitting criteria, and when might you prefer one over the other?
3. Why do single decision trees tend to overfit? What mechanisms control this?
4. What is bagging and how does it reduce variance without increasing bias?
5. What is the key difference between bagging (Random Forest) and boosting (Gradient Boosting)?
6. How does Gradient Boosting fit trees to residuals (negative gradients)?

## Algorithms to Implement from Scratch

- [ ] Decision Tree (CART — use Gini for classification, variance reduction for regression)
- [ ] Random Forest (bagging + feature subsampling)
- [ ] Gradient Boosting (from scratch)

## Mathematical Derivations to Complete

- Derive Gini impurity and Shannon entropy; show they are both valid measures of node impurity
- Derive the information gain formula for a candidate split
- Derive the Gradient Boosting update rule: show that each new tree fits the negative gradient of the loss

## Edge Cases & Tests to Consider

- Decision Tree with no depth limit on noisy data — observe overfitting
- How does feature subsampling in Random Forest reduce correlation between trees?
- Gradient Boosting with a high learning rate — what goes wrong?
- Single-feature dataset: does the tree still work?

## Completion Criteria

You are done with this chapter when you can:

- [ ] Build a decision tree that correctly classifies a simple dataset
- [ ] Explain the difference between Gini and entropy and when each is appropriate
- [ ] Implement Random Forest and show it outperforms a single tree on noisy data
- [ ] Explain Gradient Boosting's additive training in your own words
- [ ] Visualise or print a learned tree structure

## Notes

_Space for your own observations as you work through this chapter._
