# Chapter 2: Classification

> **Hard Prerequisites:** Gradient descent, loss functions, and parameter updates (Chapter 1).

---

## Learning Objectives

After completing this chapter, you should be able to:

- Explain decision boundaries and what shapes different classifiers produce
- Explain why cross-entropy replaces MSE for classification tasks
- Implement probabilistic, distance-based, and margin-based classifiers from scratch
- Reason about multi-class strategies and the curse of dimensionality

## Questions to Answer in Your Own Words

1. How does logistic regression differ from linear regression, despite sharing the name?
2. What is a decision boundary and what shapes can different classifiers produce?
3. Why is cross-entropy used instead of MSE for classification?
4. What assumptions does Naive Bayes make and when do they break down?
5. What is a support vector and why does SVM maximise the margin?
6. How does KNN make predictions without learning parameters? What is the curse of dimensionality?

## Algorithms to Implement from Scratch

- [ ] Logistic Regression (binary classification)
- [ ] K-Nearest Neighbors (KNN)
- [ ] Gaussian Naive Bayes
- [ ] Support Vector Machine (linear kernel only)

## Mathematical Derivations to Complete

- Derive the sigmoid function and show that its derivative is σ(x)(1 − σ(x))
- Derive binary cross-entropy loss and its gradient with respect to weights
- Derive Bayes' theorem and the Gaussian class-conditional likelihood for Naive Bayes
- Formulate SVM margin maximisation as a constrained optimisation problem

## Edge Cases & Tests to Consider

- Class imbalance: how does it affect each classifier differently?
- KNN: what happens when K = 1 vs K = N?
- Multi-class extension: one-vs-rest vs one-vs-one — when to use which?
- Logistic regression on linearly inseparable data — what happens?

## Completion Criteria

You are done with this chapter when you can:

- [ ] Train logistic regression on a binary dataset and plot the decision boundary
- [ ] Explain why cross-entropy is the right loss for classification (not MSE)
- [ ] Run KNN and explain how K affects bias vs variance
- [ ] Derive the sigmoid gradient without notes
- [ ] Explain the Naive Bayes independence assumption and give an example where it fails

## Notes

_Space for your own observations as you work through this chapter._
