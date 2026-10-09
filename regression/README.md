# Chapter 1: Regression

> **Hard Prerequisites:** None. This is the entry point.

---

## Learning Objectives

After completing this chapter, you should be able to:

- Explain the supervised learning setup: features, targets, hypothesis function, parameters
- Implement gradient descent from scratch and reason about its convergence
- Define and compute loss functions (MSE) and explain their geometric meaning
- Explain regularisation and the bias–variance tradeoff
- Distinguish between different regression variants and when to use each

## Questions to Answer in Your Own Words

1. What is a hypothesis function and why does its form matter?
2. What does minimising MSE actually mean geometrically?
3. Why does gradient descent need a learning rate? What happens if it's too large or too small?
4. What is regularisation trying to prevent? How do L1 and L2 penalties differ in effect?
5. What is the difference between interpolation and extrapolation?

## Algorithms to Implement from Scratch

- [x] Linear Regression (gradient descent)
- [x] Log-Transformed Exponential Regression
- [ ] Polynomial Regression
- [ ] Ridge Regression (L2)
- [ ] Lasso Regression (L1)

## Mathematical Derivations to Complete

- Derive the MSE gradient with respect to w and b
- Derive the normal equation (closed-form solution) and explain when it is preferable to gradient descent
- Show why L1 regularisation produces sparse weights and L2 does not (hint: consider the geometry of the constraint region)

## Edge Cases & Tests to Consider

- What happens when features are on very different scales? (feature scaling)
- What happens when two features are perfectly correlated? (multicollinearity)
- Test convergence: does loss decrease monotonically with a well-chosen learning rate?
- What happens with a learning rate that is too high?

## Completion Criteria

You are done with this chapter when you can:

- [ ] Explain gradient descent to someone who knows calculus but not ML
- [ ] Run your linear regression on new data and interpret the learned w and b
- [ ] Derive the MSE gradient from scratch without looking at notes
- [ ] Explain when to use Ridge vs Lasso and why
- [ ] Your implementations converge to reasonable parameters on test data

## Notes

_Space for your own observations as you work through this chapter._
