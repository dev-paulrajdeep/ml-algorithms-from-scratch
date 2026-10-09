# Chapter 7: Advanced Probabilistic Models

## Chapter Objectives
This chapter transitions from basic statistical learning algorithms to advanced probabilistic generative models, focusing heavily on latent variables and sequence models. You will explore:
- The Expectation-Maximization (EM) algorithm as a robust framework for dealing with unobserved (latent) variables.
- How to model complex data distributions using a mixture of simpler probability distributions.
- Sequence models that capture temporal or sequential dependencies using the Markov assumption.
- The trinity of Hidden Markov Model (HMM) algorithms: Forward-Backward (evaluation/learning) and Viterbi (decoding).

## Prerequisites
**Hard Prerequisites (Concepts, not prior chapters):**
- **Probability Theory**: Bayes' theorem, conditional probability, joint and marginal distributions.
- **Statistics**: Expectation, variance, covariance matrices, maximum likelihood estimation (MLE).
- **Core Math**: Jensen's inequality, latent-variable modeling concepts.

**Recommended Preparation:**
- *Chapter 4 (Clustering)*: Understanding K-Means is extremely helpful, as it functions as a special case of Gaussian Mixture Models (hard assignment vs soft assignment).
- *Chapter 6 (Optimization)*: Maturity with iterative optimization techniques will help you debug EM convergence.
- *Chapter 2 (Naive Bayes)*: Familiarity with Gaussian likelihoods will make the GMM E-step much more intuitive.

## Study Sequence
1. [Gaussian Mixture Model (`gaussian_mixture_model.py`)](./gaussian_mixture_model.py): Start by grasping EM in the context of clustering.
2. [HMM Viterbi Decoding (`hmm_viterbi.py`)](./hmm_viterbi.py): Learn how to decode the most likely hidden state sequence using dynamic programming.
3. [HMM Forward-Backward (`hmm_forward_backward.py`)](./hmm_forward_backward.py): Synthesize EM (Baum-Welch) and HMMs to learn sequence model parameters from scratch.

## Cross-Chapter Conceptual Questions
1. How does a generative model like a GMM or HMM differ philosophically from discriminative models like Logistic Regression or SVMs?
2. Compare the EM algorithm used in GMMs and Baum-Welch. Can you identify the exact parallel components (E-step and M-step) in both algorithms?
3. How does the concept of "soft assignments" (responsibilities) bridge the gap between K-Means (Chapter 4) and GMMs?

## Mathematical Learning Goals
By the end of this chapter, you should be comfortable deriving and implementing:
- The Evidence Lower Bound (ELBO) and using Jensen's Inequality to prove EM convergence.
- The recursive dynamic programming formulas for Viterbi decoding.
- The Forward and Backward recurrences and their mathematical relationship to the sequence log-likelihood.
- The Baum-Welch expected count updates for HMM parameters.

## Completion Checklist
- [ ] Gaussian Mixture Model (E-step, M-step, ELBO convergence)
- [ ] Log-sum-exp trick implemented correctly for numerical stability
- [ ] HMM Viterbi Decoder (Log-space dynamic programming)
- [ ] HMM Forward-Backward (Scaling or Log-space implemented)
- [ ] Baum-Welch EM algorithm fits multiple observation sequences
- [ ] All tests pass without underflow errors on long sequences

## Personal Notes
> Use this section to log your insights, tricky bugs, and conceptual breakthroughs.
