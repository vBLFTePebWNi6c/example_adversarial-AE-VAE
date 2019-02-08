# [Adversarial Autoencoders](https://arxiv.org/abs/1511.05644.pdf) & [Adversarial Variational Bayes](https://arxiv.org/abs/1701.04722.pdf) code playground on Keras

In this repo I managed several experimnets to empirically proove paper's statements and deeply understand the concept.

## [Adversarial Autoencoders](https://arxiv.org/abs/1511.05644.pdf)

Authors propose peculiar way to shape distribution of intermediate layer in autoencoders with Adversarial learning technique. Given this, if there is any distribution P from which we can sample, then we can establish two player game in which encoder should produce new distribution P' similar to P as much as possible and discriminative network should distinguish from which distribution come arbitrary sample.

3 types of generative models has been proposed:
* Deterministic AAE -- just AE + Adversarial component.
* Gaussian Prior AAE -- VAE-like architecture, but KL divergence replaced with Adversarial component.
* Universal approximator posterior -- compromiss between AE and VAE: there is no explicit parameters of distributionб but noise injection still persists.

## [Adversarial Variational Bayes](https://arxiv.org/abs/1701.04722.pdf)

Authors impose strong theoretical background about connection between Adversarial component and VAE, prooves that Adversarial Variational Bayes "yields an exact maximum-likelihood assignment for the parameters of the generative model, as well as the exact posterior distribution over the latent variables given an observation".

But during my experiments I found that this framework barely capable handle it's pormises in practice. I tried to find more criticism of this paper in internet and (as I think) this user on Reddit completely understand the situation: https://www.reddit.com/r/MachineLearning/comments/5p9ism/d_thoughts_on_adversarial_variational_bayes/dcplm55

http://paulrubenstein.co.uk/variational-autoencoders-are-not-autoencoders/