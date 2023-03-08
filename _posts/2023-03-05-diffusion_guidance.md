---
layout: post
title: Classifier-free diffusion guidance
date: 2023-03-05 11:59:00-0400
description: A short derivation of classifier free diffusion guidance
giscus_comments: false
related_posts: false
tags: generative_models
---

In this post, I'll describe a simple method on how to condition diffusion models called classifier-free guidance. I won't go into the SDE side of things because I don't understand it yet.

## Score function

The score function for an arbitrary step along our markov chain $$\textbf{x}_t$$ is defined as $$s_\theta(\textbf{x}_t, t) = \nabla_{\textbf{x}_t} \log q(\textbf{x}_t)$$. Intuitively, this quantity tells us how to change the noisy $$\textbf{x}_t$$ to make it more likely under the true data distribution. 

In diffusion models, we have a forward diffusion distribution $$q(\textbf{x}_t \mid \textbf{x}_0) = \mathcal{N}(\sqrt{\bar{\alpha_t}} \textbf{x}_0, (1- \bar{\alpha_t}) \textbf{I})$$. The gradient of the $$\log$$ pdf of a gaussian $$\mathcal{N}(\textbf{x}; \mu, \sigma^2)$$ can be computed as:

$$
\begin{align*}
& \nabla_{\textbf{x}} p(\textbf{x}) = \nabla_\textbf{x} \left (  - \frac{1}{2 \sigma^2} (\textbf{x} - \mathbf{\mu})^2 \right) = -\frac{\textbf{x} - \mathbf{\mu}}{\sigma^2}\\ 

&= -\frac{\mathbf{\epsilon}}{\sigma} \hspace{2 cm} (\textbf{x} = \mathbf{\mu} + \sigma \odot \mathbf{\epsilon}, \mathbf{\epsilon} \sim \mathcal{N}(\textbf{0, I}))
\end{align*}
$$

Therefore, we can express the score function of a sample $$\textbf{x}_t$$ as:

$$
\begin{align*}

&\mathbf{s}_\theta(\mathbf{x}_t, t) 
\approx \nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t) = - \frac{\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}{\sqrt{1 - \bar{\alpha}_t}}
\end{align*}
$$

I simply plugged the variance of $$q(\textbf{x}_t \mid \textbf{x}_0)$$ and the diffusion model we train $$\mathbf{\epsilon}_{\theta} (\textbf{x}_t, t)$$ is supposed to match the noise $$\mathbf{\epsilon}$$, so I substitute that in as well. 

Therefore, I now have an equivalence of the score function in terms of our diffusion model's U-Net. Learning the diffusion model as stated in DDPM is the same thing as learning the score function.

## Classifier Guidance

Given our equivalence of the score function and noise prediction network, we can intuitively understand conditioning. 

If we have some auxillary input $$y$$ that we want to condition on, we the need to model the score function $$\nabla_{\textbf{x}_t} \log q(\textbf{x}_t \mid \textbf{y})$$. Hence, using bayes rule we can write this as:

$$
\begin{align*}
& q(\textbf{x}_t \mid \textbf{y}) = \frac{q(\textbf{y} \mid \textbf{x}_t) q(\textbf{x}_t )}{q(\textbf{y})} \\

& \implies \log q(\textbf{x}_t \mid \textbf{y}) = \log q(\textbf{y} \mid \textbf{x}_t)  + \log q(\textbf{x}_t ) - \log q(\textbf{y}) \\

& \implies \nabla_{\textbf{x}_t} \log q(\textbf{x}_t \mid \textbf{y}) = \nabla_{\textbf{x}_t} \log q(\textbf{y} \mid \textbf{x}_t)  + \nabla_{\textbf{x}_t} \log q(\textbf{x}_t )

\end{align*}
$$

It's evident here that $$\nabla_{\textbf{x}_t} \log q(\textbf{y} \mid \textbf{x}_t)$$ can be computed using a differentiable approximator, such as a softmax classifier (in the case of labels). We can add a hyperparameter $$s$$ (called "guidance"), which controls how much influence this classifier has on our final prediction. 

$$
\nabla_{\textbf{x}_t} \log q(\textbf{x}_t \mid \textbf{y}) = \nabla_{\textbf{x}_t} \log q(\textbf{x}_t ) + s \cdot \nabla_{\textbf{x}_t} \log q(\textbf{y} \mid \textbf{x}_t)
$$

The issue is, our $$\textbf{x}_t$$ can be arbitrarily noisy and our classifier will not be able to be accurate at high levels of noise. 


## Classifier-Free Guidance
Hence, we seek to eliminate our dependence on a classifier, so we use bayes rule once again in the other direction: 

$$

\begin{align*}
& q(\textbf{y} \mid \textbf{x}_t) = \frac{q(\textbf{x}_t \mid \textbf{y}) q(\textbf{y})}{q(\textbf{x}_t)} \\

& \implies \log q(\textbf{y} \mid \textbf{x}_t) = \log q(\textbf{x}_t \mid \textbf{y}) + \log q(\textbf{y}) - \log q(\textbf{x}_t)  \\

& \implies \nabla_{\textbf{x}_t} \log q(\textbf{y} \mid \textbf{x}_t) = \nabla_{\textbf{x}_t} \log q(\textbf{x}_t \mid \textbf{y}) - \nabla_{\textbf{x}_t} \log q(\textbf{x}_t)  \\
\end{align*}
$$

Plugging this back into our equation from Classifier Guidance:

$$
\begin{align*}
 \nabla_{\textbf{x}_t} \log q(\textbf{x}_t \mid \textbf{y}) &= \nabla_{\textbf{x}_t} \log q(\textbf{x}_t ) + s \cdot (\nabla_{\textbf{x}_t} \log q(\textbf{x}_t \mid \textbf{y}) - \nabla_{\textbf{x}_t} \log q(\textbf{x}_t)) \\

&= (1-s) \cdot \nabla_{\textbf{x}_t} \log q(\textbf{x}_t ) + s \cdot \nabla_{\textbf{x}_t} \log q(\textbf{x}_t \mid \textbf{y})

\end{align*}
$$


<!-- end with results -->







