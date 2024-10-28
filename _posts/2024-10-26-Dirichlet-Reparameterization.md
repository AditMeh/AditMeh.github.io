---
layout: post
title: Implicit gradients and Dirichlet uncertainty
date: 2024-10-26 11:59:00-0400
description: 
giscus_comments: false
related_posts: false
tags: deep_learning_tricks
---

Goal is to compute 

$$\nabla_\theta \mathcal{L} = \nabla_\theta \mathbb{E}_{q_\theta(z)}[f_\theta(z)]$$


The transform of $$z \sim q_\theta(z)$$ by the cdf $$F_\theta(z)$$ is uniformly distributed.

Let $$u \sim U(0,1)$$ be a randomly sampled uniform value. Therefore, for some $$z \in \mathbb{R}$$, we have that

$$u = F_\theta(x) = \int_{-\infty}^z q_\theta(z') dz'$$

Taking the derivative of both sides, we get that:

$$0 = \dfrac{\partial z}{\partial \theta} q_\theta(z) + \int_{-\infty}^z \dfrac{\partial}{\partial \theta } q_\theta(z')dz'$$

Hence, 

$$\dfrac{\partial z}{\partial \theta} = \frac{-\dfrac{\partial}{\partial \theta} F_\theta(z)}{q_\theta(z)}$$


??? (I don't understand how to fill in the steps here)

Profit:
$$\nabla_\theta \mathcal{L}=\mathbb{E}_{q_\theta(z)}\left[\frac{d f_\theta(z)}{d z} \frac{d z}{d \theta}+\frac{\partial f_\theta(z)}{\partial \theta}\right]$$
