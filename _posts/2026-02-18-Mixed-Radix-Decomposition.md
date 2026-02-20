---
layout: post
title: Mixed Radix Decomposition
date: 2026-02-18 11:59:00-0400
description: Converting a many-class problem into multiple smaller subproblems
giscus_comments: false
related_posts: false
tags: computer_vision
related_publications: false
---

This was a trick I studied when looking at TabICL's codebase, that actually had a lot more depth than I imagined.

## Problem Statement

You can only train a classifier of max $$C$$ classes, but you want to get a classifier for $$N > C$$ classes. You can train multiple classifiers, but each is subject to the restriction of having $$C$$ classes, how do you do this?

We'd want to somehow break the class index $$c \in \{0, \dots, N-1\}$$ into a tuple of elements, each of which vary from $$0 \dots C$$. We then can train a classifier on each element of the tuple.

## Theory

To represent our own number system, we break down an integer $$x$$ as an addition of powers of 10, with "digits" $$x_0 \dots x_{M-1}$$:

$$x = 10^{M-1} x_{M-1} + 10^{M-2} x_{M-2} + \dots + 10^0 x_0, \quad 0 \leq x_j \leq 9$$

We can also do the same for binary:

$$x = 2^{M-1} x_{M-1} + 2^{M-2} x_{M-2} + \dots + 2^0 x_{0}, \quad 0 \leq x_j \leq 1$$

Mixed radix numeral systems are a generalization of this, where we specify a base $$[k_0, \dots, k_{M-1}]$$, $$k_i \geq 2$$ such that for $$0 \leq x_i < k_i$$:

$$x = (k_0 k_1 k_2 \dots k_{M-2})x_{M-1} + (k_0k_1k_2\dots k_{M-3})x_{M-2} + \dots + k_{0} x_{1} + x_{0}$$

We could just hardcode all $$k_i$$'s to 10 or 2, getting us the decimal and binary number systems directly, so this is a generalization of the previously mentioned number systems. Much like in decimal, we have a set of "digits" $$x_0, \dots, x_{M-1}$$ in mixed-radix numeral systems.

In this number system, we can represent any integer from $$0$$ to $$\prod_{i=0}^{M-1} k_i - 1$$. Before we use this for our usecase, I'll need to show a few basic properties first:

1. Proof that the maximum possible value is $$\prod_{i=0}^{M-1} k_i - 1$$
2. Proof that every integer $$0 \leq x \leq \prod_{i=0}^{M-1} k_i - 1$$ has a unique decomposition (existence and uniqueness)
3. Show how to go from an integer to digits in a mixed-radix numeral system

### Notation

Let $$w_i$$ be defined as

$$w_i = \prod_{j=0}^{i-1} k_j, \quad w_0 = 1$$

### Maximum Possible Value

We have that an arbitrary integer $$x$$ expressed in mixed radix notation with digits $$[x_1, \dots, x_{M-1}]$$ is,

$$x = \sum_{i=0}^{M-1} x_i w_i$$

The maximum integer value that $$x$$ can take is when all $$x_i = k_i - 1$$. So we get that:

$$x_{max} = \sum_{i=0}^{M-1} (k_i - 1) w_i$$

We observe that $$w_i k_i = w_{i+1}$$, therefore via telescoping sums,

$$x_{max} = \sum_{i=1}^{M-1} w_{i+1} - w_i = w_{M} - w_0 = \prod_{j=0}^{M-1} k_j - 1$$

Hence, we've shown the maximum possible integer value attainable by a specific mixed-radix base.

### Existence

Let $$0 \leq x < \prod_{j=0}^{M-1} k_j = k_0 \dots k_{M-1}$$, we want to show that there are digits $$0 \leq x_i < k_i$$ such that

$$x = (k_0 k_1 k_2 \dots k_{M-2})x_{M-1} + (k_0k_1k_2\dots k_{M-3})x_{M-2} + \dots + k_{0} x_{1} + x_{0}$$

There exist unique integers $$x_0$$ and $$y_1$$ such that

$$x = x_0 + k_0 y_1$$

Since $$x = x_0 + k_0 y_1 < k_0 \dots k_{M-1}$$ and $$x_0 < k_0$$, that means that $$y_1 < k_1k_2\dots k_{M-1}$$.

We can repeat this process, getting us unique integers $$x_1, y_2$$ such that $$y_1 = x_1 + k_1 y_2$$, and $$y_2 < k_2 \dots k_{M-1}$$ and $$x_1 < k_1$$.

Ultimately, we hit $$y_{M-1} < k_{M-1}$$. If we try to do the same trick, getting integers $$x_{M-1}, y_{M}$$ such that $$y_{M-1} = x_{M-1} + k_{M-1} y_{M}$$, we see that $$y_{M-1} < k_{M-1}$$, so $$y_{M} = 0$$ and $$x_{M-1} = y_{M-1}$$. Thus, we can't divide further.

Back substituting this all, we get:

$$x = x_0 + k_0 (x_1 + k_1(x_2 + k_2(\dots))) = x_0 + k_0 x_1 + (k_0k_1) x_2 + \dots + (k_0\dots k_{M-2}) x_{M-1}$$

Hence, we've proven that we have a set of $$x_i < k_i$$ digits which produce any arbitrary integer $$0 \leq x < \prod_{j=0}^{M-1} k_j = k_0 \dots k_{M-1}$$ in this number system.

### Uniqueness

We want to show if there are two integers $$x, y < \prod_{j=1}^{M-1} k_j$$ such that $$x = y$$, then their digit representations $$[a_0,\dots, a_{M-1}]$$ and $$[b_0, \dots, b_{M-1}]$$ are the same (uniqueness). In essence for $$0 \leq a_i, b_i < k_i$$:

$$\sum_{i=0}^{M-1} a_i w_i = \sum_{i=1}^{M-1} b_i w_i \implies \forall i \in [0, \dots, M-1], \quad a_i=b_i$$

Assume that

$$\sum_{i=0}^{M-1} a_i w_i = \sum_{i=1}^{M-1} b_i w_i$$

So we have that

$$\sum_{i=0}^{M-1} (a_i-b_i) w_i = 0$$

Let $$c_i = a_i - b_i$$, so we have that $$-k_i < c_i < k_i$$ and $$\sum_{i=0}^{M-1} c_i w_i = 0$$.

Case 1, if all $$c_i = 0$$, we are done. Assume that there exist an index $$j$$ such that $$j = \min \{i : c_i \neq 0 \}$$, so $$c_0 = \dots = c_{j-1} = 0$$ and $$c_j \neq 0$$. We then can rewrite the sum as:

$$c_jw_j + \sum_{i = j+1}^{M-1} c_iw_i = 0$$

$$c_jw_j + \sum_{i = j+1}^{M-1} c_i w_j \prod_{n=j}^{i-1} k_n = 0 \implies c_j + \sum_{i = j+1}^{M-1} c_i \prod_{n=j}^{i-1} k_n = 0$$

When we modulo $$\sum_{i = j+1}^{M-1} c_i \prod_{n=j}^{i-1} k_n$$ with $$k_j$$, we get 0, meaning it's divisible by $$k_j$$. So reducing the equation under modulo $$k_j$$, we get that $$c_j \equiv 0 \mod k_j$$, hence $$c_j$$ is divisible by $$k_j$$.

However, $$-k_j < c_j < k_j$$, so $$c_j$$ must be zero. Hence we have a contradiction. Therefore all $$c_j$$'s must be zero.

### Converting an Integer to a Radix Representation

To get the digit $$d_i(x)$$ for integer $$x$$ at index $$i$$, we can compute the following:

$$d_i(x) = \left\lfloor \frac{x}{w_i} \right\rfloor \mod k_i$$

To reason about how this works, we can solve for it algebraically. Recall that our digit representation can be converted to an integer like this:

$$x = w_{M-1} x_{M-1} + w_{M-2}x_{M-2} + \dots + w_1 x_{1} + w_0 x_{0}$$

Dividing $$x$$ by $$w_i$$ gets us,

$$\frac{x}{w_i} = \frac{w_{M-1}}{w_i} x_{M-1} + \dots + x_i + \dots + \frac{w_0}{w_i}x_{0}$$

First, I'll show that the sum of all terms $$j < i$$ is less than 1. Specifically,

$$\sum_{j=0}^{i-1} \frac{w_j}{w_i} x_j < 1$$

We can expand everything out first,

$$\sum_{j=0}^{i-1} \frac{w_j}{w_i} x_j = \sum_{j=0}^{i-1} \frac{\prod_{n=0}^{j-1} k_n}{\prod_{m=0}^{i-1} k_m} x_j = \sum_{j=0}^{i-1} \frac{x_j}{\prod_{m=j}^{i-1} k_m}$$

Since we know that $$x_j \leq k_j - 1$$, we have a telescoping series,

$$\sum_{j=0}^{i-1} \frac{x_j}{\prod_{m=j}^{i-1} k_m} \leq \sum_{j=0}^{i-1} \frac{k_j - 1}{\prod_{m=j}^{i-1} k_m}$$

$$= \left(1 - \frac{1}{k_{i-1}}\right) + \left(\frac{1}{k_{i-1}} - \frac{1}{k_{i-1}k_{i-2}}\right) + \dots + \left(\frac{1}{\prod_{m=1}^{i-1} k_m} - \frac{1}{\prod_{m=0}^{i-1} k_m}\right)$$

$$= 1 - \frac{1}{\prod_{m=0}^{i-1} k_m} < 1$$

Now for all terms $$j > i$$,

$$\frac{x}{w_i} = x_{M-1} \prod_{n=i}^{M-2}k_n + x_{M-2} \prod_{n=i}^{M-3}k_n + \dots + x_i + \underbrace{\dots + \frac{w_0}{w_i} x_0}_{\text{sums up to} < 1}$$

Now it's easy to see that every single term with $$j > i$$ is an integer, that also has $$k_i$$ as a factor.

$$\frac{x}{w_i} = \underbrace{x_{M-1} \prod_{n=i}^{M-2}k_n + x_{M-2} \prod_{n=i}^{M-3}k_n + \dots}_{\text{integers divisible by } k_i} + x_i + \underbrace{\dots + \frac{w_0}{w_i} x_0}_{\text{sums up to} < 1}$$

Now we have enough information to understand what the digit-extraction operation is doing. In $$\left\lfloor \frac{x}{w_i} \right\rfloor \mod k_i$$, the floor gets rid of the decimal component from summing the terms with indexes $$j < i$$. Then, the $$\mod k_i$$ operation gets rid of the terms with higher indexes $$j > i$$,

$$\left\lfloor \frac{x}{w_i} \right\rfloor \mod k_i = \left(\left(\underbrace{x_{M-1} \prod_{n=i}^{M-2}k_n + \dots}_{\text{integers divisible by } k_i}\right) \mod k_i + x_i \mod k_i\right) \mod k_i$$

$$= x_i \mod k_i = x_i$$

The last step follows since our digits satisfy $$0 \leq x_i < k_i$$.

## Using It in Practice

### Picking Bases and Number of Digits

After all that math, let's now discuss how to actually use this. Assume we have a classifier that supports $$C$$ classes, and we have $$N$$ classes. We can construct a set of digits $$[k_0, \dots]$$, where each $$k_i < C$$, allowing us to fit a classifier "per digit".

There are still two things we need to figure out: what our bases $$k_0, \dots$$ will be and how many of them we should have.

Theoretically, the maximum allowed value for each of the bases $$k$$ is $$C-1$$, since that's how many classes our classifier can take. So assuming this, we can compute the minimum number of digits required to represent an integer $$0 \leq x < N$$ as

$$M = \left\lceil \frac{\log{N}}{\log{C}} \right\rceil \implies M \geq \frac{\log N}{\log C}$$

We then get that,

$$M\log C \geq \log N \implies C^M \geq N$$

Using our equation of maximum possible value expressed by a base in mixed-radix, we get that

$$\prod_{j=0}^{M-1} C - 1 = C^{M} - 1 \geq N - 1$$

Thus, our system can represent any of the $$N$$ classes $$0 \leq x \leq N-1$$ with $$M$$ digits.

We can do a little bit better though. By taking the maximum to be $$\lceil N^{1/M}\rceil$$, we have that our maximum attainable value is still within the correct range,

$$\prod_{j=0}^{M-1} \lceil N^{1/M} \rceil - 1 \geq (N^{1/M})^{M} - 1 \geq N - 1$$

In fact, this is the minimum possible maximum value per digit, if we want even bases. We can easily prove this by assuming there is some $$A < \lceil N^{1/M}\rceil$$ that works:

$$A < \lceil N^{1/M}\rceil \leq N^{1/M} \implies A^M < N \implies A^M - 1 < N - 1$$

Hence, we have a contradiction and we've established a tight bound.

### Usage in TabICL

The use of this is in encoding labels that exceed the model's $$C$$ - the max label it can output.

```python
# Compute balanced bases for mixed-radix decomposition
bases = self._compute_mixed_radix_bases(num_classes)
num_digits = len(bases)
src_accum = torch.zeros_like(src)
src_with_y = src.clone()

# Run the set transformer for each digit, accumulate, and average
for digit_idx in range(num_digits):
    y_digit = self._extract_mixed_radix_digit(y_train, digit_idx, bases)
    y_emb = self.y_encoder(y_digit.float())
    src_with_y[..., :train_size, :] = src[..., :train_size, :] + y_emb
    src_accum = src_accum + self.tf_col(src_with_y, train_size=None if embed_with_test else train_size)

src = src_accum / num_digits
```


So first it constructs the bases, and then embeds the mixed radix digit for each index using a SetTransformer. These embeddings are accumulated in `src_accum`. Finally, they normalize by the number of digits. This approach scales to many-class problems because it just costs more iterations of the for loop. However, this probably loses signal if the number of classes goes into the tens of thousands, which is honestly not that realistic of a case.