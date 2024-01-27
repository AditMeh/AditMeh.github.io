---
layout: post
title: An introduction to Slot Attention
date: 2024-01-22
description: Going over the basics of Slot Attention, and covering some recent literature in the area
giscus_comments: false
related_posts: false
tags: computer_vision
---

In this blog post, I'll cover the basic slot attention mechanism and go over some intuition as to why it works.

The problem we tackle is learning representations for particular regions of images, like a representation for a box, sphere or background of the scene.

## Intuition, k-means clustering

### Hard K-means clustering:

K-means clustering in the most simple form has the following steps:

**Setup**: Randomly initialize clusters by assigning each vector $$\mathbf{x}_n$$ to one of $$N$$ clusters

**Repeat**:

1. Compute cluster means $$\boldsymbol{\mu}_n$$ by averaging all points assigned to cluster $$n$$
2. Reassign each each point to the cluster corresponding to it closest mean $$\boldsymbol{\mu}_k$$.
3. If some no new assignments were made, we've converged and can stop.

While the example I gave above was for 2D Points, this algorithm has been used for images as well, for segmenting regions of the image by the pixel values in some colour space.

When this algorithm converges, we expect to have $$N$$ centroids, where each one should have a high affinity to the points closest to it.

### Soft K-Means clustering

The above algorithm doesn't take into account that a certain point may share characteristics of more than one cluster. Hard K-Means is not expressive enough to capture this sort of relation, so we use Soft K-Means instead.

**Setup**: Randomly initialize clusters by assigning each vector $$\mathbf{x}_n$$ to one of $$N$$ clusters. We also define a distance measure $$\phi_n(i)$$, that measures the "closeness" of $$\mathbf{x}_n$$ to cluster $$\boldsymbol{\mu}_i$$.

**Repeat**:
Iterate the following

1. Updating $$\phi$$:

   $$
   \phi_n(i)=\frac{\exp \left\{-\frac{1}{\beta}\left\|\mathbf{x}_n-\boldsymbol{\mu}_i\right\|^2\right\}}{\sum_j \exp \left\{-\frac{1}{\beta}\left\|\mathbf{x}_n-\boldsymbol{\mu}_j\right\|^2\right\}}, \text { for } i=1, \ldots, N
   $$

2. Update $$\boldsymbol{\mu}_i$$ : For each $$i$$, update $$\boldsymbol{\mu}_k$$ with the weighted average

$$
\boldsymbol{\mu}_i=\frac{\sum_n \mathbf{x}_n \phi_n(i)}{\sum_n \phi_n(i)}
$$

Now we have a more expressive model, but notice a few things:

- The model is dependent on the data that it has fit to. Eg: Say you run this on image A and get some clusters. In order to recieve new clusters, you need to rerun the algorithm.
- Highly dependent on initialization. For some cluster initializations, you can get a very poor clustering performance
- The cluster mean might not be the best representation of the vectors inside the cluster itself

This motivates the question, can we use parameters and highly expressive neural networks to perform much better than K-Means?

## Slot Attention

### How to compute slots:

The slot attention operation works as follows:

**Setup:** Initialize embedding weights for key, query and value projection $$q( \cdot ), k(\cdot), v(\cdot)$$. Also initialize $$N_\text{slots}$$ slots of embedding dimension $$D$$, basically a matrix of shape $$N_{\text{slots}} \times D$$. These can be sampled from an isotropic normal with learned mean $$\boldsymbol{\mu} \in \mathbb{R}^{D}$$

**Repeat T times**:

1. Given image of shape $$B\times 3 \times H \times W$$, use a CNN encoder to encode this into a feature map of dimension $$B\times D \times H \times W$$. Then flatten this into a sequence of tokens of shape $$N_{\text{data}} \times D$$, call this $$\text{inputs}$$.
2. Compute softmax weights over the data embeddings:

   $$\text{Softmax}(\frac{1}{\sqrt{D}} k(\text{inputs}) q(\text{slots})^T, \text{axis='slots'})$$

   Unpacking this, we have $$\text{inputs} \in \mathbb{R}^{N_{data} \times D}$$ and $$\text{slots} \in \mathbb{R}^{N_{slots} \times D}$$ (excluding batch dimension). Therefore, our softmax weights are of shape $$(N_{\text{data}}, N_{\text{slots}})$$ and the softmax is done across the $$N_{\text{slots}}$$ axis. This is different from regular attention, which is done over the $$N_{\text{data}}$$ axis, as this promotes "competition" across the slots. I'll explain the intuition for that later.

3. Compute slot updates as using a weighted average, the shape of which is $$(N_{\text{slots}},D)$$. Note that is the same shape as our slots.

   $$\text{Softmax}(\frac{1}{\sqrt{D}} k(\text{inputs}) q(\text{slots})^T, \text{axis='slots'})^T v(\text{inputs})$$

4. Update the previous slots using a small GRU network and the slot updates as the input.

### Intuition

As you can see above, each slot update is a linear combination of $$\text{inputs}$$. The coefficients of input embedding vector is determined by the softmax matrix, very similar to the regular attention mechanism we all know and love.

To see why the slots enforce competition, we need to take a look at the softmax matrix in more detail. Denote $$I_i$$ as the $$i^{th}$$ input vector, and $$S_j$$ as the $$j^{th}$$ slot vector.

$$
k(\text{inputs}) q(\text{slots})^T = \begin{pmatrix}
I_1\cdot S_1 & \cdots & I_1 \cdot S_{N_{\text{slots}}} \\
\vdots &  \ddots & \vdots \\
I_{N_{\text{data}}} \cdot S_1 & \cdots & I_{N_{\text{data}}}\cdot S_{N_{\text{slots}}}
\end{pmatrix}
$$

For which direction to take the softmax in, we have two options, either row wise (on the data axis) or column wise (on the slot axis):

- If we normalize across the data axis, each row will sum up to one. So when we right multiply by $$v(\text{inputs})$$, each slot update will be a convex linear combination of the input vectors. This is exactly what is used in the regular attention mechanism. However, each slot is unrestricted in what parts of the input sequence it can attend to. For example, the first embedding could have a softmax weight of $$1.0$$.

- If we normalize across the slot dimension, each column will sum up to one. Now when we right multiply by $$v(\text{inputs})$$, each slot update won't be a convex linear combination anymore. However, now we are constraining the attention weights for each embedding across all slots. For example, if slot $$S$$ has a high attention weight $$\approx 1$$ for embedding $$I$$, then it must be the case that the other slots have attention weights $$\approx 0$$ for $$I$$. This promotes "competition" for input vectors among the slots, as only a few slots will be able to have a high weight for any given input vector due to the properties of softmax.

Ultimately, if a slot has high coefficients for a set of input embedding vectors, it should be representative of those input embedding vectors.

Therefore, I view this as a more expressive version of the K-Means operation we covered earlier, as the goal of both is to compute embeddings for distinct regions of the input images, and both do so via linear combinations of the input data. I believe expressive comes from the high degree of nonlinearity within the $$q,k,v$$ projections and $$\text{Softmax}$$.

### What does this actually do?

So now we run slot attention on a given image, and receive $$N_{\text{slots}}$$ slots. What do we do now?

Well, we need some sort of signal to update these weights, and quantify how good of a representation we learned. A simple answer (and what is used in the original paper) is to simply reconstruct the original image.

So assume each slot has learned to have high affinity (high inner product) with a particular region of the image. Eg: One slot binded to a sphere, another to a cube. Then, if we reconstruct each slot into an image, we should get a set of images for each object the slot binds to.

This is exactly what is done in the paper. Here is the sequence of steps to decode:

1. For each slot of shape $$(D, )$$, use add two new spatial dimensions and repeat, getting a shape of $$(D, \hat{H}, \hat{W})$$
2. Run this through a series of transpose convolutions, to get images of shape $$(4, H, W)$$.
3. The first 3 channels are RGB respectively, the last is an alpha channel. So for each slot $$i \in \{1, \dots, K\}$$, we split each feature map into $$C_i \in \mathbb{R}^{3\times H \times W}$$ and $$\alpha_i \in \mathbb{R}^{3 \times H \times W}$$ (we simply took our alpha channel and repeated it 3 times, so the shape lines up with $$C_i$$).
4. Finally we compose these by to get predicted image $$\hat{Y}$$.

$$ \hat{Y} = \sum\_{i=1}^K \alpha_i \odot C_i$$

Our loss function is the simple MSE loss between original image $$Y$$ and our predicted image $$\hat{Y}$$:

$$L(\hat{Y}, Y) = \left | \left | \hat{Y} - Y \right | \right |^2_2$$

### Visuals:

Here's a visual of the pipeline I presented above for this image with 7 objects and 7 slots. While the decoded image isn't perfect, notice how each slot representation, when decoded, has roughly learned to represent a specific object in the scene.

<!-- ![alt text](https://github.com/AditMeh/AditMeh.github.io/blob/master/assets/img/slot-attention/slot_attn_diagram.png?raw=true "An explanation") -->

<div class="equation">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="https://github.com/AditMeh/AditMeh.github.io/blob/master/assets/img/slot-attention/slot_attn_diagram.png?raw=true" title="example image" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

### Shortcomings and future directions:

To be added later...
