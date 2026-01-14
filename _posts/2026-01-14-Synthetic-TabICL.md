---
layout: post
title: Using Synthetic images as in-context samples for TabICL
date: 2026-01-13 11:59:00-0400
description: A quick way to fit probes using self-supervised data using TabICL
giscus_comments: false
related_posts: false
tags: computer_vision
related_publications: true
---

# Motivation


## In context learning in LLMs

In context learning is a term that is often associated with the ability for LLMs to use examples of a task to perform novel instances of that task. For example, say I want the LLM to translate a document from Alice's writing style to Bob's writing style. I can construct a prompt that has 10-20 samples of both of their writing styles + the author's name, then the document's to be translated and finally "translate the attached document's writing style to Bob's". LLMs have been shown to have a remarkable ability to perform tasks with just a few examples. 

## TabICL

Along these lines, tabular in-context learning models, or TabICL for short, is a transformer model that's able to use in-context samples of sample-label pairs to classify novel datapoints. However unlike LLMs, these models are much more general - **they operate on any arbitrary vector**, not just tokenized text. 

More precisely, given $$(x,y)$$ pairs, where $$x \in \mathbb{R}^n$$ is a sample and for a set of classes $$C \subset \mathbb{N}^{>0}$$, $$y \in C$$ is the corresponding label, TabICL can predict label $$\hat{y}$$ for an unseen sample $$\hat{x}$$.

There's a lot more details, like how TabICL is trained and implemented, but these will maybe be in a future blog post. For now know this, **TabICL has not seen a single image during it's training**. 


##  Using TabICL on images

Let's try a simple test. Let's take CIFAR10 and embed the images using a DINOV2 backbone, getting a train/test splits of 4096 dimensional embeddings. Now that we're in the format TabICL expects, let's take increasing percentages of our train set as ICL samples and plot how the test accuracy evolves as a function of number of ICL samples. 

<div class="equation">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="https://github.com/AditMeh/AditMeh.github.io/blob/master/assets/img/ICL_SSL/cifar.png?raw=True" title="example image" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

We see that we've managed to get a classifier that has a 90% accuracy with only 20% of the train set, along with a clear scaling of test performance with ICL samples. 

Since the backbone of TabICL is a transformer, we can also obseve a scaling law with inference time w.r.t the number of training samples 

<div class="equation">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="https://github.com/AditMeh/AditMeh.github.io/blob/master/assets/img/ICL_SSL/cifar_time.png?raw=True" title="example image" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

While the equation looks like a linear relationship, it's quadratic in reality since self-attention is $$O(N^2)$$. Inference becomes VERY slow with a lot of training samples, a key drawback of this method. 

# Method

One clear problem that arises with TabICL is that since you have quadratic scaling, there will come a point where you'll need to construct your set of ICL samples effectively, in order to minimize runtime but maximize performance. Finding the optimal set of samples is an NP hard problem. 


## Linear Gradient Matching

To circumvent this, I took advantage of recent works in dataset distillation. To summarize, dataset distillation is a technique that constructs synthetic samples paired with labels, containing the _maximal_ information about their class—much more than any individual sample. To do this, we draw upon a paper called "Dataset Distillation for Pre‑Trained Self‑Supervised Vision Models", which aims to construct images that encode discriminative information about the class they belong to, which in-theory should lead to maximially informative SSL embeddings.

<div class="equation">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="https://github.com/AditMeh/AditMeh.github.io/blob/master/assets/img/ICL_SSL/linear_dd.png?raw=True" title="example image" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

Above is their key figure. Given an SSL model $$\phi$$, they construct batches of real and synthetic images, pass them through the SSL model and a linear projector $W$. The objective is to reduce the cosine distance between the gradients of the classification loss for both batches.

Using this technique, we can construct ICL samples using perfectly synthetic data, completely avoiding picking subsets and entierly relying on synthetic data. 

Let's benchmark this technique on the imagenette subset, a 10-class subset of imagenet. Here's what a randomly sampled image per class looks like:

<div class="equation">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="https://github.com/AditMeh/AditMeh.github.io/blob/master/assets/img/ICL_SSL/imagenette2_grid.png?raw=True" title="example image" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

Now here's what our synthetic samples per class look like:

<div class="equation">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="https://github.com/AditMeh/AditMeh.github.io/blob/master/assets/img/ICL_SSL/fake_grid.png?raw=True" title="example image" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>


# Results

## Naive Baselines

Comparing this to two baselines:
- Selecting a random sample per class to be the representative 
- Computing the closest sample to the centroid of each class, and using that sample as the class representative 

For the random sample baseline, here's the distribution of test performance across 50 runs, with the red line being the mean:

<div class="equation">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="https://github.com/AditMeh/AditMeh.github.io/blob/master/assets/img/ICL_SSL/random.png?raw=True" title="example image" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

Finally, the centroid baseline gets us 92%.


## Synthetic samples

With these ICL samples, we manage to get a 96% accuracy on the test set, using only one synthetic sample per class, showing a performance gain over our baselines. 

# Future work

This was a test on one dataset with easily discriminated classes. Testing this method on more complex datasets with more classes and distribution shift should show increased benefits of using synthetic data.

Additionally, we should also be able to plot decision boundaries (or atleast attention maps), to see how swapping train set samples with synthetic samples makes a difference. 




