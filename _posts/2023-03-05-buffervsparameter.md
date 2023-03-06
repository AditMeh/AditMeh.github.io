---
layout: post
title: Torch buffer vs Parameter
date: 2023-03-05 11:59:00-0400
description: A short writeup about the difference between torch buffers and parameters
giscus_comments: false
related_posts: false
tags: programming
---

# Torch parameter vs buffer

### Torch Parameters

You can create a parameter in your torch model by using the following in your init method. 


{% highlight python  %}

torch.nn.Parameter(torch.randn(size=(3,3)))

{% endhighlight %}

When I save the model's `state_dict`, I'll find this within the `model.parameters()`. However, what if we had something that didn't need gradient and hence did not need to be a parameter? An example would the mean and variance used in batch normalization. That's where the next idea comes into play:


### Torch Buffers

You can create a buffer in your torch model by using the following in your init method of your `nn.Module` 

{% highlight python  %}# Usage: self.register_buffer(k, v)
self.register_buffer("some_tensor", torch.ones(3,3)) 
{% endhighlight %}

In this example, `k` is a string `"some_tensor"` and `v` is a tensor of ones. I can now access this tensor via `self.some_tensor`, kind of like a python dictionary. The ones tensor never recieves and gradient, and will be stored in your state dict. 


### Conclusion
So to wrap up, you should use parameters for things that require gradient and buffers for things that don't. Of course, instead of buffers you can use `nn.Parameter` and set `requires_grad = False`, but your optimizer will need to check the `requires_grad` attribute of these tensors during the weight update, which is an unnecesary step. 








