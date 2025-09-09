---
layout: post
title: NDC projection matrices in OpenGL vs gsplat
date: 2025-09-04 11:59:00-0400
description: Explaining the difference between conventional camera space to ndc projection matrices in OpenGL and Gsplat's
giscus_comments: false
related_posts: false
tags: computer_vision
---
I was reading the [gsplat technical report](https://arxiv.org/abs/2312.02121) {% cite ye2024gsplatopensourcelibrarygaussian %}  and saw this formula for the camera-space to clip space conversion matrix. If we want to get to NDC space, we simply divide everything by the $$w$$ coordinate.

$$P=\left[\begin{array}{cccc}
2 f_x / w & 0 & 0 & 0 \\
0 & 2 f_y / h & 0 & 0 \\
0 & 0 & (f+n) /(f-n) & -2 f n /(f-n) \\
0 & 0 & 1 & 0
\end{array}\right] \hspace{1 cm} \text{Gsplat Cam2Clip}$$

This confused me, because I'm used to the OpenGL notation for this sort of matrix, where $$t$$ is the top of the viewing volume on the y-axis and $$r$$ is the right of the viewing volume (from the camera center) on the x-axis.  

$$P =\left[\begin{array}{cccc}
\frac{n}{r} & 0 & 0 & 0 \\
0 & \frac{n}{t} & 0 & 0 \\
0 & 0 & \frac{-(f+n)}{f-n} & \frac{-2 f n}{f-n} \\
0 & 0 & -1 & 0
\end{array}\right] \hspace{1 cm} \text{OpenGL CamtoClip}$$

Before we start, some terms:

- Camera space: After applying W2C transform to world coordinates
- Clip space: Perspective projection (minus the step of dividing by $$w_{camera} = z_{camera}$$, aka depth) and dividing by near plane half-width
- Screen space: After perspective projection, but no other scaling. Basically just existing on the near plane, with the depth being equal to the focal length. 
- NDC space: Perspective projection and dividing by near plane half-width, to get a point in $$[-1,1]$$.

All derivations around OpenGL and NDC space are done by referncing the bible {% cite Ahn %}

## Explaining the difference

The key difference is that in gsplat, the positive z-axis faces into the scene, so our clipping planes are $$[near,far]$$ rather than $$[-near, -far]$$, where $$near = n = f$$ (focal length). I'm overloading notation, so $$f$$ is not the far clipping plane but rather the near plane, which is the focal length.

### Deriving gsplat's x/y in NDC
Firstly, check our the w coordinate of our Gsplat projection matrix. It's equal to the z-coordinate of our point in camera space, so $$w_{clip} = z_{camera}$$. In OpenGL, this is $$-z_c$$. If we rederive our NDC space $$x_n$$, with our $$w$$ coordinate being $$z_c$$ and $$x_p$$ being the projected point onto the near plane:

$$
\begin{aligned}
x_{ndc} & =\frac{2 x_p}{r-l}-\frac{r+l}{r-l} \quad\left(x_p=\frac{n x_{camera}}{z_{camera}}\right) \\
& =\frac{2 \cdot \frac{n \cdot x_{camera}}{z_{camera}}}{r-l}-\frac{r+l}{r-l} \\
& =\frac{2 n \cdot x_{camera}}{(r-l)\left(-z_{camera}\right)}-\frac{r+l}{r-l} \\
& =\frac{\frac{2 n}{r-l} \cdot x_{camera}}{z_{camera}}-\frac{r+l}{r-l} \\
& =\frac{\frac{2 n}{r-l} \cdot x_{camera}}{z_{camera}}+\frac{\frac{r+l}{r-l} \cdot z_{camera}}{z_{camera}} \\
& =(\underbrace{\frac{2 n}{r-l} \cdot x_{camera}+\frac{r+l}{r-l} \cdot z_{camera}}_{x_{clip}}) / z_{camera}
\end{aligned}
$$

$$r$$ is the right of the near plane on the x axis and $$l$$ is the left. Assuming a symmetric viewing volume, we get that $$r+l = 0 \implies r = -l$$, so finally our clip space coordinate $$x_n$$ is equal to $$\frac{2n}{r-l} = \frac{n}{r} x_{camera}$$. Therefore, our ndc space $$x$$ under the OpenGL projection is equal to:

$$x_{ndc}^{OpenGL} = \underbrace{(n\frac{x_{camera}}{z_{camera}})}_{\text{persp. projected }x_{camera}} \cdot \underbrace{\frac{1}{r}}_{\text{scaling point to [-1, -1]}}$$


Now, if we look at the gsplat matrix, the top left part is $$\frac{2 f_x}{w}$$. So the final NDC space x-coordinate is given by:

$$x_{ndc}^{gsplat} = \underbrace{(f \frac{x_{camera}}{z_{camera}})}_{\text{persp. projected } x_{camera}} \cdot \underbrace{\frac{1}{s_x}}_{\text{projected screen-space x coordinate in pixels}} \cdot \underbrace{\frac{1}{w/2}}_{\text{scaling pixel to [-1,1]}}$$

These operations are doing the same thing, except the gsplat one operates in pixel space, because we divide the perspective projected x-coordinate by width of a pixel, converting the unit from whatever the axes of the screen space represent to pixels. Also, instead of scaling by $$r$$, which is the half-width of the near plane in screen space units, we scale by $$w/2$$, which is the half width of the near plane in pixels units. Since there are only $$w/2$$ pixels in either direction of the x-axis, this rescales our screen space coordinates to $$[-1,1]$$. Finally, $$near = n = f$$, so our perspective projection equations are the same! 

At the end of the day, we're getting an output in [-1,1], it's just that the intermediate units are pixels for gsplat and screen space units for OpenGL. 

The same analysis can be repeated for $$y_{ndc}$$.

### Deriving gsplat's z in NDC

Now, $$f$$ will refer to the far clipping plane rather than focal length.

For the z-coordinate, the NDC space derivation works a similar way to the x/y coordinates. Points close to the far clipping plane with camera space values of $$(\cdot, \cdot, f)$$ are assigned to $$z_{ndc} \approx 1$$ and points close to the near clipping plane with values of $$(\cdot , \cdot, n)$$ are assigned to $$z_{ndc} \approx -1$$. 

Now, we know that $$z_{ndc}$$ only relies on $$z_{camera}$$ and $$w_{camera}$$, which is always equal to 1. We get the following equation after we perform the homogenous coordinate division:

$$z_{ndc} = \frac{Az_{camera} + B}{w_{clip}} = \frac{Az_{camera} + B}{z_{camera}}$$

Hence, using the input/outputs we had before, we get the following equations.

$$
\left\{\begin{array} { l } 
{ \frac { A n + B } { n } = - 1 } \\
{ \frac { A f + B } { f } = 1 }
\end{array} \rightarrow \left\{\begin{array}{l}
A n+B=-n \\
A f+B=f
\end{array}\right.\right.
$$

Skipping some steps, this gets us that $$A = \frac{f+n}{f-n}$$ and that $$B = \frac{-2fn}{f-n}$$. These are exactly the two values of the gsplat matrix in the z-coordinate column.

To summarize, once we rederive everything with the z-axis facing into the scene, we recover the gsplat matrix.


## Conclusion

We explained why the gsplat matrix is the same as the opengl matrix, just with a different axis direction for $$z$$ and intermediate units for NDC conversion.