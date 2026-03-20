# Blog Post 3: L2 regularisation



**Structure:**
- [ ] What is L2-regularisation actually doing: Linear model: Y = X@beta + epsilon
   1. you might be familiar with linear model, MSE loss, ... let's see why this is the case
   2. assumption: each datapoint we observed have gaussian noise, we want to maximise the likelihood: max P(data | model)
   3. leads to minimizing MSE -> gradient -> solution
   4. but you see, what we really want is max P(model | data) -> needs a prior
   5. what is a good prior? zero mean, some variance around it -> gaussian
   6. derive the L2-regularised loss -> gradient -> solution

- [ ] Why does introducing this prior improves stability
  1. a coding example, 1D ground truth function + observation near x=1 + noise
  2. try to do fit: ill-conditioned system -> high variance in weight
  3. introducing prior, much more stable solutions

- [ ] What about dropout?
    1. https://chatgpt.com/s/t_69bbd2ac418481918a64e9f472f1ea5c
    2. see how similar it is compared to L2 regularisation!






My friend Mr.K who studies chemistry. The other day he asked me a question: "Why do we use L2 regularisation in ML, how does it magically make the model training more stable / reduce overfitting?"

Let's try to answer this question together from a probabilistic point of view.


## What is L2 Regularisation?



### Linear Model

Let's consider the simplest case: a linear model. Suppose we have $N$ pairs of data points $\{(\mathbf{x}^{(i)}, y^{(i)})\}_{i=1}^N$, where $\mathbf{x}^{(i)} \in \mathbb{R}^m$ is an $m$-dimensional input vector and $y^{(i)} \in \mathbb{R}$ is a scalar output.

We define our linear model for a single point as:

$$\hat{y}^{(i)} = w_0 + w_1 x_1^{(i)} + \dots + w_{m} x_{m}^{(i)}$$

By prepending a $1$ to each input vector $\mathbf{x}^{(i)}$ to account for the bias term $w_0$, we can stack the inputs into a design matrix $X \in \mathbb{R}^{N \times (m+1)}$ and the outputs into a vector $\mathbf{y} \in \mathbb{R}^{N}$. The model becomes:

$$\mathbf{\hat{y}} = X \mathbf{w}$$

You might already be familiar with this setup. To find the "best" $\mathbf{w}$, we typically minimize the Mean Squared Error (MSE):

$$\text{MSE}(\mathbf{w}) := \frac{1}{2}\|\mathbf{\hat{y}} - \mathbf{y}\|^2 = \frac{1}{2}\|X\mathbf{w} - \mathbf{y}\|^2$$

The closed-form solution that minimizes this loss is the well-known OLS (Ordinary Least Squares) estimator:

$$\mathbf{w}^* = (X^TX)^{-1}X^T\mathbf{y}$$

Cool—but *why* do we do it this way?

### Rethink Linear Regression with Probability

The equation above assumes a deterministic relationship, but in reality, our observations are often noisy:

$$\mathbf{y} = X \mathbf{w} + \boldsymbol{\epsilon}$$

where $\boldsymbol{\epsilon} \in \mathbb{R}^{N}$ is a random noise vector representing the factors we cannot capture with our model.

How should we model this noise? Ideally, we want a few properties:
1. **Unbiased**: The noise should have a zero mean so it doesn't systematically shift our predictions.
2. **Concentrated**: Small errors should be more likely than large ones.
3. **Independent**: The noise in one observation shouldn't tell us anything about the noise in another.

These requirements naturally lead us to model the noise as **Independent and Identically Distributed (i.i.d.) Gaussian** random variables with zero mean and variance $\sigma^2$:

$$\epsilon^{(i)} \sim \mathcal{N}(0, \sigma^2) \implies \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})$$

In other words, our **assumption** is that the residuals follow a Gaussian distribution.

Now comes the interesting part: for any given parameter $\mathbf{w}$, the probability (likelihood) of observing our data $\mathbf{y}$ is determined by the probability of those residuals $\boldsymbol{\epsilon} = \mathbf{y} - X\mathbf{w}$ occurring under our Gaussian assumption:

$$P(\text{data} | \text{model}) = P(\mathbf{y} | X, \mathbf{w}) = \prod_{i=1}^N \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left( -\frac{(y^{(i)} - \mathbf{x}^{(i)\top}\mathbf{w})^2}{2\sigma^2} \right)$$

This turns our regression problem into a Maximum Likelihood Estimation (MLE) problem: can we find the parameter $\mathbf{w}^*$ that maximizes this probability?

### From Likelihood to MSE

Maximizing a product of exponentials is mathematically identical to maximizing the **Log-Likelihood**, which is much easier to work with:

$$
\log p(\mathbf{y} | X, \mathbf{w}) = \sum_{i=1}^N \left[ \underbrace{-\log\sqrt{2\pi\sigma^2}}_{\text{constant}} - \frac{(y^{(i)} - \mathbf{x}^{(i)\top}\mathbf{w})^2}{2\sigma^2} \right]
$$

When we maximize this with respect to $\mathbf{w}$, the constant term disappears. We are left with:

$$\mathbf{w}^* = \arg\max_{\mathbf{w}} \left( -\frac{1}{2\sigma^2} \sum_{i=1}^N (y^{(i)} - \mathbf{x}^{(i)\top}\mathbf{w})^2 \right)$$

Removing the negative sign and the constant $1/2\sigma^2$ (which doesn't change the location of the maximum), we get:

$$\mathbf{w}^* = \arg\min_{\mathbf{w}} \frac{1}{2} \sum_{i=1}^N (y^{(i)} - \mathbf{x}^{(i)\top}\mathbf{w})^2$$

**And there it is!** Maximizing the likelihood under a Gaussian noise assumption is exactly the same as minimizing the Mean Squared Error. We've just proven that MSE isn't just an arbitrary choice—it's the mathematically "correct" thing to do if you believe your errors are Gaussian.

But this still doesn't explain L2 regularization. To get there, we need to go one step further: from Likelihood to **Posterior**.





## How Does L2 Regularisation Improves Stability?

## What About Dropout?

## Reference