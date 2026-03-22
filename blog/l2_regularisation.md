# Blog Post 3: L2 regularisation

<!-- 

**Structure:**
- [x] What is L2-regularisation actually doing: Linear model: Y = X@beta + epsilon
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
    2. see how similar it is compared to L2 regularisation! -->






My chemist friend Mr.K who asked me a question: "Why do we use L2 regularisation in ML, how does it magically make the model training more stable / reduce overfitting?"

Let's try to answer this question together from a probabilistic point of view.



## Linear Model

Let's consider the simplest case: a linear model. Suppose we have $N$ pairs of data points $\{(\mathbf{x}^{(i)}, y^{(i)})\}_{i=1}^N$, where $\mathbf{x}^{(i)} \in \mathbb{R}^m$ is an $m$-dimensional input vector and $y^{(i)} \in \mathbb{R}$ is a scalar output.

We define our linear model for a single point as:

$$\hat{y}^{(i)} = w_0 + w_1 x_1^{(i)} + \dots + w_{m} x_{m}^{(i)} \tag{1}$$

By prepending a $1$ to each input vector $\mathbf{x}^{(i)}$ to account for the bias term $w_0$, we can stack the inputs into a design matrix $X \in \mathbb{R}^{N \times (m+1)}$ and the outputs into a vector $\mathbf{y} \in \mathbb{R}^{N}$. The model becomes:

$$\mathbf{\hat{y}} = X \mathbf{w} \tag{2}$$

You might already be familiar with this setup. To find the "best" $\mathbf{w}$, we typically minimize the Mean Squared Error (MSE):

$$\text{MSE}(\mathbf{w}) := \frac{1}{2}\|\mathbf{y} - \mathbf{\hat{y}}\|^2 = \frac{1}{2}\| \mathbf{y} - X\mathbf{w} \|^2 \tag{3}$$

The closed-form solution that minimizes this loss is the well-known OLS (Ordinary Least Squares) estimator:

$$\mathbf{w}^* = (X^TX)^{-1}X^T\mathbf{y} \tag{4}$$

Cool—but *why* do we do it this way?

## Rethink Linear Regression as Maximum Likelihood Problem

The equation above assumes a deterministic relationship, but in reality, our observations are often noisy:

$$\mathbf{y} = X \mathbf{w} + \boldsymbol{\epsilon} \tag{5}$$

where $\boldsymbol{\epsilon} \in \mathbb{R}^{N}$ is a random noise vector representing the factors we cannot capture with our model.

How should we model this noise? Ideally, we want a few properties:
1. **Unbiased**: The noise should have a zero mean so it doesn't systematically shift our predictions.
2. **Concentrated**: Small errors should be more likely than large ones.
3. **Independent**: The noise in one observation shouldn't tell us anything about the noise in another.

These requirements naturally lead us to model the noise as **Independent and Identically Distributed (i.i.d.) Gaussian** random variables with zero mean and variance $\sigma^2$:

$$\epsilon^{(i)} \sim \mathcal{N}(0, \sigma^2) \implies \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I}) \tag{6}$$

In other words, our **assumption** is that the residuals follow a Gaussian distribution.

Now comes the interesting part: for any given parameter $\mathbf{w}$, the probability (likelihood) of observing our data $\mathbf{y}$ is determined by the probability of those residuals $\boldsymbol{\epsilon} = \mathbf{y} - X\mathbf{w}$ occurring under our Gaussian assumption:

$$
P(\text{data} \mid \text{model}) = P(\mathbf{y} \mid X, \mathbf{w})
$$
$$
= \prod_{i=1}^N \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left( -\frac{(y^{(i)} - \mathbf{x}^{(i)\top}\mathbf{w})^2}{2\sigma^2} \right) \tag{7}
$$

This turns our regression problem into a Maximum Likelihood Estimation (MLE) problem: can we find the parameter $\mathbf{w}^*$ that maximizes this probability?

### From Likelihood to MSE


Taking logs, dropping the additive constant and positive factor $1/(2\sigma^2)$ gives

$$
\mathbf{w}^* = \arg\max_{\mathbf{w}} \log p(\mathbf{y} \mid X, \mathbf{w})
$$
$$
= \arg\min_{\mathbf{w}} \frac{1}{2}\sum_{i=1}^N \left(y^{(i)} - \mathbf{x}^{(i)\top}\mathbf{w}\right)^2 \tag{8}
$$

and this scalar form is exactly the earlier vector MSE in Eq. (3), since

$$\sum_{i=1}^N \left(y^{(i)} - \mathbf{x}^{(i)\top}\mathbf{w}\right)^2 = \|\mathbf{y} - X\mathbf{w}\|^2  \tag{9}$$

**And there it is!** Maximizing the likelihood under a Gaussian noise assumption is exactly the same as minimizing the Mean Squared Error. We've just proven that MSE isn't just an arbitrary choice—it's the mathematically "correct" thing to do under our probabilistic point of view.



### Solve it!

Now that we know MLE leads to the MSE objective, let's actually solve it. Recall the loss

$$
L(\mathbf{w}) = \frac{1}{2}\|\mathbf{y} - X\mathbf{w}\|^2
\quad
\tag{10}
$$

Taking the gradient with respect to $\mathbf{w}$ gives

$$
\frac{\partial L}{\partial \mathbf{w}}
= -X^\top(\mathbf{y} - X\mathbf{w})
= X^\top X \mathbf{w} - X^\top \mathbf{y}
\tag{11}
$$

Setting the gradient to zero gives the normal equation

$$
X^\top X \mathbf{w} = X^\top \mathbf{y}
\tag{12}
$$

and if $X^\top X$ is invertible, we obtain

$$
\mathbf{w}^* = (X^\top X)^{-1}X^\top \mathbf{y}
\tag{13}
$$

which is exactly the OLS solution from Eq. (4).

> **Matrix Calculus Reminder**: If matrix calculus feels a bit intimidating, a very practical sanity check is to track the shapes. Here, $\frac{\partial L}{\partial \mathbf{w}}$ must have the same shape as $\mathbf{w}$, namely $(m+1, 1)$.
>
> Looking only at the shape change,
> $$
> X^\top(\mathbf{y} - X\mathbf{w})
> : (m+1, N)(N, 1) \to (m+1, 1)
> $$
>
> so the gradient has exactly the shape we expect. This is a simple way to catch mistakes when doing matrix derivatives.



## One Step Further: From Likelihood to Posterior

So far, we have been doing **maximum likelihood estimation**:

$$P(\text{data} \mid \text{model}) \tag{14}$$


But this still only asks: *which parameter makes the observed data most likely?* What we really want is slightly different:

$$P(\text{model} \mid \text{data}) \tag{15}$$

This is the **posterior**: the probability of the model parameters after we have seen the data.

Bayes' rule tells us

$$P(\text{model} \mid \text{data}) = \frac{P(\text{data} \mid \text{model}) P(\text{model})}{P(\text{data})} \tag{16}$$

and since $P(\text{data})$ does not depend on the model, maximizing the posterior is equivalent to maximizing

$$P(\text{model} \mid \text{data}) \propto P(\text{data} \mid \text{model}) P(\text{model}) \tag{17}$$

So the posterior combines two pieces:

- the **likelihood** $P(\text{data} \mid \text{model})$, which tells us how well the model explains the observed data
- the **prior** $P(\text{model})$, which encodes what kinds of models we consider plausible before seeing any data

For linear regression, the "model" is just the parameter vector $\mathbf{w}$. A very natural choice is then a zero-mean Gaussian prior on $\mathbf{w}$:

$$P(\mathbf{w}) = \mathcal{N}(\mathbf{0}, \tau^2 I) \propto \exp\left(-\frac{\|\mathbf{w}\|^2}{2\tau^2}\right) \tag{18}$$

This says that, before seeing the data, we believe weights near zero are more likely than very large weights. That is exactly the kind of preference we want if we would like to discourage overly complicated fits.

Now substitute the Gaussian likelihood and Gaussian prior into Bayes' rule. The log-posterior is then

$$\log P(\mathbf{w} \mid \mathbf{y}, X) = -\frac{1}{2\sigma^2}\|\mathbf{y} - X\mathbf{w}\|^2 - \frac{1}{2\tau^2}\|\mathbf{w}\|^2 + \text{const} \tag{19}$$

Using the same simplification as in Eq. (8), where we drop additive constants and positive scaling factors that do not change the optimizer, and defining $\lambda := \frac{\sigma^2}{\tau^2}$, we immediately get the familiar L2-regularised objective:

$$\mathbf{w}^*_{\mathrm{MAP}} = \arg\min_{\mathbf{w}} \left[ \frac{1}{2}\|\mathbf{y} - X\mathbf{w}\|^2 + \frac{\lambda}{2}\|\mathbf{w}\|^2 \right] \tag{21}$$

And that is the key connection: **L2 regularisation is exactly what you get when you do MAP estimation with a zero-mean Gaussian prior on the weights.**


## How Does L2 Regularisation Improves Numerical Stability?

So far, we have shown that L2 regularisation comes from a Gaussian prior. But why does it improve numerical stability?

The answer is easiest to see by looking at the matrix we need to invert.

Recall that the OLS solution is

$$\mathbf{w}^* = (X^\top X)^{-1}X^\top \mathbf{y} \tag{22}$$

while the L2-regularised solution is

$$\mathbf{w}^*_{\mathrm{L2}} = (X^\top X + \lambda I)^{-1}X^\top \mathbf{y} \tag{23}$$

So the question reduces to: how stable is the inverse of $X^\top X$, and what changes when we replace it by $X^\top X + \lambda I$?

### Step 1: Condition Number Measures Stability

Consider solving a linear system

$$A\mathbf{x} = \mathbf{b} \tag{24}$$

and suppose the right-hand side is perturbed to $\mathbf{b} + \delta \mathbf{b}$. Then the solution changes from $\mathbf{x}$ to $\mathbf{x} + \delta \mathbf{x}$, where

$$(\mathbf{x} + \delta \mathbf{x}) = A^{-1}(\mathbf{b} + \delta \mathbf{b}) \qquad \Longrightarrow \qquad \delta \mathbf{x} = A^{-1}\delta \mathbf{b} \tag{25}$$

Taking norms gives the standard perturbation bound

$$\frac{\|\delta \mathbf{x}\|_2}{\|\mathbf{x}\|_2} \le \kappa_2(A)\frac{\|\delta \mathbf{b}\|_2}{\|\mathbf{b}\|_2}, \qquad \kappa_2(A) := \|A\|_2\|A^{-1}\|_2  = \frac{\|\mu\_{\max}\|}{\|\mu\_{\min}\|} \tag{26}$$


In other words, even though $\mu\_{\text{min}} > 0$ is suffcient for the the matrix $A$ to be invertible, the conditional number  $\frac{\|\mu\_{\max}\|}{\|\mu\_{\min}\|} $ determines how easy can we solve it.



### Step 2: Eigenvalues of $X^\top X$

For any vector $\mathbf{v} \in \mathbb{R}^{m+1}$,

$$\mathbf{v}^\top X^\top X \mathbf{v} = \|X\mathbf{v}\|^2 \ge 0 \tag{30}$$

so $X^\top X$ is symmetric positive semi-definite. Therefore all its eigenvalues satisfy

$$\mu_1, \dots, \mu_{m+1} \ge 0 \tag{31}$$

However, this is not suffcient for $X^{\top} X$ to be invertible, and even so, a very small $\mu\_{\text{min}}$ can make it highly unstable.

### Step 3: L2 Regularisation Shifts the Spectrum

Consider any eigenvector $v^{*}$ of $X^{\top}X$ with eigenvalue 

$\mu^{*}$, we have:

$$(X^\top X + \lambda I)v  = X^\top Xv + \lambda I v = (\mu^* + \lambda)v$$ 


i.e. $X^\top X + \lambda I$ shares the same eigenvectors, but with the eigenvalues all increase $\lambda$.


For any positive $\lambda$, we now not only guarantee invertible, and also get a nicer conditoinal number:


$$\kappa_2(X^\top X + \lambda I) = \frac{\mu_{\max} + \lambda}{\mu_{\min} + \lambda} \tag{39}$$

Splendid!



### Coding Example

Check out this 1D example to demonstrate how adding L2 regularisation improves the stability.



## What About Dropout?

Another popular regularisation technique you might have heard of is *dropout*. Dropout randomly deactivates neurons during training, aiming to preventing them from co-adapting and reducing overfitting. 

At this point, it is natural to ask: *is dropout doing something similar to L2 regularisation?*

The answer is **yes**, and let's work through the derivation with a simple example.

### Setup

Let's begin by clarifying our setup:

Let $X \in \mathbb{R}^{N \times (m+1)}, \quad \mathbf{y} \in \mathbb{R}^{N}, \quad \mathbf{w} \in \mathbb{R}^{m+1} $ just as before.

To model dropout, we place a random mask directly on the entries of the data matrix $X$. On each forward pass, each entry is either kept or dropped independently, and the surviving entries are rescaled so that the expected input stays the same.

Let $M \in \mathbb{R}^{N \times (m+1)}$ be the random mask, with entries

$$M_{ij} = \frac{z_{ij}}{1-p}, \qquad z_{ij} \sim \operatorname{Bernoulli}(1-p) \tag{27}$$

independently for each data point $i = 1, \dots, N$ and feature $j = 0, \dots, m$.

So with probability $p$, an entry is dropped to zero; with probability $1-p$, it is kept and rescaled by $1/(1-p)$.

The dropped-out predictor is

$$\hat{\mathbf{y}} = (X \odot M)\mathbf{w} \tag{28}$$

and the loss for one realization of the dropout mask is

$$L_M(\mathbf{w}) = \|\mathbf{y} - (X \odot M)\mathbf{w}\|^2 \tag{29}$$

What we really optimize in expectation is

$$L_{\mathrm{eff}}(\mathbf{w}) = \mathbb{E}_M\left[\|\mathbf{y} - (X \odot M)\mathbf{w}\|^2\right] \tag{30}$$

### Expand the Expected Loss

Because the loss is a sum over data points, it is convenient to look at one sample at a time. For the $i$-th data point,

$$\hat{y}^{(i)} = \sum_{j=0}^{m} X_{ij} M_{ij} w_j \tag{31}$$

Since $\mathbb{E}[M_{ij}] = 1$, the dropped-out prediction is unbiased:

$$\mathbb{E}[\hat{y}^{(i)}] = \sum_{j=0}^{m} X_{ij} w_j = \mathbf{x}^{(i)\top}\mathbf{w} \tag{32}$$

Now use the identity

$$\mathbb{E}\left[(y^{(i)} - \hat{y}^{(i)})^2\right] = \left(y^{(i)} - \mathbb{E}[\hat{y}^{(i)}]\right)^2 + \operatorname{Var}(\hat{y}^{(i)}) \tag{33}$$

So the problem reduces to computing the variance of $\hat{y}^{(i)}$.

Because the masks are independent across features,

$$\operatorname{Var}(\hat{y}^{(i)}) = \sum_{j=0}^{m} X_{ij}^2 w_j^2 \operatorname{Var}(M_{ij}) \tag{34}$$

and since

$$\operatorname{Var}(M_{ij}) = \frac{1}{1-p} - 1 = \frac{p}{1-p} \tag{35}$$

we get

$$\operatorname{Var}(\hat{y}^{(i)}) = \frac{p}{1-p}\sum_{j=0}^{m} X_{ij}^2 w_j^2 \tag{36}$$

### The Effective Objective

Plugging Eqs. (32) and (36) into Eq. (33), we obtain

$$\mathbb{E}\left[(y^{(i)} - \hat{y}^{(i)})^2\right] = \left(y^{(i)} - \mathbf{x}^{(i)\top}\mathbf{w}\right)^2 + \frac{p}{1-p}\sum_{j=0}^{m} X_{ij}^2 w_j^2 \tag{37}$$

Now sum over all data points $i = 1, \dots, N$:

$$L_{\mathrm{eff}}(\mathbf{w}) = \|\mathbf{y} - X\mathbf{w}\|^2 + \frac{p}{1-p}\sum_{j=0}^{m}\left(\sum_{i=1}^{N} X_{ij}^2\right) w_j^2 \tag{38}$$

But $\sum_{i=1}^{N} X_{ij}^2$ is exactly the $j$-th diagonal entry of $X^\top X$, so this simplifies to

$$L_{\mathrm{eff}}(\mathbf{w}) = \|\mathbf{y} - X\mathbf{w}\|^2 + \frac{p}{1-p}\mathbf{w}^\top \operatorname{diag}(X^\top X)\mathbf{w} \tag{39}$$

This is the main result. Dropout gives the usual least-squares loss plus a **weighted quadratic penalty**.

### Solve for the Exact Minimizer

Differentiate Eq. (39):

$$\nabla L_{\mathrm{eff}}(\mathbf{w}) = 2\left[\left(X^\top X + \frac{p}{1-p}\operatorname{diag}(X^\top X)\right)\mathbf{w} - X^\top \mathbf{y}\right] \tag{40}$$

Setting the gradient to zero gives

$$\left(X^\top X + \frac{p}{1-p}\operatorname{diag}(X^\top X)\right)\mathbf{w}^* = X^\top \mathbf{y} \tag{41}$$

and therefore

$$\mathbf{w}^*_{\mathrm{drop}} = \left(X^\top X + \frac{p}{1-p}\operatorname{diag}(X^\top X)\right)^{-1}X^\top \mathbf{y} \tag{42}$$

This is the exact minimizer of the expected dropout objective.

### Relation to L2 Regularisation

Compare Eq. (39) with ordinary L2 regularisation:

$$L_{\mathrm{L2}}(\mathbf{w}) = \|\mathbf{y} - X\mathbf{w}\|^2 + \lambda \|\mathbf{w}\|^2 \tag{43}$$

The difference is subtle but important:

- L2 regularisation uses the isotropic penalty $\lambda \|\mathbf{w}\|^2 = \lambda \mathbf{w}^\top I \mathbf{w}$
- dropout uses the data-dependent penalty $\frac{p}{1-p}\mathbf{w}^\top \operatorname{diag}(X^\top X)\mathbf{w}$

So dropout is not exactly the same as standard L2 regularisation in general. Instead, it is a **feature-wise weighted L2 penalty**, where the amount of shrinkage depends on the scale of each feature column.

If the columns of $X$ are normalized so that

$$\operatorname{diag}(X^\top X) = cI \tag{44}$$

for some constant $c$, then Eq. (39) becomes

$$L_{\mathrm{eff}}(\mathbf{w}) = \|\mathbf{y} - X\mathbf{w}\|^2 + \frac{pc}{1-p}\|\mathbf{w}\|^2 \tag{45}$$

which is exactly ordinary L2 regularisation.

So the clean summary is:

- a Gaussian prior leads exactly to ordinary L2 regularisation
- dropout leads, in expectation, to a weighted version of L2 regularisation
