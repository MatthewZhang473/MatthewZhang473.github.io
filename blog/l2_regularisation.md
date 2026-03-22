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






My friend Mr.K who studies chemistry. The other day he asked me a question: "Why do we use L2 regularisation in ML, how does it magically make the model training more stable / reduce overfitting?"

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


## How Does L2 Regularisation Improves Stability?

Linear algebra PoV:
- [ ] why $(X^TX)^{-1}$ is theoretically solvable but unstable
- [ ] how does introducing l2 reg make it stable

- [ ] Coding example:



## What About Dropout?

At this point, it is natural to ask: *is dropout doing something similar to L2 regularisation?*

The answer is **yes, but with an important caveat**. In expectation, dropout adds a quadratic penalty, so it behaves like a kind of ridge regularisation. However, the penalty is generally **data-dependent**, so it is not exactly the same as the isotropic $\|\mathbf{w}\|^2$ penalty we derived above.

To see this clearly, let's work through the derivation in the setting of **inverted dropout**.

### Setup

Let

$$X \in \mathbb{R}^{N \times d}, \quad \mathbf{y} \in \mathbb{R}^{N}, \quad \mathbf{w} \in \mathbb{R}^{d} \tag{22}$$

and define a random diagonal dropout matrix

$$D = \operatorname{diag}(\delta_1, \dots, \delta_d), \qquad \delta_j = \frac{z_j}{1-p}, \qquad z_j \sim \operatorname{Bernoulli}(1-p) \tag{23}$$

independently for each feature $j$.

This is the standard inverted-dropout convention: with probability $p$, a feature is dropped to zero; with probability $1-p$, it is kept and rescaled by $1/(1-p)$.

The dropped-out predictor is

$$\hat{\mathbf{y}} = XD\mathbf{w} \tag{24}$$

and the loss for one realization of the dropout mask is

$$L_D(\mathbf{w}) = \|\mathbf{y} - XD\mathbf{w}\|^2 \tag{25}$$

What we really optimize in expectation is

$$L_{\mathrm{eff}}(\mathbf{w}) = \mathbb{E}_D\left[\|\mathbf{y} - XD\mathbf{w}\|^2\right] \tag{26}$$

### Expand the Expected Loss

Expanding the square gives

$$L_{\mathrm{eff}}(\mathbf{w}) = \mathbf{y}^\top \mathbf{y} - 2\mathbf{y}^\top X \, \mathbb{E}[D] \, \mathbf{w} + \mathbf{w}^\top \mathbb{E}[DX^\top XD] \mathbf{w} \tag{27}$$

So the problem reduces to computing two quantities:

- $\mathbb{E}[D]$
- $\mathbb{E}[DX^\top XD]$

### First Key Identity: $\mathbb{E}[D] = I$

Since $\delta_j = z_j/(1-p)$ and $\mathbb{E}[z_j] = 1-p$, we have

$$\mathbb{E}[\delta_j] = 1 \qquad \Longrightarrow \qquad \mathbb{E}[D] = I \tag{28}$$

This is exactly why inverted dropout is convenient: the mean prediction is unchanged.

### Second Key Identity: $\mathbb{E}[DX^\top XD]$

Let

$$G := X^\top X \tag{29}$$

Then we need to compute $\mathbb{E}[DGD]$. Its $(i,j)$-entry is

$$\left(DGD\right)_{ij} = \delta_i \delta_j G_{ij} \tag{30}$$

Now consider two cases:

- If $i \neq j$, independence gives $\mathbb{E}[\delta_i \delta_j] = \mathbb{E}[\delta_i]\mathbb{E}[\delta_j] = 1$.
- If $i = j$, then $\delta_i^2 = z_i/(1-p)^2$, so $\mathbb{E}[\delta_i^2] = 1/(1-p)$.

Therefore the off-diagonal entries are unchanged, while the diagonal entries are inflated by a factor $1/(1-p)$. Equivalently,

$$\mathbb{E}[DGD] = G + \left(\frac{1}{1-p} - 1\right)\operatorname{diag}(G) = G + \frac{p}{1-p}\operatorname{diag}(G) \tag{31}$$

Substituting back $G = X^\top X$, we get the key identity

$$\mathbb{E}[DX^\top XD] = X^\top X + \frac{p}{1-p}\operatorname{diag}(X^\top X) \tag{32}$$

### The Effective Objective

Plugging Eqs. (28) and (32) into Eq. (27), we obtain

$$L_{\mathrm{eff}}(\mathbf{w}) = \mathbf{y}^\top \mathbf{y} - 2\mathbf{y}^\top X\mathbf{w} + \mathbf{w}^\top\left(X^\top X + \frac{p}{1-p}\operatorname{diag}(X^\top X)\right)\mathbf{w} \tag{33}$$

Using

$$\|\mathbf{y} - X\mathbf{w}\|^2 = \mathbf{y}^\top \mathbf{y} - 2\mathbf{y}^\top X\mathbf{w} + \mathbf{w}^\top X^\top X \mathbf{w} \tag{34}$$

this simplifies to

$$L_{\mathrm{eff}}(\mathbf{w}) = \|\mathbf{y} - X\mathbf{w}\|^2 + \frac{p}{1-p}\mathbf{w}^\top \operatorname{diag}(X^\top X)\mathbf{w} \tag{35}$$

This is the main result. Dropout gives the usual least-squares loss plus a **weighted quadratic penalty**.

### Solve for the Exact Minimizer

Differentiate Eq. (35):

$$\nabla L_{\mathrm{eff}}(\mathbf{w}) = 2\left[\left(X^\top X + \frac{p}{1-p}\operatorname{diag}(X^\top X)\right)\mathbf{w} - X^\top \mathbf{y}\right] \tag{36}$$

Setting the gradient to zero gives

$$\left(X^\top X + \frac{p}{1-p}\operatorname{diag}(X^\top X)\right)\mathbf{w}^* = X^\top \mathbf{y} \tag{37}$$

and therefore

$$\mathbf{w}^*_{\mathrm{drop}} = \left(X^\top X + \frac{p}{1-p}\operatorname{diag}(X^\top X)\right)^{-1}X^\top \mathbf{y} \tag{38}$$

This is the exact minimizer of the expected dropout objective.

### Relation to L2 Regularisation

Compare Eq. (35) with ordinary ridge regression:

$$L_{\mathrm{ridge}}(\mathbf{w}) = \|\mathbf{y} - X\mathbf{w}\|^2 + \lambda \|\mathbf{w}\|^2 \tag{39}$$

The difference is subtle but important:

- ridge uses the isotropic penalty $\lambda \|\mathbf{w}\|^2 = \lambda \mathbf{w}^\top I \mathbf{w}$
- dropout uses the data-dependent penalty $\frac{p}{1-p}\mathbf{w}^\top \operatorname{diag}(X^\top X)\mathbf{w}$

So dropout is not exactly the same as standard L2 regularisation in general. Instead, it is a **feature-wise weighted ridge penalty**, where the amount of shrinkage depends on the scale of each feature column.

If the columns of $X$ are normalized so that

$$\operatorname{diag}(X^\top X) = cI \tag{40}$$

for some constant $c$, then Eq. (35) becomes

$$L_{\mathrm{eff}}(\mathbf{w}) = \|\mathbf{y} - X\mathbf{w}\|^2 + \frac{pc}{1-p}\|\mathbf{w}\|^2 \tag{41}$$

which is exactly ridge regression.

So the clean summary is:

- a Gaussian prior leads exactly to ordinary L2 regularisation
- inverted dropout leads, in expectation, to a weighted version of L2 regularisation
- after feature normalization, the two become much closer

## Reference
