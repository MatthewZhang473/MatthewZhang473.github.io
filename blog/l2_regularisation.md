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


Let's consider a simplest linear model. Say we have $N$ pairs of datapoints $(\mathbf{x}^{(i)}, y^{(i)}), i = 1, ..., N$, where $\mathbf{x}_i$ is $m$-dimensional input, i.e. $\mathbf{x}^{(i)} = [x_1^{(i)}, x_2^{(i)}, ..., x_m^{(i)}]$ and $y_i$ is scalar output. 

We can then define a linear model:


$$\hat{y}^{(i)} = w_0 + w_1 x_1^{(i)} + ... + w_{m} x_{m}^{(i)}$$


We can stack them up into $X \in R^{N,m+1}, \mathbf{y} \in R^{N}$, and get the following linear model:


$$\mathbf{\hat{y}} = X \mathbf{w} $$


You might already be familiar with such model: let's compute the mean-square error (MSE):

$$\text{MSE} := \frac{1}{2}||\hat{y} - y||^2 $$


and find solution that minimise it:

$$\mathbf{w} = (X^TX)^{-1}X^Ty$$



cool, but why do we do it?


### Derive the linear model


Back to the Eqn.2, what we are really saying is that:


$$\mathbf{y} = X \mathbf{w} + \boldsymbol{\epsilon}$$

, where $\boldsymbol{\epsilon} \in R^{N}$ is the random noise that we cannot capture with our model.


How do we model this noise? There are several conditions we want to get:

1. We want our prediction to be *unbiased*, hence this noise should have zero mean.
2. ideally the magitude of the noise is small, ideally within one standard deviation $\sigma$ from 0.
3. the noise from each pair of data point $(\mathbf{x}^{(i)}, y^{(i)})$ should be independent

This natually gives us a way to model this noise, 
how about we make it a identitical and independent (i.i.d.) Gaussian random noise with zero mean and some fixed standard deviation $\sigma$.

i.e.

$$\boldsymbol{\epsilon} \sim N(\mathbb{0}, \Sigma)$$.



in other words, here is our **assumption** on the residues: for our linear model, the distribution of the residuals are i.i.d. Gaussian noise.


Now comes the interesting bit, with every possible parameter $\mathbf{w}$,
we can compute the residuals $\boldsymbol{\epsilon} = \mathbf{y} - X\mathbf{w}$
then given the noise are gaussianly distributed: $\boldsymbol{\epsilon} \sim N(\mathbb{0}, \Sigma)$,
we can find $P(\boldsymbol{\epsilon})$ by substituting the residuals into the distribution:
$$N(\boldsymbol{\epsilon} |\mathbb{0}, \Sigma) = (... here give the gaussian formula)$$


This turns our linear regression problem into a probability problem: can we find the parameter $\mathbf{w^*}$ that maximizes this probability?





## How Does L2 Regularisation Improves Stability?

## What About Dropout?

## Reference