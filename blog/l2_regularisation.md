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






I have a good friend Mr.K who studies chemistry. The other day he asked me a question: "Why do we use L2 regularisation in ML, how does it magically make the model training more stable / reduce overfitting?"

Let's try to answer this question together from a probabilistic point of view.


## What is L2 Regularisation?

## How Does L2 Regularisation Improves Stability?

## What About Dropout?

## Reference