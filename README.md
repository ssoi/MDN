# Mixture Density Network with TensorFlow (R API)
## Intro
---
Implementation of Chris Bishop's [Mixture Density Networks](http://www.cedar.buffalo.edu/~srihari/CSE574/Chap5/Chap5.7-MixDensityNetworks.pdf)
based on an inspiring [blog post](http://blog.otoro.net/2015/11/24/mixture-density-networks-with-tensorflow/) and RStudio's recently
released [TensorFlow API](https://rstudio.github.io/tensorflow/).

## Results (Preliminary)
Using a relatively simple test case, estimating x = 7sin(3y/4) + y/2 (with Gaussian noise), TensorFlow successfully converges and fits the model. 
An example of sample (B=1000) drawn from the predicted distribution of y given a random test sample
is shown:
![Prediction](../assets/prediction.png)

## TODO
*  Employ magrittr %>% paradigm to make construction of TensorFlow model easier to read
*  Move code for constructing TensorFlow model into functions
*  Create a Jupyter notebook with illustrative code
*  Find a more interesting example!
*  Explore extensions of model
  *  Higher-dimension covariates
  *  More complex neural architecture
  *  Multivariate Gaussian (non-diagonal covariance)

