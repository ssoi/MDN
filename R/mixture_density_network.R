# Implementation of Chris Bishop's Mixture Density Network on TensorFlow
# Based on this tutorial- http://blog.otoro.net/2015/11/24/mixture-density-networks-with-tensorflow/

library(tensorflow)

# set hyperparameters
K <- 24 # num of Gaussian components
sigma <- 0.5 # st. dev of Gaussian components
n_hidden <- 24 # num of hidden units
n_out <- K * 3

# placeholder variables for data
x <- tf$placeholder(tf$float32, shape(NULL, 1L))
y <- tf$placeholder(tf$float32, shape(NULL, 1L))

# hidden layer (Gaussian units)
Wh <- tf$Variable(tf$random_normal(shape(1, n_hidden), stddev=sigma, dtype=tf$float32))
bh <- tf$Variable(tf$random_normal(shape(1, n_hidden), stddev=sigma, dtype=tf$float32))

# hidden layer (Gaussian units)
Wo <- tf$Variable(tf$random_normal(shape(n_hidden, n_out), stddev=sigma, dtype=tf$float32))
bo <- tf$Variable(tf$random_normal(shape(1, n_out), stddev=sigma, dtype=tf$float32))

# NN w/ tanh activation
hidden_layer <- tf$nn$tanh(tf$matmul(x, Wh) + bh)
output <- tf$matmul(hidden_layer, Wo) + bo

# method to map K*3 dimension output to Gaussian mixture parameters
get_mixture_coef <- function(output) {
  out_pi <- tf$placeholder(dtype=tf$float32, shape=shape(NULL,K), name="mixparam")
  out_sigma <- tf$placeholder(dtype=tf$float32, shape=shape(NULL,K), name="mixparam")
  out_mu <- tf$placeholder(dtype=tf$float32, shape=shape(NULL,K), name="mixparam")

  out_pi <- tf$split(1L, 3L, output)[[1]]
  out_mu <- tf$split(1L, 3L, output)[[2]]
  out_sigma <- tf$split(1L, 3L, output)[[3]]

  max_pi <- tf$reduce_max(out_pi, 1L, keep_dims=TRUE)
  out_pi <- tf$sub(out_pi, max_pi)

  out_pi <- tf$exp(out_pi)

  normalize_pi <- tf$inv(tf$reduce_sum(out_pi, 1L, keep_dims=TRUE))
  out_pi <- tf$mul(normalize_pi, out_pi)

  out_sigma <- tf$exp(out_sigma)

  return(list(pi=out_pi, sigma=out_sigma, mu=out_mu))
}

# Gaussian likelihood
tf_normal <- function(y, mu, sigma) {
  result <- tf$sub(y, mu)
  result <- tf$mul(result, tf$inv(sigma))
  result <- -tf$square(result)/2

  tf$mul(tf$exp(result), tf$inv(sigma))
}

# negative log-likelihood
get_lossfunc <- function(out_pi, out_sigma, out_mu, y) {
  result <- tf_normal(y, out_mu, out_sigma)
  result <- tf$mul(result, out_pi)
  result <- tf$reduce_sum(result, 1L, keep_dims=TRUE)
  result <- -tf$log(result)

  tf$reduce_mean(result)
}

out_params <- get_mixture_coef(output)

# set up optimizer
lossfunc <- get_lossfunc(out_params$pi, out_params$sigma, out_params$mu, y)
train_op <- tf$train$AdamOptimizer()$minimize(lossfunc)

# simulate training data
y_data <- matrix(ncol=1, data=runif(n=1e4L, min=-10.5, max=10.5))
x_data <- 7.0 * sin(0.75 * y_data) + 0.5 * y_data + rnorm(n=1e4L)

# activate TensorFlow session
sess <- tf$InteractiveSession()
sess$run(tf$initialize_all_variables())

# train model
loglik <- numeric(2e3L)
for(i in seq_along(loglik)) {
  sess$run(train_op, feed_dict=dict(x=x_data, y=y_data))
  loglik[i] <- sess$run(lossfunc, feed_dict=dict(x=x_data, y=y_data))
}

# generate test data and predictions
x_test <- matrix(ncol=1, data=runif(n=1e3L, min=-14, max=14))
test_params <- sess$run(get_mixture_coef(output), feed_dict=dict(x=x_test))

# sample from predicted mixture of normals given some test x
sample_idx <- sample(1:K, size=100, replace=TRUE, prob=test_params$pi[1,])
sample_pred <- rnorm(n=length(sample_idx),
                     mean=test_params$mu[1, sample_idx],
                     sd=test_params$sigma[1, sample_idx])

# plot data and sample predictied distribution
png(file="../assets/prediction.png", height=6, width=6, unit="in", res=300)
plot(x_data, y_data,
     main="x = 7sin(3y/4) + y/2",
     xlab="x",
     ylab="y",
     pch=20,
     cex=0.75,
     bty="n")
abline(v=x_test[1,1], lwd=2, lty=2, col="lightgray")
points(x=rep(x_test[1,1], 100), y=sample_pred,
             pch=20,
             cex=2,
             col=rgb(0.7, 0, 0, 0.2))
legend(x=-14, y=9,
       legend=sprintf("y=f(%.1f)", x_test[1,1]),
       box.col=NA,
       pt.cex=2.5,
       pch=20,
       col=rgb(0.7, 0, 0, 0.2))
dev.off()
