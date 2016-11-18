# Implementation of Chris Bishop's Mixture Density Network on TensorFlow
# Based on this tutorial- http://blog.otoro.net/2015/11/24/mixture-density-networks-with-tensorflow/

mixture_density_network_fit <- function(session, xs, ys, K, n_hidden, iters=200L, save=NULL) {
  stopifnot(length(ys) == nrow(xs))
  stopifnot(is.integer(K) & n_hidden > 1L)
  stopifnot(is.integer(n_hidden) & n_hidden > 1L)
  stopifnot(ifelse(!is.null(save), dir.exists(dirname(save)), TRUE))

  require(tensorflow)

  # data dimensions
  n_obs <- nrow(xs)
  n_covars <- ncol(xs)
  n_out <- K * 3

  # set hyperparameters
  sigma <- 0.5 # st. dev of Gaussian components

  # placeholder variables for data
  x <- tf$placeholder(tf$float32, shape(NULL, n_covars))
  y <- tf$placeholder(tf$float32, shape(NULL, 1L))

  # hidden layer (Gaussian units)
  Wh <- tf$Variable(tf$random_normal(shape(n_covars, n_hidden), stddev=sigma, dtype=tf$float32))
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

  # initialize TensorFlow session
  session$run(tf$initialize_all_variables())

  # train model
  fitted <- list()
  loglik <- numeric(iters)
  for(i in seq_along(loglik)) {
    session$run(train_op, feed_dict=dict(x=xs, y=matrix(ncol=1, data=ys)))
    loglik[i] <- session$run(lossfunc, feed_dict=dict(x=xs, y=matrix(ncol=1, data=ys)))
  }
  fitted <- session$run(out_params, feed_dict=dict(x=xs, y=matrix(ncol=1, data=ys)))

  if(!is.null(save) & is.character(save)) {
    saver <- tf$train$Saver
    save$save(session, save)
  }

  return(list(loglik=loglik, fitted=fitted))
}
