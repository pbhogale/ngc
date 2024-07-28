## Created by Prasanna Bhogale, 7th May 2024
## Based on the work of Tank et al. https://arxiv.org/abs/1802.05842
print("Sourcing cmlp.R")
library(torch)
library(luz)

#' Helper function for activation functions
#'
#' @param activation Character, name of the activation function ('sigmoid', 'tanh', 'relu', 'leakyrelu', or NULL).
#' @return Activation function.
activation_helper <- function(activation) {
  if (activation == "sigmoid") {
    nn_sigmoid()
  } else if (activation == "tanh") {
    nn_tanh()
  } else if (activation == "relu") {
    nn_relu()
  } else if (activation == "leakyrelu") {
    nn_leaky_relu()
  } else if (is.null(activation)) {
    function(x) x
  } else {
    stop("Unsupported activation: ", activation)
  }
}


#' MLP Module
#'
#' This module implements a multi-layer perceptron (MLP) for time series data using 1D convolutional layers.
#'
#' @param num_series Integer, number of input time series.
#' @param lag Integer, number of time steps to look back.
#' @param hidden List, number of hidden units per layer.
#' @param activation Character, name of the activation function to use ('sigmoid', 'tanh', 'relu', 'leakyrelu', or NULL).
#'
#' @details
#' The MLP consists of multiple 1D convolutional layers. The first layer has a kernel size equal to the lag parameter,
#' and subsequent layers have a kernel size of 1. The activation function is applied to all but the first layer.
#'
#' @return An MLP module for time series data.
MLP <- nn_module(
  initialize = function(num_series, lag, hidden, activation) {
    self$activation <- activation_helper(activation)
    
    # Set up network
    self$layers <- nn_module_list()
    self$layers$append(nn_conv1d(in_channels = num_series, out_channels = hidden[[1]], kernel_size = lag))
    
    for (i in seq_along(hidden)) {
      in_channels <- hidden[[i]]
      out_channels <- ifelse(i == length(hidden), 1, hidden[[i + 1]])
      self$layers$append(nn_conv1d(in_channels = in_channels, out_channels = out_channels, kernel_size = 1))
    }
  },
  
  #' Forward pass
  #'
  #' @param X Tensor, input data of shape (batch, T, p).
  #' @return Tensor, output data.
  forward = function(X) {
    X <- torch_transpose(X, 3, 2)  # Change from (batch, T, p) to (batch, p, T)
    for (i in seq_along(self$layers)) {
      if (i != 1) {
        X <- self$activation(X)
      }
      X <- self$layers[[i]](X)
    }
    X <- torch_transpose(X, 2, 3)  # Change back to (batch, T, p)
    X
  }
)


#' cMLP Module
#'
#' This module implements a collection of MLPs for time series data using 1D convolutional layers.
#'
#' @param num_series Integer, dimensionality of the multivariate time series.
#' @param lag Integer, number of previous time points to use in prediction.
#' @param hidden List, number of hidden units per layer.
#' @param activation Character, nonlinearity at each layer.
#'
#' @details
#' The cMLP consists of multiple MLPs, each applied to a different time series. The output of each MLP is concatenated along the third dimension.
#'
#' @return A cMLP module for time series data.
cMLP <- nn_module(
  initialize = function(num_series, lag, hidden, activation = "relu") {
    self$p <- num_series
    self$lag <- lag
    self$activation <- activation_helper(activation)
    
    # Set up networks
    self$networks <- nn_module_list(
      lapply(seq_len(num_series), function(i) MLP(num_series, lag, hidden, activation))
    )
  },
  
  forward = function(X) {
    torch_cat(lapply(self$networks, function(network) network(X)), dim = 3)
  },
  
  GC = function(threshold = TRUE, ignore_lag = TRUE) {
    if (ignore_lag) {
      GC <- lapply(self$networks, function(net) torch_norm(net$layers[[1]]$weight, dim = c(1, 3)))
    } else {
      GC <- lapply(self$networks, function(net) torch_norm(net$layers[[1]]$weight, dim = 1))
    }
    GC <- torch_stack(GC)
    if (threshold) {
      (GC > 0)$to(dtype = torch_int())
    } else {
      GC
    }
  }
)



# basic_model <- cMLP(num_series = X$shape[3], lag = 10, hidden = c(2), activation = 'relu')
# X$shape
# basic_model$forward(X) -> X_out
# X_out$shape
# basic_model$GC()

