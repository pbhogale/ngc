print("Sourcing train_cmlp.R")
source("rtorch/cmlp.R")


#' Perform proximal update on the first layer weight matrix
#'
#' @param network MLP network.
#' @param lam Numeric, regularization parameter.
#' @param lr Numeric, learning rate.
#' @param penalty Character, type of penalty ('GL', 'GSGL', 'H').
prox_update <- function(network, lam, lr, penalty) {
  W <- network$layers[[1]]$weight
  hidden <- W$size(1)
  p <- W$size(2)
  lag <- W$size(3)
  
  if (penalty == "GL") {
    norm <- torch_norm(W, dim = c(1, 3), keepdim = TRUE)
    W$data <- (W / torch_clamp(norm, min = (lr * lam))) * torch_clamp(norm - (lr * lam), min = 0.0)
  } else if (penalty == "GSGL") {
    norm <- torch_norm(W, dim = 1, keepdim = TRUE)
    W$data <- (W / torch_clamp(norm, min = (lr * lam))) * torch_clamp(norm - (lr * lam), min = 0.0)
    norm <- torch_norm(W, dim = c(1, 3), keepdim = TRUE)
    W$data <- (W / torch_clamp(norm, min = (lr * lam))) * torch_clamp(norm - (lr * lam), min = 0.0)
  } else if (penalty == "H") {
    for (i in seq_len(lag)) {
      norm <- torch_norm(W[, , 1:i], dim = c(1, 3), keepdim = TRUE)
      W$data[, , 1:i] <- (W$data[, , 1:i] / torch_clamp(norm, min = (lr * lam))) * torch_clamp(norm - (lr * lam), min = 0.0)
    }
  } else {
    stop("Unsupported penalty: ", penalty)
  }
}

#' Calculate regularization term for the first layer weight matrix
#'
#' @param network MLP network.
#' @param lam Numeric, regularization parameter.
#' @param penalty Character, type of penalty ('GL', 'GSGL', 'H').
#' @return Numeric, regularization term.
regularize <- function(network, lam, penalty) {
  W <- network$layers[[1]]$weight
  hidden <- W$size(1)
  p <- W$size(2)
  lag <- W$size(3)
  
  if (penalty == "GL") {
    lam * torch_sum(torch_norm(W, dim = c(1, 3)))
  } else if (penalty == "GSGL") {
    lam * (torch_sum(torch_norm(W, dim = c(1, 3))) + torch_sum(torch_norm(W, dim = 1)))
  } else if (penalty == "H") {
    lam * sum(sapply(seq_len(lag), function(i) torch_sum(torch_norm(W[, , 1:i], dim = c(1, 3)))))
  } else {
    stop("Unsupported penalty: ", penalty)
  }
}

#' Apply ridge penalty at all subsequent layers
#'
#' @param network MLP network.
#' @param lam Numeric, regularization parameter.
#' @return Numeric, ridge regularization term.
ridge_regularize <- function(network, lam) {
  answer <- 0
  for (i in 2:length(network$layers)){
    temp <- torch_sum({network$layers[[i]]$weight ^ 2})
    answer <- answer + temp
  }
  return(answer)
}

#' Restore parameters from best_model to model
#'
#' @param model MLP model.
#' @param best_model MLP model with the best parameters.
restore_parameters <- function(model, best_model) {
  for (params in model$parameters()) {
    best_params <- best_model$parameters()[[which(model$parameters() == params)]]
    params$data <- best_params$data
  }
}


#' Calculate the loss without falling out of autograd
#' 
#' @param networks The networks of the CMLP model
#' @param loss_fn the torch loss function
#' @param lam_ridge the regularization parameter
smooth_loss <- function(networks, loss_fn, lam_ridge){
  loss <- 0
  for (i in 1:p) {
    loss <- loss + loss_fn(networks[[i]](X[, 1:(dim(X)[2] - 1), ]), X[, (lag + 1):dim(X)[2], i, drop = FALSE])
  }
  # loss |> print()
  ridge <- 0
  for (i in 1:p) {
    ridge <- ridge + ridge_regularize(networks[[i]], lam_ridge)
  }
  smooth <- loss + ridge
  return(smooth)
}


#' Train model with ISTA
#'
#' @param cmlp cMLP model.
#' @param X Tensor, input data.
#' @param lr Numeric, learning rate.
#' @param max_iter Integer, maximum number of iterations.
#' @param lam Numeric, regularization parameter (default: 0).
#' @param lam_ridge Numeric, ridge regularization parameter (default: 0).
#' @param penalty Character, type of penalty ('H', default).
#' @param lookback Integer, lookback period for early stopping (default: 5).
#' @param check_every Integer, frequency of checks for early stopping (default: 100).
#' @param verbose Integer, verbosity level (default: 1).
#' @return Numeric vector, training loss history.
train_model_ista <- function(cmlp, X, lr, max_iter, lam = 0, lam_ridge = 0, penalty = "H", lookback = 5, check_every = 100, verbose = 1) {
  lag <- cmlp$lag
  p <- X$size()[3]
  loss_fn <- nn_mse_loss(reduction = "mean")
  train_loss_list <- c()
  
  # For early stopping
  best_it <- NULL
  best_loss <- Inf
  best_model <- NULL
  
  smooth <- smooth_loss(cmlp$networks, loss_fn, lam_ridge)
  
  for (it in seq_len(max_iter)) {
    # Take gradient step
    smooth$backward()
    
    for (i in 1:length(cmlp$parameters)) {
      with_no_grad({
        cmlp$parameters[[i]]$sub_(lr * cmlp$parameters[[i]]$grad)
        cmlp$parameters[[i]]$grad$zero_
      })
    }
    
    # Take prox step
    if (lam > 0) {
      for (net in cmlp$networks) {
        prox_update(net, lam, lr, penalty)
      }
    }
    
    cmlp$zero_grad()
    
    # Calculate smooth error
    smooth <- smooth_loss(cmlp$networks, loss_fn, lam_ridge)
    
    # Check progress
    if ((it %% check_every) == 0) {
      # Add nonsmooth penalty
      nonsmooth <- sum(sapply(cmlp$networks, function(net) regularize(net, lam, penalty)))
      mean_loss <- (smooth + nonsmooth) / p
      train_loss_list <- c(train_loss_list, mean_loss$item())
      
      if (verbose > 0) {
        cat(sprintf("%sIter = %d%s\n", strrep("-", 10), it, strrep("-", 10)))
        cat("Loss = ", mean_loss$item(), "\n")
        cat("Variable usage = ", 100 * mean(as.numeric(cmlp$GC())), "%\n")
      }
      
      # Check for early stopping
      if (mean_loss < best_loss) {
        best_loss <- mean_loss
        best_it <- it
        best_model <- deepcopy(cmlp)
      } else if ((it - best_it) >= lookback * check_every) {
        if (verbose) {
          cat("Stopping early\n")
        }
        break
      }
    }
  }
  
  # Restore best model
  restore_parameters(cmlp, best_model)
  
  train_loss_list
}