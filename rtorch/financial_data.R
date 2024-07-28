# Load required libraries
library(torch)
library(ggplot2)
library(tidyr)
library(BatchGetSymbols)
library(dagitty)
library(zoo)

# Source the file containing the functions and classes
source("/home/prasanna/Documents/development/freya/ngc/rtorch/train_cmlp.R")

#' Load financial data
#'
#' @param symbols Character vector, stock ticker symbols.
#' @param period Character, period for data retrieval.
#' @return List of data frames with historical stock prices.
load_data <- function(symbols, period = "5y") {
  data <- BatchGetSymbols::BatchGetSymbols(tickers = symbols, 
    first.date = Sys.Date() - as.numeric(gsub("y", "", period)) * 365, 
    last.date = Sys.Date())
  df <- tibble(ticker = data$df.tickers$ticker, 
    price = data$df.tickers$price.adjusted, 
    date_day = data$df.tickers$ref.date)
  spread(df, key = ticker, value = price)
}

#' Normalize financial data
#'
#' @param data List of data frames, historical stock prices.
#' @return List of normalized data frames.
normalize_data <- function(data) {
  for (i in 1:ncol(data)) {
    symbol <- colnames(data)[i]
    print(symbol)
    if (class(data[[symbol]])=="numeric"){
      avg_value <- mean(data[[symbol]], na.rm = TRUE)
      sd_value <- sd(data[[symbol]], na.rm = TRUE)
      data[[symbol]] <- (data[[symbol]] - avg_value) / sd_value
      print("numeric")
    }
  }
  data
}

#' Create time series tensor
#'
#' @param normalized_data List of normalized data frames.
#' @return Tensor of shape (1, number_timesteps, number_time_series).
create_time_series_tensor <- function(normalized_data) {
  normalized_data |> select(-date_day) -> normalized_data
  min_length <- min(sapply(normalized_data, length))
  trimmed_data <- lapply(normalized_data, function(x) tail(x, min_length))
  time_series_data <- do.call(cbind, lapply(trimmed_data, function(x) x[1:min_length]))
  time_series_data <- t(time_series_data)
  time_series_data <- array(time_series_data, dim = c(1, nrow(time_series_data), ncol(time_series_data)))
  torch_tensor(time_series_data, dtype = torch_float32())
}

# Function to interpolate missing values in numeric columns
interpolate_na <- function(data) {
  data[] <- lapply(data, function(x) {
    if (is.numeric(x)) {
      na.approx(x, na.rm = FALSE)
    } else {
      x
    }
  })
  return(data)
}

# Load data
symbols <- c(
  'TTF=F', 'NG=F'
  #, 'BAYRY', 'VWAGY', '^GDAXI', 'LNG', 'SHEL', 'EURUSD=X',
  #'RUB=X', 'RELIANCE.BO', 'BZ=F', 'MTF=F', 'BP', 
  #'TTE', 'EONGY', 'RWEOY', 'BASFY', 'SIEGY', 'VWDRY'
)
data <- load_data(symbols)

# Normalize data
normalized_data <- normalize_data(data) |> interpolate_na()

# Create time series tensor
X <- create_time_series_tensor(normalized_data)
X$permute(c(1,3,2)) -> X

print(X$shape)

# Define cMLP model
cmlp <- cMLP(num_series = X$shape[3], lag = 10, hidden = c(2), activation = 'relu')

# Train model with ISTA
train_loss_list <- train_model_ista(
  cmlp, X, lam = 0.002, lam_ridge = 1e-2, lr = 5e-2, penalty = 'H', 
  max_iter = 5,
  check_every = floor(sqrt(1))
)

GC_est <- as_array(cmlp$GC()$cpu())

# Human-readable names for the series
series_names <- c(
  'TTF Gas Futures', 'Natural Gas (NA)', 'Bayer Stock', 'Volkswagen Stock',
  'DAX Index', 'Cheniere Energy', 'Shell Stock', 'EUR/USD Exchange Rate',
  'RUB/EUR Exchange Rate', 'Reliance Stock', 
  'Brent Crude Oil', 'Coal Prices', 'BP Stock', 
  'TotalEnergies Stock', 'E.ON Stock', 'RWE Stock', 'BASF Stock', 
  'Siemens Stock', 'Vestas Wind Systems Stock'
)

# Plot Granger Causality Estimation
plot_gc_est <- function(GC_est, series_names) {
  gc_df <- as.data.frame(GC_est)
  colnames(gc_df) <- series_names
  gc_df$Series <- series_names
  gc_long <- pivot_longer(gc_df, -Series, names_to = "Causal_Series", values_to = "Value")
  
  ggplot(gc_long, aes(x = Causal_Series, y = Series, fill = Value)) +
    geom_tile() +
    scale_fill_gradient(low = "white", high = "blue") +
    labs(title = "Granger Causality Estimation", x = "Causal Series (X-axis)", y = "Affected Series (Y-axis)") +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
}

# Save plot
ggsave("granger_causality_estimation.png", plot_gc_est(GC_est, series_names), dpi = 300, width = 10, height = 10)

# Create Granger Causality Graph with dagitty
create_gc_dagitty <- function(GC_est, series_names) {
  dagitty_str <- "dag {"
  for (x in seq_len(nrow(GC_est))) {
    for (y in seq_len(ncol(GC_est))) {
      if (GC_est[x, y] == 1) {
        dagitty_str <- paste0(dagitty_str, series_names[y], " -> ", series_names[x], "\n")
      }
    }
  }
  dagitty_str <- paste0(dagitty_str, "}")
  dagitty::dagitty(dagitty_str)
}

# Plot Granger Causality Graph with dagitty
plot_gc_dagitty <- function(gc_dagitty) {
  plot(gc_dagitty, main = "Granger Causality Graph", layout = "circle")
}

# Create and plot graph
gc_dagitty <- create_gc_dagitty(GC_est, series_names)
png("granger_causality_est_graph.png", width = 1200, height = 1000, res = 300)
plot_gc_dagitty(gc_dagitty)
dev.off()
