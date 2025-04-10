# Clean Comparison of RcppKPCA with kernlab and standard PCA
# This script provides a structured comparison between the implementations

# Load required libraries
library(RcppKPCA)
library(kernlab)
library(stats)
library(ggplot2)
library(gridExtra)
library(cluster)
library(microbenchmark)

# Set random seed for reproducibility
set.seed(123)

# ---- DATASET GENERATION FUNCTIONS ----

# Half-moons dataset
generate_half_moons <- function(n_samples = 200, noise = 0.1) {
  n_samples_out <- round(n_samples/2)
  n_samples_in <- n_samples - n_samples_out
  
  # Outer half moon - more challenging randomized approach
  theta_out <- runif(n_samples_out, 0, pi)
  r_out <- 1 + rnorm(n_samples_out, 0, noise/2)
  x_out <- r_out * cos(theta_out)
  y_out <- r_out * sin(theta_out)
  
  # Inner half moon - more challenging randomized approach
  theta_in <- runif(n_samples_in, 0, pi)
  r_in <- 1 + rnorm(n_samples_in, 0, noise/2)
  x_in <- 1 - r_in * cos(theta_in)
  y_in <- 1 - r_in * sin(theta_in) - 0.5
  
  # Create data frame
  x <- c(x_out, x_in) + rnorm(n_samples, sd = noise)
  y <- c(y_out, y_in) + rnorm(n_samples, sd = noise)
  
  data <- data.frame(
    x = x,
    y = y,
    class = factor(c(rep("A", n_samples_out), rep("B", n_samples_in)))
  )
  
  return(data)
}

# Concentric circles dataset
generate_circles <- function(n_samples = 200, noise = 0.1) {
  n_samples_inner <- round(n_samples/2)
  n_samples_outer <- n_samples - n_samples_inner
  
  # Inner circle
  theta_inner <- runif(n_samples_inner, 0, 2*pi)
  r_inner <- 0.5 + rnorm(n_samples_inner, 0, noise)
  x_inner <- r_inner * cos(theta_inner)
  y_inner <- r_inner * sin(theta_inner)
  
  # Outer circle
  theta_outer <- runif(n_samples_outer, 0, 2*pi)
  r_outer <- 2 + rnorm(n_samples_outer, 0, noise)
  x_outer <- r_outer * cos(theta_outer)
  y_outer <- r_outer * sin(theta_outer)
  
  data <- data.frame(
    x = c(x_inner, x_outer),
    y = c(y_inner, y_outer),
    class = factor(c(rep("A", n_samples_inner), rep("B", n_samples_outer)))
  )
  
  return(data)
}

# ---- EVALUATION FUNCTIONS ----

# Calculate separation metrics
calculate_separation_metrics <- function(projection, class_labels) {
  projection <- as.matrix(projection)
  class_levels <- levels(class_labels)
  
  # Calculate class indices
  class_A_indices <- which(class_labels == class_levels[1])
  class_B_indices <- which(class_labels == class_levels[2])
  
  # Calculate centroids
  centroid_A <- colMeans(projection[class_A_indices, , drop=FALSE])
  centroid_B <- colMeans(projection[class_B_indices, , drop=FALSE])
  
  # Calculate between-class distance
  between_class_dist <- sqrt(sum((centroid_A - centroid_B)^2))
  
  # Calculate within-class distances
  within_A_dist <- mean(apply(projection[class_A_indices, , drop=FALSE], 1, 
                              function(x) sqrt(sum((x - centroid_A)^2))))
  within_B_dist <- mean(apply(projection[class_B_indices, , drop=FALSE], 1, 
                              function(x) sqrt(sum((x - centroid_B)^2))))
  
  # Calculate separation ratio (higher is better)
  separation_ratio <- between_class_dist / (within_A_dist + within_B_dist)
  
  # Calculate silhouette coefficient
  sil <- silhouette(as.numeric(class_labels), dist(projection))
  avg_silhouette <- mean(sil[,3])
  
  return(list(
    between_class_dist = between_class_dist,
    within_A_dist = within_A_dist,
    within_B_dist = within_B_dist,
    separation_ratio = separation_ratio,
    silhouette = avg_silhouette
  ))
}

# ---- COMPARISON FUNCTION ----

compare_implementations <- function(data, gamma_values = c(1, 5, 10), dataset_name = "Dataset") {
  X <- as.matrix(data[, c("x", "y")])
  n_components <- 2
  
  # Store results
  results <- list()
  
  # Original data plot
  p_orig <- ggplot(data, aes(x = x, y = y, color = class)) +
    geom_point(size = 2, alpha = 0.7) +
    scale_color_manual(values = c("A" = "blue", "B" = "red")) +
    labs(title = paste0("Original ", dataset_name),
         x = "X", y = "Y") +
    theme_minimal()
  
  results$original_plot <- p_orig
  print(p_orig)
  
  # Store metrics
  all_metrics <- data.frame()
  
  # For each gamma value
  for (gamma in gamma_values) {
    cat("\nAnalyzing with gamma =", gamma, "\n")
    
    # Run RcppKPCA
    rcpp_time <- system.time({
      rcpp_kpca_result <- fastKPCA(X, gamma = gamma, n_components = n_components)
      rcpp_projected <- predict(rcpp_kpca_result, X)
    })
    
    # Run kernlab KPCA
    kernlab_time <- system.time({
      kernlab_kpca_result <- kernlab::kpca(X, kernel = "rbfdot", 
                                          kpar = list(sigma = gamma/2),
                                          features = n_components)
      kernlab_projected <- kernlab::pcv(kernlab_kpca_result)
    })
    
    # Run standard PCA
    pca_time <- system.time({
      pca_result <- prcomp(X, center = TRUE, scale. = FALSE)
      pca_projected <- pca_result$x[, 1:n_components]
    })
    
    # Calculate metrics
    rcpp_metrics <- calculate_separation_metrics(rcpp_projected, data$class)
    kernlab_metrics <- calculate_separation_metrics(kernlab_projected, data$class)
    pca_metrics <- calculate_separation_metrics(pca_projected, data$class)
    
    # Print results
    cat("RcppKPCA - Time:", rcpp_time[3], "s, Separation:", 
        round(rcpp_metrics$separation_ratio, 3), 
        "Silhouette:", round(rcpp_metrics$silhouette, 3), "\n")
    
    cat("kernlab  - Time:", kernlab_time[3], "s, Separation:", 
        round(kernlab_metrics$separation_ratio, 3), 
        "Silhouette:", round(kernlab_metrics$silhouette, 3), "\n")
    
    cat("PCA      - Time:", pca_time[3], "s, Separation:", 
        round(pca_metrics$separation_ratio, 3), 
        "Silhouette:", round(pca_metrics$silhouette, 3), "\n")
    
    # Store metrics
    gamma_metrics <- rbind(
      data.frame(
        gamma = gamma,
        method = "RcppKPCA",
        time = rcpp_time[3],
        separation_ratio = rcpp_metrics$separation_ratio,
        silhouette = rcpp_metrics$silhouette
      ),
      data.frame(
        gamma = gamma,
        method = "kernlab",
        time = kernlab_time[3],
        separation_ratio = kernlab_metrics$separation_ratio,
        silhouette = kernlab_metrics$silhouette
      ),
      data.frame(
        gamma = gamma,
        method = "PCA",
        time = pca_time[3],
        separation_ratio = pca_metrics$separation_ratio,
        silhouette = pca_metrics$silhouette
      )
    )
    
    all_metrics <- rbind(all_metrics, gamma_metrics)
    
    # Create plot data
    rcpp_df <- data.frame(
      PC1 = rcpp_projected[, 1],
      PC2 = rcpp_projected[, 2],
      class = data$class
    )
    
    kernlab_df <- data.frame(
      PC1 = kernlab_projected[, 1],
      PC2 = kernlab_projected[, 2],
      class = data$class
    )
    
    pca_df <- data.frame(
      PC1 = pca_projected[, 1],
      PC2 = pca_projected[, 2],
      class = data$class
    )
    
    # Create plots
    p1 <- ggplot(rcpp_df, aes(x = PC1, y = PC2, color = class)) +
      geom_point(size = 2, alpha = 0.7) +
      scale_color_manual(values = c("A" = "blue", "B" = "red")) +
      labs(title = paste0("RcppKPCA (gamma=", gamma, ")"),
           x = "PC1", y = "PC2") +
      theme_minimal()
    
    p2 <- ggplot(kernlab_df, aes(x = PC1, y = PC2, color = class)) +
      geom_point(size = 2, alpha = 0.7) +
      scale_color_manual(values = c("A" = "blue", "B" = "red")) +
      labs(title = paste0("kernlab KPCA (gamma=", gamma, ")"),
           x = "PC1", y = "PC2") +
      theme_minimal()
    
    p3 <- ggplot(pca_df, aes(x = PC1, y = PC2, color = class)) +
      geom_point(size = 2, alpha = 0.7) +
      scale_color_manual(values = c("A" = "blue", "B" = "red")) +
      labs(title = "Standard PCA",
           x = "PC1", y = "PC2") +
      theme_minimal()
    
    # Display the plots
    print(grid.arrange(p1, p2, p3, ncol = 2))
  }
  
  # Return results
  results$metrics <- all_metrics
  
  # Plot performance metrics
  p_sil <- ggplot(all_metrics[all_metrics$method != "PCA",], 
                  aes(x = gamma, y = silhouette, color = method)) +
    geom_line() +
    geom_point() +
    labs(title = "Silhouette Score by Gamma",
         x = "Gamma", y = "Silhouette Score") +
    theme_minimal()
  
  p_sep <- ggplot(all_metrics[all_metrics$method != "PCA",], 
                  aes(x = gamma, y = separation_ratio, color = method)) +
    geom_line() +
    geom_point() +
    labs(title = "Separation Ratio by Gamma",
         x = "Gamma", y = "Separation Ratio") +
    theme_minimal()
  
  p_time <- ggplot(all_metrics, aes(x = method, y = time, fill = method)) +
    geom_bar(stat = "identity") +
    labs(title = "Computation Time Comparison",
         x = "Method", y = "Time (seconds)") +
    theme_minimal()
  
  print(grid.arrange(p_sil, p_sep, p_time, ncol = 2))
  
  results$plots <- list(
    silhouette = p_sil,
    separation = p_sep,
    time = p_time
  )
  
  return(results)
}

# ---- RUN COMPARISONS ----

# Generate datasets
n_samples <- 200
noise_level <- 0.1

# Create datasets
moons_data <- generate_half_moons(n_samples, noise_level)
circles_data <- generate_circles(n_samples, noise_level)

# Run comparisons
gamma_values <- c(1, 5, 10)

cat("\n--- HALF-MOONS DATASET ---\n")
moons_results <- compare_implementations(moons_data, gamma_values, "Half-Moons")

cat("\n--- CONCENTRIC CIRCLES DATASET ---\n")
circles_results <- compare_implementations(circles_data, gamma_values, "Concentric Circles")

# ---- BENCHMARKING ----

# Performance comparison with increasing data size
cat("\n--- PERFORMANCE SCALING ---\n")

sizes <- c(100, 500, 1000)
rcpp_times <- numeric(length(sizes))
kernlab_times <- numeric(length(sizes))

for (i in seq_along(sizes)) {
  n <- sizes[i]
  data <- generate_half_moons(n, noise_level)
  X <- as.matrix(data[, c("x", "y")])
  
  cat("Sample size:", n, "\n")
  
  # RcppKPCA
  rcpp_time <- system.time({
    rcpp_kpca_result <- fastKPCA(X, gamma = 5, n_components = 2)
    rcpp_projected <- predict(rcpp_kpca_result, X)
  })
  rcpp_times[i] <- rcpp_time[3]
  
  # kernlab
  kernlab_time <- system.time({
    kernlab_kpca_result <- kernlab::kpca(X, kernel = "rbfdot", 
                                        kpar = list(sigma = 5/2),
                                        features = 2)
    kernlab_projected <- kernlab::pcv(kernlab_kpca_result)
  })
  kernlab_times[i] <- kernlab_time[3]
  
  cat("RcppKPCA:", rcpp_times[i], "s\n")
  cat("kernlab:", kernlab_times[i], "s\n")
}

# Plot scaling behavior
scaling_df <- data.frame(
  size = rep(sizes, 2),
  time = c(rcpp_times, kernlab_times),
  method = factor(rep(c("RcppKPCA", "kernlab"), each = length(sizes)))
)

p_scaling <- ggplot(scaling_df, aes(x = size, y = time, color = method)) +
  geom_line() +
  geom_point() +
  labs(title = "Performance Scaling with Dataset Size",
       x = "Sample Size", y = "Time (seconds)") +
  theme_minimal()

print(p_scaling)

# ---- SUMMARY ----

cat("\n--- SUMMARY ---\n")
cat("Feature                 | RcppKPCA      | kernlab       | Standard PCA\n")
cat("------------------------|---------------|---------------|-------------\n")
cat("Handles non-linearity   | Yes           | Yes           | No          \n")
cat("RBF kernel              | Yes           | Yes           | N/A         \n")
cat("Other kernels           | No            | Yes           | N/A         \n")
cat("Memory efficiency       | High          | Medium        | High        \n")
cat("Speed                   | High          | Medium        | Very High   \n")
cat("Easy parameter tuning   | Yes           | Yes           | N/A         \n")
cat("Pre-image calculation   | No            | Yes           | N/A         \n")
cat("Implementation language | C++/R         | R/C           | R/Fortran   \n") 