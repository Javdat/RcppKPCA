# =============================================================================
# RcppKPCA Basic Demo
# This demo showcases the various features of the RcppKPCA package
# =============================================================================

# Load the package
library(RcppKPCA)

# Set seed for reproducibility
set.seed(123)

# =============================================================================
# PART 1: GENERATE SAMPLE DATASETS
# =============================================================================
cat("\n===== GENERATING SAMPLE DATASETS =====\n")

# ----- Half-moons dataset -----
cat("Creating half-moons dataset...\n")
n_samples <- 200
n_half <- n_samples / 2

# Using the more challenging randomized approach
theta_out <- runif(n_half, 0, pi)
r_out <- 1 + rnorm(n_half, 0, 0.1)
x1 <- r_out * cos(theta_out)
y1 <- r_out * sin(theta_out)

theta_in <- runif(n_half, 0, pi)
r_in <- 1 + rnorm(n_half, 0, 0.1)
x2 <- 0.5 - r_in * cos(theta_in)
y2 <- 0.5 - r_in * sin(theta_in) - 0.5

X_moons <- rbind(cbind(x1, y1), cbind(x2, y2))
classes_moons <- factor(c(rep(1, n_half), rep(2, n_half)))

# ----- Concentric circles dataset -----
cat("Creating concentric circles dataset...\n")
noise <- 0.1

# Inner circle
n_samples_inner <- n_samples/2
theta_inner <- runif(n_samples_inner, 0, 2*pi)
r_inner <- 0.5 + rnorm(n_samples_inner, 0, noise)
x_inner <- r_inner * cos(theta_inner)
y_inner <- r_inner * sin(theta_inner)

# Outer circle
n_samples_outer <- n_samples - n_samples_inner
theta_outer <- runif(n_samples_outer, 0, 2*pi)
r_outer <- 2 + rnorm(n_samples_outer, 0, noise)
x_outer <- r_outer * cos(theta_outer)
y_outer <- r_outer * sin(theta_outer)

X_circles <- rbind(cbind(x_inner, y_inner), cbind(x_outer, y_outer))
classes_circles <- factor(c(rep(1, n_samples_inner), rep(2, n_samples_outer)))

# =============================================================================
# PART 2: BASIC KPCA WITH DATA SCALING
# =============================================================================
cat("\n===== BASIC KPCA WITH DATA SCALING =====\n")

# Plot original half-moons data
cat("Visualizing original half-moons dataset...\n")
par(mfrow = c(1, 2))
plot(X_moons, col = classes_moons, pch = 19, 
     main = "Original Half-Moons Dataset",
     xlab = "X1", ylab = "X2")

# Apply KPCA with data scaling (new feature)
cat("Performing KPCA with data scaling...\n")
kpca_result <- fastKPCA(X_moons, gamma = 15, n_components = 2, scale_data = TRUE)

# Project the data
projected <- predict(kpca_result, X_moons)

# Plot the projected data
plot(projected, col = classes_moons, pch = 19,
     main = "KPCA Projection (gamma = 15, with scaling)",
     xlab = "PC1", ylab = "PC2")

# Reset plot layout
par(mfrow = c(1, 1))

# =============================================================================
# PART 3: EFFECT OF GAMMA PARAMETER
# =============================================================================
cat("\n===== EFFECT OF GAMMA PARAMETER =====\n")

# Try different gamma values 
gamma_values <- c(5, 10, 15, 20)
par(mfrow = c(2, 2))

for (gamma in gamma_values) {
  cat(sprintf("Testing gamma = %d...\n", gamma))
  # Perform KPCA with data scaling
  kpca_result <- fastKPCA(X_moons, gamma = gamma, n_components = 2, scale_data = TRUE)
  
  # Project the data
  projected <- predict(kpca_result, X_moons)
  
  # Plot the projected data
  plot(projected, col = classes_moons, pch = 19,
       main = paste("KPCA Projection (gamma =", gamma, ")"),
       xlab = "PC1", ylab = "PC2")
}

# Reset plot layout
par(mfrow = c(1, 1))

# =============================================================================
# PART 4: COMPARING WITH AND WITHOUT DATA SCALING
# =============================================================================
cat("\n===== COMPARING WITH AND WITHOUT DATA SCALING =====\n")

# No scaling
cat("Performing KPCA without scaling...\n")
kpca_no_scale <- fastKPCA(X_moons, gamma = 15, n_components = 2, scale_data = FALSE)
proj_no_scale <- predict(kpca_no_scale, X_moons)

# With scaling
cat("Performing KPCA with scaling...\n")
kpca_with_scale <- fastKPCA(X_moons, gamma = 15, n_components = 2, scale_data = TRUE)
proj_with_scale <- predict(kpca_with_scale, X_moons)

# Compare results
par(mfrow = c(1, 2))

# Without scaling
plot(proj_no_scale, col = classes_moons, pch = 19,
     main = "Without Scaling",
     xlab = "PC1", ylab = "PC2")

# With scaling
plot(proj_with_scale, col = classes_moons, pch = 19,
     main = "With Scaling",
     xlab = "PC1", ylab = "PC2")

# Reset plot layout
par(mfrow = c(1, 1))

# =============================================================================
# PART 5: FINDING OPTIMAL GAMMA WITH CROSS-VALIDATION
# =============================================================================
cat("\n===== FINDING OPTIMAL GAMMA WITH CROSS-VALIDATION =====\n")

# Check if the cluster package is available for silhouette metric
has_cluster <- requireNamespace("cluster", quietly = TRUE)
if (has_cluster) {
  cat("Finding optimal gamma using silhouette metric...\n")
  result <- find_optimal_gamma(X_moons, classes_moons, 
                              gamma_range = c(1, 5, 10, 15, 20, 25), 
                              metric = "silhouette",
                              scale_data = TRUE)
  
  cat("Optimal gamma:", result$optimal_gamma, "\n")
  cat("Metrics for each gamma value:\n")
  print(result$metrics)
  
  # Visualize metrics for different gamma values
  plot(result$metrics$gamma, result$metrics$metric_value, 
       type = "b", col = "blue", lwd = 2,
       xlab = "Gamma", ylab = "Silhouette Score",
       main = "Effect of Gamma on Silhouette Score")
  
  # Use optimal gamma for KPCA
  cat("Performing KPCA with optimal gamma...\n")
  optimal_kpca <- fastKPCA(X_moons, gamma = result$optimal_gamma, 
                          n_components = 2, scale_data = TRUE)
  
  # Enhanced visualization with plot_kpca function
  cat("Creating enhanced visualization with decision boundaries...\n")
  plot_kpca(optimal_kpca, X_moons, classes_moons, 
           main = paste("Optimal KPCA (gamma =", result$optimal_gamma, ")"))
  
} else {
  cat("The 'cluster' package is needed for silhouette calculation. Using separation ratio instead.\n")
  
  # Use separation ratio instead
  cat("Finding optimal gamma using separation ratio...\n")
  result <- find_optimal_gamma(X_moons, classes_moons, 
                              gamma_range = c(1, 5, 10, 15, 20, 25), 
                              metric = "separation",
                              scale_data = TRUE)
  
  cat("Optimal gamma:", result$optimal_gamma, "\n")
  cat("Metrics for each gamma value:\n")
  print(result$metrics)
  
  # Visualize metrics for different gamma values
  plot(result$metrics$gamma, result$metrics$metric_value, 
       type = "b", col = "red", lwd = 2,
       xlab = "Gamma", ylab = "Separation Ratio",
       main = "Effect of Gamma on Separation Ratio")
  
  # Use optimal gamma for KPCA
  cat("Performing KPCA with optimal gamma...\n")
  optimal_kpca <- fastKPCA(X_moons, gamma = result$optimal_gamma, 
                          n_components = 2, scale_data = TRUE)
  
  # Standard projection plot
  optimal_projected <- predict(optimal_kpca, X_moons)
  plot(optimal_projected, col = classes_moons, pch = 19,
       main = paste("Optimal KPCA (gamma =", result$optimal_gamma, ")"),
       xlab = "PC1", ylab = "PC2")
}

# =============================================================================
# PART 6: ADVANCED VISUALIZATION OPTIONS
# =============================================================================
cat("\n===== ADVANCED VISUALIZATION OPTIONS =====\n")

if (has_cluster) {
  # Use our optimal kpca model from before
  
  # Basic plot with default settings
  cat("Creating basic visualization...\n")
  plot_kpca(optimal_kpca, X_moons, classes_moons, 
           main = "Basic Visualization")
  
  # Fine-grained contours
  cat("Creating visualization with fine-grained contours...\n")
  plot_kpca(optimal_kpca, X_moons, classes_moons, 
           contour_grid_size = 150,
           main = "Fine-Grained Contours")
  
  # Custom color palette
  cat("Creating visualization with custom colors...\n")
  custom_palette <- c("darkred", "darkblue")
  plot_kpca(optimal_kpca, X_moons, classes_moons,
           palette = custom_palette,
           main = "Custom Color Palette")
}

# =============================================================================
# PART 7: CONCENTRIC CIRCLES EXAMPLE
# =============================================================================
cat("\n===== CONCENTRIC CIRCLES EXAMPLE =====\n")

# Plot original circles dataset
cat("Visualizing original circles dataset...\n")
plot(X_circles, col = classes_circles, pch = 19, 
     main = "Original Concentric Circles Dataset",
     xlab = "X1", ylab = "X2")

# Find optimal gamma for circles
cat("Finding optimal gamma for circles dataset...\n")
circles_result <- find_optimal_gamma(X_circles, classes_circles, 
                                    gamma_range = c(0.5, 1, 2, 5, 10), 
                                    metric = "separation",
                                    scale_data = TRUE)

cat("Optimal gamma for circles:", circles_result$optimal_gamma, "\n")
cat("Metrics for each gamma value:\n")
print(circles_result$metrics)

# Run KPCA with optimal gamma
cat("Performing KPCA with optimal gamma...\n")
kpca_circles <- fastKPCA(X_circles, gamma = circles_result$optimal_gamma, 
                        n_components = 2, scale_data = TRUE)

# Plot circles projection
cat("Visualizing KPCA projection of circles...\n")
if (has_cluster) {
  # Enhanced visualization
  plot_kpca(kpca_circles, X_circles, classes_circles, 
           main = paste("KPCA Projection (gamma =", circles_result$optimal_gamma, ")"))
} else {
  # Standard projection plot
  projected_circles <- predict(kpca_circles, X_circles)
  plot(projected_circles, col = classes_circles, pch = 19,
       main = paste("KPCA Projection (gamma =", circles_result$optimal_gamma, ")"),
       xlab = "PC1", ylab = "PC2")
}

# =============================================================================
# PART 8: BENCHMARKING PERFORMANCE
# =============================================================================
cat("\n===== BENCHMARKING PERFORMANCE =====\n")
cat("Benchmarking RcppKPCA for different sample sizes...\n")

sizes <- c(100, 500, 1000, 2000)
times <- numeric(length(sizes))
times_with_scale <- numeric(length(sizes))

for (i in seq_along(sizes)) {
  n <- sizes[i]
  
  # Generate larger datasets
  n_half <- n / 2
  
  # Using the more challenging randomized approach
  theta_out <- runif(n_half, 0, pi)
  r_out <- 1 + rnorm(n_half, 0, 0.1)
  x1 <- r_out * cos(theta_out)
  y1 <- r_out * sin(theta_out)
  
  theta_in <- runif(n_half, 0, pi)
  r_in <- 1 + rnorm(n_half, 0, 0.1)
  x2 <- 0.5 - r_in * cos(theta_in)
  y2 <- 0.5 - r_in * sin(theta_in) - 0.5
  
  X_large <- rbind(
    cbind(x1, y1),
    cbind(x2, y2)
  )
  
  # Time the KPCA computation without scaling
  cat(sprintf("Testing sample size %d without scaling...\n", n))
  start_time <- Sys.time()
  kpca_result <- fastKPCA(X_large, gamma = 15, n_components = 2, scale_data = FALSE)
  projected <- predict(kpca_result, X_large)
  end_time <- Sys.time()
  
  times[i] <- as.numeric(difftime(end_time, start_time, units = "secs"))
  
  # Time the KPCA computation with scaling
  cat(sprintf("Testing sample size %d with scaling...\n", n))
  start_time <- Sys.time()
  kpca_result <- fastKPCA(X_large, gamma = 15, n_components = 2, scale_data = TRUE)
  projected <- predict(kpca_result, X_large)
  end_time <- Sys.time()
  
  times_with_scale[i] <- as.numeric(difftime(end_time, start_time, units = "secs"))
  
  cat(sprintf("Sample size %d: %.4f seconds (no scaling), %.4f seconds (with scaling)\n", 
              n, times[i], times_with_scale[i]))
}

# Plot the timing results
plot(sizes, times, type = "b", col = "blue", lwd = 2,
     xlab = "Sample Size", ylab = "Time (seconds)",
     main = "RcppKPCA Performance",
     ylim = range(c(times, times_with_scale)) * 1.1)
lines(sizes, times_with_scale, type = "b", col = "red", lwd = 2)
legend("topleft", 
       legend = c("Without Scaling", "With Scaling"),
       col = c("blue", "red"),
       lty = 1,
       lwd = 2)

# Show the features of the output object
cat("\n===== OUTPUT OBJECT STRUCTURE =====\n")
cat("The fastKPCA function returns an S3 object with the following structure:\n")
print(str(kpca_result))

cat("\n===== DEMO COMPLETED =====\n")
cat("Thank you for exploring the RcppKPCA package!\n") 