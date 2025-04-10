# RcppKPCA: Fast Kernel PCA in R

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

A computationally efficient implementation of Kernel Principal Component Analysis (KPCA) using the Gaussian (RBF) kernel. This package leverages Rcpp and the Armadillo C++ library for fast matrix operations.

## Features

- Fast KPCA implementation using C++ and Armadillo
- RBF kernel for non-linear dimensionality reduction
- Significantly faster than other R implementations
- Data scaling option for improved performance
- Cross-validation for optimal gamma parameter selection
- Enhanced visualization with contour plots
- Optimized kernel matrix centering

## Quick Start

```r
# Install the package
# install.packages("devtools")
devtools::install_github("Javdat/RcppKPCA")

# Load the package
library(RcppKPCA)

# Generate half-moon dataset
set.seed(123)
n <- 200
n_half <- n/2

# Create half-moons using randomized approach
theta_out <- runif(n_half, 0, pi)
r_out <- 1 + rnorm(n_half, 0, 0.1)
x1 <- r_out * cos(theta_out)
y1 <- r_out * sin(theta_out)

theta_in <- runif(n_half, 0, pi)
r_in <- 1 + rnorm(n_half, 0, 0.1)
x2 <- 0.5 - r_in * cos(theta_in)
y2 <- 0.5 - r_in * sin(theta_in) - 0.5

X <- rbind(cbind(x1, y1), cbind(x2, y2))
classes <- factor(c(rep(1, n_half), rep(2, n_half)))

# Perform KPCA with recommended settings
kpca_result <- fastKPCA(X, gamma = 15, n_components = 2, scale_data = TRUE)

# Project data and visualize results
projected <- predict(kpca_result, X)
plot(projected, col = classes, pch = 19,
     main = "KPCA Projection", xlab = "PC1", ylab = "PC2")
```

## Installation

### From GitHub

```r
# install.packages("devtools")
devtools::install_github("Javdat/RcppKPCA")
```

### From Source

```r
# Clone the repository
git clone https://github.com/Javdat/RcppKPCA.git

# Install the package
install.packages("RcppKPCA", repos = NULL, type = "source")
```

## Detailed Usage Example

### Step 1: Data Preparation and Basic KPCA

```r
library(RcppKPCA)

# Create sample data (two half-moons)
set.seed(123)
n_samples <- 200
noise <- 0.1

# Generate the two half-moons
n_samples_out <- n_samples %/% 2
n_samples_in <- n_samples - n_samples_out

# First moon (outer half-moon)
theta_out <- runif(n_samples_out, 0, pi)
r_out <- 1 + rnorm(n_samples_out, 0, noise)
outer_circ_x <- r_out * cos(theta_out)
outer_circ_y <- r_out * sin(theta_out)

# Second moon (inner half-moon)
theta_in <- runif(n_samples_in, 0, pi)
r_in <- 1 + rnorm(n_samples_in, 0, noise)
inner_circ_x <- 1 - r_in * cos(theta_in)
inner_circ_y <- 1 - r_in * sin(theta_in) - 0.5

# Combine the moons
X <- rbind(
  cbind(outer_circ_x, outer_circ_y),
  cbind(inner_circ_x, inner_circ_y)
)

# Create class labels
classes <- factor(c(rep(1, n_samples_out), rep(2, n_samples_in)))

# Plot original data
plot(X, col = classes, pch = 19, 
     main = "Original Half-Moons Dataset",
     xlab = "X1", ylab = "X2")

# Run KPCA with data scaling (recommended)
kpca_result <- fastKPCA(X, gamma = 15, n_components = 2, scale_data = TRUE)

# Project the data
projected <- predict(kpca_result, X)

# Plot the KPCA projection
plot(projected, col = classes, pch = 19, 
     main = "KPCA Projection of Half-Moons Dataset",
     xlab = "PC1", ylab = "PC2")
```

### Step 2: Finding the Optimal Gamma Parameter

One of the key new features is the ability to automatically find the optimal gamma parameter through cross-validation:

```r
# Find optimal gamma using silhouette score
optimal_result <- find_optimal_gamma(X, classes, 
                                    gamma_range = c(1, 5, 10, 15, 20, 25), 
                                    n_components = 2,
                                    metric = "silhouette",
                                    scale_data = TRUE)

# Display results
cat("Optimal gamma:", optimal_result$optimal_gamma, "\n")
print(optimal_result$metrics)

# Use the optimal gamma for KPCA
optimal_kpca <- fastKPCA(X, gamma = optimal_result$optimal_gamma, 
                         n_components = 2, scale_data = TRUE)

# Project data
optimal_projected <- predict(optimal_kpca, X)

# Compare results with and without optimal gamma
par(mfrow = c(1, 2))

# Standard gamma
plot(projected, col = classes, pch = 19,
     main = "Standard gamma = 15",
     xlab = "PC1", ylab = "PC2")

# Optimal gamma
plot(optimal_projected, col = classes, pch = 19,
     main = paste("Optimal gamma =", optimal_result$optimal_gamma),
     xlab = "PC1", ylab = "PC2")

par(mfrow = c(1, 1))
```

### Step 3: Enhanced Visualization with Decision Boundaries

The new plotting function creates enhanced visualizations with contour lines showing decision boundaries:

```r
# Create enhanced visualization with decision boundaries
plot_kpca(optimal_kpca, X, classes, 
          main = paste("KPCA with Optimal Gamma =", optimal_result$optimal_gamma))

# Try with different contour grid size for finer detail
plot_kpca(optimal_kpca, X, classes, 
          contour_grid_size = 150,
          main = "KPCA with Fine-Grained Contours")

# Custom color palette
custom_palette <- c("darkred", "darkblue")
plot_kpca(optimal_kpca, X, classes,
          palette = custom_palette,
          main = "KPCA with Custom Color Palette")
```

### Step 4: Comparison with and without Data Scaling

Data scaling can significantly improve results, especially with real-world datasets:

```r
# Without scaling
kpca_no_scale <- fastKPCA(X, gamma = 15, n_components = 2, scale_data = FALSE)
proj_no_scale <- predict(kpca_no_scale, X)

# With scaling
kpca_with_scale <- fastKPCA(X, gamma = 15, n_components = 2, scale_data = TRUE)
proj_with_scale <- predict(kpca_with_scale, X)

# Compare results
par(mfrow = c(1, 2))

# Without scaling
plot(proj_no_scale, col = classes, pch = 19,
     main = "Without Scaling",
     xlab = "PC1", ylab = "PC2")

# With scaling
plot(proj_with_scale, col = classes, pch = 19,
     main = "With Scaling",
     xlab = "PC1", ylab = "PC2")

par(mfrow = c(1, 1))
```

## Effect of Gamma Parameter

The gamma parameter controls the width of the RBF kernel. Here's how different values affect the projection:

```r
# Try different gamma values (higher values work better for half-moons)
gamma_values <- c(5, 10, 15, 20)
par(mfrow = c(2, 2))

for (gamma in gamma_values) {
  # Perform KPCA
  kpca_result <- fastKPCA(X, gamma = gamma, n_components = 2, scale_data = TRUE)
  
  # Project the data
  projected <- predict(kpca_result, X)
  
  # Plot the projected data
  plot(projected, col = classes, pch = 19,
       main = paste("KPCA Projection (gamma =", gamma, ")"),
       xlab = "PC1", ylab = "PC2")
}

par(mfrow = c(1, 1))
```

## Finding Optimal Gamma with Cross-Validation

The package includes a cross-validation function to find the optimal gamma value:

```r
# Find optimal gamma using silhouette score
optimal_result <- find_optimal_gamma(X, classes, 
                                    gamma_range = c(1, 5, 10, 15, 20, 25), 
                                    n_components = 2,
                                    metric = "silhouette")
print(optimal_result$optimal_gamma)

# Visualize the metrics for different gamma values
plot(optimal_result$metrics$gamma, optimal_result$metrics$metric_value, 
     type = "b", col = "blue",
     xlab = "Gamma", ylab = "Silhouette Score",
     main = "Effect of Gamma on Separation Quality")

# Use the optimal gamma for KPCA
optimal_kpca <- fastKPCA(X, gamma = optimal_result$optimal_gamma, 
                         n_components = 2, scale_data = TRUE)

# Plot the result with contours to show decision boundaries
plot_kpca(optimal_kpca, X, classes, 
          main = paste("Optimal KPCA (gamma =", optimal_result$optimal_gamma, ")"))
```

## Complex Datasets

RcppKPCA excels at challenging non-linear datasets where standard PCA fails:

```r
# Generate concentric circles
n_samples <- 200
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

X_circles <- rbind(
  cbind(x_inner, y_inner),
  cbind(x_outer, y_outer)
)
classes_circles <- factor(c(rep(1, n_samples_inner), rep(2, n_samples_outer)))

# Find optimal gamma for circles dataset
circles_gamma <- find_optimal_gamma(X_circles, classes_circles, 
                                   gamma_range = c(0.5, 1, 5, 10, 15))$optimal_gamma

# Run KPCA on circles dataset with optimal gamma
kpca_circles <- fastKPCA(X_circles, gamma = circles_gamma, 
                        n_components = 2, scale_data = TRUE)

# Create enhanced visualization with decision boundaries
plot_kpca(kpca_circles, X_circles, classes_circles,
          main = paste("KPCA Projection of Concentric Circles (gamma =", circles_gamma, ")"))
```

## Performance Comparison

RcppKPCA significantly outperforms other implementations:

```r
library(microbenchmark)
library(kernlab)

# Benchmark the implementations
X <- as.matrix(X)
gamma_value <- 15
n_components <- 2

timing_comparison <- microbenchmark(
  RcppKPCA = {
    kpca_result <- fastKPCA(X, gamma = gamma_value, n_components = n_components)
    projected <- predict(kpca_result, X)
  },
  kernlab = {
    kpca_result <- kernlab::kpca(X, kernel = "rbfdot", 
                                 kpar = list(sigma = gamma_value/2),
                                 features = n_components)
    projected <- kernlab::pcv(kpca_result)
  },
  times = 10
)

print(timing_comparison)
```

## Demos and Vignettes

For more examples and detailed comparisons, check out:

```r
# Run the basic demo
demo("basic_kpca", package = "RcppKPCA")

# View the vignette
vignette("kpca-comparison", package = "RcppKPCA")
```

## How It Works

RcppKPCA performs kernel PCA in the following steps:

1. (Optional) Scale the input data for better performance
2. Compute the kernel matrix K using the RBF kernel
3. Center the kernel matrix in feature space using efficient matrix operations
4. Perform eigendecomposition of the centered kernel matrix
5. Project the data onto the principal components

The implementation uses Armadillo for efficient matrix operations and eigendecomposition.

## Key Improvements

Recent updates to the package include:

- Improved kernel matrix centering using matrix operations
- Data scaling option for better performance
- Cross-validation for optimal gamma parameter selection
- Enhanced visualization with decision boundaries
- Higher default gamma values for half-moon datasets (10-15)

## References

- Schölkopf, B., Smola, A. and Müller, K.R., 1998. Nonlinear component analysis as a kernel eigenvalue problem. Neural computation, 10(5), pp.1299-1319.
- Eddelbuettel, D. and Sanderson, C., 2014. RcppArmadillo: Accelerating R with high-performance C++ linear algebra. Computational statistics & data analysis, 71, pp.1054-1063.
- Raschka, S., 2014. Kernel Principal Component Analysis. https://sebastianraschka.com/Articles/2014_kernel_pca.html