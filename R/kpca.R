#' Fast Kernel Principal Component Analysis (KPCA)
#'
#' Performs Kernel PCA using the Gaussian (RBF) kernel.
#' This function leverages RcppArmadillo for efficient computation.
#'
#' @param X A numeric matrix or data frame (observations x features).
#' @param gamma The parameter for the RBF kernel (exp(-gamma * ||x - y||^2)).
#' @param n_components The number of principal components to return.
#' @param scale_data Logical, whether to scale the data before applying KPCA (default: FALSE).
#'
#' @return An object of class \code{fastKPCA} containing:
#' \describe{
#'   \item{eigenvalues}{A numeric vector of the top positive eigenvalues of the centered kernel matrix.}
#'   \item{pcv}{A matrix containing the principal component vectors (alpha), which are the scaled eigenvectors \eqn{\alpha = V \Lambda^{-1/2}}. Used for projection.}
#'   \item{gamma}{The kernel parameter used.}
#'   \item{n_components_requested}{The number of components initially requested.}
#'   \item{n_components_returned}{The actual number of components returned (after filtering non-positive eigenvalues).}
#'   \item{X_train}{The original training data matrix used.}
#'   \item{K_train_col_sums}{Column sums of the uncentered training kernel matrix K.}
#'   \item{K_train_total_sum}{Sum of all elements in the uncentered training kernel matrix K.}
#'   \item{n_train}{Number of training samples.}
#'   \item{scaling_params}{Scaling parameters if scale_data=TRUE.}
#' }
#'
#' @examples
#' # Generate sample data
#' set.seed(123)
#' X_train <- rbind(matrix(rnorm(100, mean = 0), 50, 2),
#'                  matrix(rnorm(100, mean = 3), 50, 2))
#'
#' # Perform fast KPCA
#' kpca_result <- fastKPCA(X_train, gamma = 0.1, n_components = 2)
#'
#' # Print results
#' print(kpca_result)
#'
#' # Generate new data
#' X_new <- matrix(rnorm(20, mean = 1.5), 10, 2)
#'
#' # Project new data
#' projected_data <- predict(kpca_result, X_new)
#' print(head(projected_data))
#'
#' @export
#' @useDynLib RcppKPCA, .registration = TRUE
#' @importFrom Rcpp sourceCpp
#' @importFrom stats sd
fastKPCA <- function(X, gamma, n_components, scale_data = FALSE) {
  # Input validation
  if (!is.matrix(X) && !is.data.frame(X)) {
    stop("Input X must be a matrix or data frame.")
  }
  if (is.data.frame(X)) {
      # Check if all columns are numeric before converting
      if (!all(sapply(X, is.numeric))) {
          stop("All columns in the data frame X must be numeric.")
      }
      X_mat <- as.matrix(X)
  } else {
      X_mat <- X
  }
  if (!is.numeric(X_mat)) {
      stop("Input X must contain numeric values.")
  }
  if (!is.numeric(gamma) || length(gamma) != 1 || gamma <= 0) {
    stop("gamma must be a single positive numeric value.")
  }
  if (!is.numeric(n_components) || length(n_components) != 1 || n_components <= 0 || floor(n_components) != n_components) {
    stop("n_components must be a single positive integer.")
  }
  if (!is.logical(scale_data)) {
    stop("scale_data must be logical (TRUE/FALSE).")
  }

  n_train <- nrow(X_mat)
  if (n_train == 0) stop("Input matrix X has 0 rows.")
  n_features <- ncol(X_mat)
  if (n_features == 0) stop("Input matrix X has 0 columns.")

  # Apply scaling if requested
  scaling_params <- NULL
  if (scale_data) {
    # Center and scale the data
    scaling_center <- colMeans(X_mat)
    scaling_scale <- apply(X_mat, 2, sd)
    # Handle zero or near-zero standard deviations
    scaling_scale[scaling_scale < 1e-10] <- 1
    X_mat <- scale(X_mat, center = scaling_center, scale = scaling_scale)
    scaling_params <- list(center = scaling_center, scale = scaling_scale)
  }

  # 1. Calculate Kernel Matrix (using C++ function)
  K_train <- rbf_kernel(X_mat, gamma)

  # 2. Calculate summary stats needed for centering during prediction
  K_train_col_sums <- colSums(K_train)
  K_train_total_sum <- sum(K_train)

  # 3. Center Kernel Matrix (using C++ function)
  Kc_train <- center_kernel_matrix(K_train)

  # 4. Perform Eigenvalue Decomposition (using C++ function)
  eigen_result <- kpca_eigen_decomp(Kc_train, as.integer(n_components))

  # 5. Structure the output object
  result <- list(
    eigenvalues = eigen_result$eigenvalues,
    pcv = eigen_result$pcv,
    gamma = gamma,
    n_components_requested = n_components,
    n_components_returned = length(eigen_result$eigenvalues),
    X_train = X_mat, # Store the original training data
    K_train_col_sums = K_train_col_sums, # Store for predict
    K_train_total_sum = K_train_total_sum, # Store for predict
    n_train = n_train, # Store for predict
    scaling_params = scaling_params # Store scaling parameters if used
  )

  class(result) <- "fastKPCA"
  return(result)
}

#' Print method for fastKPCA results
#'
#' @param x An object of class \code{fastKPCA}.
#' @param ... Additional arguments passed to print.
#' @keywords internal
#' @export
#' @importFrom utils head
print.fastKPCA <- function(x, ...) {
  cat("Fast Kernel PCA Results\n")
  cat("-----------------------\n")
  cat("Training Samples:", x$n_train, "\n")
  cat("Input Features:", ncol(x$X_train), "\n")
  cat("Kernel Parameter (gamma):", x$gamma, "\n")
  cat("Data Scaled:", !is.null(x$scaling_params), "\n")
  cat("Components Requested:", x$n_components_requested, "\n")
  cat("Components Returned (Positive Eigenvalues):", x$n_components_returned, "\n\n")
  
  if (x$n_components_returned > 0) {
      cat("Top Eigenvalues:\n")
      print(head(x$eigenvalues))
      if (length(x$eigenvalues) > 6) cat("...\n")
      
      cat("\nPrincipal Component Vectors (alpha) dimensions:", paste(dim(x$pcv), collapse = " x "), "\n")
  } else {
      cat("No components returned (check warnings, input data, or gamma).\n")
  }
  invisible(x)
}

#' Predict method for fastKPCA objects
#'
#' Projects new data points onto the kernel principal components.
#'
#' @param object A fitted object of class \code{fastKPCA}.
#' @param newdata A numeric matrix or data frame containing the new data points 
#'   (observations x features). Must have the same number of features as the training data.
#' @param ... Additional arguments (currently ignored).
#'
#' @return A matrix containing the projections of the new data onto the principal components.
#'   The number of columns corresponds to \code{n_components_returned} in the \code{object}.
#'
#' @details
#' The projection is calculated as \eqn{K_{new, centered} \alpha}, where 
#' \eqn{K_{new, centered}} is the kernel matrix between the new data and the 
#' original training data, centered using the statistics from the training kernel matrix,
#' and \eqn{\alpha} are the scaled principal component vectors stored in \code{object$pcv}.
#'
#' Centering formula for the new kernel matrix \eqn{K_{new}} (dimensions m x n):
#' \eqn{K_{new, centered} = K_{new} - \mathbf{1}_{m x n} diag(\frac{1}{n} \sum_{j=1}^n K_{:,j}) - diag(\frac{1}{n} \sum_{i=1}^n K_{new, i:}) \mathbf{1}_{m x n} + \mathbf{1}_{m x n} (\frac{1}{n^2} \sum_{i,j} K_{i,j})} \\ 
#' where \eqn{K} is the original training kernel matrix, \eqn{K_{new}} is the kernel matrix between \code{newdata} and \code{object$X_train},
#' \eqn{\mathbf{1}_{m x n}} is an m x n matrix of ones, and n is the number of training samples.
#' This formula is adapted from Scholkopf et al. (1998) and applied efficiently using saved summary statistics.
#'
#' @export
predict.fastKPCA <- function(object, newdata, ...) {
  # Input validation
  if (!inherits(object, "fastKPCA")) {
    stop("Input 'object' must be of class 'fastKPCA'.")
  }
  if (missing(newdata)) {
      stop("Argument 'newdata' is missing.")
  }
  if (!is.matrix(newdata) && !is.data.frame(newdata)) {
    stop("Input 'newdata' must be a matrix or data frame.")
  }
  if (is.data.frame(newdata)) {
      if (!all(sapply(newdata, is.numeric))) {
          stop("All columns in the data frame 'newdata' must be numeric.")
      }
      X_new <- as.matrix(newdata)
  } else {
      X_new <- newdata
  }
  if (!is.numeric(X_new)) {
    stop("Input 'newdata' must contain numeric values.")
  }
  if (ncol(X_new) != ncol(object$X_train)) {
    stop(sprintf("Number of columns in 'newdata' (%d) does not match training data (%d).",
                 ncol(X_new), ncol(object$X_train)))
  }
  if (object$n_components_returned == 0 || is.null(object$pcv) || nrow(object$pcv) == 0) {
      warning("The fastKPCA object contains no principal components. Returning empty matrix.")
      return(matrix(numeric(0), nrow = nrow(X_new), ncol = 0))
  }

  # Apply the same scaling if it was used during training
  if (!is.null(object$scaling_params)) {
    X_new <- scale(X_new, 
                  center = object$scaling_params$center, 
                  scale = object$scaling_params$scale)
  }

  m <- nrow(X_new) # Number of new samples
  n <- object$n_train # Number of training samples

  # 1. Calculate Kernel Matrix between new data and training data
  K_new <- rbf_kernel_predict(X_new, object$X_train, object$gamma)

  # 2. Center the new kernel matrix K_new (m x n)
  # Using the formula: K_new_centered = K_new - 1_m * K_train_col_means'/n - K_new_row_means * 1_n'/n + K_train_total_sum / n^2 * 1_m * 1_n'
  # where 1_m is m x 1 vector of ones, 1_n is n x 1 vector of ones
  # K_train_col_means is 1 x n row vector (needs transpose in formula)
  # K_train_col_sums is n x 1 col vector
  # K_new_row_sums is m x 1 col vector

  ones_m <- matrix(1, nrow = m, ncol = 1)
  ones_n <- matrix(1, nrow = 1, ncol = n)
  
  K_new_row_sums <- rowSums(K_new)

  # Efficient centering using outer products and recycling
  # term2: Subtracts the mean column vector of K_train from each column of K_new
  # We need K_train_col_sums as a 1 x n row vector for the multiplication
  term2 <- ones_m %*% t(object$K_train_col_sums / n) 
  
  # term3: Subtracts the mean row vector of K_new from each row of K_new
  term3 <- (K_new_row_sums / n) %*% ones_n
  
  # term4: Adds back the overall mean of K_train
  term4 <- matrix(object$K_train_total_sum / (n * n), nrow = m, ncol = n)
  
  K_new_centered <- K_new - term2 - term3 + term4

  # 3. Project onto principal components
  # Projection = K_new_centered %*% alpha
  projected_data <- K_new_centered %*% object$pcv

  colnames(projected_data) <- paste0("PC", 1:ncol(projected_data))
  return(projected_data)
}

#' Find optimal gamma parameter for KPCA through cross-validation
#'
#' Uses a separation metric to find the optimal gamma value for a given dataset.
#'
#' @param X A numeric matrix or data frame (observations x features).
#' @param y A factor or vector of class labels.
#' @param gamma_range A numeric vector of gamma values to evaluate.
#' @param n_components The number of principal components to use.
#' @param metric The separation metric to use. Either "silhouette" or "separation" (default: "silhouette").
#' @param scale_data Logical, whether to scale the data before applying KPCA (default: TRUE).
#'
#' @return A list containing:
#' \describe{
#'   \item{optimal_gamma}{The gamma value that maximizes the chosen metric.}
#'   \item{metrics}{A data frame of all gamma values and their corresponding metrics.}
#' }
#'
#' @examples
#' # Generate sample data
#' set.seed(123)
#' X <- rbind(matrix(rnorm(100, mean = 0), 50, 2),
#'           matrix(rnorm(100, mean = 3), 50, 2))
#' y <- factor(c(rep(1, 50), rep(2, 50)))
#'
#' # Find optimal gamma
#' result <- find_optimal_gamma(X, y, gamma_range = c(0.1, 0.5, 1, 5, 10))
#' print(result$optimal_gamma)
#'
#' @export
#' @importFrom stats dist predict
find_optimal_gamma <- function(X, y, gamma_range = c(1, 5, 10, 15, 20, 25), 
                              n_components = 2, metric = "silhouette", 
                              scale_data = TRUE) {
  
  if (!inherits(y, "factor")) {
    y <- as.factor(y)
  }
  
  if (!metric %in% c("silhouette", "separation")) {
    stop("Metric must be either 'silhouette' or 'separation'.")
  }
  
  # Function to calculate separation ratio
  calculate_separation_ratio <- function(projected, class_labels) {
    class_levels <- levels(class_labels)
    
    # Calculate class indices
    class_A_indices <- which(class_labels == class_levels[1])
    class_B_indices <- which(class_labels == class_levels[2])
    
    # Calculate centroids
    centroid_A <- colMeans(projected[class_A_indices, , drop=FALSE])
    centroid_B <- colMeans(projected[class_B_indices, , drop=FALSE])
    
    # Calculate between-class distance
    between_class_dist <- sqrt(sum((centroid_A - centroid_B)^2))
    
    # Calculate within-class distances
    within_A_dist <- mean(apply(projected[class_A_indices, , drop=FALSE], 1, 
                                function(x) sqrt(sum((x - centroid_A)^2))))
    within_B_dist <- mean(apply(projected[class_B_indices, , drop=FALSE], 1, 
                                function(x) sqrt(sum((x - centroid_B)^2))))
    
    # Calculate separation ratio (higher is better)
    separation_ratio <- between_class_dist / (within_A_dist + within_B_dist)
    
    return(separation_ratio)
  }
  
  # Evaluate each gamma
  results <- data.frame(gamma = gamma_range, metric_value = NA)
  
  for(i in seq_along(gamma_range)) {
    gamma <- gamma_range[i]
    
    # Perform KPCA
    tryCatch({
      kpca_result <- fastKPCA(X, gamma = gamma, n_components = n_components, 
                             scale_data = scale_data)
      
      if (kpca_result$n_components_returned > 0) {
        # Project the data
        projected <- predict(kpca_result, X)
        
        # Calculate metric
        if (metric == "silhouette") {
          # Check if cluster package is installed
          if (!requireNamespace("cluster", quietly = TRUE)) {
            stop("The 'cluster' package is needed for silhouette calculation. Please install it.")
          }
          
          # Calculate silhouette coefficient
          sil <- cluster::silhouette(as.numeric(y), dist(projected))
          metric_value <- mean(sil[,3])
        } else {
          # Calculate separation ratio
          metric_value <- calculate_separation_ratio(projected, y)
        }
        
        results$metric_value[i] <- metric_value
      }
    }, error = function(e) {
      warning(paste("Error evaluating gamma =", gamma, ":", e$message))
      results$metric_value[i] <- NA
    })
  }
  
  # Find optimal gamma
  valid_results <- !is.na(results$metric_value)
  if (sum(valid_results) == 0) {
    stop("No valid gamma values were found. Try a different range or check your data.")
  }
  
  optimal_index <- which.max(results$metric_value[valid_results])
  optimal_gamma <- results$gamma[valid_results][optimal_index]
  
  return(list(
    optimal_gamma = optimal_gamma,
    metrics = results
  ))
}

#' Plot KPCA Projections with Decision Boundaries
#'
#' Creates a visualization of KPCA projections with optional decision boundary contours.
#'
#' @param kpca_result A fitted object of class \code{fastKPCA}.
#' @param X The original data matrix used for training or new data to project.
#' @param y A factor or vector of class labels.
#' @param plot_contours Logical, whether to plot contour lines (default: TRUE).
#' @param contour_grid_size Integer, number of grid points in each dimension for contour plotting (default: 100).
#' @param main Title for the plot (default: "KPCA Projection").
#' @param palette Vector of colors for different classes (default: NULL, uses standard palette).
#' @param ... Additional arguments passed to plot.
#'
#' @return Invisibly returns a list with projected data and contour information if requested.
#'
#' @examples
#' # Generate sample data
#' set.seed(123)
#' X <- rbind(matrix(rnorm(100, mean = 0), 50, 2),
#'           matrix(rnorm(100, mean = 3), 50, 2))
#' y <- factor(c(rep(1, 50), rep(2, 50)))
#'
#' # Perform KPCA
#' kpca_result <- fastKPCA(X, gamma = 5, n_components = 2)
#'
#' # Plot projections with decision boundaries
#' plot_kpca(kpca_result, X, y)
#'
#' @export
#' @importFrom graphics plot points contour legend
#' @importFrom grDevices colorRampPalette
#' @importFrom stats predict
plot_kpca <- function(kpca_result, X, y, plot_contours = TRUE, 
                      contour_grid_size = 100, main = "KPCA Projection",
                      palette = NULL, ...) {
  
  if (!inherits(kpca_result, "fastKPCA")) {
    stop("Input 'kpca_result' must be of class 'fastKPCA'.")
  }
  
  if (!is.factor(y)) {
    y <- as.factor(y)
  }
  
  # Project data
  projected <- predict(kpca_result, X)
  
  # Define colors if not provided
  if (is.null(palette)) {
    palette <- c("red", "blue", "green", "purple", "orange")
  }
  # Ensure palette is long enough
  if (length(levels(y)) > length(palette)) {
    palette <- rep(palette, length.out = length(levels(y)))
  }
  
  # Create base plot
  plot(projected[, 1], projected[, 2], 
       col = palette[as.numeric(y)], 
       pch = 19,
       main = main,
       xlab = "PC1", 
       ylab = "PC2",
       ...)
  
  # Add contour lines if requested
  contour_data <- NULL
  if (plot_contours && ncol(projected) >= 2) {
    # Get bounds for grid
    x_range <- range(projected[, 1])
    y_range <- range(projected[, 2])
    
    # Add some padding
    x_padding <- diff(x_range) * 0.1
    y_padding <- diff(y_range) * 0.1
    
    x_grid <- seq(x_range[1] - x_padding, x_range[2] + x_padding, length.out = contour_grid_size)
    y_grid <- seq(y_range[1] - y_padding, y_range[2] + y_padding, length.out = contour_grid_size)
    
    # Create grid of points
    grid_points <- expand.grid(PC1 = x_grid, PC2 = y_grid)
    
    # Calculate density for each class
    class_densities <- list()
    
    for (class_idx in seq_along(levels(y))) {
      class_label <- levels(y)[class_idx]
      class_points <- projected[y == class_label, , drop = FALSE]
      
      if (nrow(class_points) > 0) {
        # Calculate KDE for this class
        density_est <- kde2d_custom(
          class_points[, 1], 
          class_points[, 2], 
          n = contour_grid_size,
          lims = c(min(x_grid), max(x_grid), min(y_grid), max(y_grid))
        )
        
        # Add contour lines
        contour(
          density_est$x, 
          density_est$y, 
          density_est$z,
          add = TRUE,
          col = palette[class_idx],
          lwd = 1.5,
          drawlabels = FALSE
        )
        
        class_densities[[class_label]] <- density_est
      }
    }
    
    contour_data <- list(
      x_grid = x_grid,
      y_grid = y_grid,
      class_densities = class_densities
    )
  }
  
  # Add legend
  legend("topright", 
         legend = levels(y),
         col = palette[1:length(levels(y))],
         pch = 19,
         title = "Classes")
  
  # Return projected data and contour info invisibly
  invisible(list(
    projected = projected,
    contours = contour_data
  ))
}

#' Custom implementation of kde2d
#'
#' A simplified version of MASS::kde2d for density estimation.
#'
#' @param x x-coordinates
#' @param y y-coordinates
#' @param h bandwidth (smoothing parameter)
#' @param n number of grid points in each dimension
#' @param lims limits of the grid
#'
#' @return A list with components x, y, and z for plotting with contour
#'
#' @keywords internal
#' @importFrom stats dnorm
kde2d_custom <- function(x, y, h, n = 25, lims = c(range(x), range(y))) {
  nx <- length(x)
  if (nx < 2) 
    stop("Need at least 2 points")
  
  if (missing(h)) {
    h <- c(bandwidth_nrd(x), bandwidth_nrd(y))
  } else if (is.numeric(h) && length(h) == 1) {
    h <- rep(h, 2)
  }
  
  gx <- seq(lims[1], lims[2], length.out = n)
  gy <- seq(lims[3], lims[4], length.out = n)
  
  # Compute 2D density
  h <- h * 4
  ax <- outer(gx, x, "-") / h[1]
  ay <- outer(gy, y, "-") / h[2]
  
  z <- matrix(0, n, n)
  for (i in 1:nx) {
    z <- z + outer(dnorm(ax[, i]), dnorm(ay[, i]))
  }
  
  z <- z / (nx * h[1] * h[2])
  
  return(list(x = gx, y = gy, z = z))
}

#' Bandwidth estimator for kernel density
#'
#' @param x numeric vector
#' @return estimated bandwidth
#' @keywords internal
#' @importFrom stats quantile sd
bandwidth_nrd <- function(x) {
  r <- quantile(x, c(0.25, 0.75))
  h <- 0.9 * min(sd(x), (r[2] - r[1])/1.34) * length(x)^(-0.2)
  return(max(h, 0.001 * diff(range(x))))
} 