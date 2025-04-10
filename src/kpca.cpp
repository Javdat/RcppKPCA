#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;

// Function to calculate the Gaussian (RBF) kernel matrix (Train x Train)
// K(x, y) = exp(-gamma * ||x - y||^2)
// [[Rcpp::export]]
arma::mat rbf_kernel(const arma::mat& X, double gamma) {
    int n = X.n_rows;
    arma::mat K(n, n, fill::zeros);
    for (int i = 0; i < n; ++i) {
        // Symmetric matrix: only compute upper triangle + diagonal
        for (int j = i; j < n; ++j) {
            double squared_dist = accu(square(X.row(i) - X.row(j)));
            double kv = exp(-gamma * squared_dist);
            K(i, j) = kv;
            K(j, i) = kv; 
        }
    }
    return K;
}

// Function to calculate the Gaussian (RBF) kernel matrix (New x Train)
// K(x_new, x_train) = exp(-gamma * ||x_new - x_train||^2)
// [[Rcpp::export]]
arma::mat rbf_kernel_predict(const arma::mat& X_new, const arma::mat& X_train, double gamma) {
    int m = X_new.n_rows; // Number of new samples
    int n = X_train.n_rows; // Number of training samples
    int d = X_new.n_cols;

    if (X_train.n_cols != d) {
        Rcpp::stop("X_new and X_train must have the same number of columns (features).");
    }

    arma::mat K_new(m, n);

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            double squared_dist = accu(square(X_new.row(i) - X_train.row(j)));
            K_new(i, j) = exp(-gamma * squared_dist);
        }
    }
    return K_new;
}

// Function to center the kernel matrix
// [[Rcpp::export]]
arma::mat center_kernel_matrix(const arma::mat& K) {
    int n = K.n_rows;
    if (n == 0) return arma::mat(); // Handle empty matrix
    if (K.n_cols != n) Rcpp::stop("Input K to center_kernel_matrix must be square.");

    // More efficient centering using matrix operations
    // K_centered = K - 1_n*K - K*1_n + 1_n*K*1_n
    arma::mat ones_n = arma::ones(n, n) / n;
    return K - ones_n * K - K * ones_n + ones_n * K * ones_n;
}

// Function to perform eigenvalue decomposition on centered kernel matrix
// [[Rcpp::export]]
Rcpp::List kpca_eigen_decomp(const arma::mat& Kc, int n_components) {
    int n = Kc.n_rows;
    if (n == 0) {
        Rcpp::warning("Input centered kernel matrix is empty. Returning empty results.");
        return Rcpp::List::create(
            Rcpp::Named("eigenvalues") = arma::vec(),
            Rcpp::Named("pcv") = arma::mat()
        );
    }
    if (Kc.n_cols != n) Rcpp::stop("Input Kc to kpca_eigen_decomp must be square.");

    // Eigenvalue Decomposition
    arma::vec eigval;
    arma::mat eigvec;
    try {
        // Use eig_sym for symmetric matrices for stability and efficiency
        arma::eig_sym(eigval, eigvec, Kc);
    } catch (const std::exception& e) {
        Rcpp::stop("Eigenvalue decomposition failed: %s", e.what());
    } catch (...) {
        Rcpp::stop("Eigenvalue decomposition failed due to an unknown error.");
    }

    // Armadillo sorts eigenvalues in ascending order, reverse for descending order
    eigval = arma::reverse(eigval);
    eigvec = arma::fliplr(eigvec); // fliplr reverses columns

    // Keep only the top n_components
    int max_possible_components = eigval.n_elem;
    int actual_components = std::min(max_possible_components, n_components);

    if (actual_components < n_components) {
      Rcpp::warning("Requested %d components, but only %d available. Returning %d components.", 
                    n_components, actual_components, actual_components);
    }
    
    // Handle cases where no components are possible (e.g., n_components = 0 or matrix issue)
    if (actual_components <= 0) {
        Rcpp::warning("Number of components requested or available is zero or less. Returning empty results.");
        return Rcpp::List::create(
            Rcpp::Named("eigenvalues") = arma::vec(),
            Rcpp::Named("pcv") = arma::mat()
        );
    }

    arma::vec top_eigval = eigval.head(actual_components);
    arma::mat top_eigvec = eigvec.cols(0, actual_components - 1);

    // Filter out eigenvalues very close to zero or negative due to numerical precision
    double tolerance = 1e-10;
    arma::uvec positive_indices = arma::find(top_eigval > tolerance);

    int n_positive = positive_indices.n_elem;

    if (n_positive == 0) {
        Rcpp::warning("All top eigenvalues are close to zero or negative after centering. Check data or gamma. Returning empty results.");
        return Rcpp::List::create(
            Rcpp::Named("eigenvalues") = arma::vec(),
            Rcpp::Named("pcv") = arma::mat()
        );
    }
    
    if (n_positive < actual_components) {
        Rcpp::warning("Removed %d component(s) due to non-positive eigenvalues (tolerance %e).", actual_components - n_positive, tolerance);
    }

    arma::vec final_eigval = top_eigval(positive_indices);
    arma::mat final_eigvec = top_eigvec.cols(positive_indices);

    // Calculate scaled eigenvectors (alpha) used for projection:
    // alpha = V * Lambda^(-1/2)
    // Ensure eigenvalues are strictly positive before taking sqrt
    arma::mat alpha = final_eigvec * arma::diagmat(1.0 / arma::sqrt(final_eigval));

    return Rcpp::List::create(
        Rcpp::Named("eigenvalues") = final_eigval,
        Rcpp::Named("pcv") = alpha // Principal Component Vectors (alpha)
    );
}
