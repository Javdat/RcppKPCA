// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// rbf_kernel
arma::mat rbf_kernel(const arma::mat& X, double gamma);
RcppExport SEXP _RcppKPCA_rbf_kernel(SEXP XSEXP, SEXP gammaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< double >::type gamma(gammaSEXP);
    rcpp_result_gen = Rcpp::wrap(rbf_kernel(X, gamma));
    return rcpp_result_gen;
END_RCPP
}
// rbf_kernel_predict
arma::mat rbf_kernel_predict(const arma::mat& X_new, const arma::mat& X_train, double gamma);
RcppExport SEXP _RcppKPCA_rbf_kernel_predict(SEXP X_newSEXP, SEXP X_trainSEXP, SEXP gammaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X_new(X_newSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type X_train(X_trainSEXP);
    Rcpp::traits::input_parameter< double >::type gamma(gammaSEXP);
    rcpp_result_gen = Rcpp::wrap(rbf_kernel_predict(X_new, X_train, gamma));
    return rcpp_result_gen;
END_RCPP
}
// center_kernel_matrix
arma::mat center_kernel_matrix(const arma::mat& K);
RcppExport SEXP _RcppKPCA_center_kernel_matrix(SEXP KSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type K(KSEXP);
    rcpp_result_gen = Rcpp::wrap(center_kernel_matrix(K));
    return rcpp_result_gen;
END_RCPP
}
// kpca_eigen_decomp
Rcpp::List kpca_eigen_decomp(const arma::mat& Kc, int n_components);
RcppExport SEXP _RcppKPCA_kpca_eigen_decomp(SEXP KcSEXP, SEXP n_componentsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type Kc(KcSEXP);
    Rcpp::traits::input_parameter< int >::type n_components(n_componentsSEXP);
    rcpp_result_gen = Rcpp::wrap(kpca_eigen_decomp(Kc, n_components));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_RcppKPCA_rbf_kernel", (DL_FUNC) &_RcppKPCA_rbf_kernel, 2},
    {"_RcppKPCA_rbf_kernel_predict", (DL_FUNC) &_RcppKPCA_rbf_kernel_predict, 3},
    {"_RcppKPCA_center_kernel_matrix", (DL_FUNC) &_RcppKPCA_center_kernel_matrix, 1},
    {"_RcppKPCA_kpca_eigen_decomp", (DL_FUNC) &_RcppKPCA_kpca_eigen_decomp, 2},
    {NULL, NULL, 0}
};

RcppExport void R_init_RcppKPCA(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
