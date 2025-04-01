#ifndef NFFT4GP_GP_PREDICT_H
#define NFFT4GP_GP_PREDICT_H

/**
 * @file gp_predict.h
 * @brief Gaussian process prediction functions.
 */

#include "../utils/utils.h"
#include "../linearalg/kernels.h"
#include "../solvers/solvers.h"
#include "../preconds/precond.h"
#include "transform.h"

/**
 * @brief   Use the current parameters to make prediction.
 * @details Use the current parameters to make prediction.
 * @param [in]       x               Pointer to the vector of parameters (before transformation).
 * @param [in]       data            Pointer to the data matrix.
 * @param [in]       label           Pointer to the label vector.
 * @param [in]       n               Number of data points.
 * @param [in]       ldim            Dimension of the data points.
 * @param [in]       d               Dimension of the parameter space.
 * @param [in]       data_predict    Pointer to the data matrix for prediction.
 * @param [in]       n_predict       Number of data points for prediction.
 * @param [in]       ldim_predict    Dimension of the data points for prediction.
 * @param [in]       fkernel         Kernel function.
 * @param [in]       vfkernel_data   Kernel function data structure.
 * @param [in]       matvec          Matrix vector product function (for symmetric kernel matrix).
 * @param [in]       matvec_predict  Matrix vector product function (for general kernel matrix).
 * @param [in]       precond_setup   Preconditioner setup function.
 * @param [in]       precond_solve   Preconditioner solve function.
 * @param [in]       precond_data    Preconditioner data (with setup already done).
 * @param [in]       atol            Absolute tolerance for the solver.
 * @param [in]       tol             Relative tolerance for the solver.
 * @param [in]       maxits          Maximum number of iterations.
 * @param [in]       transform       Transformation type.
 * @param [in]       print_level     Print level.
 * @param [in]       dwork           Working array.
 * @param [out]      label_predictp  Pointer to the pointer to the prediction labels.
 * @param [out]      std_predictp    Pointer to the pointer to the standard deviation of the prediction.
 * @return           Return error code.
 */
int Nfft4GPGpPredict(
   NFFT4GP_DOUBLE *x, 
   NFFT4GP_DOUBLE *data,
   NFFT4GP_DOUBLE *label,
   int n,
   int ldim,
   int d,
   NFFT4GP_DOUBLE *data_predict,
   int n_predict,
   int ldim_predict,
   func_kernel fkernel,
   void* vfkernel_data,
   func_symmatvec matvec,
   func_matvec matvec_predict,
   precond_kernel_setup precond_setup,
   func_solve precond_solve,
   void *precond_data,
   int atol,
   NFFT4GP_DOUBLE tol,
   int maxits,
   nfft4gp_transform_type transform,
   int print_level,
   NFFT4GP_DOUBLE *dwork,
   NFFT4GP_DOUBLE **label_predictp,
   NFFT4GP_DOUBLE **std_predictp
);

/*********************************************
 * Below are some specific prediction functions
 *********************************************/

/**
 * @brief   Use the current parameters to make prediction with Gaussian RAN kernel and SoftPlus transformation.
 * @details Use the current parameters to make prediction using a Gaussian RAN kernel with SoftPlus transformation.
 * @param [in]       x               Pointer to the vector of parameters (before transformation).
 * @param [in]       data            Pointer to the data matrix.
 * @param [in]       label           Pointer to the label vector.
 * @param [in]       n               Number of data points.
 * @param [in]       ldim            Dimension of the data points.
 * @param [in]       d               Dimension of the parameter space.
 * @param [in]       data_predict    Pointer to the data matrix for prediction.
 * @param [in]       n_predict       Number of data points for prediction.
 * @param [in]       ldim_predict    Dimension of the data points for prediction.
 * @param [in]       permn           Permutation of RAN.
 * @param [in]       k               Rank of RAN.
 * @param [in]       atol            Absolute tolerance for the solver.
 * @param [in]       tol             Relative tolerance for the solver.
 * @param [in]       maxits          Maximum number of iterations.
 * @param [out]      label_predictp  Pointer to the pointer to the prediction labels.
 * @param [out]      std_predictp    Pointer to the pointer to the standard deviation of the prediction.
 * @return           Return error code.
 */
int Nfft4GPGpPredictGaussianRANSoftPlus(
   NFFT4GP_DOUBLE *x, 
   NFFT4GP_DOUBLE *data,
   NFFT4GP_DOUBLE *label,
   int n,
   int ldim,
   int d,
   NFFT4GP_DOUBLE *data_predict,
   int n_predict,
   int ldim_predict,
   int *permn,
   int k,
   int atol,
   NFFT4GP_DOUBLE tol,
   int maxits,
   NFFT4GP_DOUBLE **label_predictp,
   NFFT4GP_DOUBLE **std_predictp
);

#endif