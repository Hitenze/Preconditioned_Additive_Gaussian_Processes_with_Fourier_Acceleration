#ifndef NFFT4GP_GP_LOSS_H
#define NFFT4GP_GP_LOSS_H

/**
 * @file gp_loss.h
 * @brief Gaussian process loss functions.
 */

#include "../utils/utils.h"
#include "../linearalg/kernels.h"
#include "../solvers/solvers.h"
#include "../preconds/precond.h"
#include "transform.h"

/**
 * @brief   Compute the Gaussian process loss and gradient.
 * @details Compute the Gaussian process loss and gradient.
 * @param [in]       x                            Pointer to the vector of parameters (before transformation).
 * @param [in]       data                         Pointer to the data matrix.
 * @param [in]       label                        Pointer to the label vector.
 * @param [in]       n                            Number of data points.
 * @param [in]       ldim                         Dimension of the data points.
 * @param [in]       d                            Dimension of the parameter space.
 * @param [in]       fkernel                      Kernel function.
 * @param [in]       vfkernel_data                Kernel function data structure.
 * @param [in]       kernel_data_free             Kernel function data structure free function.
 * @param [in]       matvec                       Matrix vector product function (for kernel matrix).
 * @param [in]       dmatvec                      Matrix vector product function (for gradient matrix).
 * @param [in]       precond_fkernel              Preconditioner kernel function.
 * @param [in]       precond_vfkernel_data        Preconditioner kernel function data structure.
 * @param [in]       precond_kernel_data_free     Preconditioner kernel function data structure free function.
 * @param [in]       precond_setup                Preconditioner setup function.
 * @param [in]       precond_solve                Preconditioner solve function.
 * @param [in]       precond_trace                Preconditioner trace function.
 * @param [in]       precond_logdet               Preconditioner log determinant function.
 * @param [in]       precond_dvp                  Preconditioner derivative function.
 * @param [in]       precond_reset                Preconditioner reset function, ready for reuse. User is responsible for freeing the preconditioner data after use.
 * @param [in]       precond_data                 Preconditioner data (with setup already done).
 * @param [in]       atol                         Absolute tolerance for the solver (for the first part, TODO: no need in the block version).
 * @param [in]       tol                          Relative tolerance for the solver (for the first part, TODO: no need in the block version).
 * @param [in]       wsize                        Window size for reorthogonalization.
 * @param [in]       maxits                       Maximum number of iterations.
 * @param [in]       nvecs                        Number of vectors for lanczos quadrature.
 * @param [in]       radamacher                   Radamacher vector matrix. If NULL, will be generated randomly. Otherwise each column of the matrix is a radamacher vector.
 * @param [in]       transform                    Transformation type.
 * @param [in]       mask                         Mask for the parameters (0: grad is set to 0, otherwise: grad is computed).
 * @param [in]       print_level                  Print level.
 * @param [in]       dwork                        Working array. Set to NULL to not use. Otherwise at least should be 4*n*n + 4*n.
 * @param [out]      loss                         Pointer to the loss.
 * @param [out]      grad                         Pointer to the gradient.
 * @return           Return error code.
 */
int Nfft4GPGpLoss(
   NFFT4GP_DOUBLE *x, 
   NFFT4GP_DOUBLE *data,
   NFFT4GP_DOUBLE *label,
   int n,
   int ldim,
   int d,
   func_kernel fkernel,
   void* vfkernel_data,
   func_free kernel_data_free,
   func_symmatvec matvec,
   func_symmatvec dmatvec,
   func_kernel precond_fkernel,
   void* precond_vfkernel_data,
   func_free precond_kernel_data_free,
   precond_kernel_setup precond_setup,
   func_solve precond_solve,
   func_trace precond_trace,
   func_logdet precond_logdet,
   func_dvp precond_dvp,
   func_free precond_reset,
   void *precond_data,
   int atol,
   NFFT4GP_DOUBLE tol,
   int wsize,
   int maxits,
   int nvecs,
   NFFT4GP_DOUBLE *radamacher,
   nfft4gp_transform_type transform,
   int *mask,
   int print_level,
   NFFT4GP_DOUBLE *dwork,
   NFFT4GP_DOUBLE *loss,
   NFFT4GP_DOUBLE *grad
);

/*********************************************
 * Below are some specific loss functions
 *********************************************/

/**
 * @brief   Compute the Gaussian process loss and gradient with Gaussian RAN kernel and SoftPlus transformation.
 * @details Compute the Gaussian process loss and gradient using a Gaussian RAN kernel with SoftPlus transformation.
 * @param [in]       x               Pointer to the vector of parameters (before transformation).
 * @param [in]       data            Pointer to the data matrix.
 * @param [in]       label           Pointer to the label vector.
 * @param [in]       n               Number of data points.
 * @param [in]       ldim            Dimension of the data points.
 * @param [in]       d               Dimension of the parameter space.
 * @param [in]       permn           Permutation of RAN.
 * @param [in]       k               Rank of RAN.
 * @param [in]       atol            Absolute tolerance for the solver.
 * @param [in]       tol             Relative tolerance for the solver.
 * @param [in]       wsize           Window size for reorthogonalization.
 * @param [in]       maxits          Maximum number of iterations.
 * @param [in]       nvecs           Number of vectors for lanczos quadrature.
 * @param [in]       radamacher      Radamacher vector matrix. If NULL, will be generated randomly. Otherwise each column of the matrix is a radamacher vector.
 * @param [in]       mask            Mask for the parameters (0: grad is set to 0, otherwise: grad is computed).
 * @param [out]      loss            Pointer to the loss.
 * @param [out]      grad            Pointer to the gradient.
 * @return           Return error code.
 */
int Nfft4GPGpLossGaussianRANSoftPlus(
   NFFT4GP_DOUBLE *x, 
   NFFT4GP_DOUBLE *data,
   NFFT4GP_DOUBLE *label,
   int n,
   int ldim,
   int d,
   int *permn,
   int k,
   int atol,
   NFFT4GP_DOUBLE tol,
   int wsize,
   int maxits,
   int nvecs,
   NFFT4GP_DOUBLE *radamacher,
   int *mask,
   NFFT4GP_DOUBLE *loss,
   NFFT4GP_DOUBLE *grad
);

#endif