#ifndef NFFT4GP_NFFT_INTERFACE_H
#define NFFT4GP_NFFT_INTERFACE_H

/**
 * @file nfft_interface.h
 * @brief NFFT interface, depends on utils.h, kernels.h, solvers.h, precond.h, optimizer.h
 * @details Interface to NFFT library
 */

#include "config.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#ifdef HAVE_COMPLEX_H
#include <complex.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#include "fastsum.h"
#include "kernels.h"
#include "infft.h"

#include "../utils/utils.h"
#include "../linearalg/kernels.h"
#include "../solvers/solvers.h"
#include "../solvers/fgmres.h"
#include "../preconds/precond.h"
#include "../optimizer/optimizer.h"

typedef struct
{
   int               _kernel;//0: Guass, 1: Matern1/2
   int               _d;
   NFFT4GP_DOUBLE    *_sigma;
   NFFT4GP_DOUBLE    _mu;
   int               _N;      // 16 32 64
   int               _p;      // 1 1 8
   int               _m;      // 2 4 7
   NFFT4GP_DOUBLE    _eps; // 0 0 0

   int               _n;

   int               _NN;

   NFFT4GP_DOUBLE    *_x;
   NFFT4GP_DOUBLE    _scale;

   NFFT4GP_DOUBLE    _kernel_scale;

   fastsum_plan      *_fastsum_original;
   fastsum_plan      *_fastsum_derivative;
} str_adj, *pstr_adj;

/**
 * @brief   Create a nfft kernel structure. Parameters are used to preallocate memory.
 * @details Create a nfft kernel structure. Parameters are used to preallocate memory.
 * @param [in]       max_n:   maximum number of points.
 * @param [in]       dim:     data dimension
 * @return           Return the kernel structure.
 */
void *Nfft4GPNFFTKernelParamCreate(int max_n, int dim);

/**
 * @brief   Free a nfft kernel structure.
 * @details Free a nfft kernel structure.
 * @param [in]       kernel:   kernel structure.
 */
void Nfft4GPNFFTKernelParamFree(void *kernel);

/**
 * @brief   Free a kernel matrix (NFFT matrix).
 * @details Free a kernel matrix (NFFT matrix).
 * @param [in,out]   str:     kernel matrix.
 * @return           Return error code.
 */
void Nfft4GPNFFTKernelFree(void *str);

/**
 * @brief   Free NFFT kernel.
 * @details Free NFFT kernel.
 * @param [in]       kernel:   kernel structure.
 * @return           Return error code.
 */
int Nfft4GPNFFTKernelParamFreeNFFTKernel(void *kernel);

/**
 * @brief   Remove points from the kernel structure.
 * @details Remove points from the kernel structure.
 * @param [in]       kernel:   kernel structure.
 * @return           Return error code.
 */
int Nfft4GPNFFTKernelParamRemovePoints(void *kernel);

/**
 * @brief   Compute Gaussian kernel matrix using NFFT.
 * @details Compute Gaussian kernel matrix using NFFT.
 * @param [in]       str:     kernel structure.
 * @param [in]       data:    data matrix (ldim by d).
 * @param [in]       n:       number of points.
 * @param [in]       ldim:    leading dimension of the data matrix.
 * @param [in]       d:       dimension of data.
 * @param [in]       permr:   select rows. If set to NULL the whole matrix is generated.
 *                            If not NULL, when permc is NULL, generate K(permr, permr).
 * @param [in]       kr:      length of permr.
 * @param [in]       permc:   select columns. Only works if permr is not NULL, generate K(permr, permc).
 * @param [in]       kc:      length of permc.
 * @param [in,out]   Kp:      pointer to the Kernel matrix. If set to NULL, will not be generated.
 *                            If space preallocated, it will be used. Otherwise, it will be allocated.
 * @param [in,out]   dKp:     pointer to the derivative of the Kernel matrix. If set to NULL, will not be generated.
 *                            If space preallocated, it will be used. Otherwise, it will be allocated.
 * @return           Return error code.
 */
int Nfft4GPNFFTKernelGaussianKernel(
   void *str, 
   NFFT4GP_DOUBLE *data, 
   int n, 
   int ldim, 
   int d, 
   int *permr, 
   int kr, 
   int *permc, 
   int kc, 
   NFFT4GP_DOUBLE **Kp, 
   NFFT4GP_DOUBLE **dKp);

/**
 * @brief   Compute Matern 1/2 kernel matrix using NFFT.
 * @details Compute Matern 1/2 kernel matrix using NFFT.
 * @param [in]       str:     kernel structure.
 * @param [in]       data:    data matrix (ldim by d).
 * @param [in]       n:       number of points.
 * @param [in]       ldim:    leading dimension of the data matrix.
 * @param [in]       d:       dimension of data.
 * @param [in]       permr:   select rows. If set to NULL the whole matrix is generated.
 *                            If not NULL, when permc is NULL, generate K(permr, permr).
 * @param [in]       kr:      length of permr.
 * @param [in]       permc:   select columns. Only works if permr is not NULL, generate K(permr, permc).
 * @param [in]       kc:      length of permc.
 * @param [in,out]   Kp:      pointer to the Kernel matrix. If set to NULL, will not be generated.
 *                            If space preallocated, it will be used. Otherwise, it will be allocated.
 * @param [in,out]   dKp:     pointer to the derivative of the Kernel matrix. If set to NULL, will not be generated.
 *                            If space preallocated, it will be used. Otherwise, it will be allocated.
 * @return           Return error code.
 */
int Nfft4GPNFFTKernelMatern12Kernel(
   void *str, 
   NFFT4GP_DOUBLE *data, 
   int n, 
   int ldim, 
   int d, 
   int *permr, 
   int kr, 
   int *permc, 
   int kc, 
   NFFT4GP_DOUBLE **Kp, 
   NFFT4GP_DOUBLE **dKp);

/**
 * @brief   Perform matrix-vector multiplication with NFFT kernel.
 * @details Perform matrix-vector multiplication with NFFT kernel.
 * @param [in]       data:    kernel structure.
 * @param [in]       n:       dimension of the matrix.
 * @param [in]       alpha:   scalar multiplier for the matrix-vector product.
 * @param [in]       x:       input vector.
 * @param [in]       beta:    scalar multiplier for the output vector.
 * @param [in,out]   y:       output vector, y = alpha*A*x + beta*y.
 * @return           Return error code.
 */
int Nfft4GPNFFTMatSymv(
   void *data, 
   int n, 
   NFFT4GP_DOUBLE alpha, 
   NFFT4GP_DOUBLE *x, 
   NFFT4GP_DOUBLE beta, 
   NFFT4GP_DOUBLE *y);

/**
 * @brief   Perform gradient matrix-vector multiplication with NFFT kernel.
 * @details Perform gradient matrix-vector multiplication with NFFT kernel.
 * @param [in]       data:    kernel structure.
 * @param [in]       n:       dimension of the matrix.
 * @param [in]       alpha:   scalar multiplier for the matrix-vector product.
 * @param [in]       x:       input vector.
 * @param [in]       beta:    scalar multiplier for the output vector.
 * @param [in,out]   y:       output vector, y = alpha*A*x + beta*y.
 * @return           Return error code.
 */
int Nfft4GPNFFTGradMatSymv(
   void *data, 
   int n, 
   NFFT4GP_DOUBLE alpha, 
   NFFT4GP_DOUBLE *x, 
   NFFT4GP_DOUBLE beta, 
   NFFT4GP_DOUBLE *y);

/**
 * @brief   Create an additive kernel parameter structure.
 * @details Create an additive kernel parameter structure.
 * @param [in]       data:       data matrix.
 * @param [in]       n:          number of points.
 * @param [in]       ldim:       leading dimension of the data matrix.
 * @param [in]       d:          dimension of data.
 * @param [in]       windows:    window indices for each dimension.
 * @param [in]       nwindows:   number of windows.
 * @param [in]       dwindows:   dimension of windows.
 * @return           Return the kernel structure.
 */
void* Nfft4GPNFFTAdditiveKernelParamCreate(
   NFFT4GP_DOUBLE *data, 
   int n, 
   int ldim, 
   int d, 
   int *windows, 
   int nwindows, 
   int dwindows);

/**
 * @brief   Compute Gaussian kernel matrix using additive NFFT.
 * @details Compute Gaussian kernel matrix using additive NFFT.
 * @param [in]       str:     kernel structure.
 * @param [in]       data:    data matrix (ldim by d).
 * @param [in]       n:       number of points.
 * @param [in]       ldim:    leading dimension of the data matrix.
 * @param [in]       d:       dimension of data.
 * @param [in]       permr:   select rows. If set to NULL the whole matrix is generated.
 *                            If not NULL, when permc is NULL, generate K(permr, permr).
 * @param [in]       kr:      length of permr.
 * @param [in]       permc:   select columns. Only works if permr is not NULL, generate K(permr, permc).
 * @param [in]       kc:      length of permc.
 * @param [in,out]   Kp:      pointer to the Kernel matrix. If set to NULL, will not be generated.
 *                            If space preallocated, it will be used. Otherwise, it will be allocated.
 * @param [in,out]   dKp:     pointer to the derivative of the Kernel matrix. If set to NULL, will not be generated.
 *                            If space preallocated, it will be used. Otherwise, it will be allocated.
 * @return           Return error code.
 */
int Nfft4GPNFFTAdditiveKernelGaussianKernel(
   void *str, 
   NFFT4GP_DOUBLE *data, 
   int n, 
   int ldim, 
   int d, 
   int *permr, 
   int kr, 
   int *permc, 
   int kc, 
   NFFT4GP_DOUBLE **Kp, 
   NFFT4GP_DOUBLE **dKp);

/**
 * @brief   Append two data matrices.
 * @details Append two data matrices.
 * @param [in]       X1:      first data matrix.
 * @param [in]       n1:      number of points in first matrix.
 * @param [in]       ldim1:   leading dimension of first matrix.
 * @param [in]       d:       dimension of data.
 * @param [in]       X2:      second data matrix.
 * @param [in]       n2:      number of points in second matrix.
 * @param [in]       ldim2:   leading dimension of second matrix.
 * @return           Return the appended data matrix.
 */
NFFT4GP_DOUBLE* Nfft4GPNFFTAppendData(
   NFFT4GP_DOUBLE *X1, 
   int n1, 
   int ldim1, 
   int d, 
   NFFT4GP_DOUBLE *X2, 
   int n2, 
   int ldim2);

/**
 * @brief   Use NFFT kernel to make predictions.
 * @details Use NFFT kernel to make Gaussian Process predictions with training data and labels.
 * @param [in]       x:                     hyperparameters.
 * @param [in]       data:                  training data.
 * @param [in]       label:                 training labels.
 * @param [in]       n:                     number of training points.
 * @param [in]       ldim:                  leading dimension of training data.
 * @param [in]       d:                     dimension of data.
 * @param [in]       data_predict:          prediction points.
 * @param [in]       n_predict:             number of prediction points.
 * @param [in]       ldim_predict:          leading dimension of prediction data.
 * @param [in]       data_all:              combined training and prediction data.
 * @param [in]       fkernel:               kernel function.
 * @param [in]       vfkernel_data:         kernel function data.
 * @param [in]       vfkernel_data_l:       kernel function data for labels.
 * @param [in]       kernel_data_free:      function to free kernel data.
 * @param [in]       matvec:                matrix-vector product function.
 * @param [in]       precond_fkernel:       preconditioner kernel function.
 * @param [in]       precond_vfkernel_data: preconditioner kernel data.
 * @param [in]       precond_kernel_data_free: function to free preconditioner kernel data.
 * @param [in]       precond_setup:         preconditioner setup function.
 * @param [in]       precond_solve:         preconditioner solve function.
 * @param [in]       precond_data:          preconditioner data.
 * @param [in]       atol:                  absolute tolerance.
 * @param [in]       tol:                   relative tolerance.
 * @param [in]       maxits:                maximum number of iterations.
 * @param [in]       transform:             transform type.
 * @param [in]       print_level:           print level.
 * @param [in]       dwork:                 working array.
 * @param [out]      label_predictp:        predicted labels.
 * @param [out]      std_predictp:          standard deviation of predictions.
 * @return           Return error code.
 */
int Nfft4GPAdditiveNFFTGpPredict(
   NFFT4GP_DOUBLE *x, 
   NFFT4GP_DOUBLE *data,
   NFFT4GP_DOUBLE *label,
   int n,
   int ldim,
   int d,
   NFFT4GP_DOUBLE *data_predict,
   int n_predict,
   int ldim_predict,
   NFFT4GP_DOUBLE *data_all,
   func_kernel fkernel,
   void* vfkernel_data,
   void* vfkernel_data_l,
   func_free kernel_data_free,
   func_symmatvec matvec,
   func_kernel precond_fkernel,
   void* precond_vfkernel_data,
   func_free precond_kernel_data_free,
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
#endif
