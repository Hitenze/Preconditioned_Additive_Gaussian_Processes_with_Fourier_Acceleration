#ifndef NFFT4GP_EXTERNAL_HEADER_H
#define NFFT4GP_EXTERNAL_HEADER_H

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

#include "_utils.h"
#include "_linearalg.h"
#include "_solvers.h"
#include "_preconds.h"
#include "_optimizer.h"

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
   // no eigs_tol needed

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
 * @param [in] max_n:   maximum number of points.
 * @param [in] dim:     data dimension
 * @return           Return the kernel structure.
 */
void *Nfft4GPNFFTKernelParamCreate(int max_n, int dim);

/**
 * @brief   Free a nfft kernel structure.
 * @details Free a nfft kernel structure.
 * @param [in] kernel:   kernel structure.
 */
void Nfft4GPNFFTKernelParamFree(void *kernel);

/**
 * @brief   Free a kernel matrix (NFFT matrix).
 * @details Free a kernel matrix (NFFT matrix).
 * @param [in,out]   str: kernel matrix.
 * @return           Return error code.
 */
void Nfft4GPNFFTKernelFree(void *str);

/**
 * @brief   Free NFFT kernel.
 * @details Free NFFT kernel.
 * @param [in] kernel:   kernel structure.
 * @return           Return error code.
 */
int Nfft4GPNFFTKernelParamFreeNFFTKernel(void *kernel);

/**
 * @brief   Remove points from the kernel structure.
 * @details Remove points from the kernel structure.
 * @param [in] kernel:   kernel structure.
 * @return           Return error code.
 */
int Nfft4GPNFFTKernelParamRemovePoints(void *kernel);

/**
 * @brief   Gaussian kernel function K(x,y) = ff^2 * [exp(-||x-y||/(2*kk^2)) + I*noise], possibly with gradient.
 * @details Gaussian kernel function K(x,y) = ff^2 * [exp(-||x-y||/(2*kk^2)) + I*noise], possibly with gradient.
 * @param [in]       str: helper data structure, a pointer to the kernel data structure.
 * @param [in]       data: data matrix (ldim by d).
 * @param [in]       n: number of points.
 * @param [in]       ldim: leading dimension of the data matrix
 * @param [in]       d: dimension of data
 * @param [in]       permr: select rows. If set to NULL the whole matrix is generated. \n
 *                   If not NULL, when permc is NULL, generate K(permr, permr).
 * @param [in]       kr: length of permr.
 * @param [in]       permc: select columns. Only works if permr is not NULL, generate K(permr, permc).
 * @param [in]       kr: length of permc.
 * @param [in,out]   Kp: pointer to the Kernel matrix. If set to NULL, will not be generated.\n 
 *                   If space preallocated, it will be used. Otherwise, it will be allocated.
 * @param [in,out]   dKp: pointer to the derivative of the Kernel matrix. If set to NULL, will not be generated.\n
 *                   If space preallocated, it will be used. Otherwise, it will be allocated. \n
 *                   Note that df, dl, and dmu are stored contiguously in memory, one after another.
 * @return           Return error code.
 */
int Nfft4GPNFFTKernelGaussianKernel(void *str, NFFT4GP_DOUBLE *data, int n, int ldim, int d, int *permr, int kr, int *permc, int kc, NFFT4GP_DOUBLE **Kp, NFFT4GP_DOUBLE **dKp);

/**
 * @brief   Matern 1/2 kernel function + I*noise, possibly with gradient.
 * @details Matern 1/2 kernel function + I*noise, possibly with gradient.
 * @param [in]       str: helper data structure, a pointer to the kernel data structure.
 * @param [in]       data: data matrix (ldim by d).
 * @param [in]       n: number of points.
 * @param [in]       ldim: leading dimension of the data matrix
 * @param [in]       d: dimension of data
 * @param [in]       permr: select rows. If set to NULL the whole matrix is generated. \n
 *                   If not NULL, when permc is NULL, generate K(permr, permr).
 * @param [in]       kr: length of permr.
 * @param [in]       permc: select columns. Only works if permr is not NULL, generate K(permr, permc).
 * @param [in]       kr: length of permc.
 * @param [in,out]   Kp: pointer to the Kernel matrix. If set to NULL, will not be generated.\n
 *                   If space preallocated, it will be used. Otherwise, it will be allocated.
 * @param [in,out]   dKp: pointer to the derivative of the Kernel matrix. If set to NULL, will not be generated.\n
 *                   If space preallocated, it will be used. Otherwise, it will be allocated. \n
 *                   Note that df, dl, and dmu are stored contiguously in memory, one after another.
 * @return           Return error code.
 */
int Nfft4GPNFFTKernelMatern12Kernel(void *str, NFFT4GP_DOUBLE *data, int n, int ldim, int d, int *permr, int kr, int *permc, int kc, NFFT4GP_DOUBLE **Kp, NFFT4GP_DOUBLE **dKp);

/**
 * @brief   Compute the symmetrix matrix-vector product y = alpha*A*x + beta*y.
 * @details Compute the symmetrix matrix-vector product y = alpha*A*x + beta*y. The matrix is stored in the lower triangular part.
 * @param[in]     data: Pointer to the matrix. The matrix is stored in the lower triangular part.
 * @param[in]     n: Dimension of the matrix.
 * @param[in]     alpha: Scaling factor for the matrix.
 * @param[in]     x: Pointer to the vector.
 * @param[in]     beta: Scaling factor for the vector.
 * @param[in,out] y: Pointer to the vector.
 * @return  0 if success.
 */
int Nfft4GPNFFTMatSymv(void *data, int n, NFFT4GP_DOUBLE alpha, NFFT4GP_DOUBLE *x, NFFT4GP_DOUBLE beta, NFFT4GP_DOUBLE *y);

/**
 * @brief   Compute the symmetrix matrix-vector product y = alpha*A*x + beta*y.
 * @details Compute the symmetrix matrix-vector product y = alpha*A*x + beta*y. The matrix is stored in the lower triangular part.
 * @param[in]     data: Pointer to the matrix. The matrix is stored in the lower triangular part.
 * @param[in]     n: Dimension of the matrix.
 * @param[in]     alpha: Scaling factor for the matrix.
 * @param[in]     x: Pointer to the vector.
 * @param[in]     beta: Scaling factor for the vector.
 * @param[in,out] y: Pointer to the vector.
 * @return  0 if success.
 */
int Nfft4GPNFFTGradMatSymv(void *data, int n, NFFT4GP_DOUBLE alpha, NFFT4GP_DOUBLE *x, NFFT4GP_DOUBLE beta, NFFT4GP_DOUBLE *y);

/**
 * @brief   Additive kernel function K(x,y) = sum_i K_i(x,y).
 * @details Additive kernel function K(x,y) = sum_i K_i(x,y).
 * @param [in]     data: data matrix (ldim by d).
 * @param [in]     n: number of points.
 * @param [in]     ldim: leading dimension of the data matrix
 * @param [in]     d: dimension of data
 * @param [in]     windows: feature index for each kernel
 * @param [in]     nwindows: number of kernels
 * @param [in]     dwindows: dimension of each kernel
 * @param [in]     fkernel: kernel function for each kernel
 * @return           Return the kernel structure.
 */
void* Nfft4GPNFFTAdditiveKernelParamCreate(NFFT4GP_DOUBLE *data, int n, int ldim, int d, int *windows, int nwindows, int dwindows);

/**
 * @brief   Gaussian kernel function K(x,y) = ff^2 * [exp(-||x-y||/(2*kk^2)) + I*noise], possibly with gradient.
 * @details Gaussian kernel function K(x,y) = ff^2 * [exp(-||x-y||/(2*kk^2)) + I*noise], possibly with gradient.
 * @param [in]       str: helper data structure, a pointer to the kernel data structure.
 * @param [in]       data: data matrix (ldim by d).
 * @param [in]       n: number of points.
 * @param [in]       ldim: leading dimension of the data matrix
 * @param [in]       d: dimension of data
 * @param [in]       permr: select rows. If set to NULL the whole matrix is generated. \n
 *                   If not NULL, when permc is NULL, generate K(permr, permr).
 * @param [in]       kr: length of permr.
 * @param [in]       permc: select columns. Only works if permr is not NULL, generate K(permr, permc).
 * @param [in]       kr: length of permc.
 * @param [in,out]   Kp: pointer to the Kernel matrix. If set to NULL, will not be generated.\n 
 *                   If space preallocated, it will be used. Otherwise, it will be allocated.
 * @param [in,out]   dKp: pointer to the derivative of the Kernel matrix. If set to NULL, will not be generated.\n
 *                   If space preallocated, it will be used. Otherwise, it will be allocated. \n
 *                   Note that df, dl, and dmu are stored contiguously in memory, one after another.
 * @return           Return error code.
 */
int Nfft4GPNFFTAdditiveKernelGaussianKernel(void *str, NFFT4GP_DOUBLE *data, int n, int ldim, int d, int *permr, int kr, int *permc, int kc, NFFT4GP_DOUBLE **Kp, NFFT4GP_DOUBLE **dKp);

/**
 * @brief   Matern 1/2 kernel function + I*noise, possibly with gradient.
 * @details Matern 1/2 kernel function + I*noise, possibly with gradient.
 * @param [in]       str: helper data structure, a pointer to the kernel data structure.
 * @param [in]       data: data matrix (ldim by d).
 * @param [in]       n: number of points.
 * @param [in]       ldim: leading dimension of the data matrix
 * @param [in]       d: dimension of data
 * @param [in]       permr: select rows. If set to NULL the whole matrix is generated. \n
 *                   If not NULL, when permc is NULL, generate K(permr, permr).
 * @param [in]       kr: length of permr.
 * @param [in]       permc: select columns. Only works if permr is not NULL, generate K(permr, permc).
 * @param [in]       kr: length of permc.
 * @param [in,out]   Kp: pointer to the Kernel matrix. If set to NULL, will not be generated.\n 
 *                   If space preallocated, it will be used. Otherwise, it will be allocated.
 * @param [in,out]   dKp: pointer to the derivative of the Kernel matrix. If set to NULL, will not be generated.\n
 *                   If space preallocated, it will be used. Otherwise, it will be allocated. \n
 *                   Note that df, dl, and dmu are stored contiguously in memory, one after another.
 * @return           Return error code.
 */
int Nfft4GPNFFTAdditiveKernelMatern12Kernel(void *str, NFFT4GP_DOUBLE *data, int n, int ldim, int d, int *permr, int kr, int *permc, int kc, NFFT4GP_DOUBLE **Kp, NFFT4GP_DOUBLE **dKp);

/**
 * @brief   Compute the symmetrix matrix-vector product y = alpha*A*x + beta*y.
 * @details Compute the symmetrix matrix-vector product y = alpha*A*x + beta*y. The matrix is stored in the lower triangular part.
 * @param[in]   data    Pointer to the matrix. The matrix is stored in the lower triangular part.
 * @param[in]   n       Dimension of the matrix.
 * @param[in]   alpha   Scaling factor for the matrix.
 * @param[in]   x       Pointer to the vector.
 * @param[in]   beta    Scaling factor for the vector.
 * @param[in,out]   y   Pointer to the vector.
 * @return  0 if success.
 */
int Nfft4GPAdditiveNFFTMatSymv(void *data, int n, NFFT4GP_DOUBLE alpha, NFFT4GP_DOUBLE *x, NFFT4GP_DOUBLE beta, NFFT4GP_DOUBLE *y);

/**
 * @brief   Compute the symmetrix matrix-vector product y = alpha*A*x + beta*y.
 * @details Compute the symmetrix matrix-vector product y = alpha*A*x + beta*y. The matrix is stored in the lower triangular part.
 * @param[in]   data    Pointer to the matrix. The matrix is stored in the lower triangular part.
 * @param[in]   n       Dimension of the matrix.
 * @param[in]   alpha   Scaling factor for the matrix.
 * @param[in]   x       Pointer to the vector.
 * @param[in]   beta    Scaling factor for the vector.
 * @param[in,out]   y   Pointer to the vector.
 * @return  0 if success.
 */
int Nfft4GPAdditiveNFFTGradMatSymv(void *data, int n, NFFT4GP_DOUBLE alpha, NFFT4GP_DOUBLE *x, NFFT4GP_DOUBLE beta, NFFT4GP_DOUBLE *y);

/**
 * @brief   Free a kernel matrix (NFFT matrix).
 * @details Free a kernel matrix (NFFT matrix).
 * @param [in,out]   str: kernel matrix.
 * @return           Return error code.
 */
void Nfft4GPAdditiveNFFTKernelFree(void *str);

/**
 * @brief   Connect two data matrices.
 * @details Details TBA.
 */
NFFT4GP_DOUBLE* Nfft4GPNFFTAppendData(NFFT4GP_DOUBLE *X1, int n1, int ldim1, int d, NFFT4GP_DOUBLE *X2, int n2, int ldim2);

/**
 * @brief   Use NFFT kernel to make predictions.
 * @details Details TBA.
 */
int Nfft4GPAdditiveNFFTGpPredict(NFFT4GP_DOUBLE *x, 
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