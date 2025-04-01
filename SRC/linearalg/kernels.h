#ifndef NFFT4GP_KERNELS_H
#define NFFT4GP_KERNELS_H

/**
 * @file kernels.h
 * @brief Kernel functions, depends on utils.h
 */

#include "../utils/utils.h"
#include "../utils/protos.h"
#include "vecops.h"
// #include "../solvers/solvers.h"

/* function pointers */

/**
 * @brief   Function to compute the distance between two points.
 * @details Function to compute the distance between two points.
 * @param [in]       str:     helper data structure
 * @param [in]       data1:   pointer to the first point.
 * @param [in]       ldim1:   leading dimension of data1.
 * @param [in]       data2:   pointer to the second point.
 * @param [in]       ldim2:   leading dimension of data2.
 * @param [in]       d:       dimension of data
 * @return           Return the distance.
 */
typedef NFFT4GP_DOUBLE (*func_dist)(void *str, double *data1, int ldim1, double *data2, int ldim2, int d);

/**
 * @brief   Given a data matrix (ldim by d), evaluate the kernel matrix, possibly with gradient.
 * @details Given a data matrix (ldim by d), evaluate the kernel matrix, possibly with gradient.
 * @param [in]       str:     helper data structure
 * @param [in]       data:    data matrix (ldim by d).
 * @param [in]       n:       number of points.
 * @param [in]       ldim:    leading dimension of the data matrix
 * @param [in]       d:       dimension of data
 * @param [in]       permr:   select rows. If set to NULL the whole matrix is generated. \n
 *                            If not NULL, when permc is NULL, generate K(permr, permr).
 * @param [in]       kr:      length of permr.
 * @param [in]       permc:   select columns. Only works if permr is not NULL, generate K(permr, permc).
 * @param [in]       kc:      length of permc.
 * @param [in,out]   Kp:      pointer to the Kernel matrix. If set to NULL, will not be generated.\n
 *                            If space preallocated, it will be used. Otherwise, it will be allocated.
 * @param [in,out]   dKp:     pointer to the derivative of the Kernel matrix. If set to NULL, will not be generated.\n
 *                            If space preallocated, it will be used. Otherwise, it will be allocated. \n
 *                            Note that df, dl, and dmu are stored contiguously in memory, one after another.
 * @return           Return error code.
 */
typedef int (*func_kernel)(void *str, NFFT4GP_DOUBLE *data, int n, int ldim, int d, int *permr, int kr, int *permc, int kc, NFFT4GP_DOUBLE **Kp, NFFT4GP_DOUBLE **dKp);

/**
 * @brief   Compute l such that if dist(i,j) > l, K(i,j) < tol. This funcfion is used to estimate the drop tolerance given a distance value.
 * @details Compute l such that if dist(i,j) > l, K(i,j) < tol. This funcfion is used to estimate the drop tolerance given a distance value
 * @param [in]       str:     helper data structure
 * @param [in]       tol:     drop tolerance.
 * @return           Return the tolerance.
 */
typedef NFFT4GP_DOUBLE (*func_val2dist)(void *str, NFFT4GP_DOUBLE tol);

/* general kernel struct */
#ifndef NFFT4GP_KERNEL_MAX_PARAMS
#define NFFT4GP_KERNEL_MAX_PARAMS 5
#endif

typedef struct NFFT4GP_KERNEL_STRUCT
{
   // kernerl parameters, typically just lengthscale
   // double parameters are 
   NFFT4GP_DOUBLE _params[NFFT4GP_KERNEL_MAX_PARAMS];
   int _iparams[NFFT4GP_KERNEL_MAX_PARAMS];
   int _max_n;
   int _omp;

   // noise level
   NFFT4GP_DOUBLE _noise_level;

   // pointers
   int _own_buffer;
   NFFT4GP_DOUBLE* _buffer;
   int _own_dbuffer;
   NFFT4GP_DOUBLE* _dbuffer;
   func_kernel _fkernel_buffer;
   int **_ibufferp; // integer buffer
   size_t *_libufferp; // length of integer buffer
   int _own_fkernel_buffer_params;
   void *_fkernel_buffer_params;

   // working buffer
   size_t _ldwork;
   NFFT4GP_DOUBLE *_dwork;

   // external data
   void *_external;

} nfft4gp_kernel, *pnfft4gp_kernel;

/**
 * @brief   Create a kernel structure. Parameters are used to preallocate memory.
 * @details Create a kernel structure. Parameters are used to preallocate memory.
 * @param [in]       max_n:   maximum number of points.
 * @param [in]       omp:     will this kernel be used in parallel
 * @return           Return the kernel structure.
 */
void* Nfft4GPKernelParamCreate(int max_n, int omp);

/**
 * @brief   Destroy a kernel structure.
 * @details Destroy a kernel structure.
 * @param [in]       str:     kernel structure.
 * @return           Return error code.
 */
void Nfft4GPKernelParamFree(void *str);

/**
 * @brief   Free a kernel matrix.
 * @details Free a kernel matrix.
 * @param [in,out]   str:     kernel matrix.
 * @return           Return error code.
 */
void Nfft4GPKernelFree(void *str);

/**********************************************************************************
 * 
 * Euclidean distance
 * 
 **********************************************************************************/

/**
 * @brief   Compute Euclidean distance between two points.
 * @details Compute Euclidean distance between two points.
 * @param [in]       null_params: not used, can be NULL.
 * @param [in]       data1:       pointer to the first point.
 * @param [in]       ldim1:       leading dimension of data1.
 * @param [in]       data2:       pointer to the second point.
 * @param [in]       ldim2:       leading dimension of data2.
 * @param [in]       d:           dimension of data.
 * @return           Return the Euclidean distance.
 */
NFFT4GP_DOUBLE Nfft4GPDistanceEuclid(
   void *null_params, 
   double *data1, 
   int ldim1, 
   double *data2, 
   int ldim2, 
   int d);

/* Euclidean distance helper functions */

/**
 * @brief   Compute pairwise Euclidean distances between two sets of points.
 * @details Compute pairwise Euclidean distances between two sets of points.
 * @param [in]       X:       first set of points, size nX x d.
 * @param [in]       Y:       second set of points, size nY x d.
 * @param [in]       ldimX:   leading dimension of X.
 * @param [in]       ldimY:   leading dimension of Y.
 * @param [in]       nX:      number of points in X.
 * @param [in]       nY:      number of points in Y.
 * @param [in]       d:       dimension of data.
 * @param [in,out]   XYp:     pointer to the distance matrix. If NULL, will be allocated.
 */
void Nfft4GPDistanceEuclidXY(
   NFFT4GP_DOUBLE *X, 
   NFFT4GP_DOUBLE *Y, 
   int ldimX, 
   int ldimY, 
   int nX, 
   int nY, 
   int d, 
   NFFT4GP_DOUBLE **XYp);

/**
 * @brief   Compute sum of squared Euclidean distances for a set of points.
 * @details Compute sum of squared Euclidean distances for a set of points.
 * @param [in]       X:       set of points, size n x d.
 * @param [in]       ldimX:   leading dimension of X.
 * @param [in]       n:       number of points.
 * @param [in]       d:       dimension of data.
 * @param [in,out]   XXp:     pointer to the sum of squared distances. If NULL, will be allocated.
 */
void Nfft4GPDistanceEuclidSumXX(
   NFFT4GP_DOUBLE *X, 
   int ldimX, 
   int n, 
   int d, 
   NFFT4GP_DOUBLE **XXp);

/**
 * @brief   Assemble kernel matrix from distance components.
 * @details Assemble kernel matrix from distance components.
 * @param [in]       XX:      sum of squared distances for first set of points.
 * @param [in]       nX:      number of points in first set.
 * @param [in]       YY:      sum of squared distances for second set of points.
 * @param [in]       nY:      number of points in second set.
 * @param [in]       scale:   scaling factor for the kernel.
 * @param [in,out]   K:       kernel matrix, size nX x nY.
 */
void Nfft4GPDistanceEuclidMatrixAssemble(
   NFFT4GP_DOUBLE *XX, 
   int nX, 
   NFFT4GP_DOUBLE *YY, 
   int nY, 
   NFFT4GP_DOUBLE scale, 
   NFFT4GP_DOUBLE *K);

/**
 * @brief   Find k-nearest neighbors for each point in a dataset.
 * @details Find k-nearest neighbors for each point in a dataset.
 * @param [in]       data:    data matrix, size n x d.
 * @param [in]       n:       number of points.
 * @param [in]       ldim:    leading dimension of data.
 * @param [in]       d:       dimension of data.
 * @param [in]       lfil:    number of neighbors to find.
 * @param [out]      pS_i:    row indices of neighbors.
 * @param [out]      pS_j:    column indices of neighbors.
 */
void Nfft4GPDistanceEuclidKnn(
   NFFT4GP_DOUBLE *data, 
   int n, 
   int ldim, 
   int d, 
   int lfil, 
   int **pS_i, 
   int **pS_j);

/**
 * @brief   Find k-nearest neighbors from a distance matrix.
 * @details Find k-nearest neighbors from a distance matrix.
 * @param [in]       matrix:  distance matrix, size n x n.
 * @param [in]       n:       number of points.
 * @param [in]       ldim:    leading dimension of matrix.
 * @param [in]       lfil:    number of neighbors to find.
 * @param [out]      pS_i:    row indices of neighbors.
 * @param [out]      pS_j:    column indices of neighbors.
 */
void Nfft4GPDistanceEuclidMatrixKnn(
   NFFT4GP_DOUBLE *matrix, 
   int n, 
   int ldim, 
   int lfil, 
   int **pS_i, 
   int **pS_j);

/**********************************************************************************
 * 
 * Gaussian Kernel functions
 * Gaussian kernel function K(x,y) = ff^2 * [exp(-||x-y||/(2*kk^2)) + I*noise].
 * 
 * Kernel functions are data -> matrix mappings
 * 
 * WARNING: If the stadard kernel is used, we create only the LOWER part 
 * when permc == NULL since the matrix is symmetric
 * 
 **********************************************************************************/

/**
 * @brief   Gaussian kernel function K(x,y) = ff^2 * [exp(-||x-y||/(2*kk^2)) + I*noise], possibly with gradient.
 * @details Gaussian kernel function K(x,y) = ff^2 * [exp(-||x-y||/(2*kk^2)) + I*noise], possibly with gradient.
 * @param [in]       str:     helper data structure, a pointer to the kernel data structure.
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
 *                            Note that df, dl, and dmu are stored contiguously in memory, one after another.
 * @return           Return error code.
 */
int Nfft4GPKernelGaussianKernel(
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
 * @brief   Compute distance threshold for Gaussian kernel.
 * @details Compute l such that if dist(i,j) > l, K(i,j) < tol for Gaussian kernel.
 * @param [in]       str:     kernel structure.
 * @param [in]       tol:     tolerance value.
 * @return           Return the distance threshold.
 */
NFFT4GP_DOUBLE Nfft4GPKernelGaussianKernelVal2Dist(void *str, NFFT4GP_DOUBLE tol);

/**********************************************************************************
 * 
 * Matern 3/2  Kernel functions 
 * K(x,y) = ff^2 * [(kk+sqrt(3)kk*||x-y||)exp(-sqrt(3)*kk*||x-y||) + I*noise].
 * 
 * Kernel functions are data -> matrix mappings
 * 
 * WARNING: If the stadard kernel is used, we create only the LOWER part 
 * when permc == NULL since the matrix is symmetric
 * 
 **********************************************************************************/
#ifndef NFFT4GP_MATERN32_SQRT3
#define NFFT4GP_MATERN32_SQRT3 1.7320508075688772
#endif

/**
 * @brief   Matern kernel function K(x,y) = ff^2 * [(kk+sqrt(3)kk*||x-y||)exp(-sqrt(3)*kk*||x-y||) + I*noise], possibly with gradient.
 * @details Matern kernel function K(x,y) = ff^2 * [(kk+sqrt(3)kk*||x-y||)exp(-sqrt(3)*kk*||x-y||) + I*noise], possibly with gradient.
 * @param [in]       str: helper data structure, here is the pointer to a floating point array of length 2. First one is lengthscale, second one is noise.
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
int Nfft4GPKernelMatern32Kernel(void *str, NFFT4GP_DOUBLE *data, int n, int ldim, int d, int *permr, int kr, int *permc, int kc, NFFT4GP_DOUBLE **Kp, NFFT4GP_DOUBLE **dKp);

/**
 * @brief   Matern kernel function.
 * @details Matern kernel function.
 * @param [in]       str: helper data structure
 * @param [in]       tol: drop tolerance.
 * @return           Return the real distance.
 */
NFFT4GP_DOUBLE Nfft4GPKernelMatern32KernelVal2Dist(void *str, NFFT4GP_DOUBLE tol);

/**********************************************************************************
 * 
 * Matern 1/2  Kernel functions (Exponential Kernel)
 * Also known as exponential kernel
 * K(x,y) = ff^2 * [exp(-||x-y||/kk) + I*noise].
 * 
 * Kernel functions are data -> matrix mappings
 * 
 * WARNING: If the stadard kernel is used, we create only the LOWER part 
 * when permc == NULL since the matrix is symmetric
 * 
 **********************************************************************************/

/**
 * @brief   Matern kernel function K(x,y) = ff^2 * [exp(-||x-y||/kk) + I*noise], possibly with gradient.
 * @details Matern kernel function K(x,y) = ff^2 * [exp(-||x-y||/kk) + I*noise], possibly with gradient.
 * @param [in]       str: helper data structure, here is the pointer to a floating point array of length 2. First one is lengthscale, second one is noise.
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
int Nfft4GPKernelMatern12Kernel(void *str, NFFT4GP_DOUBLE *data, int n, int ldim, int d, int *permr, int kr, int *permc, int kc, NFFT4GP_DOUBLE **Kp, NFFT4GP_DOUBLE **dKp);

/**
 * @brief   Matern kernel function.
 * @details Matern kernel function.
 * @param [in]       str: helper data structure
 * @param [in]       tol: drop tolerance.
 * @return           Return the real distance.
 */
NFFT4GP_DOUBLE Nfft4GPKernelMatern12KernelVal2Dist(void *str, NFFT4GP_DOUBLE tol);

/**********************************************************************************
 * 
 * Additive kernel: add multiple kernels together to form a new kernel
 * 
 **********************************************************************************/

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
 * @param [in]       fkernel:    kernel function.
 * @return           Return the kernel structure.
 */
void* Nfft4GPKernelAdditiveKernelParamCreate(
   NFFT4GP_DOUBLE *data, 
   int n, 
   int ldim, 
   int d,
   int *windows, 
   int nwindows, 
   int dwindows, 
   func_kernel fkernel);

/**
 * @brief   Compute additive kernel matrix.
 * @details Compute additive kernel matrix.
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
int Nfft4GPKernelAdditiveKernel(
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

/**********************************************************************************
 * 
 * Schur complement combined kernel
 * 
 **********************************************************************************/

/**
 * @brief   Create a Schur complement kernel parameter structure.
 * @details Create a Schur complement kernel parameter structure.
 * @param [in]       data:           data matrix.
 * @param [in]       n:              number of points.
 * @param [in]       ldim:           leading dimension of the data matrix.
 * @param [in]       d:              dimension of data.
 * @param [in]       perm:           permutation vector.
 * @param [in]       k:              length of permutation vector.
 * @param [in]       chol_K11:       Cholesky factorization of K11.
 * @param [in]       GdK11G:         G*dK11*G.
 * @param [in]       fkernel:        kernel function.
 * @param [in]       fkernel_params: kernel function parameters.
 * @param [in]       omp:            whether to use OpenMP.
 * @param [in]       requires_grad:  whether gradient is required.
 * @return           Return the kernel structure.
 */
void* Nfft4GPKernelSchurCombineKernelParamCreate(
   NFFT4GP_DOUBLE *data, 
   int n, 
   int ldim, 
   int d, 
   int *perm, 
   int k, 
   NFFT4GP_DOUBLE *chol_K11, 
   NFFT4GP_DOUBLE *GdK11G,
   func_kernel fkernel, 
   void *fkernel_params, 
   int omp, 
   int requires_grad);

/**
 * @brief   Compute Schur complement kernel matrix.
 * @details Compute Schur complement kernel matrix.
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
int Nfft4GPKernelSchurCombineKernel(
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

/**********************************************************************************
 * 
 * Additive kernel
 * 
 **********************************************************************************/

/* Not yet available in this branch */

#endif