#ifndef NFFT4GP_LINEARALG_HEADER_H
#define NFFT4GP_LINEARALG_HEADER_H

/**
 * @file _linearalg.h
 * @brief Wrapper of header files for external use.
 */


#include "stdio.h"
#include "stdlib.h"
#include "_utils.h"
// #include "_solvers.h"
// #include "_preconds.h"

/*------------------------------------------
 * other datastrs
 *------------------------------------------*/

/**
 * @brief   Simple integer vector.
 * @details Simple integer vector.
 */
typedef struct NFFT4GP_INTVEC_STRUCT
{
   int   _len;
   int   _max_len;
   int   *_data;
} vec_int,*pvec_int;

/**
 * @brief   Create an integer vector.
 * @details Create an integer vector.
 * @return  Pointer to the created vector.
 */
pvec_int Nfft4GPVecIntCreate();

/**
 * @brief   Initialize an integer vector.
 * @details Initialize an integer vector.
 * @param[in,out]   vec     Pointer to the vector.
 * @param[in]       n_max   Maximum length of the vector.
 */
void Nfft4GPVecIntInit(pvec_int vec, int n_max);

/**
 * @brief   Add an integer to the end of the vector.
 * @details Add an integer to the end of the vector.
 * @param[in,out]   vec     Pointer to the vector.
 * @param[in]       val     Value to be added to the vector.
 */
void Nfft4GPVecIntPushback(pvec_int vec, int val);

/**
 * @brief   Free the memory of an integer vector.
 * @details Free the memory of an integer vector.
 * @param[in,out]   vec     Pointer to the vector.
 */
void Nfft4GPVecIntFree(pvec_int vec);

/**
 * @brief   Simple double vector.
 * @details Simple double vector.
 */
typedef struct NFFT4GP_DOUBLEVEC_STRUCT
{
   int _len;
   int _max_len;
   NFFT4GP_DOUBLE *_data;
} vec_double,*pvec_double;

/**
 * @brief   Create a double vector.
 * @details Create a double vector.
 * @return  Pointer to the created vector.
 */
pvec_double Nfft4GPVecDoubleCreate();

/**
 * @brief   Initialize a double vector.
 * @details Initialize a double vector.
 * @param[in,out]   vec     Pointer to the vector.
 * @param[in]       n_max   Maximum length of the vector.
 */
void Nfft4GPVecDoubleInit(pvec_double vec, int n_max);

/**
 * @brief   Add a double value to the end of the vector.
 * @details Add a double value to the end of the vector.
 * @param[in,out]   vec     Pointer to the vector.
 * @param[in]       val     Value to be added to the vector.
 */
void Nfft4GPVecDoublePushback(pvec_double vec, NFFT4GP_DOUBLE val);

/**
 * @brief   Free the memory of a double vector.
 * @details Free the memory of a double vector.
 * @param[in,out]   vec     Pointer to the vector.
 */
void Nfft4GPVecDoubleFree(pvec_double vec);

/**
 * @brief   Data structure of a max-heap.
 * @details Data structure of a max-heap. Containing points from 0 to _max_len - 1. \n
 *          _max_len: length of arrays \n
 *          _len: current length \n
 *          _dist_v: distance array \n
 *          _index_v: index array, index[i] is the point number of dist[i] \n
 *          _rindex_v: reverse index array, rindex[i] is the location of point i in the heap
 * @note    This data structure is used in the FPS algorithm.
 */
typedef struct NFFT4GP_FPS_HEAP_STRUCT
{
   int _max_len;// max length of the heap
   int _len;

   NFFT4GP_DOUBLE *_dist_v; // dist, index, and rindex
   int *_index_v;
   int *_rindex_v;
} heap,*pheap;

/**
 * @brief   Create a max-heap.
 * @details Create a max-heap.
 * @return  Pointer to the created heap.
 */
pheap Nfft4GPHeapCreate();

/**
 * @brief   Initialize a max-heap.
 * @details Initialize a max-heap.
 * @param[in,out]   fpsheap     Pointer to the heap.
 * @param[in]       n           Maximum length of the heap.
 */
void Nfft4GPHeapInit(pheap fpsheap, int n);

/**
 * @brief   Add a (idx, dist) pair to the heap.
 * @details Add a (idx, dist) pair to the heap.
 * @param[in,out]   fpsheap     Pointer to the heap.
 * @param[in]       dist        Distance.
 * @param[in]       idx         Index.
 */
void Nfft4GPHeapAdd(pheap fpsheap, NFFT4GP_DOUBLE dist, int idx);

/**
 * @brief   Get the largest entry from the heap and remove it.
 * @details Get the largest entry from the heap and remove it.
 * @param[in,out]   fpsheap     Pointer to the heap.
 * @param[out]      dist        Distance.
 * @param[out]      idx         Index.
 */
void Nfft4GPHeapPop(pheap fpsheap, NFFT4GP_DOUBLE *dist, int *idx);

/**
 * @brief   Decrease the (idx, dist) pair of the heap if already inside.
 * @details Decrease the (idx, dist) pair of the heap if already inside.
 * @param[in,out]   fpsheap     Pointer to the heap.
 * @param[in]       dist        Distance.
 * @param[in]       idx         Index.
 */
void Nfft4GPHeapDecrease(pheap fpsheap, NFFT4GP_DOUBLE dist, int idx);

/**
 * @brief   Clean the heap.
 * @details Clean the heap.
 * @param[in,out]   fpsheap     Pointer to the heap.
 */
void Nfft4GPHeapClear(pheap fpsheap);

/**
 * @brief   Compute the 2-norm of a vector.
 * @details Compute the 2-norm of a vector.
 * @param[in]   x   Pointer to the vector.
 * @param[in]   n   Length of the vector.
 * @return  The 2-norm of the vector.
 */
NFFT4GP_DOUBLE Nfft4GPVecNorm2(NFFT4GP_DOUBLE *x, int n);

/**
 * @brief   Compute the dot product of two vectors.
 * @details Compute the dot product of two vectors.
 * @param[in]   x   Pointer to the first vector.
 * @param[in]   n   Length of the vectors.
 * @param[in]   y   Pointer to the second vector.
 * @return  The dot product of the two vectors.
 */
NFFT4GP_DOUBLE Nfft4GPVecDdot(NFFT4GP_DOUBLE *x, int n, NFFT4GP_DOUBLE *y);

/**
 * @brief   Compute the AXPY of two vectors. y = alpha*x + y.
 * @details Compute the AXPY of two vectors. y = alpha*x + y.
 * @param[in]   alpha   Scaling factor for the first vector.
 * @param[in]   x       Pointer to the first vector.
 * @param[in]   n       Length of the vectors.
 * @param[in]   y       Pointer to the second vector.
 */
void Nfft4GPVecAxpy(NFFT4GP_DOUBLE alpha, NFFT4GP_DOUBLE *x, size_t n, NFFT4GP_DOUBLE *y);

/**
 * @brief   Fill a vector with random values between 0 and 1.
 * @details Fill a vector with random values between 0 and 1.
 * @param[in,out]   x       Pointer to the vector.
 * @param[in]       n       Length of the vector.
 */
void Nfft4GPVecRand(NFFT4GP_DOUBLE *x, int n);

/**
 * @brief   Fill a vector with random values between -1 and 1.
 * @details Fill a vector with random values between -1 and 1.
 * @param[in,out]   x       Pointer to the vector.
 * @param[in]       n       Length of the vector.
 */
void Nfft4GPVecRadamacher(NFFT4GP_DOUBLE *x, int n);

/**
 * @brief   Fill a vector with a constant value.
 * @details Fill a vector with a constant value.
 * @param[in,out]   x       Pointer to the vector.
 * @param[in]       n       Length of the vector.
 * @param[in]       val     Value to be filled in the vector.
 */
void Nfft4GPVecFill(NFFT4GP_DOUBLE *x, size_t n, NFFT4GP_DOUBLE val);

/**
 * @brief   Scale a vector.
 * @details Scale a vector.
 * @param[in,out]   x       Pointer to the vector.
 * @param[in]       n       Length of the vector.
 * @param[in]       scale   Scaling factor.
 */
void Nfft4GPVecScale(NFFT4GP_DOUBLE *x, size_t n, NFFT4GP_DOUBLE scale);

/**
 * @brief   Fill a int vector with a constant value.
 * @details Fill a int vector with a constant value.
 * @param[in,out]   x       Pointer to the vector.
 * @param[in]       n       Length of the vector.
 * @param[in]       val     Value to be filled in the vector.
 */
void Nfft4GPIVecFill(int *x, int n, int val);

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
int Nfft4GPDenseMatSymv(void *data, int n, NFFT4GP_DOUBLE alpha, NFFT4GP_DOUBLE *x, NFFT4GP_DOUBLE beta, NFFT4GP_DOUBLE *y);

/**
 * @brief   Compute three symmetrix matrix-vector product y = alpha*A*x + beta*y. A, A+n*n, and A+2*n*n are three gradient matrices.
 * @details Compute three symmetrix matrix-vector product y = alpha*A*x + beta*y. A, A+n*n, and A+2*n*n are three gradient matrices.\n 
 *          The matrix is stored in the lower triangular part.
 * @param[in]   data    Pointer to the matrix. The matrix is stored in the lower triangular part.
 * @param[in]   n       Dimension of the matrix.
 * @param[in]   alpha   Scaling factor for the matrix.
 * @param[in]   x       Pointer to the vector.
 * @param[in]   beta    Scaling factor for the vector.
 * @param[in,out]   y   Pointer to the vector.
 * @return  0 if success.
 */
int Nfft4GPDenseGradMatSymv(void *data, int n, NFFT4GP_DOUBLE alpha, NFFT4GP_DOUBLE *x, NFFT4GP_DOUBLE beta, NFFT4GP_DOUBLE *y);

/**
 * @brief   Compute the general matrix-vector product y = alpha*A*x + beta*y.
 * @details Compute the general matrix-vector product y = alpha*A*x + beta*y.
 * @param[in]   data    Pointer to the matrix.
 * @param[in]   m       Number of rows of the matrix.
 * @param[in]   n       Number of columns of the matrix.
 * @param[in]   alpha   Scaling factor for the matrix.
 * @param[in]   x       Pointer to the vector.
 * @param[in]   beta    Scaling factor for the y vector.
 * @param[in,out]   y   Pointer to the vector.
 * @return  0 if success.
 */
int Nfft4GPDenseMatGemv(void *data, char trans, int m, int n, NFFT4GP_DOUBLE alpha, NFFT4GP_DOUBLE *x, NFFT4GP_DOUBLE beta, NFFT4GP_DOUBLE *y);

/**
 * @brief   Compute the inverse of a lower triangular matrix.
 * @details Compute the inverse of a lower triangular matrix.
 * @param[in,out]   data    Pointer to the matrix. The matrix is stored in the lower triangular part.
 * @param[in]   lda     Leading dimension of the matrix.
 * @param[in]   n       Dimension of the matrix.
 */
int AFnTrilNystromInv(void *data, int lda, int n);

/**
 * @brief   Compute the matrix-matrix product B = alpha*B*op(A) where A is a lower triangular matrix.
 * @details Compute the matrix-matrix product B = alpha*B*op(A).
 * @param[in]   a       Pointer to the matrix.
 * @param[in]   lda     Leading dimension of the matrix.
 * @param[in,out]   b   Pointer to the second matrix.
 * @param[in]   ldb     Leading dimension of the second matrix.
 * @param[in]   m       Number of rows of the second matrix.
 * @param[in]   n       Number of columns of the second matrix.
 * @param[in]   alpha   Scaling factor for the matrix.
 * @return  0 if success.
 */
int Nfft4GPTrilNystromMm( void *a, int lda, void *b, int ldb, int m, int n, NFFT4GP_DOUBLE alpha);

/**
 * @brief   Compute the svd of a matrix. Instead of computing it directly, it computes the eig of the matrix A'*A.
 * @details Compute the svd of a matrix. Instead of computing it directly, it computes the eig of the matrix A'*A.
 * @param[in]   a       Pointer to the matrix.
 * @param[in]   lda     Leading dimension of the matrix.
 * @param[in]   m       Number of rows of the matrix.
 * @param[in]   n       Number of columns of the matrix.
 * @param[in,out]   s   Pointer to the vector of singular values.
 * @return  0 if success.
 */
int Nfft4GPTrilNystromSvd( void *a, int lda, int m, int n, void **s);

/**
 * @brief   Compute the matrix-vector product y = alpha*A*x + beta*y of CSR matrix.
 * @details Compute the matrix-vector product y = alpha*A*x + beta*y of CSR matrix.
 * @param[in]   ia      Pointer to the row pointer.
 * @param[in]   ja      Pointer to the column indices.
 * @param[in]   aa      Pointer to the non-zero values.
 * @param[in]   nrows   Number of rows of the matrix.
 * @param[in]   ncols   Number of columns of the matrix.
 * @param[in]   trans   Transpose flag.
 * @param[in]   alpha   Scaling factor for the matrix.
 * @param[in]   x       Pointer to the x vector.
 * @param[in]   beta    Scaling factor for the y vector.
 * @param[in,out]   y   Pointer to the y vector.
 * @return  0 if success.
 */
int Nfft4GPCsrMv( int *ia, int *ja, NFFT4GP_DOUBLE *aa, int nrows, int ncols, char trans, NFFT4GP_DOUBLE alpha, NFFT4GP_DOUBLE *x, NFFT4GP_DOUBLE beta, NFFT4GP_DOUBLE *y);

/* OTHER OPERATIONS */

/**
 * @brief   Modified Gram-Schmidt orthogonalization for Arnoldi.
 * @details Modified Gram-Schmidt orthogonalization for Arnoldi.
 * @param[in,out] w           Pointer to the vector.
 * @param[in]     n           Dimension of the vector.
 * @param[in]     kdim        Dimension of the Krylov subspace.
 * @param[in]     V           Pointer to the matrix of the Krylov subspace.
 * @param[in]     H           Pointer to the Hessberg matrix.
 * @param[in]     t           Vector norm of the crruent vector.
 * @param[in]     k           Current iteration.
 * @param[in]     tol_orth    Tolerance for the orthogonality.
 * @param[in]     tol_reorth  Tolerance for the reorthogonalization.
 * @return  0 if success.
 */
int Nfft4GPModifiedGS( NFFT4GP_DOUBLE *w, int n, int kdim, NFFT4GP_DOUBLE *V, NFFT4GP_DOUBLE *H, NFFT4GP_DOUBLE *t, int k, NFFT4GP_DOUBLE tol_orth, NFFT4GP_DOUBLE tol_reorth);

/**
 * @brief   Modified Gram-Schmidt orthogonalization for Lanczos.
 * @details Modified Gram-Schmidt orthogonalization for Lanczos.
 * @param[in,out] w           Pointer to the vector.
 * @param[in]     n           Dimension of the vector.
 * @param[in]     V           Pointer to the matrix of the Krylov subspace of preconditioned vector
 * @param[in]     Z           Pointer to the matrix of the Krylov subspace of original vector
 * @param[in]     TD          Pointer to the diagonal of the tridiagonal matrix.
 * @param[in]     TE          Pointer to the subdiagonal of the tridiagonal matrix.
 * @param[in]     t           Vector norm of the crruent vector.
 * @param[in]     k           Current iteration.
 * @param[in]     tol_orth    Tolerance for the orthogonality.
 * @param[in]     tol_reorth  Tolerance for the reorthogonalization.
 * @return  0 if success.
 */
int Nfft4GPModifiedGS2( NFFT4GP_DOUBLE *w, int n, NFFT4GP_DOUBLE *V, NFFT4GP_DOUBLE *Z, NFFT4GP_DOUBLE *TD, NFFT4GP_DOUBLE *TE, NFFT4GP_DOUBLE *t, int k, NFFT4GP_DOUBLE tol_orth, NFFT4GP_DOUBLE tol_reorth);

/**
 * @brief   Function to compute the distance between two points.
 * @details Function to compute the distance between two points.
 * @param [in]       str: helper data structure
 * @param [in]       data1: pointer to the first point.
 * @param [in]       ldim1: leading dimension of data1.
 * @param [in]       data1: pointer to the second point.
 * @param [in]       ldim1: leading dimension of data2.
 * @param [in]       d: dimension of data
 * @return           Return the distance.
 */
typedef NFFT4GP_DOUBLE (*func_dist)(void *str, double *data1, int ldim1, double *data2, int ldim2, int d);

/**
 * @brief   Given a data matrix (ldim by d), evaluate the kernel matrix, possibly with gradient.
 * @details Given a data matrix (ldim by d), evaluate the kernel matrix, possibly with gradient.
 * @param [in]       str: helper data structure
 * @param [in]       data: data matrix (ldim by d).
 * @param [in]       n: number of points.
 * @param [in]       ldim: leading dimension of the data matrix
 * @param [in]       d: dimension of data
 * @param [in]       permr: select rows. If set to NULL the whole matrix is generated. \n
 *                   If not NULL< when permc is NULL, generate K(permr, permr).
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
typedef int (*func_kernel)(void *str, NFFT4GP_DOUBLE *data, int n, int ldim, int d, int *permr, int kr, int *permc, int kc, NFFT4GP_DOUBLE **Kp, NFFT4GP_DOUBLE **dKp);

/**
 * @brief   Compute l such that if dist(i,j) > l, K(i,j) < tol. This funcfion is used to estimate the drop tolerance given a distance value.
 * @details Compute l such that if dist(i,j) > l, K(i,j) < tol. This funcfion is used to estimate the drop tolerance given a distance value
 * @param [in]       str: helper data structure
 * @param [in]       tol: droptolerance.
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
 * @param [in] max_n:   maximum number of points.
 * @param [in] omp:     will this kernel be used in parallel
 * @return           Return the kernel structure.
 */
void* Nfft4GPKernelParamCreate( int max_n, int omp);

/**
 * @brief   Destroy a kernel structure.
 * @details Destroy a kernel structure.
 * @param [in]       str: kernel structure.
 * @return           Return error code.
 */
void Nfft4GPKernelParamFree(void *str);

/**
 * @brief   Free a kernel matrix.
 * @details Free a kernel matrix.
 * @param [in,out]   str: kernel matrix.
 * @return           Return error code.
 */
void Nfft4GPKernelFree(void *str);

/**********************************************************************************
 * 
 * Euclidean distance
 * 
 **********************************************************************************/

/**
 * @brief   Euclidean distance between two points.
 * @details Euclidean distance between two points.
 * @param [in]       null_params: not used
 * @param [in]       data1: data1
 * @param [in]       ldim1: leading dimension of data1
 * @param [in]       data2: data2
 * @param [in]       ldim2: leading dimension of data2
 * @param [in]       d: dimension of data
 * @return           Return the distance.
 */
NFFT4GP_DOUBLE Nfft4GPDistanceEuclid(void *null_params, double *data1, int ldim1, double *data2, int ldim2, int d);

/* Euclidean distance helper functions */

/**
 * @brief   Helper function for computing the distance matrix. This function computes the multiplication 2*X*Y'
 * @details Helper function for computing the distance matrix. This function computes the multiplication 2*X*Y'
 * @param [in]       X: data1
 * @param [in]       Y: data2
 * @param [in]       ldimX: leading dimension of data1
 * @param [in]       ldimY: leading dimension of data2
 * @param [in]       nX: number of points in data1
 * @param [in]       nY: number of points in data2
 * @param [in]       d: dimension of data
 * @param [in,out]   XYp: pointer to the result, size nX by nY. Can not be NULL. If space preallocated, it will be used. Otherwise, it will be allocated.
 * @return           Return the distance.
 */
void Nfft4GPDistanceEuclidXY(NFFT4GP_DOUBLE *X, NFFT4GP_DOUBLE *Y, int ldimX, int ldimY, int nX, int nY, int d, NFFT4GP_DOUBLE **XYp);

/**
 * @brief   Helper function for computing the distance matrix. This function computes the multiplication X.^2
 * @details Helper function for computing the distance matrix. This function computes the multiplication X.^2
 * @param [in]       X: data
 * @param [in]       ldimX: leading dimension of data1
 * @param [in]       n: number of points in data1
 * @param [in]       d: dimension of data
 * @param [in,out]   XXp: pointer to the result, size n. Can not be NULL. If space preallocated, it will be used. Otherwise, it will be allocated.
 * @return           Return the distance.
 */
void Nfft4GPDistanceEuclidSumXX(NFFT4GP_DOUBLE *X, int ldimX, int n, int d, NFFT4GP_DOUBLE **XXp);

/**
 * @brief   Helper function for computing the distance matrix.
 * @details Helper function for computing the distance matrix.
 * @param [in]       XX: X.^2
 * @param [in]       nX: size of XX
 * @param [in]       YY: Y.^2
 * @param [in]       nY: size of YY
 * @param [in]       scale: scale
 * @param [in,out]   K: pointer to the result, when input should be the output of EuclidXY.
 * @return           Return the distance.
 */
void Nfft4GPDistanceEuclidMatrixAssemble(NFFT4GP_DOUBLE *XX, int nX, NFFT4GP_DOUBLE *YY, int nY, NFFT4GP_DOUBLE scale, NFFT4GP_DOUBLE *K);

/**
 * @brief   Helper function for computing KNN.
 * @details Helper function for computing KNN.
 * @param [in]       data: data matrix (ldim by d).
 * @param [in]       n: number of points.
 * @param [in]       ldim: leading dimension of the data matrix
 * @param [in]       d: dimension of data
 * @param [in]       lfil: number of neighbors
 * @param [in,out]   pS_i: pointer to the I of the CSR format. Can not be NULL. Should not preallocated.
 * @param [in,out]   pS_j: pointer to the J of the CSR format. Can not be NULL. Should not preallocated.
 * @return           Return the distance.
 */
void Nfft4GPDistanceEuclidKnn(NFFT4GP_DOUBLE *data, int n, int ldim, int d, int lfil, int **pS_i, int **pS_j);

/**
 * @brief   Helper function for computing KNN.
 * @details Helper function for computing KNN.
 * @param [in]       data: data matrix (ldim by d).
 * @param [in]       n: number of points.
 * @param [in]       ldim: leading dimension of the data matrix
 * @param [in]       d: dimension of data
 * @param [in]       lfil: number of neighbors
 * @param [in,out]   pS_i: pointer to the I of the CSR format. Can not be NULL. Should not preallocated.
 * @param [in,out]   pS_j: pointer to the J of the CSR format. Can not be NULL. Should not preallocated.
 * @return           Return the distance.
 */
void Nfft4GPDistanceEuclidMatrixKnn(NFFT4GP_DOUBLE *matrix, int n, int ldim, int lfil, int **pS_i, int **pS_j);

/**********************************************************************************
 * 
 * Gaussian Kernel functions
 * Gaussian kernel function K(x,y) = ff^2 * exp(-||x-y||/(2*kk^2)) + I*noise.
 * 
 * Kernel functions are data -> matrix mappings
 * 
 * WARNING: If the stadard kernel is used, we create only the LOWER part 
 * when permc == NULL since the matrix is symmetric
 * 
 **********************************************************************************/

/**
 * @brief   Gaussian kernel function K(x,y) = ff^2 * exp(-||x-y||/(2*kk^2)) + I*noise, possibly with gradient.
 * @details Gaussian kernel function K(x,y) = ff^2 * exp(-||x-y||/(2*kk^2)) + I*noise, possibly with gradient.
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
int Nfft4GPKernelGaussianKernel(void *str, NFFT4GP_DOUBLE *data, int n, int ldim, int d, int *permr, int kr, int *permc, int kc, NFFT4GP_DOUBLE **Kp, NFFT4GP_DOUBLE **dKp);

/**
 * @brief   Gaussian kernel function.
 * @details Gaussian kernel function.
 * @param [in]       str: helper data structure
 * @param [in]       tol: drop tolerance.
 * @return           Return the real distance.
 */
NFFT4GP_DOUBLE Nfft4GPKernelGaussianKernelVal2Dist(void *str, NFFT4GP_DOUBLE tol);

/**********************************************************************************
 * 
 * Matern 3/2  Kernel functions 
 * K(x,y) = ff^2 * (kk+sqrt(3)kk*||x-y||)exp(-sqrt(3)*kk*||x-y||) + I*noise.
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
 * @brief   Matern kernel function K(x,y) = ff^2 * (kk+sqrt(3)kk*||x-y||)exp(-sqrt(3)*kk*||x-y||) + I*noise, possibly with gradient.
 * @details Matern kernel function K(x,y) = ff^2 * (kk+sqrt(3)kk*||x-y||)exp(-sqrt(3)*kk*||x-y||) + I*noise, possibly with gradient.
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
 * Matern 1/2  Kernel functions 
 * K(x,y) = ff^2 * [(kk+kk*||x-y||)exp(-kk*||x-y||) + I*noise].
 * 
 * Kernel functions are data -> matrix mappings
 * 
 * WARNING: If the stadard kernel is used, we create only the LOWER part 
 * when permc == NULL since the matrix is symmetric
 * 
 **********************************************************************************/

/**
 * @brief   Matern kernel function K(x,y) = ff^2 * [(kk+kk*||x-y||)exp(-kk*||x-y||) + I*noise], possibly with gradient.
 * @details Matern kernel function K(x,y) = ff^2 * [(kk+kk*||x-y||)exp(-kk*||x-y||) + I*noise], possibly with gradient.
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
 * @brief   Additive kernel function K(x,y) = sum_i K_i(x,y).
 * @details Additive kernel function K(x,y) = sum_i K_i(x,y).
 * @param [in]       data: data matrix (ldim by d).
 * @param [in]       n: number of points.
 * @param [in]       ldim: leading dimension of the data matrix
 * @param [in]       d: dimension of data
 * @param [in]       windows: feature index for each kernel
 * @param [in]       nwindows: number of kernels
 * @param [in]       dwindows: dimension of each kernel
 * @param [in]       fkernel: kernel function for each kernel
 * @return           Return the kernel structure.
 */
void* Nfft4GPKernelAdditiveKernelParamCreate(NFFT4GP_DOUBLE *data, int n, int ldim, int d,
                                          int *windows, int nwindows, int dwindows, func_kernel fkernel);

/**
 * @brief   Additive kernel function K(x,y) = sum_i K_i(x,y).
 * @details Additive kernel function K(x,y) = sum_i K_i(x,y).
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
int Nfft4GPKernelAdditiveKernel(void *str, NFFT4GP_DOUBLE *data, int n, int ldim, int d, int *permr, int kr, int *permc, int kc, NFFT4GP_DOUBLE **Kp, NFFT4GP_DOUBLE **dKp);

/**********************************************************************************
 * 
 * Schur complement combined kernel
 * 
 **********************************************************************************/

/**
 * @brief   Since Schur complement can be complex, we recommend using this wrapper.
 * @details Since Schur complement can be complex, we recommend using this wrapper.
 * @param [in]       data: data matrix (ldim by d).
 * @param [in]       n: number of points.
 * @param [in]       ldim: leading dimension of the data matrix
 * @param [in]       d: dimension of data
 * @param [in]       perm: permutation array for the Schur complement reordering, size of n
 * @param [in]       k: size of K11
 * @param [in]       chol_K11: Cholesky factor of K11 (lower part)
 * @param [in]       GdK11G: L\dK11/L^T
 * @param [in]       fkernel: kernel function
 * @param [in]       fkernel_params: kernel function parameters
 * @param [in]       omp: will this kernel be used in parallel
 * @param [in]       requires_grad: requires gradient?
 */
void* Nfft4GPKernelSchurCombineKernelParamCreate(NFFT4GP_DOUBLE *data, int n, int ldim, int d, 
                                             int *perm, int k, NFFT4GP_DOUBLE *chol_K11, NFFT4GP_DOUBLE *GdK11G,
                                             func_kernel fkernel, void *fkernel_params, int omp, int requires_grad);

/**
 * @brief   Schur complement combined kernel function.
 * @details Schur complement combined kernel function.
 * @param [in]       str: helper data structure, here is the pointer to a floating point array of length 2. First one is lengthscale, second one is noise.
 * @param [in]       data: data matrix (ldim by d).
 * @param [in]       n: number of points.
 * @param [in]       ldim: leading dimension of the data matrix
 * @param [in]       d: dimension of data
 * @param [in]       permr: select rows. If set to NULL the whole matrix is generated. \n
 *                  If not NULL, when permc is NULL, generate K(permr, permr).
 * @param [in]       kr: length of permr.
 * @param [in]       permc: select columns. Only works if permr is not NULL, generate K(permr, permc).
 * @param [in]       kr: length of permc.
 * @param [in,out]   Kp: pointer to the Kernel matrix. Can not be NULL. If space preallocated, it will be used. Otherwise, it will be allocated.
 * @param [in]       ldimk: leading dimension of *Kp if preallocated.
 * @return           Return error code.
 */
int Nfft4GPKernelSchurCombineKernel(void *str, NFFT4GP_DOUBLE *data, int n, int ldim, int d, int *permr, int kr, int *permc, int kc, NFFT4GP_DOUBLE **Kp, NFFT4GP_DOUBLE **dKp);

/**********************************************************************************
 * 
 * Schur complement combined kernel
 * 
 **********************************************************************************/

/**
 * @brief   General ordering function. Get a ordering of length k.
 * @details Geteral ordering function. Get a ordering of length k.
 * @param [in]       str: helper data structure
 * @param [in]       data: data matrix (ldim by d).
 * @param [in]       n: number of points.
 * @param [in]       ldim: leading dimension of the data matrix
 * @param [in]       d: dimension of data
 * @param [in,out]   k: number of selected points, out put the nember of selected points.
 * @param [out]      pperm: pointer to the permutation can not be NULL.
 */
typedef int (*func_ordering)(void *str, NFFT4GP_DOUBLE *data, int n, int ldim, int d, int *k, int **pperm);

/* FPS ordering */

typedef enum NFFT4GP_FPS_PATTERN_ENUM 
{
   kFpsPatternDefault = 0, // build both pattern
   kFpsPatternUonly // build only U pattern
}fps_pattern_enum;

typedef enum NFFT4GP_FPS_ALGORITHM_ENUM 
{
   kFpsAlgorithmParallel1 = 0, // O(nlogn) sequential algorithm
   kFpsAlgorithmSequential1 // O(n^2) parallel algorithm
}fps_algorithm_enum;

typedef struct NFFT4GP_FPS_STRUCT 
{
   /* FPS paramters */
   fps_algorithm_enum _algorithm; // algorithm to use
   double _tol; // tolerance to stop the FPS. The value is used only if lfil is not set.
   double _rho; // radius of the FPS search for the O(nlogn) algorithm
   func_dist _fdist; // fill distance function, by default is the euclidean distance
   void *_fdist_params; // parameter for the distance function
   NFFT4GP_DOUBLE *_dist; // distance array

   /* pattern paramters */
   int _build_pattern; // should we build pattern?
   int _pattern_lfil; // nnz for the nonzero pattern in the upper triangular part per column
   fps_pattern_enum _pattern_opt;

   /* data structures holding the pattern in CSC format */
   int *_S_i; // row pointer
   int *_S_j; // column index

}ordering_fps,*pordering_fps;

/**
 * @brief   Create a FPS ordering structure, and set the default parameters.
 * @details Create a FPS ordering structure, and set the default parameters.
 * @return  Pointer to the created structure.
 */
void* Nfft4GPOrdFpsCreate();

/**
 * @brief   Free the memory of the FPS ordering structure.
 * @details Free the memory of the FPS ordering structure.
 * @param [in]       str: pointer to the structure.
 */
void Nfft4GPOrdFpsFree(void *str);

/**
 * @brief   Compute the FPS of data points. This is a general interface.
 * @details Compute the FPS of data points. This is a general interface.
 * @param [in]       data: data values. n times d.
 * @param [in]       n: number of data points.
 * @param [in]       ldim: data leading dimension.
 * @param [in]       d: data dimension.
 * @param [in,out]   lfil: number of selected points, out put the nember of selected points. \n
 *                         If the input <= 0 then only tol is used.
 * @return           Return error message.
 */
int Nfft4GPSortFps(void *str, NFFT4GP_DOUBLE *data, int n, int ldim, int d, int *k, int **pperm);

/* random ordering */

typedef enum NFFT4GP_RANDORD_SEED_ENUM 
{
   kRandordDefault = 0, // do not reset seed
   kRandprdResetSeed // reset random seed
}fps_randord_seed_enum;

typedef struct NFFT4GP_RAND_STRUCT 
{
   /* FPS paramters */
   fps_randord_seed_enum _reset_seed; // should we reset the seed?
   int _seed; // seed for the random number generator
}ordering_rand,*pordering_rand;

/**
 * @brief   Create a random ordering structure, and set the default parameters.
 * @details Create a random ordering structure, and set the default parameters.
 * @return  Pointer to the created structure.
 */
void* Nfft4GPOrdRandCreate();

/**
 * @brief   Free the memory of the random ordering structure.
 * @details Free the memory of the random ordering structure.
 * @param [in]       str: pointer to the structure.
 */
void Nfft4GPOrdRandFree(void *str);

/**
 * @brief   Compute a random permutation of data points.
 * @details Compute a random permutation of data points.
 * @param [in]       data: data values. n times d.
 * @param [in]       n: number of data points.
 * @param [in]       ldim: data leading dimension.
 * @param [in]       d: data dimension.
 * @param [in,out]   lfil: number of selected points, out put the nember of selected points. \n
 *                         If the input <= 0 then only tol is used.
 * @return           Return error message.
 */
int Nfft4GPSortRand(void *str, NFFT4GP_DOUBLE *data, int n, int ldim, int d, int *k, int **pperm);

typedef struct NFFT4GP_RANKEST_STRUCT 
{
   int _nsample; // size of subsampled dataset
   int _nsample_r; // numbers we redo the subsampling
   int _max_rank; // maximum rank we consider
   NFFT4GP_DOUBLE _full_tol; // tolerance for skipping the rank estimation
   func_kernel _kernel_func; // pointer to the kernel function
   void* _kernel_str; // data for the kernel function we do not free this pointer here
   void* _ordering_str; // we do not free this pointer here
   int* _perm; // permutation of the selected data, might stored here for some algorithms
}rankest,*prankest;

/**
 * @brief   Create a helper data structure for rank estimation.
 * @details Create a helper data structure for rank estimation.
 * @return  Pointer to the created data structure.
 */
void* Nfft4GPRankestStrCreate();

/**
 * @brief   Free the memory of a helper data structure for rank estimation.
 * @details Free the memory of a helper data structure for rank estimation.
 * @param [in,out]   str: helper data structure
 */
void Nfft4GPRankestStrFree(void* str);

/**
 * @brief   Rank estimation of a given kernel. Return the estimated rank.
 * @details Rank estimation of a given kernel. Return the estimated rank. \n
 *          Note that this rank is not the numerical rank of the kernel matrix. This is a near
 *          optimal rank for the NFFT4GP preconditioner.
 * @param [in]       str: helper data structure
 * @param [in]       data: data matrix (ldim by d).
 * @param [in]       n: number of points.
 * @param [in]       ldim: leading dimension of the data matrix
 * @param [in]       d: dimension of data
 * @return           Return the tolerance of distance.
 */
typedef int (*func_rank_esimate)(void *str, double *data, int n, int ldim, int d);

/**
 * @brief   Rank estimation of a given kernel. Return the estimated rank.
 * @details Rank estimation of a given kernel. Return the estimated rank. \n
 *          Note that this rank is not the numerical rank of the kernel matrix. This is a near
 *          optimal rank for the NFFT4GP preconditioner.
 * @param [in]       str: helper data structure
 * @param [in]       data: data matrix (ldim by d).
 * @param [in]       n: number of points.
 * @param [in]       ldim: leading dimension of the data matrix
 * @param [in]       d: dimension of data
 * @return           Return the tolerance of distance.
 */
int Nfft4GPRankestDefault(void *str, double *data, int n, int ldim, int d);

/**
 * @brief   Rank estimation of a given kernel. Return the estimated rank. This version is a comprehensive version using Nystrom approximation.
 * @details Rank estimation of a given kernel. Return the estimated rank. This version is a comprehensive version using Nystrom approximation. \n
 *          Note that this rank is not the numerical rank of the kernel matrix. This is a near
 *          optimal rank for the NFFT4GP preconditioner.
 * @param [in]       str: helper data structure
 * @param [in]       data: data matrix (ldim by d).
 * @param [in]       n: number of points.
 * @param [in]       ldim: leading dimension of the data matrix
 * @param [in]       d: dimension of data
 * @return           Return the tolerance of distance.
 */
int Nfft4GPRankestNysScaled(void *str, double *data, int n, int ldim, int d);

#endif