#ifndef NFFT4GP_UTILS_HEADER_H
#define NFFT4GP_UTILS_HEADER_H

/**
 * @file _utils.h
 * @brief Wrapper of header files for external use.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

// OpenMP
#ifdef NFFT4GP_USING_OPENMP
#include "omp.h"
// note: NFFT4GP_DEFAULT_OPENMP_SCHEDULE is also defined in util.h
#ifndef NFFT4GP_DEFAULT_OPENMP_SCHEDULE
#define NFFT4GP_DEFAULT_OPENMP_SCHEDULE schedule(static)
#endif

// Some rountines are not good with too many threads
// change to #define NFFT4GP_OPENMP_REDUCED_THREADS nthreads to use all
#define NFFT4GP_OPENMP_REDUCED_THREADS 1

#endif

#define NFFT4GP_MIN(a, b, c) {\
   (c) = (a) <= (b) ? (a) : (b);\
}

#define NFFT4GP_MAX(a, b, c) {\
   (c) = (a) >= (b) ? (a) : (b);\
}

#define NFFT4GP_MALLOC(ptr, length, ...) {\
   (ptr) = (__VA_ARGS__*) Nfft4GPMalloc( (size_t)(length)*sizeof(__VA_ARGS__));\
}

#define NFFT4GP_CALLOC(ptr, length, ...) {\
   (ptr) = (__VA_ARGS__*) Nfft4GPCalloc( (size_t)(length)*sizeof(__VA_ARGS__), 1);\
}

#define NFFT4GP_REALLOC(ptr, length, ...) {\
   (ptr) = (__VA_ARGS__*) Nfft4GPRealloc( (void*)(ptr), (size_t)(length)*sizeof(__VA_ARGS__));\
}

#define NFFT4GP_MEMCPY(ptr_to, ptr_from, length, ...) {\
   Nfft4GPMemcpy( (void*)(ptr_to), (void*)(ptr_from), (size_t)(length)*sizeof(__VA_ARGS__));\
}

#define NFFT4GP_FREE( ptr) {\
   if(ptr){Nfft4GPFreeHost( (void*)(ptr));}\
   (ptr) = NULL;\
}

/**
 * @brief Allocate memory on host
 * @details Allocate memory on host
 * @param[in] size Size of memory to be allocated
 * @return Pointer to allocated memory
 */
static inline void* Nfft4GPMalloc(size_t size)
{
   void *ptr = NULL;
   ptr = malloc(size);
   return ptr;
}

/**
 * @brief Allocate memory on host and initialize to zero
 * @details Allocate memory on host and initialize to zero
 * @param[in] length Length of memory to be allocated
 * @param[in] unitsize Size of each unit of memory
 * @return Pointer to allocated memory
 */
static inline void* Nfft4GPCalloc(size_t length, int unitsize)
{
   void *ptr = NULL;
   ptr = calloc(length, unitsize);
   return ptr;
}

/**
 * @brief Reallocate memory on host
 * @details Reallocate memory on host
 * @param[in,out] ptr Pointer to memory to be reallocated
 * @param[in] size Size of memory to be allocated
 * @return Pointer to allocated memory
 */
static inline void* Nfft4GPRealloc(void *ptr, size_t size)
{
   return ptr ? realloc( ptr, size ) : malloc( size );
}

/**
 * @brief Copy memory on host
 * @details Copy memory on host
 * @param[in,out] ptr_to Pointer to memory to be copied to
 * @param[in] ptr_from Pointer to memory to be copied from
 * @param[in] size Size of memory to be copied
 */
static inline void Nfft4GPMemcpy(void *ptr_to, void *ptr_from, size_t size)
{
#ifdef NFFT4GP_USING_OPENMP
#ifndef NFFT4GP_OPENMP_NO_MEMCPY
   // use openmp to copy if possible, might not gain on all systems
   if(!omp_in_parallel())
   {
      size_t i;
      #pragma omp parallel for NFFT4GP_DEFAULT_OPENMP_SCHEDULE
      for(i = 0; i < size; i++)
      {
         ((char*)ptr_to)[i] = ((char*)ptr_from)[i];
      }
      return;
   }
#endif
#endif
   memcpy( ptr_to, ptr_from, size);
}

/**
 * @brief Free memory on host
 * @details Free memory on host
 * @param[in,out] ptr Pointer to memory to be freed
 */
static inline void Nfft4GPFreeHost(void *ptr)
{
   free(ptr);
}

// Expand factor for data allocation
#define NFFT4GP_EXPAND_FACT 1.3

// Double or Float
#ifdef NFFT4GP_USING_FLOAT32
#define NFFT4GP_DOUBLE float
#else
#define NFFT4GP_DOUBLE double
#endif

/**
 * @brief   Get time in second. Always in double precision.
 * @details Get time in second. Always in double precision.
 * @return           Return time in second.
 */
double Nfft4GPWtime();

/**
 * @brief   Swap two entries in two arrays (an int array and a real array).
 * @details Swap two entries in two arrays (an int array and a real array).
 * @param [in,out]   v_i: int array.
 * @param [in,out]   v_d: real array.
 * @param [in]       i:
 * @param [in]       j: swap v_i[i] with v_i[j], v_d[i] with v_d[j].
 * @return           No return.
 */
void Nfft4GPSwap( int *v_i, NFFT4GP_DOUBLE *v_d, int i, int j);

/**
 * @brief   Quick sort basec on v_d in ascending order between v_d[l] and v_d[r]. Also swap v_i.
 * @details Quick sort basec on v_d in ascending order between v_d[l] and v_d[r]. Also swap v_i.
 * @param [in,out]   v_i: int array.
 * @param [in,out]   v_d: real array, sort based on its value.
 * @param [in]       l:
 * @param [in]       r: sort between v_d[l] and v_d[r], includes v_d[r].
 * @return           No return.
 */
void Nfft4GPQsortAscend( int *v_i, NFFT4GP_DOUBLE *v_d, int l, int r);

/**
 * @brief   Quick split. Similar to quick sort, but stop when v_d[l] till v_d[k] are sorted.
 * @details Quick split. Similar to quick sort, but stop when v_d[l] till v_d[k] are sorted.
 * @param [in,out]   v_i: int array.
 * @param [in,out]   v_d: real array, sort based on its value.
 * @param [in]       k: stop when v_d[l] till v_d[k] are sorted.
 * @param [in]       l:
 * @param [in]       r: sort between v_d[l] and v_d[r], includes v_d[r].
 * @return           No return.
 */
void Nfft4GPQsplitAscend( int *v_i, NFFT4GP_DOUBLE *v_d, int k, int l, int r);

/**
 * @brief   Generate random permutation in [0,n-1] of length k.
 * @details Generate random permutation in [0,n-1] of length k.
 * @param [in]       n: max number is n-1.
 * @param [in]       k: length of randperm is k.
 * @return           Return the array.
 */
int* Nfft4GPRandPerm( int n, int k);

/**
 * @brief   Given a data matrix (n by d), extract several rows from it.
 * @details Given a data matrix (n by d), extract several rows from it.
 * @param [in]       data: data matrix, n by d
 * @param [in]       n: number of points in data
 * @param [in]       ldim: leading dimension of data matrix
 * @param [in]       d: dimension of data
 * @param [in]       perm: permutation vector
 * @param [in]       k: length of permutatin vector.
 * @return           Return the sub data (k by d).
 */
NFFT4GP_DOUBLE* Nfft4GPSubData( NFFT4GP_DOUBLE *data, int n, int ldim, int d, int *perm, int k);

/**
 * @brief   Given a data matrix (n by d), extract several rows from it.
 * @details Given a data matrix (n by d), extract several rows from it.
 * @param [in]       data: data matrix, n by d
 * @param [in]       n: number of points in data
 * @param [in]       ldim: leading dimension of data matrix
 * @param [in]       d: dimension of data
 * @param [in]       perm: permutation vector
 * @param [in]       k: length of permutatin vector.
 * @param [in,out]   subdata: sub data matrix, k by d, memory must be pre-allocated!
 * @return           Return error code.
 */
int Nfft4GPSubData2( NFFT4GP_DOUBLE *data, int n, int ldim, int d, int *perm, int k, NFFT4GP_DOUBLE *subdata);

/**
 * @brief   Given a kernel matrix (n by n) K and permutation p, extract a submatrix K(p,p) from it.
 * @details Given a kernel matrix (n by n) K and permutation p, extract a submatrix K(p,p) from it.
 * @param [in]       K: kernel matrix, n by n
 * @param [in]       n: size of the matrix
 * @param [in]       ldim: leading dimension of the matrix
 * @param [in]       perm: permutation vector
 * @param [in]       k: length of permutatin vector.
 * @return           Return the submatrix (k by k).
 */
NFFT4GP_DOUBLE* Nfft4GPSubMatrix( NFFT4GP_DOUBLE *K, int n, int ldim, int *perm, int k);

/**
 * @brief   Given the fisrt k element of the permutation, expand it to a permutation of length n.
 * @details Given the fisrt k element of the permutation, expand it to a permutation of length n.
 * @param [in]       perm: permutation vector
 * @param [in]       k: length of permutatin vector.
 * @return           Return the permutation array.
 */
int* Nfft4GPExpandPerm( int *perm, int k, int n);

/* create uniformly distributed points
 * data: d by n, Fortran style matrix data, each column is a data point
 */
/**
 * @brief   Create points on 2D regular grid. Column major. Grid size is 1. Each column is a dimension.
 * @details Create points on 2D regular grid. Column major. Grid size is 1. Each column is a dimension.
 * @param [in]       n: number of points
 * @param [in]       d: dimension of points
 * @return           Return the data matrix (d by n).
 */
void* Nfft4GPDatasetRegular2D(int nx, int ny);

/**
 * @brief   Create uniformly distributed points scaled by n^(1/d). Column major. Each column is a dimension.
 * @details Create uniformly distributed points scaled by n^(1/d). Column major. Each column is a dimension.
 * @param [in]       n: number of points
 * @param [in]       d: dimension of points
 * @return           Return the data matrix (d by n).
 */
void* Nfft4GPDatasetUniformRandom(int n, int d);

/* BLAS/LAPACK */

#ifdef NFFT4GP_USING_FLOAT32

#ifdef NFFT4GP_USING_MKL

#include "mkl.h"
#define NFFT4GP_DSYEV       ssyev
#define NFFT4GP_DAXPY       saxpy
#define NFFT4GP_DDOT        sdot
#define NFFT4GP_DGEMV       sgemv
#define NFFT4GP_DSYMV       ssymv
#define NFFT4GP_DPOTRF      spotrf
#define NFFT4GP_TRTRS       strtrs
#define NFFT4GP_DTRTRI      strtri
#define NFFT4GP_DTRMM       strmm
#define NFFT4GP_DGESVD      sgesvd
#define NFFT4GP_DGESVDX     sgesvdx
#define NFFT4GP_DGEMM       sgemm
#define NFFT4GP_DGESV       sgesv
#define NFFT4GP_DGETRF      sgetrf
#define NFFT4GP_DGETRI      sgetri
#define NFFT4GP_DLANGE      slange
#define NFFT4GP_DLANTB      slantb
#define NFFT4GP_DLANSY      slansy
#else
#define NFFT4GP_DSYEV       ssyev_
#define NFFT4GP_DAXPY       saxpy_
#define NFFT4GP_DDOT        sdot_
#define NFFT4GP_DGEMV       sgemv_
#define NFFT4GP_DSYMV       ssymv_
#define NFFT4GP_DPOTRF      spotrf_
#define NFFT4GP_TRTRS       strtrs_
#define NFFT4GP_DTRTRI      strtri_
#define NFFT4GP_DTRMM       strmm_
#define NFFT4GP_DGESVD      sgesvd_
#define NFFT4GP_DGESVDX     sgesvdx_
#define NFFT4GP_DGEMM       sgemm_
#define NFFT4GP_DGESV       sgesv_
#define NFFT4GP_DGETRF      sgetrf_
#define NFFT4GP_DGETRI      sgetri_
#define NFFT4GP_DLANGE      slange_
#define NFFT4GP_DLANTB      slantb_
#define NFFT4GP_DLANSY      slansy_

#endif // #ifdef NFFT4GP_USING_MKL

#ifdef NFFT4GP_USING_ARPACK

#define NFFT4GP_DSAUPD ssaupd_
#define NFFT4GP_DSEUPD sseupd_

#endif

#else // #ifdef NFFT4GP_USING_FLOAT32

#ifdef NFFT4GP_USING_MKL

#include "mkl.h"
#define NFFT4GP_DSYEV       dsyev
#define NFFT4GP_DAXPY       daxpy
#define NFFT4GP_DDOT        ddot
#define NFFT4GP_DGEMV       dgemv
#define NFFT4GP_DSYMV       dsymv
#define NFFT4GP_DPOTRF      dpotrf
#define NFFT4GP_TRTRS       dtrtrs
#define NFFT4GP_DTRTRI      dtrtri
#define NFFT4GP_DTRMM       dtrmm
#define NFFT4GP_DGESVD      dgesvd
#define NFFT4GP_DGESVDX     dgesvdx
#define NFFT4GP_DGEMM       dgemm
#define NFFT4GP_DGESV       dgesv
#define NFFT4GP_DGETRF      dgetrf
#define NFFT4GP_DGETRI      dgetri
#define NFFT4GP_DLANGE      dlange
#define NFFT4GP_DLANTB      dlantb
#define NFFT4GP_DLANSY      dlansy
#else
#define NFFT4GP_DSYEV       dsyev_
#define NFFT4GP_DAXPY       daxpy_
#define NFFT4GP_DDOT        ddot_
#define NFFT4GP_DGEMV       dgemv_
#define NFFT4GP_DSYMV       dsymv_
#define NFFT4GP_DPOTRF      dpotrf_
#define NFFT4GP_TRTRS       dtrtrs_
#define NFFT4GP_DTRTRI      dtrtri_
#define NFFT4GP_DTRMM       dtrmm_
#define NFFT4GP_DGESVD      dgesvd_
#define NFFT4GP_DGESVDX     dgesvdx_
#define NFFT4GP_DGEMM       dgemm_
#define NFFT4GP_DGESV       dgesv_
#define NFFT4GP_DGETRF      dgetrf_
#define NFFT4GP_DGETRI      dgetri_
#define NFFT4GP_DLANGE      dlange_
#define NFFT4GP_DLANTB      dlantb_
#define NFFT4GP_DLANSY      dlansy_

#endif // #ifdef NFFT4GP_USING_MKL

#ifdef NFFT4GP_USING_ARPACK

#define NFFT4GP_DSAUPD dsaupd_
#define NFFT4GP_DSEUPD dseupd_

#endif

#endif // #ifdef NFFT4GP_USING_FLOAT32

#ifndef NFFT4GP_USING_MKL

void NFFT4GP_DSYEV(char *jobz, char *uplo, int *n, NFFT4GP_DOUBLE *a, int *lda, NFFT4GP_DOUBLE *w, NFFT4GP_DOUBLE *work, int *lwork, int *info);

void NFFT4GP_DAXPY(int *n, const NFFT4GP_DOUBLE *alpha, const NFFT4GP_DOUBLE *x, int *incx, NFFT4GP_DOUBLE *y, int *incy);

NFFT4GP_DOUBLE NFFT4GP_DDOT(int *n, NFFT4GP_DOUBLE *x, int *incx, NFFT4GP_DOUBLE *y, int *incy);

void NFFT4GP_DGEMV(char *trans, int *m, int *n, const NFFT4GP_DOUBLE *alpha, const NFFT4GP_DOUBLE *a,
               int *lda, const NFFT4GP_DOUBLE *x, int *incx, const NFFT4GP_DOUBLE *beta, NFFT4GP_DOUBLE *y, int *incy);

void NFFT4GP_DSYMV(char *uplo, int *n, NFFT4GP_DOUBLE *alpha, NFFT4GP_DOUBLE *a, int *lda, NFFT4GP_DOUBLE *x, int *incx, NFFT4GP_DOUBLE *beta, NFFT4GP_DOUBLE *y, int *incy);

void NFFT4GP_DPOTRF(char *uplo, int *n, NFFT4GP_DOUBLE *a, int *lda, int *info);

void NFFT4GP_TRTRS(char *uplo, char *trans, char *diag, int *n, int *nrhs, NFFT4GP_DOUBLE *a, int *lda, NFFT4GP_DOUBLE *b, int *ldb, int *info);

void NFFT4GP_DTRTRI( char *uplo, char *diag, int *n, NFFT4GP_DOUBLE *a, int *lda, int *info);

void NFFT4GP_DTRMM( char *side, char *uplo, char *transa, char *diag, int *m, int *n, NFFT4GP_DOUBLE *alpha, NFFT4GP_DOUBLE *a, int *lda, NFFT4GP_DOUBLE *b, int *ldb);

void NFFT4GP_DGESVD(char *jobu, char *jobvt, int *m, int *n, NFFT4GP_DOUBLE *a, int *lda, NFFT4GP_DOUBLE *s, NFFT4GP_DOUBLE *u, int *ldu,
                int *vt, int *ldvt, NFFT4GP_DOUBLE *work, int *lwork, int *info);

void NFFT4GP_DGESVDX( char *jobu, char *jobvt, char *range, int *m, int *n, NFFT4GP_DOUBLE *a, int *lda, NFFT4GP_DOUBLE *vl, NFFT4GP_DOUBLE *vu,
                  int *il, int *iu, int *ns, NFFT4GP_DOUBLE *s, NFFT4GP_DOUBLE *u, int *ldu, int *vt, int *ldvt, NFFT4GP_DOUBLE *work, int *lwork, int *info);

void NFFT4GP_DGEMM(char *transa, char *transb, int *m, int *n, int *k, const NFFT4GP_DOUBLE *alpha, const NFFT4GP_DOUBLE *a, int *lda,
               const NFFT4GP_DOUBLE *b, int *ldb, const NFFT4GP_DOUBLE *beta, NFFT4GP_DOUBLE *c, int *ldc);

void NFFT4GP_DGESV( int *n, int *nrhs, NFFT4GP_DOUBLE *a, int *lda, int *ipiv, NFFT4GP_DOUBLE *b, int *ldb, int *info);

void NFFT4GP_DGETRF( int *m, int *n, NFFT4GP_DOUBLE *a, int *lda, int *ipiv, int *info);

void NFFT4GP_DGETRI( int *n, NFFT4GP_DOUBLE *a, int *lda, int *ipiv, NFFT4GP_DOUBLE *work, int *lwork, int *info);

NFFT4GP_DOUBLE NFFT4GP_DLANGE( char *norm, int *m, int *n, NFFT4GP_DOUBLE *A, int *lda, NFFT4GP_DOUBLE *work);

NFFT4GP_DOUBLE NFFT4GP_DLANTB( char *norm, char *uplo, char *diag, int *n, int *k, NFFT4GP_DOUBLE *A, int *lda, NFFT4GP_DOUBLE *work);

NFFT4GP_DOUBLE NFFT4GP_DLANSY( char *norm, char *uplo, int *n, NFFT4GP_DOUBLE *A, int *lda, NFFT4GP_DOUBLE *work);

#endif // #ifndef NFFT4GP_USING_MKL

#ifdef NFFT4GP_USING_ARPACK

void NFFT4GP_DSAUPD( int* ido, char* bmat, int* n, char* which, int* nev, NFFT4GP_DOUBLE* tol, NFFT4GP_DOUBLE* resid, int* ncv, NFFT4GP_DOUBLE* v, int* ldv, int* iparam, int* ipntr, NFFT4GP_DOUBLE* workd, NFFT4GP_DOUBLE* workl, int* lworkl, int* info);

void NFFT4GP_DSEUPD( int* rvec, char* howmny, int* select, NFFT4GP_DOUBLE* d, NFFT4GP_DOUBLE* z, int* ldz, NFFT4GP_DOUBLE* sigma, char* bmat, int* n, char* which, int* nev, NFFT4GP_DOUBLE* tol, NFFT4GP_DOUBLE* resid, int* ncv, NFFT4GP_DOUBLE* v, int* ldv, int* iparam, int* ipntr, NFFT4GP_DOUBLE* workd, NFFT4GP_DOUBLE* workl, int* lworkl, int* info);

#endif // #ifdef NFFT4GP_USING_ARPACK
 
/* function prototypes */

/**
 * @brief   Create a data structure and set it to default values.
 * @details Create a data structure and set it to default values.
 * @return           Pointer to the data structure.
 */
typedef void* (*func_create)();

/**
 * @brief   Free a data structure.
 * @details This function is used to free a data structure.
 * @param [in,out]   str: data structure to be freed.
 * @return           No return.
 */
typedef void (*func_free)(void *str);

/**
 * @brief   Print a denst matrix to terminal.
 * @details Print a denst matrix to terminal.
 * @param [in]       matrix: denst matrix, m by n
 * @param [in]       m: number of rows in matrix
 * @param [in]       n: number of columns in matrix
 * @param [in]       ldim: leading dimension of matrix
 * @return           Return the data matrix (d by n).
 */
void TestPrintMatrix(NFFT4GP_DOUBLE *matrix, int m, int n, int ldim);

/**
 * @brief   Print a denst matrix to file at a lower precision for validation.
 * @details Print a denst matrix to file at a lower precision for validation.
 * @param [in]       file: file pointer
 * @param [in]       matrix: denst matrix, m by n
 * @param [in]       m: number of rows in matrix
 * @param [in]       n: number of columns in matrix
 * @param [in]       ldim: leading dimension of matrix
 * @return           Return the data matrix (d by n).
 */
void TestPrintMatrixToFile(FILE *file, NFFT4GP_DOUBLE *matrix, int m, int n, int ldim);

/**
 * @brief   Print a dense low triangular matrix to file at a lower precision for validation.
 * @details Print a dense low triangular matrix to file at a lower precision for validation.
 * @param [in]       file: file pointer
 * @param [in]       matrix: dense matrix, m by n
 * @param [in]       m: number of rows in matrix
 * @param [in]       n: number of columns in matrix
 * @param [in]       ldim: leading dimension of matrix
 * @return           Return the data matrix (d by n).
 */
void TestPrintTrilMatrixToFile(FILE *file, NFFT4GP_DOUBLE *matrix, int m, int n, int ldim);

/**
 * @brief   Print a sparse matrix to terminal.
 * @details Print a sparse matrix to terminal.
 * @param [in]       A_i: row index of sparse matrix
 * @param [in]       A_j: column index of sparse matrix
 * @param [in]       m: number of rows in matrix
 * @param [in]       n: number of columns in matrix
 * @return           Return the data matrix (d by n).
 */
void TestPrintCSRMatrixPattern(int *A_i, int *A_j, int m, int n);

/**
 * @brief   Print a sparse matrix to file at a lower precision for validation.
 * @details Print a sparse matrix to file at a lower precision for validation.
 * @param [in]       file: file pointer
 * @param [in]       A_i: row index of sparse matrix
 * @param [in]       A_j: column index of sparse matrix
 * @param [in]       A_a: value of sparse matrix
 * @param [in]       m: number of rows in matrix
 * @param [in]       n: number of columns in matrix
 * @return           Return the data matrix (d by n).
 */
void TestPrintCSRMatrixToFile(FILE *file, int *A_i, int *A_j, NFFT4GP_DOUBLE *A_a, int m, int n);

/**
 * @brief   Print a sparse matrix to terminal.
 * @details Print a sparse matrix to terminal.
 * @param [in]       A_i: row index of sparse matrix
 * @param [in]       A_j: column index of sparse matrix
 * @param [in]       m: number of rows in matrix
 * @param [in]       n: number of columns in matrix
 * @param [in]       A_a: value of sparse matrix
 * @return           Return the data matrix (d by n).
 */
void TestPrintCSRMatrixVal(int *A_i, int *A_j, int m, int n, NFFT4GP_DOUBLE *A_a);

/**
 * @brief   Print a sparse matrix to terminal.
 * @details Print a sparse matrix to terminal.
 * @param [in]       A_i: row index of sparse matrix
 * @param [in]       A_j: column index of sparse matrix
 * @param [in]       A_a: value of sparse matrix
 * @param [in]       m: number of rows in matrix
 * @param [in]       n: number of columns in matrix
 * @return           Return the data matrix (d by n).
 */
void TestPlotCSRMatrix(int *A_i, int *A_j, NFFT4GP_DOUBLE *A_a, int m, int n, const char *datafilename);

/**
 * @brief   Print the first two dimension of a dataset as figure, and highlight the selected points.
 * @details Print the first two dimension of a dataset as figure, and highlight the selected points.
 * @param [in]       data: data matrix, n by d
 * @param [in]       n: number of points in data
 * @param [in]       d: dimension of data
 * @param [in]       ldim: leading dimension of data matrix
 * @param [in]       perm: permutation vector
 * @param [in]       k: length of permutatin vector.
 * @param [in]       datafilename: file name of the data.
 * @return           Return the data matrix (d by n).
 */
void TestPlotData(NFFT4GP_DOUBLE *data, int n, int d, int ldim, int *perm, int k, const char *datafilename);

#endif