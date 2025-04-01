#ifndef NFFT4GP_MATOPS
#define NFFT4GP_MATOPS

/**
 * @file matops.h
 * @brief Matrix operations
 */

#include "../utils/utils.h"
#include "../utils/protos.h"
#include "vector.h"
#include "vecops.h"

/**
 * @brief   Compute the symmetric matrix-vector product y = alpha*A*x + beta*y.
 * @details Compute the symmetric matrix-vector product y = alpha*A*x + beta*y. The matrix is stored in the lower triangular part.
 * @param [in]       data           Pointer to the matrix. The matrix is stored in the lower triangular part.
 * @param [in]       n              Dimension of the matrix.
 * @param [in]       alpha          Scaling factor for the matrix.
 * @param [in]       x              Pointer to the vector.
 * @param [in]       beta           Scaling factor for the vector.
 * @param [in,out]   y              Pointer to the vector.
 * @return           Return 0 if successful.
 */
int Nfft4GPDenseMatSymv(
   void *data, 
   int n, 
   NFFT4GP_DOUBLE alpha, 
   NFFT4GP_DOUBLE *x, 
   NFFT4GP_DOUBLE beta, 
   NFFT4GP_DOUBLE *y);

/**
 * @brief   Compute three symmetric matrix-vector products with gradient matrices.
 * @details Compute three symmetric matrix-vector product y = alpha*A*x + beta*y. A, A+n*n, and A+2*n*n are three gradient matrices.\n 
 *          The matrix is stored in the lower triangular part.
 * @param [in]       data           Pointer to the matrix. The matrix is stored in the lower triangular part.
 * @param [in]       n              Dimension of the matrix.
 * @param [in]       alpha          Scaling factor for the matrix.
 * @param [in]       x              Pointer to the vector.
 * @param [in]       beta           Scaling factor for the vector.
 * @param [in,out]   y              Pointer to the vector.
 * @return           Return 0 if successful.
 */
int Nfft4GPDenseGradMatSymv(
   void *data, 
   int n, 
   NFFT4GP_DOUBLE alpha, 
   NFFT4GP_DOUBLE *x, 
   NFFT4GP_DOUBLE beta, 
   NFFT4GP_DOUBLE *y);

/**
 * @brief   Compute the general matrix-vector product y = alpha*A*x + beta*y.
 * @details Compute the general matrix-vector product y = alpha*A*x + beta*y.
 * @param [in]       data           Pointer to the matrix.
 * @param [in]       trans          'N' for A*x and 'T' for A'*x.
 * @param [in]       m              Number of rows of the matrix.
 * @param [in]       n              Number of columns of the matrix.
 * @param [in]       alpha          Scaling factor for the matrix.
 * @param [in]       x              Pointer to the vector.
 * @param [in]       beta           Scaling factor for the y vector.
 * @param [in,out]   y              Pointer to the vector.
 * @return           Return 0 if successful.
 */
int Nfft4GPDenseMatGemv(
   void *data, 
   char trans, 
   int m, 
   int n, 
   NFFT4GP_DOUBLE alpha, 
   NFFT4GP_DOUBLE *x, 
   NFFT4GP_DOUBLE beta, 
   NFFT4GP_DOUBLE *y);

/**
 * @brief   Compute the inverse of a Nystrom matrix.
 * @details Compute the inverse of a Nystrom matrix.
 * @param [in,out]   data           Pointer to the matrix.
 * @param [in]       lda            Leading dimension of the matrix.
 * @param [in]       n              Dimension of the matrix.
 * @return           Return 0 if successful.
 */
int AFnTrilNystromInv(
   void *data, 
   int lda, 
   int n);

/**
 * @brief   Compute the matrix-matrix product for a Nystrom matrix.
 * @details Compute the matrix-matrix product for a Nystrom matrix.
 * @param [in,out]   a              Pointer to the first matrix.
 * @param [in]       lda            Leading dimension of the first matrix.
 * @param [in]       b              Pointer to the second matrix.
 * @param [in]       ldb            Leading dimension of the second matrix.
 * @param [in]       m              Number of rows of the first matrix.
 * @param [in]       n              Number of columns of the second matrix.
 * @param [in]       alpha          Scaling factor for the product.
 * @return           Return 0 if successful.
 */
int Nfft4GPTrilNystromMm(
   void *a, 
   int lda, 
   void *b, 
   int ldb, 
   int m, 
   int n, 
   NFFT4GP_DOUBLE alpha);

/**
 * @brief   Compute the SVD of a Nystrom matrix.
 * @details Compute the SVD of a Nystrom matrix.
 * @param [in,out]   a              Pointer to the matrix.
 * @param [in]       lda            Leading dimension of the matrix.
 * @param [in]       m              Number of rows of the matrix.
 * @param [in]       n              Number of columns of the matrix.
 * @param [out]      s              Pointer to the singular values.
 * @return           Return 0 if successful.
 */
int Nfft4GPTrilNystromSvd(
   void *a, 
   int lda, 
   int m, 
   int n, 
   void **s);

/**
 * @brief   Compute the matrix-vector product for a CSR matrix.
 * @details Compute the matrix-vector product for a CSR matrix.
 * @param [in]       ia             Pointer to the row pointers.
 * @param [in]       ja             Pointer to the column indices.
 * @param [in]       aa             Pointer to the non-zero values.
 * @param [in]       nrows          Number of rows of the matrix.
 * @param [in]       ncols          Number of columns of the matrix.
 * @param [in]       trans          'N' for A*x and 'T' for A'*x.
 * @param [in]       alpha          Scaling factor for the matrix.
 * @param [in]       x              Pointer to the vector.
 * @param [in]       beta           Scaling factor for the y vector.
 * @param [in,out]   y              Pointer to the vector.
 * @return           Return 0 if successful.
 */
int Nfft4GPCsrMv(
   int *ia, 
   int *ja, 
   NFFT4GP_DOUBLE *aa, 
   int nrows, 
   int ncols, 
   char trans, 
   NFFT4GP_DOUBLE alpha, 
   NFFT4GP_DOUBLE *x, 
   NFFT4GP_DOUBLE beta, 
   NFFT4GP_DOUBLE *y);

/**
 * @brief   Perform modified Gram-Schmidt orthogonalization.
 * @details Perform modified Gram-Schmidt orthogonalization.
 * @param [in,out]   w              Pointer to the vector to be orthogonalized.
 * @param [in]       n              Length of the vector.
 * @param [in]       kdim           Maximum dimension of the Krylov subspace.
 * @param [in,out]   V              Pointer to the orthogonal basis.
 * @param [out]      H              Pointer to the Hessenberg matrix.
 * @param [out]      t              Pointer to the temporary vector.
 * @param [in]       k              Current dimension of the Krylov subspace.
 * @param [in]       tol_orth       Tolerance for orthogonalization.
 * @param [in]       tol_reorth     Tolerance for reorthogonalization.
 * @return           Return 0 if successful.
 */
int Nfft4GPModifiedGS(
   NFFT4GP_DOUBLE *w, 
   int n, 
   int kdim, 
   NFFT4GP_DOUBLE *V, 
   NFFT4GP_DOUBLE *H, 
   NFFT4GP_DOUBLE *t, 
   int k, 
   NFFT4GP_DOUBLE tol_orth, 
   NFFT4GP_DOUBLE tol_reorth);

/**
 * @brief   Perform modified Gram-Schmidt orthogonalization with two bases.
 * @details Perform modified Gram-Schmidt orthogonalization with two bases.
 * @param [in,out]   w              Pointer to the vector to be orthogonalized.
 * @param [in]       n              Length of the vector.
 * @param [in,out]   V              Pointer to the first orthogonal basis.
 * @param [in,out]   Z              Pointer to the second orthogonal basis.
 * @param [out]      TD             Pointer to the diagonal of the tridiagonal matrix.
 * @param [out]      TE             Pointer to the off-diagonal of the tridiagonal matrix.
 * @param [out]      t              Pointer to the temporary vector.
 * @param [in]       k              Current dimension of the Krylov subspace.
 * @param [in]       tol_orth       Tolerance for orthogonalization.
 * @param [in]       tol_reorth     Tolerance for reorthogonalization.
 * @return           Return 0 if successful.
 */
int Nfft4GPModifiedGS2(
   NFFT4GP_DOUBLE *w, 
   int n, 
   NFFT4GP_DOUBLE *V, 
   NFFT4GP_DOUBLE *Z, 
   NFFT4GP_DOUBLE *TD, 
   NFFT4GP_DOUBLE *TE, 
   NFFT4GP_DOUBLE *t, 
   int k, 
   NFFT4GP_DOUBLE tol_orth, 
   NFFT4GP_DOUBLE tol_reorth);

#endif
