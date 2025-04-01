#ifndef NFFT4GP_SOLVERS_H
#define NFFT4GP_SOLVERS_H

/**
 * @file solvers.h
 * @brief Solvers including PCG and FGMRES.
 */

#include "float.h"
#include "../utils/utils.h"

/**
 * @brief   Given rhs, solve for x such that op(x) = rhs.
 * @details Given rhs, solve for x such that op(x) = rhs.
 * @param [in]       precond        Preconditioner data structure.
 * @param [in]       n              Length of vector.
 * @param [out]      x              Solution.
 * @param [in]       rhs            Right-hand side.
 * @return           Return error code.
 */
typedef int (*func_solve)(void *precond, int n, NFFT4GP_DOUBLE *x, NFFT4GP_DOUBLE *rhs);

/**
 * @brief   General matvec function y = alpha*A*x + beta*y or y = alpha*A'*x + beta*y.
 * @details General matvec function y = alpha*A*x + beta*y or y = alpha*A'*x + beta*y.
 * @param [in]       matrix         Matrix data structure.
 * @param [in]       trans          'N' for A*x and 'T' for A'*x.
 * @param [in]       m              Number of rows in the matrix.
 * @param [in]       n              Number of columns in the matrix.
 * @param [in]       alpha          Alpha value.
 * @param [in]       x              First vector.
 * @param [in]       beta           Beta value.
 * @param [out]      y              Second vector.
 * @return           Return error code.
 */
typedef int (*func_matvec)(void *matrix, char trans, int m, int n, NFFT4GP_DOUBLE alpha, NFFT4GP_DOUBLE *x, NFFT4GP_DOUBLE beta, NFFT4GP_DOUBLE *y);

/**
 * @brief   Symmetric matvec y = alpha*A*x + beta*y.
 * @details Symmetric matvec y = alpha*A*x + beta*y.
 * @param [in]       matrix         Matrix data structure.
 * @param [in]       n              The matrix is n * n.
 * @param [in]       alpha          Alpha value.
 * @param [in]       x              First vector.
 * @param [in]       beta           Beta value.
 * @param [out]      y              Second vector.
 * @return           Return error code.
 */
typedef int (*func_symmatvec)(void *matrix, int n, NFFT4GP_DOUBLE alpha, NFFT4GP_DOUBLE *x, NFFT4GP_DOUBLE beta, NFFT4GP_DOUBLE *y);

/**
 * @brief   Return the trace of M^{-1}dMd/theta for theta equals to f, l, and mu.
 * @details Return the trace of M^{-1}dMd/theta for theta equals to f, l, and mu.
 * @param [in]       str            Pointer to the preconditioner.
 * @param [out]      tracesp        Pointer to the traces.
 * @return           Return error code.
 */
typedef int (*func_trace)(void *str, NFFT4GP_DOUBLE **tracesp);

/**
 * @brief   Return the log determinant of M.
 * @details Return the log determinant of M.
 * @param [in]       str            Pointer to the preconditioner.
 * @return           Return log determinant value.
 */
typedef NFFT4GP_DOUBLE (*func_logdet)(void *str);

/**
 * @brief   Compute y{i} = M^{-1}dM(i)*x.
 * @details Compute y{i} = M^{-1}dM(i)*x.
 * @param [in]       str            Pointer to the preconditioner.
 * @param [in]       n              Size of the problem.
 * @param [in]       mask           Mask of the gradient entries. If NULL, all entries are computed.
 *                                  If preset, those marked with 0 are not computed.
 * @param [in]       x              Input vector.
 * @param [in,out]   yp             Pointer to the output vector, cannot be NULL.
 * @return           Return error code.
 */
typedef int (*func_dvp)(void *str, int n, int *mask, NFFT4GP_DOUBLE *x, NFFT4GP_DOUBLE **yp);

#endif
