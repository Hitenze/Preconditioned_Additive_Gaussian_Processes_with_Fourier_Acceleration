#ifndef NFFT4GP_SOLVERS_HEADER_H
#define NFFT4GP_SOLVERS_HEADER_H

/**
 * @file _solvers.h
 * @brief Wrapper of header files for external use.
 */


#include "float.h"
#include "_utils.h"
#include "_linearalg.h"

/**
 * @brief   Given rhs, solve for x such that op(x) = rhs.
 * @details Given rhs, solve for x such that op(x) = rhs.
 * @param [in]       precond: preconditioner data structure
 * @param [in]       n: length of vector.
 * @param [out]      x: solution.
 * @param [in]       rhs: right-hand side.
 * @return           Return error messate.
 */
typedef int (*func_solve)( void *precond, int n, double *x, double *rhs);

/**
 * @brief   General matvec function y = alpha*A*x + beta*y or y = alpha*A'*x + beta*y.
 * @details General matvec function y = alpha*A*x + beta*y or y = alpha*A'*x + beta*y.
 * @param [in]       matrix: matrix data structure.
 * @param [in]       trans: 'N' for A*x and 'T' for A'*x.
 * @param [in]       m:
 * @param [in]       n: the matrix is m * n.
 * @param [in]       alpha: alpha value.
 * @param [in]       x: first vector.
 * @param [in]       beta: beta value.
 * @param [out]      y: second vector.
 * @return           Return error messate.
 */
typedef int (*func_matvec)(void *matrix, char trans, int m, int n, double alpha, double *x, double beta, double *y);

/**
 * @brief   Symmetrix matvec y = alpha*A*x + beta*y.
 * @details Symmetrix matvec y = alpha*A*x + beta*y.
 * @param [in]       matrix: matrix data structure.
 * @param [in]       n: the matrix is n * n.
 * @param [in]       alpha: alpha value.
 * @param [in]       x: first vector.
 * @param [in]       beta: beta value.
 * @param [out]      y: second vector.
 * @return           Return error messate.
 */
typedef int (*func_symmatvec)(void *matrix, int n, double alpha, double *x, double beta, double *y);

/**
 * @brief   Return the trace of M^{-1}dMd/theta for theta equals to f, l, and mu.
 * @details Return the trace of M^{-1}dMd/theta for theta equals to f, l, and mu.
 * @param[in] str   Pointer to the preconditioner.
 * @return           Return error code.
 */
typedef int (*func_trace)(void *str, NFFT4GP_DOUBLE **tracesp);

/**
 * @brief   Return the log determinant of M.
 * @details Return the log determinant of M.
 * @param[in] str   Pointer to the preconditioner.
 * @return           Return error code.
 */
typedef NFFT4GP_DOUBLE (*func_logdet)(void *str);

/**
 * @brief   Compute y{i} = M^{-1}dM(i)*x.
 * @details Compute y{i} = M^{-1}dM(i)*x.
 * @param[in]     str   Pointer to the preconditioner.
 * @param[in]     n     Size of the problem
 * @param[in]     mask  Mask of the gradient entries. If NULL, all entries are computed.
 *                      If preset, those marked with 0 are not computed.
 * @param[in]     x     Input vector.
 * @param[in,out] yp    Pointer to the output vector, cannot be NULL.
 * @return           Return error code.
 */
typedef int (*func_dvp)(void *str, int n, int *mask, NFFT4GP_DOUBLE *x, NFFT4GP_DOUBLE **yp);

/**
 * @brief   PCG solver.
 * @details PCG solver.
 * @note    Version 1.
 * @param[in]       mat_data        Matrix data, used in the matvec function.
 * @param[in]       n               Size of the matrix.
 * @param[in]       matvec          Matrix vector produce function y = alpha*x+beta*y.
 * @param[in]       prec_data       Preconditioner data for the preconditioner.
 * @param[in]       precondfunc     Preconditioning function x = op(rhs).
 * @param[in,out]   x               Solution vector.
 * @param[in]       rhs             Right hand side vector.
 * @param[in]       maxits          Maximum number of iterations.
 * @param[in]       atol            Absolute tolerance.
 * @param[in]       tol             Relative tolerance.
 * @param[in,out]   prel_res        Pointer to the relative residual.
 * @param[in,out]   prel_res_v      Pointer to the relative residual vector.
 * @param[in,out]   piter           Pointer to the number of iterations.
 * @param[in]       print_level     Print level.
 * @return                          Flag. 
 */
int Nfft4GPSolverPcg( void *mat_data,
                     int n,
                     func_symmatvec matvec,
                     void *prec_data,
                     func_solve precondfunc,
                     NFFT4GP_DOUBLE *x,
                     NFFT4GP_DOUBLE *rhs,
                     int maxits,
                     int atol,
                     NFFT4GP_DOUBLE tol,
                     NFFT4GP_DOUBLE *prel_res,
                     NFFT4GP_DOUBLE **prel_res_v,
                     int *piter,
                     int print_level);

/**
 * @brief   FGMRES solver.
 * @details FGMRES solver.
 * @param[in]       mat_data        Matrix data, used in the matvec function.
 * @param[in]       n               Size of the matrix.
 * @param[in]       matvec          Matrix vector produce function y = alpha*x+beta*y.
 * @param[in]       prec_data       Preconditioner data for the preconditioner.
 * @param[in]       precondfunc     Preconditioning function x = op(rhs).
 * @param[in,out]   x               Solution vector.
 * @param[in]       rhs             Right hand side vector.
 * @param[in]       kdim            Dimension of the Krylov subspace.
 * @param[in]       maxits          Maximum number of iterations.
 * @param[in]       atol            Absolute tolerance.
 * @param[in]       tol             Relative tolerance.
 * @param[in,out]   prel_res        Pointer to the relative residual.
 * @param[in,out]   prel_res_v      Pointer to the relative residual vector.
 * @param[in,out]   piter           Pointer to the number of iterations.
 * @param[in]       print_level     Print level.
 * @return                          Flag.
 */
int Nfft4GPSolverFgmres( void *mat_data,
                     int n,
                     func_symmatvec matvec,
                     void *prec_data,
                     func_solve precondfunc,
                     NFFT4GP_DOUBLE *x,
                     NFFT4GP_DOUBLE *rhs,
                     int kdim,
                     int maxits,
                     int atol,
                     NFFT4GP_DOUBLE tol,
                     NFFT4GP_DOUBLE *prel_res,
                     NFFT4GP_DOUBLE **prel_res_v,
                     int *piter,
                     int print_level);

/**
 * @brief   Lanczos algorithm.
 * @details Lanczos algorithm.
 * @param[in]       mat_data        Matrix data, used in the matvec function.
 * @param[in]       n               Size of the matrix.
 * @param[in]       matvec          Matrix vector produce function y = alpha*x+beta*y.
 * @param[in]       prec_data       Preconditioner data for the preconditioner.
 * @param[in]       precondfunc     Preconditioning function x = op(rhs).
 * @param[in,out]   x               Solution vector.
 * @param[in]       rhs             Right hand side vector.
 * @param[in]       wsize           Window size for reorthogonalization.
 * @param[in]       maxits          Maximum number of iterations.
 * @param[in]       atol            Absolute tolerance for the M-norm of the residual, where M is the preconditioner.
 * @param[in]       tol             Relative tolerance for the M-norm of the residual, where M is the preconditioner.
 * @param[in,out]   prel_res        Pointer to the relative residual.
 * @param[in,out]   prel_res_v      Pointer to the relative residual vector.
 * @param[in,out]   piter           Pointer to the number of iterations.
 * @param[in,out]   tsize           Pointer to the size of the tridiagonal matrix.
 * @param[in,out]   TDp             Pointer to the diagonal of the tridiagonal matrix.
 * @param[in,out]   TEp             Pointer to the off-diagonal of the tridiagonal matrix.
 * @param[in]       print_level     Print level.
 * @return                          Flag.
 */
int Nfft4GPSolverLanczos( void *mat_data,
                     int n,
                     func_symmatvec matvec,
                     void *prec_data,
                     func_solve precondfunc,
                     NFFT4GP_DOUBLE *x,
                     NFFT4GP_DOUBLE *rhs,
                     int wsize,
                     int maxits,
                     int atol,
                     NFFT4GP_DOUBLE tol,
                     NFFT4GP_DOUBLE *prel_res,
                     NFFT4GP_DOUBLE **prel_res_v,
                     int *piter,
                     int *tsize,
                     NFFT4GP_DOUBLE **TDp,
                     NFFT4GP_DOUBLE **TEp,
                     int print_level);

/**
 * @brief   Lanczos algorithm for logdet and its derivative.
 * @details Lanczos algorithm for logdet and its derivative.
 * @param[in]       mat_data        Matrix data, used in the matvec function.
 * @param[in]       dmat_data       Derivative matrix data, used in the dmatvec function.
 * @param[in]       n               Size of the matrix.
 * @param[in]       matvec          Matrix vector produce function y = alpha*x+beta*y.
 * @param[in]       dmatvec         Derivative matrix vector produce function y = alpha*x+beta*y.
 * @param[in]       prec_data       Preconditioner data for the preconditioner.
 * @param[in]       precondfunc     Preconditioning function x = op(rhs).
 * @param[in]       tracefunc       Trace estimation function for the preconditioned matrix.
 * @param[in]       logdetfunc      Logdet estimation function for the preconditioned matrix.
 * @param[in]       dvfunc          Derivative of the trace estimation function for the preconditioned matrix.
 * @param[in]       maxits          Maximum number of iterations.
 * @param[in]       nvecs           Number of vectors to use in the trace estimation.
 * @param[in]       radamacher      Radamacher vector matrix. If NULL, will be generated randomly. Otherwise each column of the matrix is a radamacher vector.
 * @param[in]       print_level     Print level.
 * @param[in,out]   logdet          Pointer to the logdet.
 * @param[in,out]   dlogdetp        Pointer to the derivative of the logdet.
 * @return                          Return error code.
 */
int Nfft4GPLanczosQuadratureLogdet( void *mat_data,
                                 void *dmat_data,
                                 int n,
                                 func_symmatvec matvec,
                                 func_symmatvec dmatvec,
                                 void *prec_data,
                                 func_solve precondfunc,
                                 func_trace tracefunc,
                                 func_logdet logdetfunc,
                                 func_dvp dvpfunc,
                                 int maxits,
                                 int nvecs,
                                 NFFT4GP_DOUBLE *radamacher,
                                 int print_level,
                                 NFFT4GP_DOUBLE *logdet,
                                 NFFT4GP_DOUBLE **dlogdetp);

#endif