#ifndef NFFT4GP_LANCZOS_H
#define NFFT4GP_LANCZOS_H

/**
 * @file lanczos.h
 * @brief Lanczos algorithm for symmetric linear systems and trace estimation.
 */

#include "float.h"
#include "../utils/utils.h"
#include "../linearalg/matops.h"
#include "../linearalg/vecops.h"
#include "solvers.h"

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
 * @param[in]       wsize           Window size for reorthogonalization, if <= 0 use full reorthogonalization.
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