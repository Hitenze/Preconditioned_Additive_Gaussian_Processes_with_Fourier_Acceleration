#ifndef NFFT4GP_PCG_H
#define NFFT4GP_PCG_H

#include "float.h"
#include "../utils/utils.h"
#include "../linearalg/vecops.h"
#include "solvers.h"

/**
 * @brief   Preconditioned Conjugate Gradient solver.
 * @details Preconditioned Conjugate Gradient solver for symmetric positive definite systems.
 * @param [in]       mat_data        Matrix data, used in the matvec function.
 * @param [in]       n               Size of the matrix.
 * @param [in]       matvec          Matrix-vector product function y = alpha*A*x+beta*y.
 * @param [in]       prec_data       Preconditioner data for the preconditioner.
 * @param [in]       precondfunc     Preconditioning function x = op(rhs).
 * @param [in,out]   x               Solution vector.
 * @param [in]       rhs             Right-hand side vector.
 * @param [in]       maxits          Maximum number of iterations.
 * @param [in]       atol            Absolute tolerance flag (0 or 1).
 * @param [in]       tol             Relative tolerance.
 * @param [out]      prel_res        Pointer to the final relative residual.
 * @param [out]      prel_res_v      Pointer to the relative residual vector history.
 * @param [out]      piter           Pointer to the number of iterations performed.
 * @param [in]       print_level     Print level for solver output.
 * @return           Return error code.
 */
int Nfft4GPSolverPcg(
   void *mat_data,
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

#endif