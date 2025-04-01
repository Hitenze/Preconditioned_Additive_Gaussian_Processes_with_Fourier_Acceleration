#ifndef NFFT4GP_AFN_H
#define NFFT4GP_AFN_H

/**
 * @file afn.h
 * @brief AFN preconditioner.
 */

#include "../utils/utils.h"
#include "../linearalg/kernels.h"
#include "../linearalg/matops.h"
#include "../linearalg/ordering.h"
#include "../linearalg/rankest.h"
#include "precond.h"
#include "chol.h"
#include "fsai.h"
#include "nys.h"

/*------------------------------------------
 * AFN preconditioner
 *------------------------------------------*/

/**
 * @brief   Struct for the AFN preconditioner.
 * @details Struct for the AFN preconditioner.
 */
typedef struct NFFT4GP_PRECOND_AFN_STRUCT
{
   int _n; // size
   int _tits; // total number of iterations
   double _titt; // total solve time
   double _tset; // total setup time
   double _tlogdet; // total logdet time
   double _tdvp; // total M^{-1}dM/dtheta time

   int _k; // rank
   int _own_perm; // is the permutation owned by the preconditioner?
   int *_perm; // permutation
   NFFT4GP_DOUBLE *_dwork; // work array

   // A11 solver
   func_solve _fA11_solve;
   func_free _fA11_solve_free_data; // if this is NULL use AFN_FREE
   void* _fA11_solve_data;

   // A12 matvec
   int _fA12_matvec_own_data;
   func_matvec _fA12_matvec;
   func_free _fA12_matvec_free_data; // if this is NULL use AFN_FREE
   void* _fA12_matvec_data;

   // Schur complement solver
   int _fS_solve_own_data;
   func_solve _fS_solve;
   func_free _fS_solve_free_data; // if this is NULL use AFN_FREE
   void* _fS_solve_data;

} precond_afn,*pprecond_afn;

/**
 * @brief   Create the AFN preconditioner.
 * @details Create the AFN preconditioner.
 * @return  Pointer to the preconditioner (void).
 */
void* Nfft4GPPrecondAFNCreate();

/**
 * @brief   Free the AFN preconditioner.
 * @details Free the AFN preconditioner.
 * @param[in]        str Pointer to the preconditioner (void).
 */
void Nfft4GPPrecondAFNFree(void *str);

/**
 * @brief   Setup the matvec function for the AFN preconditioner.
 * @details Setup the matvec function for the AFN preconditioner.
 * @param[in]        vafn_mat Pointer to the preconditioner (void).
 * @param[in]        n Size of the linear system.
 * @param[in,out]    x Solution vector.
 * @param[in]        rhs Right-hand side vector.
 * @return           0 if successful.
 */
int Nfft4GPPrecondAFNSolve( void *vafn_mat, int n, NFFT4GP_DOUBLE *x, NFFT4GP_DOUBLE *rhs);

/**
 * @brief   Return the trace of M^{-1}dMd/theta for theta equals to f, l, and mu.
 * @details Return the trace of M^{-1}dMd/theta for theta equals to f, l, and mu.
 * @param[in] vafn_mat  Pointer to the preconditioner.
 * @return              Return error code.
 */
int Nfft4GPPrecondAFNTrace(void *vafn_mat, NFFT4GP_DOUBLE **tracesp);

/**
 * @brief   Return the log determinant of M.
 * @details Return the log determinant of M.
 * @param[in] vafn_mat  Pointer to the preconditioner.
 * @return              Return error code.
 */
NFFT4GP_DOUBLE Nfft4GPPrecondAFNLogdet(void *vafn_mat);

/**
 * @brief   Setup the AFN preconditioner with given sparsity pattern, data, and kernel.
 * @details Setup the AFN preconditioner with given sparsity pattern, data, and kernel.
 * @param[in]        data Dataset.
 * @param[in]        n Number of points in the dataset.
 * @param[in]        ldim Leading dimension of the dataset.
 * @param[in]        d Dimension of the dataset.
 * @param[in]        rank: rank of the preconditioner.
 * @param[in]        max_k Maximum rank. If the value is negative, we will use -max_rank as the given rank.
 * @param[in]        perm_opt Permutation option.
 * @param[in]        schur_opt Schur complement solver.
 * @param[in]        schur_lfil Schur complement fill-in.
 * @param[in]        fkernel Kernel function.
 * @param[in]        fkernel_params Kernel function parameters.
 * @param[in]        require_grad Should we compute the gradient?
 * @param[in,out]    vafn_mat Pointer to the preconditioner (void).
 * @return           Pointer to the preconditioner (void).
 */
void* Nfft4GPPrecondAFNSetup(NFFT4GP_DOUBLE 
   *data, 
   int n, 
   int ldim, 
   int d,
   int max_k, 
   int perm_opt, 
   int schur_opt, 
   int schur_lfil, 
   int nsamples,
   func_kernel fkernel, 
   void *fkernel_params, 
   int require_grad,
   void* vafn_mat);

/**
 * @brief Plot data and selected points for the NYS preconditioner.
 * @details Plot data and selected points for the NYS preconditioner.
 * @param[in] vnys_mat Pointer to the preconditioner (void).
 * @param[in] data Dataset.
 * @param[in] n Number of points in the dataset.
 * @param[in] ldim Leading dimension of the dataset.
 * @param[in] d Dimension of the dataset.
 */
void Nfft4GPPrecondAFNPlot(void *vafn_mat, NFFT4GP_DOUBLE *data, int n, int ldim, int d);

#endif