#ifndef NFFT4GP_NYS_H
#define NFFT4GP_NYS_H

/**
 * @file nys.h
 * @brief Nystrom preconditioner.
 */

#include "../utils/utils.h"
#include "../linearalg/kernels.h"
#include "../linearalg/matops.h"
#include "../linearalg/ordering.h"
#include "precond.h"
#include "chol.h"

/*------------------------------------------
 * NYS
 *------------------------------------------*/

/**
 * @brief   Struct for the NYS preconditioner.
 * @details Struct for the Nystrom preconditioner, which uses a low-rank approximation.
 */
typedef struct NFFT4GP_PRECOND_NYS_STRUCT
{
   /* setup */
   int _k_setup; // rank setup
   int _own_perm; // is the permutation owned by the preconditioner?
   int *_perm; // permutation

   int _n; // size
   int _tits; // total number of iterations
   double _titt; // total solve time
   double _tset; // total setup time
   double _tlogdet; // total logdet time
   double _tdvp; // total M^{-1}dM/dtheta time

   int _nys_opt; // NYS option. 0: standard, 1: stabliized.

   int _k; // rank
   NFFT4GP_DOUBLE _eta; // regularization parameter
   NFFT4GP_DOUBLE _f2; // store the scale factor

   NFFT4GP_DOUBLE *_U;
   NFFT4GP_DOUBLE *_s;
   NFFT4GP_DOUBLE *_work;

   // buffer for gradient
   NFFT4GP_DOUBLE *_K;
   NFFT4GP_DOUBLE *_dU;
   NFFT4GP_DOUBLE *_dK;
   pprecond_chol _chol_K11; // the chol of K + mu I
   int _dvp_nosolve;

} precond_nys,*pprecond_nys;

/**
 * @brief   Create the NYS preconditioner.
 * @details Create the Nystrom preconditioner for low-rank approximation.
 * @return  Pointer to the preconditioner (void).
 */
void* Nfft4GPPrecondNysCreate();

/**
 * @brief   Free the NYS preconditioner.
 * @details Free the Nystrom preconditioner and release all allocated memory.
 * @param[in,out] str Pointer to the preconditioner (void).
 */
void Nfft4GPPrecondNysFree(void *str);

/**
 * @brief   Reset the NYS preconditioner.
 * @details Reset the Nystrom preconditioner without freeing the str pointer. Ready to setup again.
 * @param[in,out] str Pointer to the preconditioner (void).
 */
void Nfft4GPPrecondNysReset(void *str);

/**
 * @brief   Set the NYS rank.
 * @details Set the rank parameter for the Nystrom approximation.
 * @param[in,out] str Pointer to the preconditioner (void).
 * @param[in]     k   Rank of the NYS preconditioner.
 */
void Nfft4GPPrecondNysSetRank(void *str, int k);

/**
 * @brief   Set the NYS permutation.
 * @details Set the permutation vector for the Nystrom preconditioner.
 * @param[in,out] str      Pointer to the preconditioner (void).
 * @param[in]     perm     Permutation vector. Full n-permutation.
 * @param[in]     own_perm Flag indicating if the preconditioner owns the permutation (1) or not (0).
 */
void Nfft4GPPrecondNysSetPerm(void *str, int *perm, int own_perm);

/**
 * @brief   Apply one solve step with the NYS preconditioner.
 * @details Apply one solve step with the Nystrom preconditioner to solve the system Mx = rhs.
 * @param[in]     vnys_mat Pointer to the preconditioner (void).
 * @param[in]     n        Size of the linear system.
 * @param[out]    x        Solution vector.
 * @param[in]     rhs      Right-hand side vector.
 * @return        Error code (0 if successful).
 */
int Nfft4GPPrecondNysSolve(
   void *vnys_mat, 
   int n, 
   NFFT4GP_DOUBLE *x, 
   NFFT4GP_DOUBLE *rhs);

/**
 * @brief   Compute y{i} = M^{-1}dM(i)*x.
 * @details Compute the product of the inverse preconditioner with its derivative and a vector.
 * @param[in]     vnys_mat Pointer to the preconditioner.
 * @param[in]     n        Size of the problem.
 * @param[in]     mask     Mask of the gradient entries. If NULL, all entries are computed.
 *                         If present, those marked with 0 are not computed.
 * @param[in]     x        Input vector.
 * @param[in,out] yp       Pointer to the output vector, cannot be NULL.
 * @return        Error code (0 if successful).
 */
int Nfft4GPPrecondNysDvp(
   void *vnys_mat, 
   int n, 
   int *mask, 
   NFFT4GP_DOUBLE *x, 
   NFFT4GP_DOUBLE **yp);

/**
 * @brief   Return the trace of M^{-1}dMd/theta.
 * @details Return the trace of M^{-1}dMd/theta for theta equals to f, l, and mu.
 * @param[in]     vnys_mat Pointer to the preconditioner.
 * @param[out]    tracesp  Pointer to store the trace values.
 * @return        Error code (0 if successful).
 */
int Nfft4GPPrecondNysTrace(
   void *vnys_mat, 
   NFFT4GP_DOUBLE **tracesp);

/**
 * @brief   Return the log determinant of M.
 * @details Return the log determinant of the preconditioner matrix M.
 * @param[in]     vnys_mat Pointer to the preconditioner.
 * @return        The log determinant value.
 */
NFFT4GP_DOUBLE Nfft4GPPrecondNysLogdet(void *vnys_mat);

/**
 * @brief   Setup the NYS preconditioner with given data and kernel.
 * @details Setup the Nystrom preconditioner with given data and kernel function.
 * @param[in]     data           Dataset containing point coordinates.
 * @param[in]     n              Number of points in the dataset.
 * @param[in]     ldim           Leading dimension of the dataset.
 * @param[in]     d              Dimension of the dataset.
 * @param[in]     fkernel        Kernel function.
 * @param[in]     fkernel_params Parameters of the kernel function.
 * @param[in]     require_grad   Should we compute the gradient (1) or not (0).
 * @param[in,out] vnys_mat       Pointer to the preconditioner (void).
 * @return        Error code (0 if successful).
 */
int Nfft4GPPrecondNysSetupWithKernel(
   NFFT4GP_DOUBLE *data, 
   int n, 
   int ldim, 
   int d,
   func_kernel fkernel, 
   void *fkernel_params, 
   int require_grad, 
   void* vnys_mat);

/**
 * @brief   Plot the NYS preconditioner.
 * @details Plot the structure of the Nystrom preconditioner for visualization.
 * @param[in]     vnys_mat Pointer to the preconditioner.
 * @param[in]     data     Dataset containing point coordinates.
 * @param[in]     n        Number of points in the dataset.
 * @param[in]     ldim     Leading dimension of the dataset.
 * @param[in]     d        Dimension of the dataset.
 */
void Nfft4GPPrecondNysPlot(
   void *vnys_mat, 
   NFFT4GP_DOUBLE *data, 
   int n, 
   int ldim, 
   int d);

#endif