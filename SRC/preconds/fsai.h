#ifndef NFFT4GP_FSAI_H
#define NFFT4GP_FSAI_H

/**
 * @file fsai.h
 * @brief Factored Sparse Approximate Inverse preconditioner.
 */

#include "../utils/utils.h"
#include "../linearalg/kernels.h"
#include "../linearalg/matops.h"
#include "precond.h"

/*------------------------------------------
 * FSAI
 *------------------------------------------*/

/**
 * @brief   Struct for the FSAI preconditioner.
 * @details Struct for the Factored Sparse Approximate Inverse preconditioner.
 */
typedef struct NFFT4GP_PRECOND_FSAI_STRUCT
{
   /* setup */
   int _lfil; // nnz per column in U

   int _n; // size
   int _tits; // total number of iterations
   double _titt; // total solve time
   double _tset; // total setup time
   double _tlogdet; // total logdet time
   double _tdvp; // total M^{-1}dM/dtheta time
   
   int *_L_i;
   int *_L_j;
   NFFT4GP_DOUBLE *_L_a;
   NFFT4GP_DOUBLE *_dL_a;
   NFFT4GP_DOUBLE *_work;
} precond_fsai,*pprecond_fsai;

/**
 * @brief   Create the FSAI preconditioner.
 * @details Create the Factored Sparse Approximate Inverse preconditioner.
 * @return  Pointer to the preconditioner (void).
 */
void* Nfft4GPPrecondFsaiCreate();

/**
 * @brief   Free the FSAI preconditioner.
 * @details Free the Factored Sparse Approximate Inverse preconditioner and release all allocated memory.
 * @param[in,out] str Pointer to the preconditioner (void).
 */
void Nfft4GPPrecondFsaiFree(void *str);

/**
 * @brief   Reset the FSAI preconditioner.
 * @details Reset the FSAI preconditioner without freeing the str pointer. Ready to setup again.
 * @param[in,out] str Pointer to the preconditioner (void).
 */
void Nfft4GPPrecondFsaiReset(void *str);

/**
 * @brief   Set lfil option of the preconditioner.
 * @details Set the number of nonzeros per column in the factorization.
 * @param[in,out] str  Pointer to the preconditioner (void).
 * @param[in]     lfil Number of nonzeros per column in U.
 */
void Nfft4GPPrecondFsaiSetLfil(void *str, int lfil);

/**
 * @brief   Apply one solve step with the FSAI preconditioner.
 * @details Apply one solve step with the FSAI preconditioner to solve the system Mx = rhs.
 * @param[in]     vfsai_mat Pointer to the preconditioner (void).
 * @param[in]     n         Size of the linear system.
 * @param[out]    x         Solution vector.
 * @param[in]     rhs       Right-hand side vector.
 * @return        0 if successful.
 */
int Nfft4GPPrecondFsaiSolve(
   void *vfsai_mat, 
   int n, 
   NFFT4GP_DOUBLE *x, 
   NFFT4GP_DOUBLE *rhs);

/**
 * @brief   Compute y{i} = M^{-1}dM(i)*x.
 * @details Compute the product of the inverse preconditioner with its derivative and a vector.
 * @param[in]     vfsai_mat Pointer to the preconditioner.
 * @param[in]     n         Size of the problem.
 * @param[in]     mask      Mask of the gradient entries. If NULL, all entries are computed.
 *                          If present, those marked with 0 are not computed.
 * @param[in]     x         Input vector.
 * @param[in,out] yp        Pointer to the output vector, cannot be NULL.
 * @return        Error code (0 if successful).
 */
int Nfft4GPPrecondFsaiDvp(
   void *vfsai_mat, 
   int n, 
   int *mask, 
   NFFT4GP_DOUBLE *x, 
   NFFT4GP_DOUBLE **yp);

/**
 * @brief   Return the trace of M^{-1}dMd/theta.
 * @details Return the trace of M^{-1}dMd/theta for theta equals to f, l, and mu.
 * @param[in]     vfsai_mat Pointer to the preconditioner.
 * @param[out]    tracesp   Pointer to store the trace values.
 * @return        Error code (0 if successful).
 */
int Nfft4GPPrecondFsaiTrace(
   void *vfsai_mat, 
   NFFT4GP_DOUBLE **tracesp);

/**
 * @brief   Return the log determinant of M.
 * @details Return the log determinant of the preconditioner matrix M.
 * @param[in]     vfsai_mat Pointer to the preconditioner.
 * @return        The log determinant value.
 */
NFFT4GP_DOUBLE Nfft4GPPrecondFsaiLogdet(void *vfsai_mat);

/**
 * @brief   Setup the FSAI preconditioner with given data and kernel.
 * @details Setup the FSAI preconditioner with given data and kernel function.
 * @param[in]     data           Dataset containing point coordinates.
 * @param[in]     n              Number of points in the dataset.
 * @param[in]     ldim           Leading dimension of the dataset.
 * @param[in]     d              Dimension of the dataset.
 * @param[in]     fkernel        Kernel function.
 * @param[in]     fkernel_params Parameters of the kernel function.
 * @param[in]     require_grad   Should we compute the gradient (1) or not (0).
 * @param[in,out] vfsai_mat      Pointer to the preconditioner (void).
 * @return        Error code (0 if successful).
 */
int Nfft4GPPrecondFsaiSetupWithKernel(
   NFFT4GP_DOUBLE *data, 
   int n, 
   int ldim, 
   int d,
   func_kernel fkernel, 
   void *fkernel_params, 
   int require_grad, 
   void* vfsai_mat);

/**
 * @brief   Setup the FSAI preconditioner with given sparsity pattern, data, and kernel.
 * @details Setup the FSAI preconditioner with given sparsity pattern, data, and kernel.
 * @param[in]     A_i            Row indices of the sparsity pattern.
 * @param[in]     A_j            Column indices of the sparsity pattern.
 * @param[in]     data           Dataset containing point coordinates.
 * @param[in]     n              Number of points in the dataset.
 * @param[in]     ldim           Leading dimension of the dataset.
 * @param[in]     d              Dimension of the dataset.
 * @param[in]     fkernel        Kernel function.
 * @param[in]     fkernel_params Parameters of the kernel function.
 * @param[in]     require_grad   Should we compute the gradient (1) or not (0).
 * @param[in,out] vfsai_mat      Pointer to the preconditioner (void).
 * @return        Error code (0 if successful).
 */
int Nfft4GPPrecondFsaiSetupWithKernelPattern(
   int *A_i, 
   int *A_j, 
   NFFT4GP_DOUBLE *data, 
   int n, 
   int ldim, 
   int d,
   func_kernel fkernel, 
   void *fkernel_params, 
   int require_grad, 
   void* vfsai_mat);

/**
 * @brief   Apply the inverse of L.
 * @details Apply the inverse of the lower triangular factor L to solve L^{-1}rhs.
 * @param[in]     vfsai_mat Pointer to the preconditioner.
 * @param[in]     n         Size of the problem.
 * @param[out]    x         Solution vector.
 * @param[in]     rhs       Right-hand side vector.
 * @return        Error code (0 if successful).
 */
int Nfft4GPPrecondFsaiInvL(
   void *vfsai_mat, 
   int n, 
   NFFT4GP_DOUBLE *x, 
   NFFT4GP_DOUBLE *rhs);

/**
 * @brief   Apply the inverse of L^T.
 * @details Apply the inverse of the transpose of the lower triangular factor L to solve L^{-T}rhs.
 * @param[in]     vfsai_mat Pointer to the preconditioner.
 * @param[in]     n         Size of the problem.
 * @param[out]    x         Solution vector.
 * @param[in]     rhs       Right-hand side vector.
 * @return        Error code (0 if successful).
 */
int Nfft4GPPrecondFsaiInvLT(
   void *vfsai_mat, 
   int n, 
   NFFT4GP_DOUBLE *x, 
   NFFT4GP_DOUBLE *rhs);

/**
 * @brief   Plot the FSAI preconditioner.
 * @details Plot the structure of the FSAI preconditioner for visualization.
 * @param[in]     vfsai_mat Pointer to the preconditioner.
 */
void Nfft4GPPrecondFsaiPlot(void *vfsai_mat);

#endif
