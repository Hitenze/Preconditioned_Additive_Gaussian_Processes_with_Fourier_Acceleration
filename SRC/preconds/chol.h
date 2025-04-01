#ifndef NFFT4GP_CHOL_H
#define NFFT4GP_CHOL_H

#include "../utils/utils.h"
#include "../linearalg/kernels.h"
#include "../linearalg/matops.h"
#include "precond.h"

/**
 * @brief   LAPACK Cholesky solver.
 * @details LAPACK Cholesky solver.
 */

/**
 * @brief   Struct for the LAPACK Cholesky solver.
 * @details Struct for the LAPACK Cholesky solver.
 */
typedef struct NFFT4GP_PRECOND_CHOL_STRUCT
{
   /* setup */
   int _stable; // Should we add small diagonal perturbation to the matrix?

   /* data */
   NFFT4GP_DOUBLE *_chol_data;
   NFFT4GP_DOUBLE *_GdKG_data;
   NFFT4GP_DOUBLE *_dchol_data;
   NFFT4GP_DOUBLE *_dK_data;

   int _n;
   int _tits; // total number of iterations
   double _titt; // total solve time
   double _tset; // total setup time
   double _tlogdet; // total logdet time
   double _tdvp; // total M^{-1}dM/dtheta time
   
   char _uplo;
   char _trans;
   char _diag;
   int _nrhs;
   int _info;

} precond_chol, *pprecond_chol;

/**
 * @brief   Create a LAPACK Cholesky solver.
 * @details Create a LAPACK Cholesky solver.
 * @return           Return pointer to the solver.
 */
void* Nfft4GPPrecondCholCreate();

/**
 * @brief   Free a LAPACK Cholesky solver.
 * @details Free a LAPACK Cholesky solver.
 * @param [in,out]   str            Pointer to the solver.
 * @return           No return.
 */
void Nfft4GPPrecondCholFree(void *str);

/**
 * @brief   Reset a LAPACK Cholesky solver.
 * @details Reset a LAPACK Cholesky solver.
 * @param [in,out]   str            Pointer to the solver.
 * @return           No return.
 */
void Nfft4GPPrecondCholReset(void *str);

/**
 * @brief   Set the stable flag for a LAPACK Cholesky solver.
 * @details Set the stable flag for a LAPACK Cholesky solver.
 * @param [in,out]   str            Pointer to the solver.
 * @param [in]       stable         Should we add small diagonal perturbation to the matrix?
 * @return           No return.
 */
void Nfft4GPPrecondCholSetStable(void *str, int stable);

/**
 * @brief   Solve a linear system using a LAPACK Cholesky solver.
 * @details Solve a linear system using a LAPACK Cholesky solver.
 * @param [in]       vchol_mat      Pointer to the solver.
 * @param [in]       n              Size of the matrix.
 * @param [out]      x              Solution.
 * @param [in]       rhs            Right-hand side.
 * @return           Return error code.
 */
int Nfft4GPPrecondCholSolve(
   void *vchol_mat, 
   int n, 
   NFFT4GP_DOUBLE *x, 
   NFFT4GP_DOUBLE *rhs);

/**
 * @brief   Compute y{i} = M^{-1}dM(i)*x using a LAPACK Cholesky solver.
 * @details Compute y{i} = M^{-1}dM(i)*x using a LAPACK Cholesky solver.
 * @param [in]       vchol_mat      Pointer to the solver.
 * @param [in]       n              Size of the problem.
 * @param [in]       mask           Mask of the gradient entries. If NULL, all entries are computed.
 * @param [in]       x              Input vector.
 * @param [in,out]   yp             Pointer to the output vector, cannot be NULL.
 * @return           Return error code.
 */
int Nfft4GPPrecondCholDvp(
   void *vchol_mat, 
   int n, 
   int *mask, 
   NFFT4GP_DOUBLE *x, 
   NFFT4GP_DOUBLE **yp);

/**
 * @brief   Return the trace of M^{-1}dMd/theta for theta equals to f, l, and mu.
 * @details Return the trace of M^{-1}dMd/theta for theta equals to f, l, and mu.
 * @param [in]       vchol_mat      Pointer to the solver.
 * @param [out]      tracesp        Pointer to the traces.
 * @return           Return error code.
 */
int Nfft4GPPrecondCholTrace(void *vchol_mat, NFFT4GP_DOUBLE **tracesp);

/**
 * @brief   Return the log determinant of M.
 * @details Return the log determinant of M.
 * @param [in]       vchol_mat      Pointer to the solver.
 * @return           Return log determinant value.
 */
NFFT4GP_DOUBLE Nfft4GPPrecondCholLogdet(void *vchol_mat);

/**
 * @brief   Setup the LAPACK Cholesky solver with a kernel.
 * @details Setup the LAPACK Cholesky solver with a kernel.
 * @param [in]       data           Dataset.
 * @param [in]       n              Number of points in the dataset.
 * @param [in]       ldim           Leading dimension of the dataset.
 * @param [in]       d              Dimension of the dataset.
 * @param [in]       fkernel        Kernel function.
 * @param [in]       fkernel_params Kernel parameters.
 * @param [in]       require_grad   Should we compute the gradient?
 * @param [in,out]   vchol_mat      Pointer to the solver.
 * @return           Return error code.
 */
int Nfft4GPPrecondCholSetupWithKernel(
   NFFT4GP_DOUBLE *data, 
   int n, 
   int ldim, 
   int d,
   func_kernel fkernel, 
   void *fkernel_params, 
   int require_grad, 
   void* vchol_mat);

#endif
