#ifndef NFFT4GP_PRECONDS_HEADER_H
#define NFFT4GP_PRECONDS_HEADER_H

/**
 * @file _preconds.h
 * @brief Wrapper of header files for external use.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "_linearalg.h"

/**
 * @brief   Enumeration of the suppotred precond.
 * @details Enumeration of the suppotred precond.
 */
typedef enum {
   NFFT4GP_PRECOND_UNDEFINED = 0,
   NFFT4GP_PRECOND_CHOL,
   NFFT4GP_PRECOND_FSAI,
   NFFT4GP_PRECOND_NYS,
   NFFT4GP_PRECOND_NFFT4GP
}nfft4gp_precond_type;

/**
 * @brief            Setup the LAPACK Cholesky solver.
 * @details          Setup the LAPACK Cholesky solver.
 * @param[in]        data Dataset.
 * @param[in]        n Number of points in the dataset.
 * @param[in]        ldim Leading dimension of the dataset.
 * @param[in]        d Dimension of the dataset.
 * @param[in]        stable Should we add small diagonal perturbation to the matrix?
 * @param[in]        fkernel Kernel function.
 * @param[in]        fkernel_params Kernel parameters.
 * @param[in]        require_grad Should we compute the gradient?
 * @param[in,out]    precond_data Pointer to the preconditioner. Should be created before calling this function.
 * @return           Return error code.
 */
typedef int (*precond_kernel_setup)(NFFT4GP_DOUBLE *data, int n, int ldim, int d, 
                                       func_kernel fkernel, void *fkernel_params, int require_grad, void* precond_data);

/**
 * @brief   Struct for the LAPACK Cholesky solver.
 * @details Struct for the LAPACK Cholesky solver.
 */
typedef struct NFFT4GP_PRECOND_CHOL_STRUCT
{
   /* setup */
   int _stable;

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

} precond_chol,*pprecond_chol;

/**
 * @brief            Create the LAPACK Cholesky solver.
 * @details          Create the LAPACK Cholesky solver.
 * @return           Pointer to the solver (void).
 */
void* Nfft4GPPrecondCholCreate();

/**
 * @brief            Free the LAPACK Cholesky solver.
 * @details          Free the LAPACK Cholesky solver.
 * @param[in,out]    str Pointer to the solver (void).
 */
void Nfft4GPPrecondCholFree(void *str);

/**
 * @brief            Reset the LAPACK Cholesky solver without free the str pointer. Ready to setup again.
 * @details          Reset the LAPACK Cholesky solver without free the str pointer. Ready to setup again.
 */
void Nfft4GPPrecondCholReset(void *str);

/**
 * @brief            Set stable option of the preconditioner.
 * @details          Set stable option of the preconditioner.
 * @param[in,out]    str Pointer to the solver (void).
 * @param[in]        stable Should we set stable CHOL
 */
void Nfft4GPPrecondCholSetStable(void *str, int stable);

/**
 * @brief            Apply one solve step with the LAPACK Cholesky solver.
 * @details          Apply one solve step with the LAPACK Cholesky solver.
 * @param[in]        vchol_mat Pointer to the solver (void).
 * @param[in]        n Size of the linear system.
 * @param[in,out]    x Solution vector.
 * @param[in]        rhs Right-hand side vector.
 * @return           0 if successful
 */
int Nfft4GPPrecondCholSolve( void *vchol_mat, int n, NFFT4GP_DOUBLE *x, NFFT4GP_DOUBLE *rhs);

/**
 * @brief   Compute y{i} = M^{-1}dM(i)*x.
 * @details Compute y{i} = M^{-1}dM(i)*x.
 * @param[in]     vchol_mat      Pointer to the preconditioner.
 * @param[in]     n              Size of the problem
 * @param[in]     mask           Mask of the gradient entries. If NULL, all entries are computed.
 *                               If preset, those marked with 0 are not computed.
 * @param[in]     x              Input vector.
 * @param[in,out] yp             Pointer to the output vector, cannot be NULL.
 * @return           Return error code.
 */
int Nfft4GPPrecondCholDvp(void *vchol_mat, int n, int *mask, NFFT4GP_DOUBLE *x, NFFT4GP_DOUBLE **yp);

/**
 * @brief   Return the trace of M^{-1}dMd/theta for theta equals to f, l, and mu.
 * @details Return the trace of M^{-1}dMd/theta for theta equals to f, l, and mu.
 * @param[in] vchol_mat Pointer to the preconditioner.
 * @return              Return error code.
 */
int Nfft4GPPrecondCholTrace(void *vchol_mat, NFFT4GP_DOUBLE **tracesp);

/**
 * @brief   Return the log determinant of M.
 * @details Return the log determinant of M.
 * @param[in] vchol_mat Pointer to the preconditioner.
 * @return              Return error code.
 */
NFFT4GP_DOUBLE Nfft4GPPrecondCholLogdet(void *vchol_mat);

/**
 * @brief            Setup the LAPACK Cholesky solver.
 * @details          Setup the LAPACK Cholesky solver.
 * @param[in]        data Dataset.
 * @param[in]        n Number of points in the dataset.
 * @param[in]        ldim Leading dimension of the dataset.
 * @param[in]        d Dimension of the dataset.
 * @param[in]        fkernel Kernel function.
 * @param[in]        fkernel_params Kernel parameters.
 * @param[in]        require_grad Should we compute the gradient?
 * @param[in,out]    vchol_mat Pointer to the preconditioner. Should be created before calling this function. \n
 *                             Can use Nfft4GPPrecondCholCreate() to create it.
 * @return           Return error code.
 */
int Nfft4GPPrecondCholSetupWithKernel(NFFT4GP_DOUBLE *data, int n, int ldim, int d,
                                    func_kernel fkernel, void *fkernel_params, int require_grad, void* vchol_mat);

/**
 * @brief   Struct for the FSAI preconditioner.
 * @details Struct for the FSAI preconditioner.
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
 * @brief            Create the FSAI preconditioner.
 * @details          Create the FSAI preconditioner.
 * @return           Pointer to the preconditioner (void).
 */
void* Nfft4GPPrecondFsaiCreate();

/**
 * @brief            Free the FSAI preconditioner.
 * @details          Free the FSAI preconditioner.
 * @param[in,out]    str Pointer to the preconditioner (void).
 */
void Nfft4GPPrecondFsaiFree(void *str);

/**
 * @brief            Reset the FSAI preconditioner without free the str pointer. Ready to setup again.
 * @details          Reset the FSAI preconditioner without free the str pointer. Ready to setup again.
 * @param[in,out]    str Pointer to the preconditioner (void).
 */
void Nfft4GPPrecondFsaiReset(void *str);

/**
 * @brief            Set lfil option of the preconditioner.
 * @details          Set lfll option of the preconditioner.
 * @param[in,out]    str Pointer to the solver (void).
 * @param[in]        lfil Number of nonzeros per column in U.
 */
void Nfft4GPPrecondFsaiSetLfil(void *str, int lfil);

/**
 * @brief            Apply one solve step with the FSAI preconditioner.
 * @details          Apply one solve step with the FSAI preconditioner.
 * @param[in]        vfsai_mat Pointer to the preconditioner (void).
 * @param[in]        n Size of the linear system.
 * @param[in,out]    x Solution vector.
 * @param[in]        rhs Right-hand side vector.
 * @return           0 if successful.
 */
int Nfft4GPPrecondFsaiSolve( void *vfsai_mat, int n, NFFT4GP_DOUBLE *x, NFFT4GP_DOUBLE *rhs);

/**
 * @brief   Compute y{i} = M^{-1}dM(i)*x.
 * @details Compute y{i} = M^{-1}dM(i)*x.
 * @param[in]     vfsai_mat      Pointer to the preconditioner.
 * @param[in]     n              Size of the problem
 * @param[in]     mask           Mask of the gradient entries. If NULL, all entries are computed.
 *                               If preset, those marked with 0 are not computed.
 * @param[in]     x              Input vector.
 * @param[in,out] yp             Pointer to the output vector, cannot be NULL.
 * @return           Return error code.
 */
int Nfft4GPPrecondFsaiDvp(void *vfsai_mat, int n, int *mask, NFFT4GP_DOUBLE *x, NFFT4GP_DOUBLE **yp);

/**
 * @brief   Return the trace of M^{-1}dMd/theta for theta equals to f, l, and mu.
 * @details Return the trace of M^{-1}dMd/theta for theta equals to f, l, and mu.
 * @param[in] vfsai_mat Pointer to the preconditioner.
 * @return              Return error code.
 */
int Nfft4GPPrecondFsaiTrace(void *vfsai_mat, NFFT4GP_DOUBLE **tracesp);

/**
 * @brief   Return the log determinant of M.
 * @details Return the log determinant of M.
 * @param[in] vfsai_mat Pointer to the preconditioner.
 * @return              Return error code.
 */
NFFT4GP_DOUBLE Nfft4GPPrecondFsaiLogdet(void *vfsai_mat);

/**
 * @brief            Setup the FSAI preconditioner with given sparsity pattern, data, and kernel.
 * @details          Setup the FSAI preconditioner with given sparsity pattern, data, and kernel.
 * @param[in]        data Dataset.
 * @param[in]        n Number of points in the dataset.
 * @param[in]        ldim Leading dimension of the dataset.
 * @param[in]        d Dimension of the dataset.
 * @param[in]        fkernel Kernel function.
 * @param[in]        fkernel_params Parameters of the kernel function.
 * @param[in]        require_grad Should we compute the gradient?
 * @param[in,out]    vfsai_mat Pointer to the preconditioner (void).
 * @return           Return error code.
 */
int Nfft4GPPrecondFsaiSetupWithKernel(NFFT4GP_DOUBLE *data, int n, int ldim, int d,
                                    func_kernel fkernel, void *fkernel_params, int require_grad, void* vfsai_mat);

/**
 * @brief            Setup the FSAI preconditioner with given sparsity pattern, data, and kernel.
 * @details          Setup the FSAI preconditioner with given sparsity pattern, data, and kernel.
 * @param[in]        A_i Row pointer of the sparsity pattern.
 * @param[in]        A_j Column indices of the sparsity pattern. WARNING: we assume that the diagonal is on the last position of each row.
 * @param[in]        data Dataset.
 * @param[in]        n Number of points in the dataset.
 * @param[in]        ldim Leading dimension of the dataset.
 * @param[in]        d Dimension of the dataset.
 * @param[in]        fkernel Kernel function.
 * @param[in]        fkernel_params Parameters of the kernel function.
 * @param[in]        require_grad Should we compute the gradient?
 * @param[in,out]    vfsai_mat Pointer to the preconditioner (void).
 * @return           Return error code.
 */
int Nfft4GPPrecondFsaiSetupWithKernelPattern(int *A_i, int *A_j, NFFT4GP_DOUBLE *data, int n, int ldim, int d,
                                    func_kernel fkernel, void *fkernel_params, int require_grad, void* vfsai_mat);

/**
 * @brief            We have L^TL \approx A^{-1}, this rountine compute x = L^{-1} rhs.
 * @details          We have L^TL \approx A^{-1}, this rountine compute x = L^{-1} rhs.
 * @note             WARNING: the diagonal entry is on the last position of each row.
 * @param[in]        vfsai_mat Pointer to the preconditioner (void).
 * @param[in]        n Size of the linear system.
 * @param[in,out]    x Solution vector.
 * @param[in]        rhs Right-hand side vector.
 * @return           0 if successful.
 */
int Nfft4GPPrecondFsaiInvL( void *vfsai_mat, int n, NFFT4GP_DOUBLE *x, NFFT4GP_DOUBLE *rhs);

/**
 * @brief            We have L^TL \approx A^{-1}, this rountine compute x = L^{-T} rhs.
 * @details          We have L^TL \approx A^{-1}, this rountine compute x = L^{-T} rhs.
 * @note             WARNING: the diagonal entry is on the last position of each row.
 * @param[in]        vfsai_mat Pointer to the preconditioner (void).
 * @param[in]        n Size of the linear system.
 * @param[in,out]    x Solution vector.
 * @param[in]        rhs Right-hand side vector.
 * @return           0 if successful.
 */
int Nfft4GPPrecondFsaiInvLT( void *vfsai_mat, int n, NFFT4GP_DOUBLE *x, NFFT4GP_DOUBLE *rhs);

/**
 * @brief            Plot the FSAI sparsity pattern.
 * @details          Plot the FSAI sparsity pattern.
 * @param[in]        vfsai_mat Pointer to the preconditioner (void).
 */
void Nfft4GPPrecondFsaiPlot(void *vfsai_mat);

/**
 * @brief   Struct for the NYS preconditioner.
 * @details Struct for the NYS preconditioner.
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
 * @brief            Create the FSAI preconditioner.
 * @details          Create the FSAI preconditioner.
 * @return           Pointer to the preconditioner (void).
 */
void* Nfft4GPPrecondNysCreate();

/**
 * @brief            Free the FSAI preconditioner.
 * @details          Free the FSAI preconditioner.
 * @param[in,out]    str Pointer to the preconditioner (void).
 */
void Nfft4GPPrecondNysFree(void *str);

/**
 * @brief            Reset the NYS preconditioner without free the str pointer. Ready to setup again.
 * @details          Reset the NYS preconditioner without free the str pointer. Ready to setup again.
 * @param[in,out]    str Pointer to the preconditioner (void).
 */
void Nfft4GPPrecondNysReset(void *str);

/**
 * @brief            Set the NYS rank.
 * @details          Set the NYS rank.
 * @param[in,out]    str Pointer to the preconditioner (void).
 * @param[in]        k Rank of the NYS preconditioner.
 */
void Nfft4GPPrecondNysSetRank(void *str, int k);

/**
 * @brief            Set the NYS permutation.
 * @details          Set the NYS permutation.
 * @param[in,out]    str Pointer to the preconditioner (void).
 * @param[in]        perm Permutation. Full n-permutation.
 * @param[in]        own_perm Does the preconditioner own the permutation?
 */
void Nfft4GPPrecondNysSetPerm(void *str, int *perm, int own_perm);

/**
 * @brief            Apply one solve step with the FSAI preconditioner.
 * @details          Apply one solve step with the FSAI preconditioner.
 * @param[in]        vfsai_mat Pointer to the preconditioner (void).
 * @param[in]        n Size of the linear system.
 * @param[in,out]    x Solution vector.
 * @param[in]        rhs Right-hand side vector.
 * @return           0 if successful.
 */
int Nfft4GPPrecondNysSolve( void *vnys_mat, int n, NFFT4GP_DOUBLE *x, NFFT4GP_DOUBLE *rhs);

/**
 * @brief   Compute y{i} = M^{-1}dM(i)*x.
 * @details Compute y{i} = M^{-1}dM(i)*x.
 * @param[in]     vnys_mat       Pointer to the preconditioner.
 * @param[in]     n              Size of the problem
 * @param[in]     mask           Mask of the gradient entries. If NULL, all entries are computed.
 *                               If preset, those marked with 0 are not computed.
 * @param[in,out] yp             Pointer to the output vector, cannot be NULL.
 * @return           Return error code.
 */
int Nfft4GPPrecondNysDvp(void *vnys_mat, int n, int *mask, NFFT4GP_DOUBLE *x, NFFT4GP_DOUBLE **yp);

/**
 * @brief   Return the trace of M^{-1}dMd/theta for theta equals to f, l, and mu.
 * @details Return the trace of M^{-1}dMd/theta for theta equals to f, l, and mu.
 * @param[in] vnys_mat  Pointer to the preconditioner.
 * @return              Return error code.
 */
int Nfft4GPPrecondNysTrace(void *vnys_mat, NFFT4GP_DOUBLE **tracesp);

/**
 * @brief   Return the log determinant of M.
 * @details Return the log determinant of M.
 * @param[in] vnys_mat  Pointer to the preconditioner.
 * @return              Return error code.
 */
NFFT4GP_DOUBLE Nfft4GPPrecondNysLogdet(void *vnys_mat);

/**
 * @brief            Setup the NYS preconditioner with given sparsity pattern, data, permutation, and kernel.
 * @details          Setup the NYS preconditioner with given sparsity pattern, data, permutation, and kernel.
 * @note             The column sample version.
 * @param[in]        data Dataset.
 * @param[in]        n Number of points in the dataset.
 * @param[in]        ldim Leading dimension of the dataset.
 * @param[in]        d Dimension of the dataset.
 * @param[in]        fkernel Kernel function.
 * @param[in]        fkernel_params Parameters of the kernel function.
 * @param[in]        require_perm Does the preconditioner require a gradient?
 * @param[in,out]    vnys_mat Pointer to the preconditioner (void).
 * @return           Return error code.
 */
int Nfft4GPPrecondNysSetupWithKernel(NFFT4GP_DOUBLE *data, int n, int ldim, int d,
                                    func_kernel fkernel, void *fkernel_params, int require_grad, void* vnys_mat);

/**
 * @brief Plot data and selected points for the NYS preconditioner.
 * @details Plot data and selected points for the NYS preconditioner.
 * @param[in] vnys_mat Pointer to the preconditioner (void).
 * @param[in] data Dataset.
 * @param[in] n Number of points in the dataset.
 * @param[in] ldim Leading dimension of the dataset.
 * @param[in] d Dimension of the dataset.
 */
void Nfft4GPPrecondNysPlot(void *vnys_mat, NFFT4GP_DOUBLE *data, int n, int ldim, int d);

#endif