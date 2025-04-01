#ifndef NFFT4GP_PRECOND_H
#define NFFT4GP_PRECOND_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "../utils/utils.h"
#include "../linearalg/kernels.h"
#include "../solvers/solvers.h"

/**
 * @brief   Enumeration of the supported preconditioners.
 * @details Enumeration of the supported preconditioners.
 */
typedef enum {
   NFFT4GP_PRECOND_UNDEFINED = 0,
   NFFT4GP_PRECOND_CHOL,
   NFFT4GP_PRECOND_FSAI,
   NFFT4GP_PRECOND_NYS,
   NFFT4GP_PRECOND_NFFT4GP
}nfft4gp_precond_type;

/**
 * @brief   Setup the LAPACK Cholesky solver.
 * @details Setup the LAPACK Cholesky solver.
 * @param [in]       data           Dataset.
 * @param [in]       n              Number of points in the dataset.
 * @param [in]       ldim           Leading dimension of the dataset.
 * @param [in]       d              Dimension of the dataset.
 * @param [in]       stable         Should we add small diagonal perturbation to the matrix?
 * @param [in]       fkernel        Kernel function.
 * @param [in]       fkernel_params Kernel parameters.
 * @param [in]       require_grad   Should we compute the gradient?
 * @param [in,out]   precond_data   Pointer to the preconditioner. Should be created before calling this function.
 * @return           Return error code.
 */
typedef int (*precond_kernel_setup)(
   NFFT4GP_DOUBLE *data, 
   int n, 
   int ldim, 
   int d, 
   func_kernel fkernel, 
   void *fkernel_params, 
   int require_grad, 
   void* precond_data);

#endif
