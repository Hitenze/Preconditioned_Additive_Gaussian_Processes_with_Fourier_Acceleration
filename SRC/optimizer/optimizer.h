#ifndef NFFT4GP_OPTIMIZER_H
#define NFFT4GP_OPTIMIZER_H

/**
 * @file optimizer.h
 * @brief Optimization function
 */

#include "../utils/utils.h"
#include "../linearalg/kernels.h"
#include "../solvers/solvers.h"
#include "../preconds/precond.h"
#include "transform.h"

/**
 * @brief   Enumeration of the supported optimizers.
 * @details Enumeration of the supported optimizers.
 */
typedef enum {
   NFFT4GP_OPTIMIZER_UNDEFINED = 0,
   NFFT4GP_OPTIMIZER_ADAM,
   NFFT4GP_OPTIMIZER_NLTGCR
}nfft4gp_optimizer_type;

/**
 * @brief   Given a point x, compute the loss function and its gradient.
 * @details Given a point x, compute the loss function and its gradient. \n 
 *          If lossp is NULL, the loss function is not computed. If dlossp is NULL, the gradient is not computed.
 * @param [in]       problem       Pointer to the problem data structure.
 * @param [in]       x             Point at which the loss function and its gradient are computed.
 * @param [out]      lossp         Pointer to the loss function. If NULL, the loss function is not computed.
 * @param [out]      dloss         Pointer to the gradient of the loss function. If NULL, the gradient is not computed.
 * @return           Return 0 if successful.
 */
typedef int (*func_loss)(void *problem, NFFT4GP_DOUBLE *x, NFFT4GP_DOUBLE *lossp, NFFT4GP_DOUBLE *dloss);

/**
 * @brief   Given a point x, proceed with an optimization step.
 * @details Given a point x, proceed with an optimization step.
 * @param [in]       optimizer     Pointer to the optimizer.
 * @return           Return 0 if successful.
 */
typedef int (*func_optimization_step)(void *optimizer);

/**
 * @brief   Perform optimization using the provided optimizer and step function.
 * @details Perform optimization using the provided optimizer and step function.
 * @param [in]       optimizer     Pointer to the optimizer.
 * @param [in]       fstep         Step function.
 * @param [in,out]   x             Initial point and final solution.
 * @param [in]       maxits        Maximum number of iterations for the optimization.
 * @param [in]       tol           Tolerance for the optimization.
 * @return           Return 0 if successful.
 */
int Nfft4GPOptimization(void *optimizer, func_optimization_step fstep, NFFT4GP_DOUBLE *x, int maxits, NFFT4GP_DOUBLE tol);

#endif