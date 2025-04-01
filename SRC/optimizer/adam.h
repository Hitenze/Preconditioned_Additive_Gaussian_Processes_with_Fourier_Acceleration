#ifndef NFFT4GP_ADAM_H
#define NFFT4GP_ADAM_H

/**
 * @file adam.h
 * @brief ADAM optimizer.
 */

#include "../utils/utils.h"
#include "optimizer.h"

/**
 * @brief   Structure for the ADAM optimizer.
 * @details Structure for the ADAM optimizer containing all parameters and state variables.
 */
typedef struct NFFT4GP_OPTIMIZER_ADAM_STRUCT
{
   // Options
   int _maxits;               // Maximum number of iterations
   NFFT4GP_DOUBLE _tol;       // Tolerance

   func_loss _floss;          // Loss function
   void *_fdata;              // Loss function data

   int _n;                    // Dimension of the problem
   
   NFFT4GP_DOUBLE _beta1;     // First moment decay rate
   NFFT4GP_DOUBLE _beta2;     // Second moment decay rate
   NFFT4GP_DOUBLE _epsilon;   // Small number to avoid division by zero
   NFFT4GP_DOUBLE _alpha;     // Learning rate
   NFFT4GP_DOUBLE *_m;        // First moment vector
   NFFT4GP_DOUBLE *_v;        // Second moment vector
   NFFT4GP_DOUBLE *_m_hat;    // Bias-corrected first moment
   NFFT4GP_DOUBLE *_v_hat;    // Bias-corrected second moment
   
   // History data
   int _nits;                 // Number of iterations
   NFFT4GP_DOUBLE *_loss_history;      // History of the loss function, the first element is the initial loss
   NFFT4GP_DOUBLE *_x_history;         // History of the points, the first element is the initial point
   NFFT4GP_DOUBLE *_grad_history;      // History of the gradient, the first element is the initial gradient
   NFFT4GP_DOUBLE *_grad_norm_history; // History of the gradient norm, the first element is the initial gradient norm

} optimizer_adam, *poptimizer_adam;

/**
 * @brief   Create an ADAM optimizer.
 * @details Create an ADAM optimizer with default parameters.
 * @return           Return pointer to the optimizer.
 */
void* Nfft4GPOptimizationAdamCreate();

/**
 * @brief   Free an ADAM optimizer.
 * @details Free an ADAM optimizer and all associated memory.
 * @param [in]       optimizer      Pointer to the optimizer.
 * @return           No return.
 */
void Nfft4GPOptimizationAdamFree(void *optimizer);

/**
 * @brief   Set up the problem for an ADAM optimization.
 * @details Set up the problem for an ADAM optimization by specifying the loss function and problem dimension.
 * @param [in,out]   optimizer      Pointer to the optimizer.
 * @param [in]       floss          Loss function.
 * @param [in]       fdata          Loss function data.
 * @param [in]       n              Dimension of the problem.
 * @return           Return 0 if successful.
 */
int Nfft4GPOptimizationAdamSetProblem(
   void *optimizer, 
   func_loss floss,
   void *fdata,
   int n);

/**
 * @brief   Set up the options for an ADAM optimization.
 * @details Set up the options for an ADAM optimization including convergence criteria and algorithm parameters.
 * @param [in,out]   optimizer      Pointer to the optimizer.
 * @param [in]       maxits         Maximum number of iterations.
 * @param [in]       tol            Tolerance for convergence.
 * @param [in]       beta1          First moment decay rate (typically 0.9).
 * @param [in]       beta2          Second moment decay rate (typically 0.999).
 * @param [in]       epsilon        Small number to avoid division by zero.
 * @param [in]       alpha          Learning rate.
 * @return           Return 0 if successful.
 */
int Nfft4GPOptimizationAdamSetOptions(
   void *optimizer, 
   int maxits,
   NFFT4GP_DOUBLE tol,
   NFFT4GP_DOUBLE beta1,
   NFFT4GP_DOUBLE beta2,
   NFFT4GP_DOUBLE epsilon,
   NFFT4GP_DOUBLE alpha);

/**
 * @brief   Initialize an ADAM optimization.
 * @details Initialize an ADAM optimization with the given starting point.
 * @param [in,out]   optimizer      Pointer to the optimizer.
 * @param [in]       x              Initial point.
 * @return           Return 0 if successful.
 */
int Nfft4GPOptimizationAdamInit(void *optimizer, NFFT4GP_DOUBLE *x);

/**
 * @brief   Perform one step of ADAM optimization.
 * @details Perform one step of ADAM optimization using the current state.
 * @param [in,out]   optimizer      Pointer to the optimizer.
 * @return           Return status code: 0 = standard step, 1 = max iterations reached, 2 = tolerance reached, -1 = error.
 */
int Nfft4GPOptimizationAdamStep(void *optimizer);

#endif