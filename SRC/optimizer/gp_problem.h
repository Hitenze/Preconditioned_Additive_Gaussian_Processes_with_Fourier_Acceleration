#ifndef NFFT4GP_GP_PROBLEM_H
#define NFFT4GP_GP_PROBLEM_H

/**
 * @file gp_problem.h
 * @brief GP Problem function.
 */

#include "../utils/utils.h"
#include "../linearalg/kernels.h"
#include "../solvers/solvers.h"
#include "../preconds/precond.h"
#include "transform.h"
#include "optimizer.h"

/**
 * @brief   Structure of wrapper for the optimizer for Gaussian processes.
 * @details Structure of wrapper for the optimizer for Gaussian processes.
 */
typedef struct NFFT4GP_GP_OPTIMIZER_STRUCT
{
   // Self data
   nfft4gp_optimizer_type _type;        // Type of optimizer
   func_optimization_step _fstep;       // Pointer to the optimization step function
   void *_optimizer;                    // Pointer to the optimizer
   NFFT4GP_DOUBLE *_dwork;              // Working array
   
   // Loss data
   // Note that this struct is not responsible for the memory management of the following pointers
   NFFT4GP_DOUBLE *_data_train;         // Training data
   int _data_train_n;                   // Number of training data
   int _data_train_ldim;                // Leading dimension of the training data
   int _data_train_d;                   // Dimension of the training data
   NFFT4GP_DOUBLE *_label_train;        // Training labels
   func_kernel _fkernel;                // Pointer to the kernel function
   void *_vfkernel_data;                // Pointer to the data of the kernel function
   func_free _kernel_data_free;         // Pointer to the free function of the kernel function data
   func_symmatvec _matvec;              // Pointer to the matrix-vector product function
   func_symmatvec _dmatvec;             // Pointer to the matrix-vector product function for the gradient
   func_kernel _precond_fkernel;        // Pointer to the preconditioner kernel function
   void *_precond_vfkernel_data;        // Pointer to the data of the preconditioner kernel function
   func_free _precond_kernel_data_free; // Pointer to the free function of the preconditioner kernel function data
   precond_kernel_setup _precond_setup; // Pointer to the preconditioner setup function
   func_solve _precond_solve;           // Pointer to the preconditioner solve function
   func_trace _precond_trace;           // Pointer to the preconditioner trace function
   func_logdet _precond_logdet;         // Pointer to the preconditioner logdet function
   func_dvp _precond_dvp;               // Pointer to the preconditioner derivative of the vector product function
   func_free _precond_reset;            // Pointer to the preconditioner reset function
   void *_precond_data;                 // Pointer to the data of the preconditioner
   int _atol_loss;                      // Absolute tolerance for the loss function
   NFFT4GP_DOUBLE _tol_loss;            // Tolerance for the loss function
   int _wsize_loss;                     // Window size for the loss function
   int _maxits_loss;                    // Maximum number of iterations for the loss function
   int _nvecs_loss;                     // Number of random vectors for the loss function
   NFFT4GP_DOUBLE *_radamacher_loss;    // Radamacher vector for the loss function
   nfft4gp_transform_type _transform;   // Transform to be used for the loss function
   int *_mask;                          // Mask for the loss function
   int _print_level_loss;               // Print level for the loss function

   // Prediction data
   // Note that this struct is not responsible for the memory management of the following pointers
   NFFT4GP_DOUBLE *_data_predict;       // Predict data
   int _data_predict_n;                 // Number of predict data
   int _data_predict_ldim;              // Leading dimension of the predict data
   int _data_predict_d;                 // Dimension of the predict data
   NFFT4GP_DOUBLE *_label_predict;      // Predict labels, not required
   func_matvec _matvec_predict;         // Pointer to the matrix-vector product function for the predict data
   int _atol_predict;                   // Absolute tolerance for the predict data
   NFFT4GP_DOUBLE _tol_predict;         // Tolerance for the predict data
   int _wsize_predict;                  // Window size for the predict data
   int _maxits_predict;                 // Maximum number of iterations for the predict data
   int _print_level_predict;            // Print level for the predict data
   int _retuire_std;                    // Require standard deviation for the predict data

} nfft4gp_gp_problem, *pnfft4gp_gp_problem;

/**
 * @brief   Create a GP problem.
 * @details Create a GP problem structure.
 * @return           Return pointer to the GP problem.
 */
void* Nfft4GPGPProblemCreate();

/**
 * @brief   Free a GP problem.
 * @details Free a GP problem and all associated memory.
 * @param [in]       gp_problem      Pointer to the GP problem.
 * @return           No return.
 */
void Nfft4GPGPProblemFree(void *gp_problem);

/**
 * @brief   Set up a GP problem.
 * @details Set up a GP problem with all necessary parameters for training and prediction.
 * @param [in,out]   gp_problem              Pointer to the GP problem.
 * @param [in]       type                    Type of optimizer.
 * @param [in]       fstep                   Optimization step function.
 * @param [in]       optimizer               Pointer to the optimizer.
 * @param [in]       data_train              Training data.
 * @param [in]       data_train_n            Number of training data points.
 * @param [in]       data_train_ldim         Leading dimension of the training data.
 * @param [in]       data_train_d            Dimension of the training data.
 * @param [in]       label_train             Training labels.
 * @param [in]       fkernel                 Kernel function.
 * @param [in]       vfkernel_data           Kernel function data.
 * @param [in]       kernel_data_free        Kernel function data free function.
 * @param [in]       matvec                  Matrix-vector product function.
 * @param [in]       dmatvec                 Matrix-vector product function for gradient.
 * @param [in]       precond_fkernel         Preconditioner kernel function.
 * @param [in]       precond_vfkernel_data   Preconditioner kernel function data.
 * @param [in]       precond_kernel_data_free Preconditioner kernel function data free function.
 * @param [in]       precond_setup           Preconditioner setup function.
 * @param [in]       precond_solve           Preconditioner solve function.
 * @param [in]       precond_trace           Preconditioner trace function.
 * @param [in]       precond_logdet          Preconditioner logdet function.
 * @param [in]       precond_dvp             Preconditioner derivative function.
 * @param [in]       precond_reset           Preconditioner reset function.
 * @param [in]       precond_data            Preconditioner data.
 * @param [in]       atol_loss               Absolute tolerance for loss function.
 * @param [in]       tol_loss                Tolerance for loss function.
 * @param [in]       wsize_loss              Window size for loss function.
 * @param [in]       maxits_loss             Maximum iterations for loss function.
 * @param [in]       nvecs_loss              Number of random vectors for loss function.
 * @param [in]       radamacher_loss         Radamacher vector for loss function.
 * @param [in]       transform               Transform type.
 * @param [in]       mask                    Mask for loss function.
 * @param [in]       print_level_loss        Print level for loss function.
 * @param [in]       data_predict            Prediction data.
 * @param [in]       data_predict_n          Number of prediction data points.
 * @param [in]       data_predict_ldim       Leading dimension of prediction data.
 * @param [in]       data_predict_d          Dimension of prediction data.
 * @param [in]       label_predict           Prediction labels (can be NULL).
 * @param [in]       matvec_predict          Matrix-vector product function for prediction.
 * @param [in]       atol_predict            Absolute tolerance for prediction.
 * @param [in]       tol_predict             Tolerance for prediction.
 * @param [in]       wsize_predict           Window size for prediction.
 * @param [in]       maxits_predict          Maximum iterations for prediction.
 * @param [in]       print_level_predict     Print level for prediction.
 * @param [in]       retuire_std             Whether to compute standard deviation.
 * @return           Return 0 if successful.
 */
int Nfft4GPGpProblemSetup(
   void *gp_problem,
   nfft4gp_optimizer_type type,
   func_optimization_step fstep,
   void *optimizer,
   NFFT4GP_DOUBLE *data_train,
   int data_train_n,
   int data_train_ldim,
   int data_train_d,
   NFFT4GP_DOUBLE *label_train,
   func_kernel fkernel,
   void *vfkernel_data,
   func_free kernel_data_free,
   func_symmatvec matvec,
   func_symmatvec dmatvec,
   func_kernel precond_fkernel,
   void* precond_vfkernel_data,
   func_free precond_kernel_data_free,
   precond_kernel_setup precond_setup,
   func_solve precond_solve,
   func_trace precond_trace,
   func_logdet precond_logdet,
   func_dvp precond_dvp,
   func_free precond_reset,
   void *precond_data,
   int atol_loss,
   NFFT4GP_DOUBLE tol_loss,
   int wsize_loss,
   int maxits_loss,
   int nvecs_loss,
   NFFT4GP_DOUBLE *radamacher_loss,
   nfft4gp_transform_type transform,
   int *mask,
   int print_level_loss,
   NFFT4GP_DOUBLE *data_predict,
   int data_predict_n,
   int data_predict_ldim,
   int data_predict_d,
   NFFT4GP_DOUBLE *label_predict,
   func_matvec matvec_predict,
   int atol_predict,
   NFFT4GP_DOUBLE tol_predict,
   int wsize_predict,
   int maxits_predict,
   int print_level_predict,
   int retuire_std
);

/**
 * @brief   Compute the loss and gradient for a GP problem.
 * @details Compute the loss and gradient for a GP problem given parameters.
 * @param [in]       problem         Pointer to the GP problem.
 * @param [in]       x               Parameters.
 * @param [out]      lossp           Pointer to store the loss value.
 * @param [out]      dloss           Pointer to store the gradient.
 * @return           Return 0 if successful.
 */
int Nfft4GPGpProblemLoss(void *problem, NFFT4GP_DOUBLE *x, NFFT4GP_DOUBLE *lossp, NFFT4GP_DOUBLE *dloss);

#endif