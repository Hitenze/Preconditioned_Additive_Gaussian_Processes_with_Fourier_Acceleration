#ifndef NFFT4GP_OPTIMIZER_HEADER_H
#define NFFT4GP_OPTIMIZER_HEADER_H

/**
 * @file _optimizer.h
 * @brief Functions related to optimization
 */

#include "_utils.h"
#include "_linearalg.h"
#include "_solvers.h"
#include "_preconds.h"

/**
 * @brief   Enumeration of the possible transformations.
 * @details Enumeration of the possible transformations.
 */
typedef enum {
   NFFT4GP_TRANSFORM_SOFTPLUS = 0,
   NFFT4GP_TRANSFORM_SIGMOID,
   NFFT4GP_TRANSFORM_EXP,
   NFFT4GP_TRANSFORM_IDENTITY
}nfft4gp_transform_type;

/**
 * @brief   Transform a value.
 * @details Transform a value.
 *          The transformation is done according to the nfft4gp_transform_type. Supported transformations are:
 *          - NFFT4GP_TRANSFORM_SOFTPLUS: softplus transformation.
 *          - NFFT4GP_TRANSFORM_SIGMOID: sigmoid transformation.
 *          - NFFT4GP_TRANSFORM_EXP: exponential transformation.
 *          - NFFT4GP_TRANSFORM_IDENTITY: identity transformation.
 * @param[in]   type    Type of transformation, see nfft4gp_transform_type.
 * @param[in]   val     Value to transform.
 * @param[in]   inverse Transform or inverse transform.
 * @param[out]  tvalp   Pointer to the transformed value.
 * @param[out]  dtvalp  Pointer to the derivative of the transformed value. Only used if inverse is false. \n
 *                      Returns a NFFT4GP_DOUBLE array of length 1.
 * @return              Return 0 if success.
 */
int Nfft4GPTransform(nfft4gp_transform_type type, NFFT4GP_DOUBLE val, int inverse, NFFT4GP_DOUBLE *tvalp, NFFT4GP_DOUBLE *dtvalp);

/**
 * @brief   Enumeration of the suppotred optimizers.
 * @details Enumeration of the suppotred optimizers.
 */
typedef enum {
   NFFT4GP_OPTIMIZER_UNDEFINED = 0,
   NFFT4GP_OPTIMIZER_ADAM,
   NFFT4GP_OPTIMIZER_NLTGCR
}nfft4gp_optimizer_type;

/**
 * @brief   Given a point x, compute the loss function and its gradient
 * @details Given a point x, compute the loss function and its gradient. \n 
 *          If lossp is NULL, the loss function is not computed. If dlossp is NULL, the gradient is not computed.
 * @param   optimizer   Pointer to the optimizer
 * @param   x           Point at which the loss function and its gradient are computed
 * @param   transform   Transform to be used
 * @param   lossp       Pointer to the loss function. If NULL, the loss function is not computed
 * @param   dloss       Pointer to the gradient of the loss function. If NULL, the gradient is not computed
 * @return  Returns 0 if successfull
 */
typedef int (*func_loss)( void *problem, NFFT4GP_DOUBLE *x, NFFT4GP_DOUBLE *lossp, NFFT4GP_DOUBLE *dloss);

/**
 * @brief   Given a point x, proceed with an optimization step
 * @details Given a point x, proceed with an optimization step.
 * @param   optimizer   Pointer to the optimizer
 * @return  Returns 0 if successfull
 */
typedef int (*func_optimization_step)( void *optimizer);

/**
 * @brief   Given a point x, proceed with an optimization step
 * @details Given a point x, proceed with an optimization step.
 * @param   optimizer   Pointer to the optimizer
 * @param   fstep       Step function
 * @param   x           Point at which the loss function and its gradient are computed
 * @param   maxits      Maximum number of iterations for the optimization
 * @param   tol         Tolerance for the optimization
 */
int Nfft4GPOptimization( void *optimizer, func_optimization_step fstep, NFFT4GP_DOUBLE *x, int maxits, NFFT4GP_DOUBLE tol);

typedef struct NFFT4GP_OPTIMIZER_ADAM_STRUCT
{
   // Options
   int _maxits; // Maximum number of iterations
   NFFT4GP_DOUBLE _tol; // Tolerance

   func_loss _floss; // loss function
   void *_fdata; // loss function data

   int _n; // dimention of the problem
   
   NFFT4GP_DOUBLE _beta1; // first moment
   NFFT4GP_DOUBLE _beta2; // second moment
   NFFT4GP_DOUBLE _epsilon; // small number to avoid division by zero
   NFFT4GP_DOUBLE _alpha; // learning rate
   NFFT4GP_DOUBLE *_m;
   NFFT4GP_DOUBLE *_v;
   NFFT4GP_DOUBLE *_m_hat;
   NFFT4GP_DOUBLE *_v_hat;
   
   // History data
   int _nits; // Number of iterations
   NFFT4GP_DOUBLE *_loss_history; // History of the loss function, the fist element is the initial loss
   NFFT4GP_DOUBLE *_x_history; // History of the points, the fist element is the initial point
   NFFT4GP_DOUBLE *_grad_history; // History of the gradient, the fist element is the initial gradient
   NFFT4GP_DOUBLE *_grad_norm_history; // History of the gradient norm, the fist element is the initial gradient norm

}optimizer_adam, *poptimizer_adam;

/**
 * @brief   Create an ADAM optimizer
 * @details Create an ADAM optimizer
 * @return  Pointer to the optimizer (void).
 */
void* Nfft4GPOptimizationAdamCreate();

/**
 * @brief   Destroy an ADAM optimizer
 * @details Destroy an ADAM optimizer
 * @param   optimizer   Pointer to the optimizer
 * @return  Returns 0 if successfull
 */
void Nfft4GPOptimizationAdamFree( void *optimizer );

/**
 * @brief   Setup the problem for an ADAM optimization
 * @details Setup the problem for an ADAM optimization
 * @param[in]   optimizer   Pointer to the optimizer
 * @param[in]   floss       Loss function
 * @param[in]   fdata       Loss function data
 * @param[in]   n           Dimension of the problem
 * @return  Returns 0 if successfull
 */
int Nfft4GPOptimizationAdamSetProblem( void *optimizer, 
                              func_loss floss,
                              void *fdata,
                              int n);

/**
 * @brief   Setup the options for an ADAM optimization
 * @details Setup the options for an ADAM optimization
 * @param[in]   optimizer   Pointer to the optimizer
 * @param[in]   maxits      Maximum number of iterations
 * @param[in]   tol         Tolerance
 * @param[in]   beta1       First moment
 * @param[in]   beta2       Second moment
 * @param[in]   epsilon     Small number to avoid division by zero
 * @param[in]   alpha       Learning rate
 * @return  Returns 0 if successfull
 */
int Nfft4GPOptimizationAdamSetOptions( void *optimizer, 
                                    int maxits,
                                    NFFT4GP_DOUBLE tol,
                                    NFFT4GP_DOUBLE beta1,
                                    NFFT4GP_DOUBLE beta2,
                                    NFFT4GP_DOUBLE epsilon,
                                    NFFT4GP_DOUBLE alpha);

/**
 * @brief   Initialize an ADAM optimization
 * @details Initialize an ADAM optimization
 * @param[in]   optimizer   Pointer to the optimizer
 * @param[in]   x           Initial point
 * @return  Returns 0 if successfull
 */
int Nfft4GPOptimizationAdamInit( void *optimizer, NFFT4GP_DOUBLE *x);

/**
 * @brief   Given a point x, proceed with an ADAM optimization step
 * @details Given a point x, proceed with an ADAM optimization step.
 * @param   optimizer   Pointer to the optimizer
 * @return  Returns flag. 0: standart step, 1: maxits reached, 2: tolerance reached, -1: error
 */
int Nfft4GPOptimizationAdamStep( void *optimizer);

/**
 * @brief   Compute the Gaussian process loss and gradient.
 * @details Compute the Gaussian process loss and gradient.
 * @param[in]   x               Pointer to the vector of parameters (before transformation).
 * @param[in]   data            Pointer to the data matrix.
 * @param[in]   label           Pointer to the label vector.
 * @param[in]   n               Number of data points.
 * @param[in]   ldim            Dimension of the data points.
 * @param[in]   d               Dimension of the parameter space.
 * @param[in]   permn           Permutation of RAN.
 * @param[in]   k               Rank of RAN.
 * @param[in]   atol            Absolute tolerance for the solver (for the first part, TODO: no need in the block version).
 * @param[in]   tol             Relative tolerance for the solver (for the first part, TODO: no need in the block version).
 * @param[in]   wsize           Window size for reorthogonalization.
 * @param[in]   maxits          Maximum number of iterations.
 * @param[in]   nvecs           Number of vectors for lanczos quadrature.
 * @param[in]   radamacher      Radamacher vector matrix. If NULL, will be generated randomly. Otherwise each column of the matrix is a radamacher vector.
 * @param[in]   mask            Mask for the parameters (0: grad is set to 0, otherwise: grad is computed).
 * @param[out]  loss            Pointer to the loss.
 * @param[out]  grad            Pointer to the gradient.
 * @return                      Return error code.
 */
int Nfft4GPGpLossGaussianRANSoftPlus(NFFT4GP_DOUBLE *x, 
                                 NFFT4GP_DOUBLE *data,
                                 NFFT4GP_DOUBLE *label,
                                 int n,
                                 int ldim,
                                 int d,
                                 int *permn,
                                 int k,
                                 int atol,
                                 NFFT4GP_DOUBLE tol,
                                 int wsize,
                                 int maxits,
                                 int nvecs,
                                 NFFT4GP_DOUBLE *radamacher,
                                 int *mask,
                                 NFFT4GP_DOUBLE *loss,
                                 NFFT4GP_DOUBLE *grad
                                 );

/**
 * @brief   Compute the Gaussian process loss and gradient.
 * @details Compute the Gaussian process loss and gradient.
 * @param[in]   x                            Pointer to the vector of parameters (before transformation).
 * @param[in]   data                         Pointer to the data matrix.
 * @param[in]   label                        Pointer to the label vector.
 * @param[in]   n                            Number of data points.
 * @param[in]   ldim                         Dimension of the data points.
 * @param[in]   d                            Dimension of the parameter space.
 * @param[in]   fkernel                      Kernel function.
 * @param[in]   vfkernel_data                Kernel function data structure.
 * @param[in]   kernel_data_free             Kernel function data structure free function.
 * @param[in]   matvec                       Matrix vector product function (for kernel matrix).
 * @param[in]   dmatvec                      Matrix vector product function (for gradient matrix).
 * @param[in]   precond_fkernel              Preconditioner kernel function.
 * @param[in]   precond_vfkernel_data        Preconditioner kernel function data structure.
 * @param[in]   precond_kernel_data_free     Preconditioner kernel function data structure free function.
 * @param[in]   precond_setup                Preconditioner setup function.
 * @param[in]   precond_solve                Preconditioner solve function.
 * @param[in]   precond_trace                Preconditioner trace function.
 * @param[in]   precond_logdet               Preconditioner log determinant function.
 * @param[in]   precond_dvp                  Preconditioner derivative function.
 * @param[in]   precond_reset                Preconditioner reset function, ready for reuse. User is responsible for freeing the preconditioner data after use.
 * @param[in]   precond_data                 Preconditioner data (with setup already done)
 * @param[in]   atol                         Absolute tolerance for the solver (for the first part, TODO: no need in the block version).
 * @param[in]   tol                          Relative tolerance for the solver (for the first part, TODO: no need in the block version).
 * @param[in]   wsize                        Window size for reorthogonalization.
 * @param[in]   maxits                       Maximum number of iterations.
 * @param[in]   nvecs                        Number of vectors for lanczos quadrature.
 * @param[in]   radamacher                   Radamacher vector matrix. If NULL, will be generated randomly. Otherwise each column of the matrix is a radamacher vector.
 * @param[in]   transform                    Transformation type.
 * @param[in]   mask                         Mask for the parameters (0: grad is set to 0, otherwise: grad is computed).
 * @param[in]   print_level                  Print level.
 * @param[in]   dwork                        Working array. Set to NULL to not use. Otherwise at least should be 4*n*n + 4*n.
 * @param[out]  loss                         Pointer to the loss.
 * @param[out]  grad                         Pointer to the gradient.
 * @return                                   Return error code.
 */
int Nfft4GPGpLoss(NFFT4GP_DOUBLE *x, 
               NFFT4GP_DOUBLE *data,
               NFFT4GP_DOUBLE *label,
               int n,
               int ldim,
               int d,
               func_kernel fkernel,
               void* vfkernel_data,
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
               int atol,
               NFFT4GP_DOUBLE tol,
               int wsize,
               int maxits,
               int nvecs,
               NFFT4GP_DOUBLE *radamacher,
               nfft4gp_transform_type transform,
               int *mask,
               int print_level,
               NFFT4GP_DOUBLE *dwork,
               NFFT4GP_DOUBLE *loss,
               NFFT4GP_DOUBLE *grad
               );

/**
 * @brief   Easy to use prediction function using Gaussian Kernel and RAN.
 * @details Easy to use prediction function using Gaussian Kernel and RAN
 * @param[in]   x               Pointer to the vector of parameters (before transformation).
 * @param[in]   data            Pointer to the data matrix.
 * @param[in]   label           Pointer to the label vector.
 * @param[in]   n               Number of data points.
 * @param[in]   ldim            Dimension of the data points.
 * @param[in]   d               Dimension of the parameter space.
 * @param[in]   data_predict    Pointer to the data matrix for prediction.
 * @param[in]   n_predict       Number of data points for prediction.
 * @param[in]   ldim_predict    Dimension of the data points for prediction.
 * @param[in]   permn           Permutation of RAN.
 * @param[in]   k               Rank of RAN.
 * @param[in]   atol            Absolute tolerance for the solver (for the first part, TODO: no need in the block version).
 * @param[in]   tol             Relative tolerance for the solver (for the first part, TODO: no need in the block version).
 * @param[in]   maxits          Maximum number of iterations.
 * @param[out]  label_predictp  Pointer to the pointer to the prediction labels.
 * @param[out]  std_predictp    Pointer to the pointer to the standard deviation of the prediction.
 * @return                      Return error code.
 */
int Nfft4GPGpPredictGaussianRANSoftPlus(NFFT4GP_DOUBLE *x, 
                                    NFFT4GP_DOUBLE *data,
                                    NFFT4GP_DOUBLE *label,
                                    int n,
                                    int ldim,
                                    int d,
                                    NFFT4GP_DOUBLE *data_predict,
                                    int n_predict,
                                    int ldim_predict,
                                    int *permn,
                                    int k,
                                    int atol,
                                    NFFT4GP_DOUBLE tol,
                                    int maxits,
                                    NFFT4GP_DOUBLE **label_predictp,
                                    NFFT4GP_DOUBLE **std_predictp
                                    );

/**
 * @brief   Use the current parameters to make prediction.
 * @details Use the current parameters to make prediction.
 * @param[in]   x               Pointer to the vector of parameters (before transformation).
 * @param[in]   data            Pointer to the data matrix.
 * @param[in]   label           Pointer to the label vector.
 * @param[in]   n               Number of data points.
 * @param[in]   ldim            Dimension of the data points.
 * @param[in]   d               Dimension of the parameter space.
 * @param[in]   data_predict    Pointer to the data matrix for prediction.
 * @param[in]   n_predict       Number of data points for prediction.
 * @param[in]   ldim_predict    Dimension of the data points for prediction.
 * @param[in]   fkernel         Kernel function.
 * @param[in]   vfkernel_data   Kernel function data structure.
 * @param[in]   matvec          Matrix vector product function (for symmetric kernel matrix).
 * @param[in]   matvec_predict  Matrix vector product function (for general kernel matrix).
 * @param[in]   precond_setup   Preconditioner setup function.
 * @param[in]   precond_solve   Preconditioner solve function.
 * @param[in]   precond_data    Preconditioner data (with setup already done)
 * @param[in]   atol            Absolute tolerance for the solver (for the first part, TODO: no need in the block version).
 * @param[in]   tol             Relative tolerance for the solver (for the first part, TODO: no need in the block version).
 * @param[in]   maxits          Maximum number of iterations.
 * @param[in]   transform       Transformation type.
 * @param[in]   print_level     Print level.
 * @param[in]   dwork           Working array. Set to NULL to not use. Otherwise at least should be 4*n*n + 4*n.
 * @param[out]  label_predictp  Pointer to the pointer to the prediction labels.
 * @param[out]  std_predictp    Pointer to the pointer to the standard deviation of the prediction.
 * @return                      Return error code.
 */
int Nfft4GPGpPredict(NFFT4GP_DOUBLE *x, 
                  NFFT4GP_DOUBLE *data,
                  NFFT4GP_DOUBLE *label,
                  int n,
                  int ldim,
                  int d,
                  NFFT4GP_DOUBLE *data_predict,
                  int n_predict,
                  int ldim_predict,
                  func_kernel fkernel,
                  void* vfkernel_data,
                  func_symmatvec matvec,
                  func_matvec matvec_predict,
                  precond_kernel_setup precond_setup,
                  func_solve precond_solve,
                  void *precond_data,
                  int atol,
                  NFFT4GP_DOUBLE tol,
                  int maxits,
                  nfft4gp_transform_type transform,
                  int print_level,
                  NFFT4GP_DOUBLE *dwork,
                  NFFT4GP_DOUBLE **label_predictp,
                  NFFT4GP_DOUBLE **std_predictp
                  );

/**
 * @brief   Structure of wrapper for the optimizer for Gaussian processes
 * @details Structure of wrapper for the optimizer for Gaussian processes
 */
typedef struct NFFT4GP_GP_OPTIMIZER_STRUCT
{
   // Self data
   nfft4gp_optimizer_type _type; // Type of optimizer
   func_optimization_step _fstep; // Pointer to the optimization step function
   void *_optimizer; // Pointer to the optimizer
   NFFT4GP_DOUBLE *_dwork; // Working array
   
   // Loss data
   // Note that this struct is not responsible for the memory management of the following pointers
   NFFT4GP_DOUBLE *_data_train; // Training data
   int _data_train_n; // Number of training data
   int _data_train_ldim; // Leading dimension of the training data
   int _data_train_d; // Dimension of the training data
   NFFT4GP_DOUBLE *_label_train; // Training labels
   func_kernel _fkernel; // Pointer to the kernel function
   void *_vfkernel_data; // Pointer to the data of the kernel function
   func_free _kernel_data_free; // Pointer to the free function of the kernel function data
   func_symmatvec _matvec; // Pointer to the matrix-vector product function
   func_symmatvec _dmatvec; // Pointer to the matrix-vector product function for the gradient
   func_kernel _precond_fkernel; // Pointer to the preconditioner kernel function
   void *_precond_vfkernel_data; // Pointer to the data of the preconditioner kernel function
   func_free _precond_kernel_data_free; // Pointer to the free function of the preconditioner kernel function data
   precond_kernel_setup _precond_setup; // Pointer to the preconditioner setup function
   func_solve _precond_solve; // Pointer to the preconditioner solve function
   func_trace _precond_trace; // Pointer to the preconditioner trace function
   func_logdet _precond_logdet; // Pointer to the preconditioner logdet function
   func_dvp _precond_dvp; // Pointer to the preconditioner derivative of the vector product function
   func_free _precond_reset; // Pointer to the preconditioner reset function
   void *_precond_data; // Pointer to the data of the preconditioner
   int _atol_loss; // Absolute tolerance for the loss function?
   NFFT4GP_DOUBLE _tol_loss; // Tolerance for the loss function
   int _wsize_loss; // Window size for the loss function
   int _maxits_loss; // Maximum number of iterations for the loss function
   int _nvecs_loss; // Number of random vectors for the loss function
   NFFT4GP_DOUBLE *_radamacher_loss; // Radamacher vector for the loss function
   nfft4gp_transform_type _transform; // Transform to be used for the loss function
   int *_mask; // Mask for the loss function
   int _print_level_loss; // Print level for the loss function

   // Prediction data
   // Note that this struct is not responsible for the memory management of the following pointers
   NFFT4GP_DOUBLE *_data_predict; // Predict data
   int _data_predict_n; // Number of predict data
   int _data_predict_ldim; // Leading dimension of the predict data
   int _data_predict_d; // Dimension of the predict data
   NFFT4GP_DOUBLE *_label_predict; // Predict labels, not required
   func_matvec _matvec_predict; // Pointer to the matrix-vector product function for the predict data
   int _atol_predict; // Absolute tolerance for the predict data?
   NFFT4GP_DOUBLE _tol_predict; // Tolerance for the predict data
   int _wsize_predict; // Window size for the predict data
   int _maxits_predict; // Maximum number of iterations for the predict data
   int _print_level_predict; // Print level for the predict data
   int _retuire_std; // Require standard deviation for the predict data?

}nfft4gp_gp_problem, *pnfft4gp_gp_problem;

/**
 * @brief   Create an optimizer for Gaussian processes and set default parameters
 * @details Create an optimizer for Gaussian processes and set default parameters.
 */
void* Nfft4GPGPProblemCreate();

/**
 * @brief   Free the optimizer
 * @details Free the optimizer.
 * @param   optimizer   Pointer to the optimizer
 */
void Nfft4GPGPProblemFree( void *gp_problem);

/**
 * @brief      Set the GP optimizer
 * @details    Set the GP optimizer.
 * @param[in]   gp_problem             Pointer to the GP optimizer str created by Nfft4GPGPOptimizerCreate
 * @param[in]   type                   Type of the optimizer, see nfft4gp_optimizer_type
 * @param[in]   fstep                  Pointer to the optimization step function
 * @param[in]   optimizer              Pointer to the optimizer
 * @param[in]   data_train             Pointer to the training data matrix
 * @param[in]   data_train_n           Number of training data
 * @param[in]   data_train_ldim        Leading dimension of the training data
 * @param[in]   data_train_d           Dimension of the training data
 * @param[in]   label_train            Pointer to the training labels
 * @param[in]   fkernel                Pointer to the kernel function
 * @param[in]   vfkernel_data          Pointer to the data of the kernel function
 * @param[in]   kernel_data_free       Pointer to the free function of the kernel function data
 * @param[in]   matvec                 Pointer to the matrix-vector product function
 * @param[in]   dmatvec                Pointer to the matrix-vector product function for the gradient
 * @param[in]   precond_fkernel        Pointer to the preconditioner kernel function
 * @param[in]   precond_vfkernel_data  Pointer to the data of the preconditioner kernel function
 * @param[in]   precond_setup          Pointer to the preconditioner setup function
 * @param[in]   precond_setup          Pointer to the preconditioner setup function
 * @param[in]   precond_solve          Pointer to the preconditioner solve function
 * @param[in]   precond_trace          Pointer to the preconditioner trace function
 * @param[in]   precond_logdet         Pointer to the preconditioner logdet function
 * @param[in]   precond_dvp            Pointer to the preconditioner derivative of the vector product function
 * @param[in]   precond_reset          Pointer to the preconditioner reset function
 * @param[in]   precond_data           Pointer to the data of the preconditioner
 * @param[in]   atol_loss              Absolute tolerance for the loss function?
 * @param[in]   tol_loss               Tolerance for the loss function
 * @param[in]   wsize_loss             Window size for the loss function
 * @param[in]   maxits_loss            Maximum number of iterations for the loss function
 * @param[in]   nvecs_loss             Number of random vectors for the loss function
 * @param[in]   radamacher_loss        Radamacher vector for the loss function
 * @param[in]   transform              Transform to be used for the loss function
 * @param[in]   mask                   Mask for the loss function
 * @param[in]   print_level_loss       Print level for the loss function
 * @param[in]   data_predict           Pointer to the predict data matrix
 * @param[in]   data_predict_n         Number of predict data
 * @param[in]   data_predict_ldim      Leading dimension of the predict data
 * @param[in]   data_predict_d         Dimension of the predict data
 * @param[in]   label_predict          Pointer to the predict labels
 * @param[in]   matvec_predict         Pointer to the matrix-vector product function for the predict data
 * @param[in]   atol_predict           Absolute tolerance for the predict data?
 * @param[in]   tol_predict            Tolerance for the predict data
 * @param[in]   wsize_predict          Window size for the predict data
 * @param[in]   maxits_predict         Maximum number of iterations for the predict data
 * @param[in]   print_level_predict    Print level for the predict data
 * @param[in]   retuire_std            Require standard deviation for the predict data?
 * return   Returns 0 if successfull
 */
int Nfft4GPGpProblemSetup(void *gp_problem,
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
                        int retuire_std);

/**
 * @brief   Given a point x, compute the GP loss function and its gradient
 * @details Given a point x, compute the GP loss function and its gradient. \n 
 *          If lossp is NULL, the loss function is not computed. If dlossp is NULL, the gradient is not computed.
 * @param   optimizer   Pointer to the optimizer
 * @param   x           Point at which the loss function and its gradient are computed
 * @param   transform   Transform to be used
 * @param   lossp       Pointer to the loss function. If NULL, the loss function is not computed
 * @param   dloss       Pointer to the gradient of the loss function. If NULL, the gradient is not computed
 * @return  Returns 0 if successfull
 */
int Nfft4GPGpProblemLoss( void *problem, NFFT4GP_DOUBLE *x, NFFT4GP_DOUBLE *lossp, NFFT4GP_DOUBLE *dloss);

#endif