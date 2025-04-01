/**
 * @file gp_problem.c
 * @brief GP Optimizer wrapper
 */

#include "gp_problem.h"
#include "gp_loss.h"
#include "gp_predict.h"

void* Nfft4GPGPProblemCreate()
{
   pnfft4gp_gp_problem str = NULL;
   NFFT4GP_MALLOC(str, 1, nfft4gp_gp_problem);

   str->_type = NFFT4GP_OPTIMIZER_UNDEFINED;
   str->_fstep = NULL;
   str->_optimizer = NULL;
   str->_dwork = NULL;
   
   str->_data_train = NULL;
   str->_data_train_n = 0;
   str->_data_train_ldim = 0;
   str->_data_train_d = 0;
   str->_label_train = NULL;
   str->_fkernel = NULL;
   str->_vfkernel_data = NULL;
   str->_matvec = NULL;
   str->_dmatvec = NULL;
   str->_precond_fkernel = NULL;
   str->_precond_vfkernel_data = NULL;
   str->_precond_setup = NULL;
   str->_precond_solve = NULL;
   str->_precond_trace = NULL;
   str->_precond_logdet = NULL;
   str->_precond_dvp = NULL;
   str->_precond_reset = NULL;
   str->_precond_data = NULL;
   str->_atol_loss = 0;
   str->_tol_loss = 1e-04;
   str->_wsize_loss = 0; // full reorthogonalization
   str->_maxits_loss = 50; // maximum 50 iterations
   str->_nvecs_loss = 5; // use 5 random vectors
   str->_radamacher_loss = NULL; // do not provide radamacher vectors
   str->_transform = NFFT4GP_TRANSFORM_SOFTPLUS; // use softplus transform
   str->_mask = NULL; // do not mask
   str->_print_level_loss = 0; // do not print

   str->_data_predict = NULL;
   str->_data_predict_n = 0;
   str->_data_predict_ldim = 0;
   str->_data_predict_d = 0;
   str->_label_predict = NULL;
   str->_matvec_predict = NULL;
   str->_atol_predict = 0;
   str->_tol_predict = 1e-06;
   str->_wsize_predict = 0; // full reorthogonalization
   str->_maxits_predict = 100; // maximum 100 iterations
   str->_print_level_predict = 0; // do not print

   return (void*)str;
}

void Nfft4GPGPProblemFree( void *gp_problem)
{
   if(gp_problem)
   {
      pnfft4gp_gp_problem str = (pnfft4gp_gp_problem)gp_problem;
      
      NFFT4GP_FREE(str->_dwork);

      NFFT4GP_FREE(str);
   }
}

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
                        int retuire_std)
{
   if(!gp_problem)
   {
      printf("Error: gp_problem is NULL.\n");
      return -1;
   }
   pnfft4gp_gp_problem str = (pnfft4gp_gp_problem)gp_problem;

   str->_type = type;
   str->_fstep = fstep;
   str->_optimizer = optimizer;

   str->_data_train = data_train;
   str->_data_train_n = data_train_n;
   str->_data_train_ldim = data_train_ldim;
   str->_data_train_d = data_train_d;
   str->_label_train = label_train;
   str->_fkernel = fkernel;
   str->_vfkernel_data = vfkernel_data;
   str->_kernel_data_free = kernel_data_free;
   str->_matvec = matvec;
   str->_dmatvec = dmatvec;
   str->_precond_fkernel = precond_fkernel;
   str->_precond_vfkernel_data = precond_vfkernel_data;
   str->_precond_kernel_data_free = precond_kernel_data_free;
   str->_precond_setup = precond_setup;
   str->_precond_solve = precond_solve;
   str->_precond_trace = precond_trace;
   str->_precond_logdet = precond_logdet;
   str->_precond_dvp = precond_dvp;
   str->_precond_reset = precond_reset;
   str->_precond_data = precond_data;
   str->_atol_loss = atol_loss;
   str->_tol_loss = tol_loss;
   str->_wsize_loss = wsize_loss;
   str->_maxits_loss = maxits_loss;
   str->_nvecs_loss = nvecs_loss;
   str->_radamacher_loss = radamacher_loss;
   str->_transform = transform;
   str->_mask = mask;
   str->_print_level_loss = print_level_loss;

   str->_data_predict = data_predict;
   str->_data_predict_n = data_predict_n;
   str->_data_predict_ldim = data_predict_ldim;
   str->_data_predict_d = data_predict_d;
   str->_label_predict = label_predict;
   str->_matvec_predict = matvec_predict;
   str->_atol_predict = atol_predict;
   str->_tol_predict = tol_predict;
   str->_wsize_predict = wsize_predict;
   str->_maxits_predict = maxits_predict;
   str->_print_level_predict = print_level_predict;
   
   return 0;
}

int Nfft4GPGpProblemLoss( void *problem, NFFT4GP_DOUBLE *x, NFFT4GP_DOUBLE *lossp, NFFT4GP_DOUBLE *dlossp)
{
   pnfft4gp_gp_problem str = (pnfft4gp_gp_problem)problem;

   //if(str->_dwork == NULL)
   //{
   //   NFFT4GP_MALLOC(str->_dwork, (size_t)(4*str->_data_train_n+4)*(str->_data_train_n), NFFT4GP_DOUBLE);
   //}

   return Nfft4GPGpLoss(x, 
                     str->_data_train,
                     str->_label_train,
                     str->_data_train_n,
                     str->_data_train_ldim,
                     str->_data_train_d,
                     str->_fkernel,
                     str->_vfkernel_data,
                     str->_kernel_data_free,
                     str->_matvec,
                     str->_dmatvec,
                     str->_precond_fkernel,
                     str->_precond_vfkernel_data,
                     str->_precond_kernel_data_free,
                     str->_precond_setup,
                     str->_precond_solve,
                     str->_precond_trace,
                     str->_precond_logdet,
                     str->_precond_dvp,
                     str->_precond_reset,
                     str->_precond_data,
                     str->_atol_loss,
                     str->_tol_loss,
                     str->_wsize_loss,
                     str->_maxits_loss,
                     str->_nvecs_loss,
                     str->_radamacher_loss,
                     str->_transform,
                     str->_mask,
                     str->_print_level_loss,
                     str->_dwork,
                     lossp,
                     dlossp
                     );
}
