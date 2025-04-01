#include "adam.h"
#include "../linearalg/vecops.h"

void* Nfft4GPOptimizationAdamCreate()
{
   poptimizer_adam adam = NULL;
   NFFT4GP_MALLOC(adam, 1, optimizer_adam);

   adam->_maxits = 1000;
   adam->_tol = 1e-6;

   adam->_floss = NULL;
   adam->_fdata = NULL;

   adam->_n = 0;

   adam->_beta1 = 0.9;
   adam->_beta2 = 0.999;
   adam->_epsilon = 1e-8;
   adam->_alpha = 0.001;
   adam->_m = NULL;
   adam->_v = NULL;
   adam->_m_hat = NULL;
   adam->_v_hat = NULL;

   adam->_nits = 0;
   adam->_loss_history = NULL;
   adam->_x_history = NULL;
   adam->_grad_history = NULL;
   adam->_grad_norm_history = NULL;

   return (void*)adam;
}

void Nfft4GPOptimizationAdamFree( void *optimizer )
{
   poptimizer_adam adam = (poptimizer_adam)optimizer;
   if(adam)
   {
      NFFT4GP_FREE(adam->_m);
      NFFT4GP_FREE(adam->_v);
      NFFT4GP_FREE(adam->_m_hat);
      NFFT4GP_FREE(adam->_v_hat);
      NFFT4GP_FREE(adam->_loss_history);
      NFFT4GP_FREE(adam->_x_history);
      NFFT4GP_FREE(adam->_grad_history);
      NFFT4GP_FREE(adam->_grad_norm_history);
   }

   NFFT4GP_FREE(adam);
}

int Nfft4GPOptimizationAdamSetProblem( void *optimizer, 
                              func_loss floss,
                              void *fdata,
                              int n)
{
   poptimizer_adam adam = (poptimizer_adam)optimizer;
   adam->_floss = floss;
   adam->_fdata = fdata;
   adam->_n = n;

   return 0;
}

int Nfft4GPOptimizationAdamSetOptions( void *optimizer, 
                                    int maxits,
                                    NFFT4GP_DOUBLE tol,
                                    NFFT4GP_DOUBLE beta1,
                                    NFFT4GP_DOUBLE beta2,
                                    NFFT4GP_DOUBLE epsilon,
                                    NFFT4GP_DOUBLE alpha)
{
   poptimizer_adam adam = (poptimizer_adam)optimizer;

   if(adam->_nits > 0)
   {
      printf("Error: cannot set options after the first optimization step.\n");
      return -1;
   }

   adam->_maxits = maxits;
   adam->_tol = tol;
   adam->_beta1 = beta1;
   adam->_beta2 = beta2;
   adam->_epsilon = epsilon;
   adam->_alpha = alpha;

   return 0;
}

int Nfft4GPOptimizationAdamInit( void *optimizer, NFFT4GP_DOUBLE *x)
{
   poptimizer_adam adam = (poptimizer_adam)optimizer;
   if(adam->_nits > 0)
   {
      printf("Error: cannot initialize after the first optimization step.\n");
      return -1;
   }

   // create necessary memory
   NFFT4GP_CALLOC(adam->_m, adam->_n, NFFT4GP_DOUBLE);
   NFFT4GP_CALLOC(adam->_v, adam->_n, NFFT4GP_DOUBLE);
   NFFT4GP_MALLOC(adam->_m_hat, adam->_n, NFFT4GP_DOUBLE);
   NFFT4GP_MALLOC(adam->_v_hat, adam->_n, NFFT4GP_DOUBLE);
   NFFT4GP_CALLOC(adam->_loss_history, adam->_maxits+1, NFFT4GP_DOUBLE);
   NFFT4GP_CALLOC(adam->_x_history, (adam->_maxits+1)*adam->_n, NFFT4GP_DOUBLE);
   NFFT4GP_CALLOC(adam->_grad_history, (adam->_maxits+1)*adam->_n, NFFT4GP_DOUBLE);
   NFFT4GP_CALLOC(adam->_grad_norm_history, adam->_maxits+1, NFFT4GP_DOUBLE);

   NFFT4GP_MEMCPY(adam->_x_history, x, adam->_n, NFFT4GP_DOUBLE);

   // print the header for information for each iteration
   // we have 3 columns: iteration | loss | grad_norm

   printf("Iteration  | Loss            | Grad norm     \n");

   return 0;
}

int Nfft4GPOptimizationAdamStep( void *optimizer)
{
   poptimizer_adam adam = (poptimizer_adam)optimizer;

   int i;

   // compute the loss function and its gradient
   int err = adam->_floss(adam->_fdata, adam->_x_history+adam->_nits*adam->_n, adam->_loss_history+adam->_nits, adam->_grad_history+adam->_nits*adam->_n);
   
   if(err != 0)
   {
      printf("Loss failed\n");
      printf("Current x:\n");
      for(i = 0 ; i < adam->_n ; i ++)
      {
         printf("%15.8e ", adam->_x_history[adam->_nits*adam->_n+i]);
      }
      printf("\n");
      return -1;
   }

   // now loss is in loss history and grad is in grad history
   // TODO: add openmp here. Note that for our GP we only have 3 parameters so not necessary at this point
   for(i = 0 ; i < adam->_n ; i ++)
   {
      adam->_m[i] = adam->_beta1*adam->_m[i] + (1-adam->_beta1)*adam->_grad_history[adam->_nits*adam->_n+i];
      adam->_v[i] = adam->_beta2*adam->_v[i] + (1-adam->_beta2)*adam->_grad_history[adam->_nits*adam->_n+i]*adam->_grad_history[adam->_nits*adam->_n+i];
      // TODO: extra flop here, but again not a big deal for our GP at this point
      adam->_m_hat[i] = adam->_m[i]/(1-pow(adam->_beta1, adam->_nits+1.0));
      adam->_v_hat[i] = adam->_v[i]/(1-pow(adam->_beta2, adam->_nits+1.0));
      adam->_x_history[(adam->_nits+1)*adam->_n+i] = 
         adam->_x_history[adam->_nits*adam->_n+i] - 
            adam->_alpha*adam->_m_hat[i]/
            (sqrt(adam->_v_hat[i])+adam->_epsilon);
   }
   
   adam->_nits++; // increase the number of iterations

   // check the stopping criteria
   adam->_grad_norm_history[adam->_nits-1] = Nfft4GPVecNorm2( adam->_grad_history+(adam->_nits-1)*adam->_n, adam->_n);

   // print the information for this iteration
   // we have 3 columns: iteration | loss | grad_norm
   // each should be 15 characters long
   // with vertical bars

   //printf("%10d | %15.8e | %15.8e\n", adam->_nits, adam->_loss_history[adam->_nits-1], adam->_grad_norm_history[adam->_nits-1]);
   
   NFFT4GP_DOUBLE x00, dx00, x01, dx01, x02, dx02;
   Nfft4GPTransform( NFFT4GP_TRANSFORM_SOFTPLUS, adam->_x_history[adam->_nits*3], 0, &x00, &dx00);
   Nfft4GPTransform( NFFT4GP_TRANSFORM_SOFTPLUS, adam->_x_history[adam->_nits*3+1], 0, &x01, &dx01);
   Nfft4GPTransform( NFFT4GP_TRANSFORM_SOFTPLUS, adam->_x_history[adam->_nits*3+2], 0, &x02, &dx02);
   printf("%10d | %15.8e | %15.8e | current params: %15.8e, %15.8e, %15.8e | current params (after transform): %15.8e, %15.8e, %15.8e\n", 
               adam->_nits, adam->_loss_history[adam->_nits-1], adam->_grad_norm_history[adam->_nits-1],
               adam->_x_history[adam->_nits*3], adam->_x_history[adam->_nits*3+1], adam->_x_history[adam->_nits*3+2],
               x00, x01, x02);
   
   if(adam->_grad_norm_history[adam->_nits-1] < adam->_tol)
   {
      printf("Iteration stopped at %d with norm %e\n", adam->_nits, adam->_grad_norm_history[adam->_nits-1]);
      return 2; // tol reached
   }

   if(adam->_nits >= adam->_maxits)
   {
      return 1; // maxits reached
   }

   return 0; // normal iteration
}
