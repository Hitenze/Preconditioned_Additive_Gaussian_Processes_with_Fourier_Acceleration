#include "gp_predict.h"
#include "../solvers/fgmres.h"
#include "../preconds/nys.h"

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
                                    )
{

   func_kernel gausskernel = &Nfft4GPKernelGaussianKernel;
   pnfft4gp_kernel gausskernel_data = (pnfft4gp_kernel)Nfft4GPKernelParamCreate(n,1);

   pprecond_nys nys_mat = (pprecond_nys) Nfft4GPPrecondNysCreate();
   Nfft4GPPrecondNysSetPerm(nys_mat, permn, 0);
   Nfft4GPPrecondNysSetRank(nys_mat, k);
   
   int err = Nfft4GPGpPredict(x, 
                  data,
                  label,
                  n,
                  ldim,
                  d,
                  data_predict,
                  n_predict,
                  ldim_predict,
                  gausskernel,
                  gausskernel_data,
                  &Nfft4GPDenseMatSymv,
                  &Nfft4GPDenseMatGemv,
                  &Nfft4GPPrecondNysSetupWithKernel,
                  &Nfft4GPPrecondNysSolve,
                  nys_mat,
                  atol,
                  tol,
                  maxits,
                  NFFT4GP_TRANSFORM_SOFTPLUS,
                  0,
                  NULL,
                  label_predictp,
                  std_predictp
                  );
   
   Nfft4GPKernelParamFree(gausskernel_data);
   return err;
}

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
                  )
{

   int i;

   /* Get the transformed parameters */
   NFFT4GP_DOUBLE tvals[3], dtvals[3];

   Nfft4GPTransform( transform, x[0], 0, tvals, dtvals);
   Nfft4GPTransform( transform, x[1], 0, tvals+1, dtvals+1);
   Nfft4GPTransform( transform, x[2], 0, tvals+2, dtvals+2);

   /* Create kernel matrix */
   pnfft4gp_kernel fkernel_data = (pnfft4gp_kernel)vfkernel_data;
   fkernel_data->_params[0] = tvals[0]; // f
   fkernel_data->_params[1] = tvals[1]; // l
   fkernel_data->_noise_level = tvals[2]; // mu

   NFFT4GP_DOUBLE *data_all = NULL;
   NFFT4GP_DOUBLE *K11_mat = NULL;
   NFFT4GP_DOUBLE *K12_mat = NULL;
   NFFT4GP_DOUBLE *K22_mat = NULL;
   int *unit_perm = Nfft4GPExpandPerm( NULL, 0, n + n_predict);
   
   NFFT4GP_MALLOC( data_all, (size_t)(n+n_predict)*d, NFFT4GP_DOUBLE);
   for(i = 0 ; i < d ; i++)
   {
      NFFT4GP_MEMCPY( data_all + (size_t)i*(n+n_predict), data + (size_t)i*ldim, n, NFFT4GP_DOUBLE);
      NFFT4GP_MEMCPY( data_all + (size_t)i*(n+n_predict)+n, data_predict + (size_t)i*ldim_predict, n_predict, NFFT4GP_DOUBLE);
   }
   
   fkernel( vfkernel_data, data, n, ldim, d, NULL, 0, NULL, 0, &K11_mat, NULL);
   fkernel( vfkernel_data, data_all, n+n_predict, n+n_predict, d, unit_perm, n, unit_perm+n, n_predict, &K12_mat, NULL);
   // K22 is not always needed, setup later

   if(precond_setup != NULL)
   {
      // setup preconditioner is provided
      precond_setup( data, n, ldim, d, fkernel, fkernel_data, 1, precond_data);
   }
   else
   {
      // no preconditioner
      precond_data = NULL;
   }

   if(std_predictp == NULL)
   {
      // only label is needed
      NFFT4GP_DOUBLE *label_predict = NULL;
      NFFT4GP_DOUBLE *iKY = NULL;

      if(*label_predictp == NULL)
      {
         NFFT4GP_MALLOC(label_predict, n_predict, NFFT4GP_DOUBLE);
      }
      else
      {
         label_predict = *label_predictp;
      }

      NFFT4GP_CALLOC(iKY, n, NFFT4GP_DOUBLE);

      NFFT4GP_DOUBLE rel_res;
      NFFT4GP_DOUBLE *rel_res_v = NULL;
      int niter;
      Nfft4GPSolverFgmres( K11_mat,                    // matrix
                        n,                            // matrix size
                        matvec,                       // matrix vector product
                        precond_data,                 // precond
                        precond_solve,                // precond function
                        iKY,                          // initial guess
                        label,                        // rhs
                        n,                            // restart dimension
                        n,                            // max iterations
                        atol,                         // absolute residual?
                        tol,                          // tolerance
                        &rel_res,                     // relative residual
                        &rel_res_v,                   // relative residual vector
                        &niter,                       // number of iterations
                        print_level);                 // print level

      matvec_predict( K12_mat, 'T', n, n_predict, 1.0, iKY, 0.0, label_predict);

      NFFT4GP_FREE(iKY);
      NFFT4GP_FREE(rel_res_v);

      if(*label_predictp == NULL)
      {
         *label_predictp = label_predict;
      }
      
   }
   else
   {
      // also need the Schur complement
      fkernel( vfkernel_data, data_predict, n_predict, ldim_predict, d, NULL, 0, NULL, 0, &K22_mat, NULL);
      
      NFFT4GP_DOUBLE *label_predict = NULL;
      NFFT4GP_DOUBLE *std_predict = NULL;
      NFFT4GP_DOUBLE *iKY = NULL;

      if(*label_predictp == NULL)
      {
         NFFT4GP_MALLOC(label_predict, n_predict, NFFT4GP_DOUBLE);
      }
      else
      {
         label_predict = *label_predictp;
      }

      if(*std_predictp == NULL)
      {
         NFFT4GP_MALLOC(std_predict, n_predict, NFFT4GP_DOUBLE);
      }
      else
      {
         std_predict = *std_predictp;
      }

      NFFT4GP_CALLOC(iKY, n, NFFT4GP_DOUBLE);

      NFFT4GP_DOUBLE rel_res;
      NFFT4GP_DOUBLE *rel_res_v = NULL;
      int niter;
      Nfft4GPSolverFgmres( K11_mat,                       // matrix
                        n,                            // matrix size
                        matvec,                       // matrix vector product
                        precond_data,                 // precond
                        precond_solve,                // precond function
                        iKY,                          // initial guess
                        label,                        // rhs
                        n,                            // restart dimension
                        n,                            // max iterations
                        atol,                         // absolute residual?
                        tol,                          // tolerance
                        &rel_res,                     // relative residual
                        &rel_res_v,                   // relative residual vector
                        &niter,                       // number of iterations
                        print_level);                 // print level

      matvec_predict( K12_mat, 'T', n, n_predict, 1.0, iKY, 0.0, label_predict);

      NFFT4GP_DOUBLE *K12iKY = NULL;
      NFFT4GP_MALLOC(K12iKY, n_predict, NFFT4GP_DOUBLE);

      for(i = 0 ; i < n_predict ; i++)
      {
         NFFT4GP_FREE(rel_res_v);
         Nfft4GPVecFill( iKY, n, 0.0);

         // TODO: update, we need to take two kernel matrices as inputs
         Nfft4GPSolverFgmres( K11_mat,                       // matrix
                           n,                            // matrix size
                           matvec,                       // matrix vector product
                           precond_data,                 // precond
                           precond_solve,                // precond function
                           iKY,                          // initial guess
                           K12_mat + i * n,              // rhs
                           n,                            // restart dimension
                           n,                            // max iterations
                           atol,                         // absolute residual?
                           tol,                          // tolerance
                           &rel_res,                     // relative residual
                           &rel_res_v,                   // relative residual vector
                           &niter,                       // number of iterations
                           print_level);                 // print level
         
         std_predict[i] = sqrt(fabs(K22_mat[i*n_predict+i] - Nfft4GPVecDdot( K12_mat + i*n, n, iKY)));

      }

      NFFT4GP_FREE(K12iKY);
      NFFT4GP_FREE(iKY);
      NFFT4GP_FREE(rel_res_v);

      if(*label_predictp == NULL)
      {
         *label_predictp = label_predict;
      }

      if(*std_predictp == NULL)
      {
         *std_predictp = std_predict;
      }
      
   }

   NFFT4GP_FREE(data_all);
   NFFT4GP_FREE(K11_mat);
   NFFT4GP_FREE(K12_mat);
   NFFT4GP_FREE(K22_mat);
   NFFT4GP_FREE(unit_perm);
   
   return 0;
}
