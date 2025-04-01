#include "gp_loss.h"
#include "../solvers/fgmres.h"
#include "../solvers/lanczos.h"
#include "../preconds/nys.h"

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
                                 )
{

   func_kernel gausskernel = &Nfft4GPKernelGaussianKernel;
   pnfft4gp_kernel gausskernel_data = (pnfft4gp_kernel)Nfft4GPKernelParamCreate(n,1);

   pprecond_nys nys_mat = (pprecond_nys) Nfft4GPPrecondNysCreate();
   Nfft4GPPrecondNysSetPerm(nys_mat, permn, 0);
   Nfft4GPPrecondNysSetRank(nys_mat, k);
   
   int err = Nfft4GPGpLoss(x, 
                        data,
                        label,
                        n,
                        ldim,
                        d,
                        gausskernel,
                        gausskernel_data,
                        &Nfft4GPKernelFree,
                        &Nfft4GPDenseMatSymv,
                        &Nfft4GPDenseGradMatSymv,
                        gausskernel,
                        gausskernel_data,
                        &Nfft4GPKernelFree,
                        &Nfft4GPPrecondNysSetupWithKernel,
                        &Nfft4GPPrecondNysSolve,
                        &Nfft4GPPrecondNysTrace,
                        &Nfft4GPPrecondNysLogdet,
                        &Nfft4GPPrecondNysDvp,
                        &Nfft4GPPrecondNysReset,
                        nys_mat,
                        atol,
                        tol,
                        wsize,
                        maxits,
                        nvecs,
                        radamacher,
                        NFFT4GP_TRANSFORM_SOFTPLUS,
                        mask,
                        0,
                        NULL,
                        loss,
                        grad
                        );
   
   Nfft4GPPrecondNysFree(nys_mat);

   Nfft4GPKernelParamFree(gausskernel_data);
   return err;
}

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
               func_free precond_vfkernel_data_free,
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
               )
{
   int i;

   /* Get the transformed parameters */
   NFFT4GP_DOUBLE tvals[3], dtvals[3];

   Nfft4GPTransform( transform, x[0], 0, tvals, dtvals);
   Nfft4GPTransform( transform, x[1], 0, tvals+1, dtvals+1);
   Nfft4GPTransform( transform, x[2], 0, tvals+2, dtvals+2);

   printf("Transform %e %e %e into %e %e %e with grad %e %e %e\n", x[0], x[1], x[2], tvals[0], tvals[1], tvals[2], dtvals[0], dtvals[1], dtvals[2]);

   /* Create kernel matrix */
   pnfft4gp_kernel fkernel_data = (pnfft4gp_kernel)vfkernel_data;
   pnfft4gp_kernel precond_fkernel_data = (pnfft4gp_kernel)precond_vfkernel_data;
   fkernel_data->_params[0] = tvals[0]; // f
   precond_fkernel_data->_params[0] = tvals[0]; // f
   fkernel_data->_params[1] = tvals[1]; // l
   precond_fkernel_data->_params[1] = tvals[1]; // l
   fkernel_data->_noise_level = tvals[2]; // mu
   precond_fkernel_data->_noise_level = tvals[2]; // mu

   NFFT4GP_DOUBLE *kernel_mat = NULL;
   NFFT4GP_DOUBLE *dkernel_mat = NULL;

   if(dwork)
   {
      kernel_mat = dwork;
      dkernel_mat = dwork + (size_t)n*n;
   }// so far used 4*n*n

   fkernel( vfkernel_data, data, n, ldim, d, NULL, 0, NULL, 0, &kernel_mat, &dkernel_mat);
   
   if(precond_setup != NULL)
   {
      // setup preconditioner is provided
      precond_setup( data, n, ldim, d, precond_fkernel, precond_vfkernel_data, 1, precond_data);
   }
   else
   {
      // no preconditioner
      precond_data = NULL;
   }
   
   /* Compute the first part of the loss */
   // first solve the linear system

   NFFT4GP_DOUBLE *iKY = NULL;
   NFFT4GP_DOUBLE *dKiKY = NULL;
   if(dwork)
   {
      iKY = dwork + (size_t)4*n*n;
      dKiKY = dwork + (size_t)4*n*n + (size_t)n;
   }// so far used 4*n*n + 4*n
   else
   {
      NFFT4GP_MALLOC(iKY, n, NFFT4GP_DOUBLE);
      NFFT4GP_MALLOC(dKiKY, 3*n, NFFT4GP_DOUBLE);
   }
   Nfft4GPVecFill( iKY, n, 0.0); // always use zero initial guess
   NFFT4GP_DOUBLE rel_res;
   NFFT4GP_DOUBLE *rel_res_v = NULL;
   int niter;
   
   int solve_kdim;
   int solve_maxits;
   NFFT4GP_MIN(n, maxits*2, solve_kdim);
   NFFT4GP_MIN(n, maxits*2, solve_maxits);

   Nfft4GPSolverFgmres( kernel_mat,                    // matrix
                     n,                            // matrix size
                     matvec,                       // matrix vector product
                     precond_data,                 // precond
                     precond_solve,                // precond function
                     iKY,                          // initial guess
                     label,                        // rhs
                     solve_kdim,                   // restart dimension
                     solve_maxits,                 // max iterations
                     atol,                         // absolute residual?
                     tol,                          // tolerance
                     &rel_res,                     // relative residual
                     &rel_res_v,                   // relative residual vector
                     &niter,                       // number of iterations
                     print_level);                 // print level

   // check the residual
   if(rel_res > 1e10)
   {
      printf("Warning: FGMRES unstable, rel_res = %f\n", rel_res);
      printf("Current parameters (after transform): f = %f, l = %f, mu = %f\n", tvals[0], tvals[1], tvals[2]);
   }

   NFFT4GP_FREE(rel_res_v);

   NFFT4GP_DOUBLE L1 = Nfft4GPVecDdot( label, n, iKY) / (NFFT4GP_DOUBLE)n; // L1 = Y'*iKY;
   NFFT4GP_DOUBLE L1_grad[3];
   dmatvec( dkernel_mat, n, 1.0, iKY, 0.0, dKiKY); // iKY'*dK*iKY*dtval;
   for(i = 0 ; i < 3 ; i++)
   {
      //matvec( dkernel_mat+(size_t)i*n*n, n, dtvals[i], iKY, 0.0, dKiKY);
      L1_grad[i] = Nfft4GPVecDdot( dKiKY+i*n, n, iKY) / (NFFT4GP_DOUBLE)n;
      L1_grad[i] *= dtvals[i];
   }
   
   // compute the second part of the loss using Lanczos quadrature
   
   // trace estimation
   NFFT4GP_DOUBLE L2;
   NFFT4GP_DOUBLE *L2_grad = NULL;
   NFFT4GP_MIN(n, maxits, maxits);
   int err = Nfft4GPLanczosQuadratureLogdet( kernel_mat,
                                          dkernel_mat,
                                          n,
                                          matvec,
                                          dmatvec,
                                          precond_data,
                                          precond_solve,
                                          precond_trace,
                                          precond_logdet,
                                          precond_dvp,
                                          maxits,
                                          nvecs,
                                          radamacher,
                                          print_level,
                                          &L2,
                                          &L2_grad);

   if(err != 0)
   {
      printf("Error in Nfft4GPLanczosQuadratureLogdet\n");
      return err;
   }

   // combine to get the loss
   // note that the loss has been divided by n
   loss[0] = 0.5*(L1 + L2 + log(2.0*3.1415926535897932384626));
   //loss[0] = 0.5*(L1 + L2);

   //printf("L1 = %f, L2 = %f, 0.5*(L1+L2) = %f\n", L1, L2, loss[0]);
   //printf("L1_grad = [%f, %f, %f]\n", L1_grad[0], L1_grad[1], L1_grad[2]);
   //printf("L2_grad = [%f, %f, %f]\n", L2_grad[0]*dtvals[0], L2_grad[1]*dtvals[1], L2_grad[2]*dtvals[2]);

   if(mask)
   {
      grad[0] = mask[0] ? 0.5*( - L1_grad[0] + L2_grad[0]*dtvals[0]) : 0.0;
      grad[1] = mask[1] ? 0.5*( - L1_grad[1] + L2_grad[1]*dtvals[1]) : 0.0;
      grad[2] = mask[2] ? 0.5*( - L1_grad[2] + L2_grad[2]*dtvals[2]) : 0.0;
   }
   else
   {
      grad[0] = 0.5*( - L1_grad[0] + L2_grad[0]*dtvals[0]);
      grad[1] = 0.5*( - L1_grad[1] + L2_grad[1]*dtvals[1]);
      grad[2] = 0.5*( - L1_grad[2] + L2_grad[2]*dtvals[2]);
   }

   if(!dwork)
   {
      NFFT4GP_FREE(iKY);
      NFFT4GP_FREE(dKiKY);
   }
   
   NFFT4GP_FREE(L2_grad);
   if(!dwork)
   {
      if(kernel_data_free)
      {
         kernel_data_free(kernel_mat);
         kernel_data_free(dkernel_mat);
      }
   }
   
   if(precond_data)
   {
      precond_reset(precond_data);
   }

   return 0;
}