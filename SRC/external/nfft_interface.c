#include "nfft_interface.h"

void *Nfft4GPNFFTKernelParamCreate(int max_n, int dim)
{
   void *kernel = Nfft4GPKernelParamCreate(max_n, 0);
   pnfft4gp_kernel kernel_data = (pnfft4gp_kernel)kernel;

   pstr_adj adj = NULL;
   NFFT4GP_MALLOC(adj, 1, str_adj);
   
   adj->_kernel = 0;
   adj->_sigma = NULL;
   NFFT4GP_CALLOC(adj->_sigma, 2, NFFT4GP_DOUBLE);
   adj->_sigma[0] = 0.0;
   adj->_sigma[1] = 0.0;
   adj->_mu = 0.0;
   adj->_d = dim;
   adj->_N = 32; // 16 32 64
   adj->_p = 1; // 1 1 8
   adj->_m = 4; // 2 4 7
   adj->_eps = 0.0;
   adj->_fastsum_derivative = NULL;
   adj->_fastsum_original = NULL;

   adj->_NN = 2;
   while (2 * adj->_N > adj->_NN)
      adj->_NN *= 2;

   printf("Initializing NFFT kernel with N = %d, p = %d, m = %d, eps = %4.2f\n", adj->_N, adj->_p, adj->_m, adj->_eps);
   printf("Problem size: %d, problem dimension: %d\n", max_n, dim);

   NFFT4GP_MALLOC( adj->_x, max_n * dim, NFFT4GP_DOUBLE);
   adj->_scale = -1.0;

   adj->_kernel_scale = 1.0;

   adj->_n = 0;

   kernel_data->_external = (void*)adj;

   return kernel;
}

void Nfft4GPNFFTKernelParamFree(void *kernel)
{
   Nfft4GPNFFTKernelParamFreeNFFTKernel(kernel);

   pnfft4gp_kernel kernel_data = (pnfft4gp_kernel)kernel;
   pstr_adj adj = (pstr_adj)kernel_data->_external;

   NFFT4GP_FREE(adj->_sigma);
   NFFT4GP_FREE(adj->_x);

   if(adj->_n)
   {
      fastsum_finalize_source_nodes(adj->_fastsum_derivative);
      fastsum_finalize_target_nodes(adj->_fastsum_derivative);
      fastsum_finalize_source_nodes(adj->_fastsum_original);
      fastsum_finalize_target_nodes(adj->_fastsum_original);

      adj->_fastsum_original->x = NULL;
      adj->_fastsum_original->y = NULL;
      adj->_fastsum_original->alpha = NULL;
      adj->_fastsum_original->f = NULL;
      adj->_fastsum_derivative->x = NULL;
      adj->_fastsum_derivative->y = NULL;
      adj->_fastsum_derivative->alpha = NULL;
      adj->_fastsum_derivative->f = NULL;
      adj->_n = 0;
   }

   NFFT4GP_FREE(adj);

   Nfft4GPKernelParamFree(kernel);

   return;
}

void Nfft4GPNFFTKernelFree(void *str)
{
   return;
}

int Nfft4GPNFFTKernelParamFreeNFFTKernel(void *kernel)
{
   Nfft4GPNFFTKernelParamRemovePoints(kernel);

   pnfft4gp_kernel kernel_data = (pnfft4gp_kernel)kernel;
   pstr_adj adj = (pstr_adj)kernel_data->_external;

   if (adj->_fastsum_original)
   {
      fastsum_finalize_kernel(adj->_fastsum_original);
      nfft_free(adj->_fastsum_original);
      adj->_fastsum_original = NULL;
      fastsum_finalize_kernel(adj->_fastsum_derivative);
      nfft_free(adj->_fastsum_derivative);
      adj->_fastsum_derivative = NULL;
   }
   return 0;
}

int Nfft4GPNFFTKernelParamRemovePoints(void *kernel)
{
   pnfft4gp_kernel kernel_data = (pnfft4gp_kernel)kernel;
   pstr_adj adj = (pstr_adj)kernel_data->_external;

   if(adj->_n)
   {
      fastsum_finalize_source_nodes(adj->_fastsum_derivative);
      fastsum_finalize_target_nodes(adj->_fastsum_derivative);
      fastsum_finalize_source_nodes(adj->_fastsum_original);
      fastsum_finalize_target_nodes(adj->_fastsum_original);

      adj->_fastsum_original->x = NULL;
      adj->_fastsum_original->y = NULL;
      adj->_fastsum_original->alpha = NULL;
      adj->_fastsum_original->f = NULL;
      adj->_fastsum_derivative->x = NULL;
      adj->_fastsum_derivative->y = NULL;
      adj->_fastsum_derivative->alpha = NULL;
      adj->_fastsum_derivative->f = NULL;
      adj->_n = 0;
   }

   return 0;
}

int Nfft4GPNFFTKernelGaussianKernel(void *str, NFFT4GP_DOUBLE *data, int n, int ldim, int d, int *permr, int kr, int *permc, int kc, NFFT4GP_DOUBLE **Kp, NFFT4GP_DOUBLE **dKp)
{
   int i, j;
   pnfft4gp_kernel kernel_data = (pnfft4gp_kernel)str;
   pstr_adj adj = (pstr_adj)kernel_data->_external;
   adj->_kernel=0;

   if(Kp == NULL || dKp == NULL)
   {
      printf("Error: NFFT kernel requires Kp and dKp to be not NULL.\n");
      return -1;
   }
   
   if(n != ldim)
   {
      printf("Error: NFFT kernel requires n == ldim.\n");
      return -1;
   }

   // why we need to remove points every time?
   // Nfft4GPNFFTKernelParamRemovePoints(str);
   if(adj->_scale < 0.0)
   {
      // compute scaling factor
      NFFT4GP_MEMCPY(adj->_x, data, n * d, NFFT4GP_DOUBLE);

      // first compute the center of the points if needed
      
      NFFT4GP_DOUBLE radius = 0.0;
      NFFT4GP_DOUBLE *points_center = NULL;
      NFFT4GP_CALLOC(points_center, d, NFFT4GP_DOUBLE);

      // loop to find the center
      for(i = 0; i < d; ++i)
      {
         for( j = 0; j < n; ++j)
         {
            points_center[i] += adj->_x[i * n + j];
         }
         points_center[i] /= (NFFT4GP_DOUBLE) n;
         for( j = 0; j < n; ++j)
         {
            adj->_x[i * n + j] -= points_center[i];
         }
      }
      NFFT4GP_FREE(points_center);

      // loop to compute the radius
      for( j = 0; j < n; ++j)
      {
         NFFT4GP_DOUBLE radius_i = 0.0;
         for(i = 0; i < d; ++i)
         {
            radius_i += adj->_x[i * n + j] * adj->_x[i * n + j];
         }
         radius_i = sqrt(radius_i);
         NFFT4GP_MAX(radius, radius_i, radius);
      }
      
      if(radius > 0.25 || radius < 0.125)
      {
         adj->_scale = 0.25 / radius;
         Nfft4GPVecScale( adj->_x, n * d, adj->_scale);
      }
      else
      {
         adj->_scale = 1.0;
      }

      //printf("Initializing NFFT kernel with scale = %g\n", adj->_scale);

      NFFT4GP_DOUBLE *x_trans = NULL;
      NFFT4GP_MALLOC( x_trans, n*d, NFFT4GP_DOUBLE);

      for(i = 0 ; i < d ; i ++)
      {
         for(j = 0 ; j < n ; j ++)
         {
            x_trans[j*d+i] = adj->_x[i*n+j];
         }
      }

      NFFT4GP_MEMCPY(adj->_x, x_trans, n*d, NFFT4GP_DOUBLE);
      NFFT4GP_FREE(x_trans);
   }
   
   // free the old kernel
   Nfft4GPNFFTKernelParamFreeNFFTKernel(str);

   // update the kernel parameters
   adj->_sigma[0] = kernel_data->_params[1] * adj->_scale * sqrt(2.0); // length scale
   adj->_sigma[1] = adj->_sigma[0]; // signal variance
   adj->_mu = kernel_data->_noise_level; // noise level
   
   adj->_fastsum_original = (fastsum_plan*) nfft_malloc(sizeof(fastsum_plan));
   adj->_fastsum_derivative = (fastsum_plan*) nfft_malloc(sizeof(fastsum_plan));

   // reset kernel with new parameters
   fastsum_init_guru_kernel(adj->_fastsum_original, adj->_d, gaussian, adj->_sigma,
                              STORE_PERMUTATION_X_ALPHA, adj->_N, adj->_p, 0.0, adj->_eps);
                              
   fastsum_init_guru_kernel(adj->_fastsum_derivative, adj->_d, xx_gaussian, adj->_sigma+1,
                              STORE_PERMUTATION_X_ALPHA, adj->_N, adj->_p, 0.0, adj->_eps);
                              
   adj->_fastsum_original->x = NULL;
   adj->_fastsum_original->y = NULL;
   adj->_fastsum_original->alpha = NULL;
   adj->_fastsum_original->f = NULL;
   adj->_fastsum_derivative->x = NULL;
   adj->_fastsum_derivative->y = NULL;
   adj->_fastsum_derivative->alpha = NULL;
   adj->_fastsum_derivative->f = NULL;

   adj->_n = n;
   fastsum_init_guru_source_nodes(adj->_fastsum_original, n, adj->_NN, adj->_m);
   fastsum_init_guru_target_nodes(adj->_fastsum_original, n, adj->_NN, adj->_m);
   fastsum_init_guru_source_nodes(adj->_fastsum_derivative, n, adj->_NN, adj->_m);
   fastsum_init_guru_target_nodes(adj->_fastsum_derivative, n, adj->_NN, adj->_m);

   NFFT4GP_MEMCPY(adj->_fastsum_original->x, adj->_x, n * d, NFFT4GP_DOUBLE);
   NFFT4GP_MEMCPY(adj->_fastsum_original->y, adj->_x, n * d, NFFT4GP_DOUBLE);
   NFFT4GP_MEMCPY(adj->_fastsum_derivative->x, adj->_x, n * d, NFFT4GP_DOUBLE);
   NFFT4GP_MEMCPY(adj->_fastsum_derivative->y, adj->_x, n * d, NFFT4GP_DOUBLE);

   fastsum_precompute(adj->_fastsum_original);
   fastsum_precompute(adj->_fastsum_derivative);

   adj->_kernel_scale = kernel_data->_params[0];

   *Kp = (NFFT4GP_DOUBLE *)adj;
   *dKp = (NFFT4GP_DOUBLE *)adj;

   return 0;

}

int Nfft4GPNFFTKernelMatern12Kernel(void *str, NFFT4GP_DOUBLE *data, int n, int ldim, int d, int *permr, int kr, int *permc, int kc, NFFT4GP_DOUBLE **Kp, NFFT4GP_DOUBLE **dKp)
{
   int i, j;
   pnfft4gp_kernel kernel_data = (pnfft4gp_kernel)str;
   pstr_adj adj = (pstr_adj)kernel_data->_external;
   adj->_kernel=1;

   if(Kp == NULL || dKp == NULL)
   {
      printf("Error: NFFT kernel requires Kp and dKp to be not NULL.\n");
      return -1;
   }
   
   if(n != ldim)
   {
      printf("Error: NFFT kernel requires n == ldim.\n");
      return -1;
   }

   // why we need to remove points every time?
   // Nfft4GPNFFTKernelParamRemovePoints(str);
   if(adj->_scale < 0.0)
   {
      // compute scaling factor
      NFFT4GP_MEMCPY(adj->_x, data, n * d, NFFT4GP_DOUBLE);

      // first compute the center of the points if needed
      
      NFFT4GP_DOUBLE radius = 0.0;
      NFFT4GP_DOUBLE *points_center = NULL;
      NFFT4GP_CALLOC(points_center, d, NFFT4GP_DOUBLE);

      // loop to find the center
      for(i = 0; i < d; ++i)
      {
         for( j = 0; j < n; ++j)
         {
            points_center[i] += adj->_x[i * n + j];
         }
         points_center[i] /= (NFFT4GP_DOUBLE) n;
         for( j = 0; j < n; ++j)
         {
            adj->_x[i * n + j] -= points_center[i];
         }
      }
      NFFT4GP_FREE(points_center);

      // loop to compute the radius
      for( j = 0; j < n; ++j)
      {
         NFFT4GP_DOUBLE radius_i = 0.0;
         for(i = 0; i < d; ++i)
         {
            radius_i += adj->_x[i * n + j] * adj->_x[i * n + j];
         }
         radius_i = sqrt(radius_i);
         NFFT4GP_MAX(radius, radius_i, radius);
      }
      
      if(radius > 0.25 || radius < 0.125)
      {
         adj->_scale = 0.25 / radius;
         Nfft4GPVecScale( adj->_x, n * d, adj->_scale);
      }
      else
      {
         adj->_scale = 1.0;
      }

      //printf("Initializing NFFT kernel with scale = %g\n", adj->_scale);

      NFFT4GP_DOUBLE *x_trans = NULL;
      NFFT4GP_MALLOC( x_trans, n*d, NFFT4GP_DOUBLE);

      for(i = 0 ; i < d ; i ++)
      {
         for(j = 0 ; j < n ; j ++)
         {
            x_trans[j*d+i] = adj->_x[i*n+j];
         }
      }

      NFFT4GP_MEMCPY(adj->_x, x_trans, n*d, NFFT4GP_DOUBLE);
      NFFT4GP_FREE(x_trans);
   }
   
   // free the old kernel
   Nfft4GPNFFTKernelParamFreeNFFTKernel(str);

   // update the kernel parameters
   adj->_sigma[0] = kernel_data->_params[1] * adj->_scale; // length scale
   adj->_sigma[1] = adj->_sigma[0]; // signal variance
   adj->_mu = kernel_data->_noise_level; // noise level
   
   adj->_fastsum_original = (fastsum_plan*) nfft_malloc(sizeof(fastsum_plan));
   adj->_fastsum_derivative = (fastsum_plan*) nfft_malloc(sizeof(fastsum_plan));

   // reset kernel with new parameters
   fastsum_init_guru_kernel(adj->_fastsum_original, adj->_d, laplacian_rbf, adj->_sigma,
                              STORE_PERMUTATION_X_ALPHA, adj->_N, adj->_p, 0.0, adj->_eps);
                              
   fastsum_init_guru_kernel(adj->_fastsum_derivative, adj->_d, der_laplacian_rbf, adj->_sigma+1,
                              STORE_PERMUTATION_X_ALPHA, adj->_N, adj->_p, 0.0, adj->_eps);
                              
   adj->_fastsum_original->x = NULL;
   adj->_fastsum_original->y = NULL;
   adj->_fastsum_original->alpha = NULL;
   adj->_fastsum_original->f = NULL;
   adj->_fastsum_derivative->x = NULL;
   adj->_fastsum_derivative->y = NULL;
   adj->_fastsum_derivative->alpha = NULL;
   adj->_fastsum_derivative->f = NULL;

   adj->_n = n;
   fastsum_init_guru_source_nodes(adj->_fastsum_original, n, adj->_NN, adj->_m);
   fastsum_init_guru_target_nodes(adj->_fastsum_original, n, adj->_NN, adj->_m);
   fastsum_init_guru_source_nodes(adj->_fastsum_derivative, n, adj->_NN, adj->_m);
   fastsum_init_guru_target_nodes(adj->_fastsum_derivative, n, adj->_NN, adj->_m);

   NFFT4GP_MEMCPY(adj->_fastsum_original->x, adj->_x, n * d, NFFT4GP_DOUBLE);
   NFFT4GP_MEMCPY(adj->_fastsum_original->y, adj->_x, n * d, NFFT4GP_DOUBLE);
   NFFT4GP_MEMCPY(adj->_fastsum_derivative->x, adj->_x, n * d, NFFT4GP_DOUBLE);
   NFFT4GP_MEMCPY(adj->_fastsum_derivative->y, adj->_x, n * d, NFFT4GP_DOUBLE);

   fastsum_precompute(adj->_fastsum_original);
   fastsum_precompute(adj->_fastsum_derivative);

   adj->_kernel_scale = kernel_data->_params[0];

   *Kp = (NFFT4GP_DOUBLE *)adj;
   *dKp = (NFFT4GP_DOUBLE *)adj;

   return 0;
}

int Nfft4GPNFFTMatSymv(void *data, int n, NFFT4GP_DOUBLE alpha, NFFT4GP_DOUBLE *x, NFFT4GP_DOUBLE beta, NFFT4GP_DOUBLE *y)
{
   int i;
   pstr_adj adj = (pstr_adj)data;
   NFFT4GP_DOUBLE ff = adj->_kernel_scale * adj->_kernel_scale;

#ifdef NFFT4GP_USING_OPENMP
   if(!omp_in_parallel())
   {
      #pragma omp parallel for private(i) NFFT4GP_DEFAULT_OPENMP_SCHEDULE
      for(i = 0; i < n; ++i)
      {
         adj->_fastsum_original->alpha[i] = alpha*x[i];
      }
   }
   else
   {
#endif
      for(i = 0; i < n; ++i)
      {
         adj->_fastsum_original->alpha[i] = alpha*x[i];
      }
#ifdef NFFT4GP_USING_OPENMP
   }
#endif

   fastsum_trafo(adj->_fastsum_original);

   if(beta == 0.0)
   {
#ifdef NFFT4GP_USING_OPENMP
      if(!omp_in_parallel())
      {
         #pragma omp parallel for private(i) NFFT4GP_DEFAULT_OPENMP_SCHEDULE
         for(i = 0; i < n; ++i)
         {
            y[i] = ff * (CREAL(adj->_fastsum_original->f[i]) + (adj->_mu) * CREAL(adj->_fastsum_original->alpha[i]));
         }
      }
      else
      {
#endif
         for (i = 0; i < n; ++i)
         {
            y[i] = ff * (CREAL(adj->_fastsum_original->f[i]) + (adj->_mu) * CREAL(adj->_fastsum_original->alpha[i]));
         }
#ifdef NFFT4GP_USING_OPENMP
      }
#endif
   }
   else if(beta == 1.0)
   {
#ifdef NFFT4GP_USING_OPENMP
      if(!omp_in_parallel())
      {
         #pragma omp parallel for private(i) NFFT4GP_DEFAULT_OPENMP_SCHEDULE
         for(i = 0; i < n; ++i)
         {
            y[i] += ff * (CREAL(adj->_fastsum_original->f[i]) + (adj->_mu) * CREAL(adj->_fastsum_original->alpha[i]));
         }
      }
      else
      {
#endif
         for (i = 0; i < n; ++i)
         {
            y[i] += ff * (CREAL(adj->_fastsum_original->f[i]) + (adj->_mu) * CREAL(adj->_fastsum_original->alpha[i]));
         }
#ifdef NFFT4GP_USING_OPENMP
      }
#endif
   }
   else
   {
      Nfft4GPVecScale(y, n, beta);
#ifdef NFFT4GP_USING_OPENMP
      if(!omp_in_parallel())
      {
         #pragma omp parallel for private(i) NFFT4GP_DEFAULT_OPENMP_SCHEDULE
         for(i = 0; i < n; ++i)
         {
            y[i] += ff * (CREAL(adj->_fastsum_original->f[i]) + (adj->_mu) * CREAL(adj->_fastsum_original->alpha[i]));
         }
      }
      else
      {
#endif
         for (i = 0; i < n; ++i)
         {
            y[i] += ff * (CREAL(adj->_fastsum_original->f[i]) + (adj->_mu) * CREAL(adj->_fastsum_original->alpha[i]));
         }
#ifdef NFFT4GP_USING_OPENMP
      }
#endif
   }

   return 0;
}

int Nfft4GPNFFTGradMatSymv(void *data, int n, NFFT4GP_DOUBLE alpha, NFFT4GP_DOUBLE *x, NFFT4GP_DOUBLE beta, NFFT4GP_DOUBLE *y)
{
   int i;
   pstr_adj adj = (pstr_adj)data;
   NFFT4GP_DOUBLE *y0, *y1, *y2;
   NFFT4GP_DOUBLE ff = adj->_kernel_scale * adj->_kernel_scale;
   NFFT4GP_DOUBLE f2 = adj->_kernel_scale * 2.0;

   y0 = y;
   y1 = y + n;
   y2 = y + 2*n;

#ifdef NFFT4GP_USING_OPENMP
   if(!omp_in_parallel())
   {
      #pragma omp parallel for private(i) NFFT4GP_DEFAULT_OPENMP_SCHEDULE
      for(i = 0; i < n; ++i)
      {
         adj->_fastsum_original->alpha[i] = alpha*x[i];
         adj->_fastsum_derivative->alpha[i] = alpha*x[i];
      }
   }
   else
   {
#endif
      for(i = 0; i < n; ++i)
      {
         adj->_fastsum_original->alpha[i] = alpha*x[i];
         adj->_fastsum_derivative->alpha[i] = alpha*x[i];
      }
#ifdef NFFT4GP_USING_OPENMP
   }
#endif

   fastsum_trafo(adj->_fastsum_original);
   fastsum_trafo(adj->_fastsum_derivative);

   NFFT4GP_DOUBLE scale = adj->_kernel == 0 ? 2.0*adj->_scale*sqrt(2.0)/adj->_fastsum_derivative->kernel_param[0] : adj->_scale/adj->_fastsum_derivative->kernel_param[0];
   scale *= ff;

   if(beta == 0.0)
   {
#ifdef NFFT4GP_USING_OPENMP
      if(!omp_in_parallel())
      {
         #pragma omp parallel for private(i) NFFT4GP_DEFAULT_OPENMP_SCHEDULE
         for(i = 0; i < n; ++i)
         {
            y0[i] = f2 * (CREAL(adj->_fastsum_original->f[i]) + (adj->_mu) * CREAL(adj->_fastsum_original->alpha[i]));
            y1[i] = scale * CREAL(adj->_fastsum_derivative->f[i]);
            y2[i] = ff * CREAL(adj->_fastsum_original->alpha[i]);
         }
      }
      else
      {
#endif
         for (i = 0; i < n; ++i)
         {
            y0[i] = f2*(CREAL(adj->_fastsum_original->f[i]) + (adj->_mu) * CREAL(adj->_fastsum_original->alpha[i]));
            y1[i] = scale * CREAL(adj->_fastsum_derivative->f[i]);
            y2[i] = ff * CREAL(adj->_fastsum_original->alpha[i]);
         }
#ifdef NFFT4GP_USING_OPENMP
      }
#endif
   }
   else if(beta == 1.0)
   {
#ifdef NFFT4GP_USING_OPENMP
      if(!omp_in_parallel())
      {
         #pragma omp parallel for private(i) NFFT4GP_DEFAULT_OPENMP_SCHEDULE
         for(i = 0; i < n; ++i)
         {
            y0[i] += f2 * (CREAL(adj->_fastsum_original->f[i]) + (adj->_mu) * CREAL(adj->_fastsum_original->alpha[i]));
            y1[i] += scale * CREAL(adj->_fastsum_derivative->f[i]);
            y2[i] += ff * CREAL(adj->_fastsum_original->alpha[i]);
         }
      }
      else
      {
#endif
         for (i = 0; i < n; ++i)
         {
            y0[i] += f2 * (CREAL(adj->_fastsum_original->f[i]) + (adj->_mu) * CREAL(adj->_fastsum_original->alpha[i]));
            y1[i] += scale * CREAL(adj->_fastsum_derivative->f[i]);
            y2[i] += ff * CREAL(adj->_fastsum_original->alpha[i]);
         }
#ifdef NFFT4GP_USING_OPENMP
      }
#endif
   }
   else
   {
      Nfft4GPVecScale(y, 3*n, beta);
#ifdef NFFT4GP_USING_OPENMP
      if(!omp_in_parallel())
      {
         #pragma omp parallel for private(i) NFFT4GP_DEFAULT_OPENMP_SCHEDULE
         for(i = 0; i < n; ++i)
         {
            y0[i] += f2 * (CREAL(adj->_fastsum_original->f[i]) + (adj->_mu) * CREAL(adj->_fastsum_original->alpha[i]));
            y1[i] += scale * CREAL(adj->_fastsum_derivative->f[i]);
            y2[i] += ff * CREAL(adj->_fastsum_original->alpha[i]);
         }
      }
      else
      {
#endif
         for (i = 0; i < n; ++i)
         {
            y0[i] += f2 * (CREAL(adj->_fastsum_original->f[i]) + (adj->_mu) * CREAL(adj->_fastsum_original->alpha[i]));
            y1[i] += scale * CREAL(adj->_fastsum_derivative->f[i]);
            y2[i] += ff * CREAL(adj->_fastsum_original->alpha[i]);
         }
#ifdef NFFT4GP_USING_OPENMP
      }
#endif
   }
   
   return 0;
}

void* Nfft4GPNFFTAdditiveKernelParamCreate(NFFT4GP_DOUBLE *data, int n, int ldim, int d, int *windows, int nwindows, int dwindows)
{
   void *kernel = Nfft4GPKernelParamCreate(3*n, 0);
   pnfft4gp_kernel kernel_data = (pnfft4gp_kernel)kernel;

   kernel_data->_iparams[0] = nwindows;
   kernel_data->_iparams[1] = dwindows;
   
   int skip_window = 1;
   kernel_data->_iparams[2] = 0;
   while(skip_window < dwindows && windows[nwindows*dwindows - skip_window] < 0)
   {
      skip_window ++;
      kernel_data->_iparams[2]++;
   }

   NFFT4GP_MALLOC( kernel_data->_ibufferp, 1, int*);
   kernel_data->_ibufferp[0] = windows;

   nfft4gp_kernel **adj = NULL;
   NFFT4GP_MALLOC(adj, nwindows, nfft4gp_kernel*);
   
   NFFT4GP_MALLOC( kernel_data->_buffer, (size_t)n*nwindows*dwindows, NFFT4GP_DOUBLE);
   kernel_data->_own_buffer = 1;

   int i, j;
   NFFT4GP_DOUBLE *data_window = kernel_data->_buffer;
   int *feature_window = windows;
   for(i = 0 ; i < nwindows ; i ++)
   {
      int actual_dim = 0;
      for(j = 0 ; j < dwindows ; j ++)
      {
         //printf("Copying data for window %d, feature %d\n", i, feature_window[0]);
         if(feature_window[0] >= 0)
         {
            //printf("Reading feature %d\n", feature_window[0]);
            NFFT4GP_MEMCPY( data_window, data + feature_window[0]*ldim, n, NFFT4GP_DOUBLE);
            feature_window++;
            data_window += n;
            actual_dim++;
         }
         else
         {
            //printf("Skip last window\n");
         }
      }
      adj[i] = (pnfft4gp_kernel)Nfft4GPNFFTKernelParamCreate(n,actual_dim);
   }
   //TestPrintMatrix(kernel_data->_buffer, n, nwindows*dwindows, n);
   kernel_data->_external = (void*)adj;
   return kernel;
}

int Nfft4GPNFFTAdditiveKernelGaussianKernel(void *str, NFFT4GP_DOUBLE *data, int n, int ldim, int d, int *permr, int kr, int *permc, int kc, NFFT4GP_DOUBLE **Kp, NFFT4GP_DOUBLE **dKp)
{
   
   pnfft4gp_kernel kernel_data = (pnfft4gp_kernel)str;
   nfft4gp_kernel **adj = (nfft4gp_kernel**)kernel_data->_external;
   int nwindows = kernel_data->_iparams[0];
   int dwindows = kernel_data->_iparams[1];
   int skip_last = kernel_data->_iparams[2];

   void *nfft_kernel;
   void *nfft_dkernel;

   int i, j;
   NFFT4GP_DOUBLE *data_window = kernel_data->_buffer;
   if(skip_last)
   {
      for(i = 0 ; i < nwindows - 1 ; i ++)
      {
         nfft_kernel = NULL;
         nfft_dkernel = NULL;

         adj[i]->_params[0] = kernel_data->_params[0]; // f
         adj[i]->_params[1] = kernel_data->_params[1]; // l
         adj[i]->_noise_level = kernel_data->_noise_level; // mu
         //printf("Set parameters for kernel %d as f = %4.2f, l = %4.2f, mu = %4.2f\n", i, adj[i]->_params[0], adj[i]->_params[1], adj[i]->_noise_level);
         //TestPrintMatrix(data_window, n, dwindows, n);
         Nfft4GPNFFTKernelGaussianKernel( (void*)adj[i], data_window, n, n, dwindows, NULL, 0, NULL, 0, (NFFT4GP_DOUBLE**)&nfft_kernel, (NFFT4GP_DOUBLE**)&nfft_dkernel);
         data_window += n*dwindows;
      }
      i = nwindows - 1;
      nfft_kernel = NULL;
      nfft_dkernel = NULL;
      adj[i]->_params[0] = kernel_data->_params[0]; // f
      adj[i]->_params[1] = kernel_data->_params[1]; // l
      adj[i]->_noise_level = kernel_data->_noise_level; // mu
      Nfft4GPNFFTKernelGaussianKernel( (void*)adj[i], data_window, n, n, dwindows-skip_last, NULL, 0, NULL, 0, (NFFT4GP_DOUBLE**)&nfft_kernel, (NFFT4GP_DOUBLE**)&nfft_dkernel);
   }
   else
   {
      for(i = 0 ; i < nwindows ; i ++)
      {
         nfft_kernel = NULL;
         nfft_dkernel = NULL;

         adj[i]->_params[0] = kernel_data->_params[0]; // f
         adj[i]->_params[1] = kernel_data->_params[1]; // l
         adj[i]->_noise_level = kernel_data->_noise_level; // mu
         //printf("Set parameters for kernel %d as f = %4.2f, l = %4.2f, mu = %4.2f\n", i, adj[i]->_params[0], adj[i]->_params[1], adj[i]->_noise_level);
         //TestPrintMatrix(data_window, n, dwindows, n);
         Nfft4GPNFFTKernelGaussianKernel( (void*)adj[i], data_window, n, n, dwindows, NULL, 0, NULL, 0, (NFFT4GP_DOUBLE**)&nfft_kernel, (NFFT4GP_DOUBLE**)&nfft_dkernel);
         data_window += n*dwindows;
      }
   }

   *Kp = (NFFT4GP_DOUBLE *)str;
   *dKp = (NFFT4GP_DOUBLE *)str;

   return 0;
}

int Nfft4GPNFFTAdditiveKernelMatern12Kernel(void *str, NFFT4GP_DOUBLE *data, int n, int ldim, int d, int *permr, int kr, int *permc, int kc, NFFT4GP_DOUBLE **Kp, NFFT4GP_DOUBLE **dKp)
{
   
   pnfft4gp_kernel kernel_data = (pnfft4gp_kernel)str;
   nfft4gp_kernel **adj = (nfft4gp_kernel**)kernel_data->_external;
   int nwindows = kernel_data->_iparams[0];
   int dwindows = kernel_data->_iparams[1];
   int skip_last = kernel_data->_iparams[2];

   void *nfft_kernel;
   void *nfft_dkernel;

   int i, j;
   NFFT4GP_DOUBLE *data_window = kernel_data->_buffer;
   if(skip_last)
   {
      for(i = 0 ; i < nwindows - 1 ; i ++)
      {
         nfft_kernel = NULL;
         nfft_dkernel = NULL;

         adj[i]->_params[0] = kernel_data->_params[0]; // f
         adj[i]->_params[1] = kernel_data->_params[1]; // l
         adj[i]->_noise_level = kernel_data->_noise_level; // mu
         //printf("Set parameters for kernel %d as f = %4.2f, l = %4.2f, mu = %4.2f\n", i, adj[i]->_params[0], adj[i]->_params[1], adj[i]->_noise_level);
         //TestPrintMatrix(data_window, n, dwindows, n);
         Nfft4GPNFFTKernelMatern12Kernel( (void*)adj[i], data_window, n, n, dwindows, NULL, 0, NULL, 0, (NFFT4GP_DOUBLE**)&nfft_kernel, (NFFT4GP_DOUBLE**)&nfft_dkernel);
         data_window += n*dwindows;
      }
      i = nwindows - 1;
      nfft_kernel = NULL;
      nfft_dkernel = NULL;
      adj[i]->_params[0] = kernel_data->_params[0]; // f
      adj[i]->_params[1] = kernel_data->_params[1]; // l
      adj[i]->_noise_level = kernel_data->_noise_level; // mu
      Nfft4GPNFFTKernelMatern12Kernel( (void*)adj[i], data_window, n, n, dwindows-skip_last, NULL, 0, NULL, 0, (NFFT4GP_DOUBLE**)&nfft_kernel, (NFFT4GP_DOUBLE**)&nfft_dkernel);
   }
   else
   {
      for(i = 0 ; i < nwindows ; i ++)
      {
         nfft_kernel = NULL;
         nfft_dkernel = NULL;

         adj[i]->_params[0] = kernel_data->_params[0]; // f
         adj[i]->_params[1] = kernel_data->_params[1]; // l
         adj[i]->_noise_level = kernel_data->_noise_level; // mu
         //printf("Set parameters for kernel %d as f = %4.2f, l = %4.2f, mu = %4.2f\n", i, adj[i]->_params[0], adj[i]->_params[1], adj[i]->_noise_level);
         //TestPrintMatrix(data_window, n, dwindows, n);
         Nfft4GPNFFTKernelMatern12Kernel( (void*)adj[i], data_window, n, n, dwindows, NULL, 0, NULL, 0, (NFFT4GP_DOUBLE**)&nfft_kernel, (NFFT4GP_DOUBLE**)&nfft_dkernel);
         data_window += n*dwindows;
      }
   }

   *Kp = (NFFT4GP_DOUBLE *)str;
   *dKp = (NFFT4GP_DOUBLE *)str;

   return 0;
}

int Nfft4GPAdditiveNFFTMatSymv(void *data, int n, NFFT4GP_DOUBLE alpha, NFFT4GP_DOUBLE *x, NFFT4GP_DOUBLE beta, NFFT4GP_DOUBLE *y)
{
   pnfft4gp_kernel kernel_data = (pnfft4gp_kernel)data;
   nfft4gp_kernel **adj = (nfft4gp_kernel**)kernel_data->_external;
   int nwindows = kernel_data->_iparams[0];
   int dwindows = kernel_data->_iparams[1];

   Nfft4GPVecFill( kernel_data->_dwork, n, 0.0);

   int i;
   NFFT4GP_DOUBLE scale = 1.0 / (NFFT4GP_DOUBLE)nwindows * alpha;
   for(i = 0 ; i < nwindows ; i ++)
   {
      pstr_adj adji = (pstr_adj)adj[i]->_external;
      Nfft4GPNFFTMatSymv( adji, n, scale, x, 1.0, kernel_data->_dwork);
   }
   
   Nfft4GPVecScale( y, n, beta);
   Nfft4GPVecAxpy( 1.0, kernel_data->_dwork, n, y);

   return 0;
}

int Nfft4GPAdditiveNFFTGradMatSymv(void *data, int n, NFFT4GP_DOUBLE alpha, NFFT4GP_DOUBLE *x, NFFT4GP_DOUBLE beta, NFFT4GP_DOUBLE *y)
{
   pnfft4gp_kernel kernel_data = (pnfft4gp_kernel)data;
   nfft4gp_kernel **adj = (nfft4gp_kernel**)kernel_data->_external;
   int nwindows = kernel_data->_iparams[0];
   int dwindows = kernel_data->_iparams[1];

   Nfft4GPVecFill( kernel_data->_dwork, 3 * n, 0.0);

   int i;
   NFFT4GP_DOUBLE scale = 1.0 / (NFFT4GP_DOUBLE)nwindows * alpha;
   for(i = 0 ; i < nwindows ; i ++)
   {
      pstr_adj adji = (pstr_adj)adj[i]->_external;
      Nfft4GPNFFTGradMatSymv( adji, n, scale, x, 1.0, kernel_data->_dwork);
   }
   
   Nfft4GPVecScale( y, 3*n, beta);
   Nfft4GPVecAxpy( 1.0, kernel_data->_dwork, 3*n, y);

   return 0;
}

void Nfft4GPAdditiveNFFTKernelFree(void *str)
{
   pnfft4gp_kernel kernel_data = (pnfft4gp_kernel)str;
   nfft4gp_kernel **adj = (nfft4gp_kernel**)kernel_data->_external;
   int nwindows = kernel_data->_iparams[0];

   int i;
   for(i = 0 ; i < nwindows ; i ++)
   {
      Nfft4GPNFFTKernelParamFree( (void*)adj[i]);
   }
   NFFT4GP_FREE(adj);

   Nfft4GPKernelParamFree(str);
}

NFFT4GP_DOUBLE* Nfft4GPNFFTAppendData(NFFT4GP_DOUBLE *X1, int n1, int ldim1, int d, NFFT4GP_DOUBLE *X2, int n2, int ldim2)
{
   int i;
   NFFT4GP_DOUBLE *X = NULL;

   NFFT4GP_MALLOC( X, (size_t)(n1+n2)*d, NFFT4GP_DOUBLE);
   for(i = 0 ; i < d ; i++)
   {
      NFFT4GP_MEMCPY( X + (size_t)i*(n1+n2), X1 + (size_t)i*ldim1, n1, NFFT4GP_DOUBLE);
      NFFT4GP_MEMCPY( X + (size_t)i*(n1+n2)+n1, X2 + (size_t)i*ldim2, n2, NFFT4GP_DOUBLE);
   }
   
   return X;
}

int Nfft4GPAdditiveNFFTGpPredict(NFFT4GP_DOUBLE *x, 
                              NFFT4GP_DOUBLE *data,
                              NFFT4GP_DOUBLE *label,
                              int n,
                              int ldim,
                              int d,
                              NFFT4GP_DOUBLE *data_predict,
                              int n_predict,
                              int ldim_predict,
                              NFFT4GP_DOUBLE *data_all,
                              func_kernel fkernel,
                              void* vfkernel_data,
                              void* vfkernel_data_l,
                              func_free kernel_data_free,
                              func_symmatvec matvec,
                              func_kernel precond_fkernel,
                              void* precond_vfkernel_data,
                              func_free precond_kernel_data_free,
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
   pnfft4gp_kernel fkernel_data_l = (pnfft4gp_kernel)vfkernel_data_l;
   fkernel_data->_params[0] = tvals[0]; // f
   fkernel_data_l->_params[0] = tvals[0]; // f
   fkernel_data->_params[1] = tvals[1]; // l
   fkernel_data_l->_params[1] = tvals[1]; // l
   fkernel_data->_noise_level = tvals[2]; // mu
   fkernel_data_l->_noise_level = tvals[2]; // mu

   NFFT4GP_DOUBLE *K11_mat = NULL;
   NFFT4GP_DOUBLE *dK11_mat = NULL;
   NFFT4GP_DOUBLE *K_mat = NULL;
   NFFT4GP_DOUBLE *dK_mat = NULL;
   
   fkernel( vfkernel_data, data, n, ldim, d, NULL, 0, NULL, 0, &K11_mat, &dK11_mat);
   fkernel( vfkernel_data_l, data_all, n+n_predict, n+n_predict, d, NULL, 0, NULL, 0, &K_mat, &dK_mat);
   // K22 is not always needed, setup later

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

   // only label is needed
   NFFT4GP_DOUBLE *label_predict = NULL;
   NFFT4GP_DOUBLE *label_helper = NULL;
   NFFT4GP_DOUBLE *iKY = NULL;

   if(*label_predictp == NULL)
   {
      NFFT4GP_MALLOC(label_predict, n_predict, NFFT4GP_DOUBLE);
   }
   else
   {
      label_predict = *label_predictp;
   }

   NFFT4GP_CALLOC(iKY, (size_t)n+n_predict, NFFT4GP_DOUBLE);
   NFFT4GP_MALLOC(label_helper, (size_t)n+n_predict, NFFT4GP_DOUBLE);

   //TestPrintMatrix(iKY, n, 1, n);

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
                     maxits,                       // restart dimension
                     maxits,                       // max iterations
                     atol,                         // absolute residual?
                     tol,                          // tolerance
                     &rel_res,                     // relative residual
                     &rel_res_v,                   // relative residual vector
                     &niter,                       // number of iterations
                     print_level);                 // print level

   // Next compute label_predict = K21 * iKY
   // this is K11 K12   iKy = not needed
   //         K21 K22 *  0  = K21 * iKY
   matvec( K_mat, n+n_predict, 1.0, iKY, 0.0, label_helper);

   NFFT4GP_MEMCPY( label_predict, label_helper + n, n_predict, NFFT4GP_DOUBLE);

   NFFT4GP_FREE(iKY);
   NFFT4GP_FREE(label_helper);
   NFFT4GP_FREE(rel_res_v);

   if(*label_predictp == NULL)
   {
      *label_predictp = label_predict;
   }

   if(std_predictp)
   {
      NFFT4GP_DOUBLE *std_predict = NULL;

      // also need the diagonal of the Schur complment
      if(*std_predictp == NULL)
      {
         NFFT4GP_MALLOC(std_predict, n_predict, NFFT4GP_DOUBLE);
      }
      else
      {
         std_predict = *std_predictp;
      }

      NFFT4GP_MALLOC(iKY, (size_t)n+n_predict, NFFT4GP_DOUBLE);
      NFFT4GP_MALLOC(label_helper, (size_t)n+n_predict, NFFT4GP_DOUBLE);

      for( i = 0 ; i < n_predict ; i ++)
      {
         // The diagonal of the Schur complement
         // K22 - K21 * K11^{-1} * K12
         // K11 K12
         // K21 K22
         // As we only need the diagonal, we can do this in the following way:
         // 1. Compute K11^{-1} * K12(:,i)
         // 2. Compute K21(i,:) * K11^{-1} * K12(:,i)
         // 3. Subtract it from K22(i, i)

         // Compute = K12 * ei
         // this is K11 K12   0   = K12 * ei => get corresponding column of K12
         //         K21 K22 * ei  = K22 * ei => get diagonal of K22

         Nfft4GPVecFill( iKY, n+n_predict, 0.0);
         iKY[n+i] = 1.0;
         matvec( K_mat, n+n_predict, 1.0, iKY, 0.0, label_helper);
         NFFT4GP_DOUBLE K22i = label_helper[n+i];

         // TODO: update, we need to take two kernel matrices as inputs
         iKY[n+i] = 0.0;
         Nfft4GPSolverFgmres( K11_mat,                       // matrix
                           n,                            // matrix size
                           matvec,                       // matrix vector product
                           precond_data,                 // precond
                           precond_solve,                // precond function
                           iKY,                          // initial guess
                           label_helper,                 // rhs
                           n,                            // restart dimension
                           n,                            // max iterations
                           atol,                         // absolute residual?
                           tol,                          // tolerance
                           &rel_res,                     // relative residual
                           &rel_res_v,                   // relative residual vector
                           &niter,                       // number of iterations
                           print_level);                 // print level
         
         // Next we need to compute K21(i, :) * our results
         std_predict[i] = sqrt(fabs(K22i - Nfft4GPVecDdot( label_helper, n, iKY)));

         NFFT4GP_FREE(rel_res_v);
      }
      NFFT4GP_FREE(iKY);
      NFFT4GP_FREE(label_helper);

      if(*std_predictp == NULL)
      {
         *std_predictp = std_predict;
      }
   }
   
   return 0;
}