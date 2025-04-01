#include "kernels.h"

/* Euclidean distance */

NFFT4GP_DOUBLE Nfft4GPDistanceEuclid(void *null_params, double *data1, int ldim1, double *data2, int ldim2, int d)
{
   int i;
   NFFT4GP_DOUBLE val = 0.0, t_val;
   for(i = 0 ; i < d ; i ++)
   {
      t_val = data1[i*ldim1] - data2[i*ldim2];
      val += t_val * t_val;
   }
   return sqrt(val);
}

void Nfft4GPDistanceEuclidXY(NFFT4GP_DOUBLE *X, NFFT4GP_DOUBLE *Y, int ldimX, int ldimY, int nX, int nY, int d, NFFT4GP_DOUBLE **XYp)
{
   NFFT4GP_DOUBLE *XY;
   if(XYp == NULL)
   {
      NFFT4GP_MALLOC(XY, (size_t)nX*nY, NFFT4GP_DOUBLE);
   }
   else
   {
      XY = *XYp;
   }

   char charn = 'N', chart = 'T';
   NFFT4GP_DOUBLE two = 2.0, zero = 0.0;

   NFFT4GP_DGEMM(&charn, &chart, &nX, &nY, &d, &two, X, &ldimX, Y, &ldimY, &zero, XY, &nX);

   if(*XYp == NULL)
   {
      *XYp = XY;
   }

   return;
}

void Nfft4GPDistanceEuclidSumXX(NFFT4GP_DOUBLE *X, int ldimX, int n, int d, NFFT4GP_DOUBLE **XXp)
{
   NFFT4GP_DOUBLE *XX;
   if(XXp == NULL)
   {
      NFFT4GP_CALLOC(XX, n, NFFT4GP_DOUBLE);
   }
   else
   {
      XX = *XXp;
      Nfft4GPVecFill(XX, (size_t)n, 0.0);
   }

   size_t i, j;
   NFFT4GP_DOUBLE *X_j;
   for(j = 0 ; j < d ; j ++)
   {
      X_j = X + j*ldimX;
#ifdef NFFT4GP_USING_OPENMP
      if(!omp_in_parallel())
      {
#pragma omp parallel for private(i) schedule(static)
#endif
         for(i = 0 ; i < n ; i ++)
         {
            XX[i] += X_j[i] * X_j[i];
         }
#ifdef NFFT4GP_USING_OPENMP
      }
      else
      {
         for(i = 0 ; i < n ; i ++)
         {
            XX[i] += X_j[i] * X_j[i];
         }
      }
#endif
   }

   if(*XXp == NULL)
   {
      *XXp = XX;
   }

   return;
}

void Nfft4GPDistanceEuclidMatrixAssemble(NFFT4GP_DOUBLE *XX, int nX, NFFT4GP_DOUBLE *YY, int nY, NFFT4GP_DOUBLE scale, NFFT4GP_DOUBLE *K)
{
   size_t i, j;
   NFFT4GP_DOUBLE* K_j;
   
   for(j = 0 ; j < nY ; j ++)
   {
      K_j = K + j * nX;
#ifdef NFFT4GP_USING_OPENMP
      if(!omp_in_parallel())
      {
         #pragma omp parallel for private(i) schedule(static)
#endif
         for(i = 0 ; i < nX ; i ++)
         {
            K_j[i] -= XX[i] + YY[j];
            K_j[i] *= scale;
         }
#ifdef NFFT4GP_USING_OPENMP
      }
      else
      {
         for(i = 0 ; i < nX ; i ++)
         {
            K_j[i] -= XX[i] + YY[j];
            K_j[i] *= scale;
         }
      }
#endif
   }
}

void Nfft4GPDistanceEuclidKnn(NFFT4GP_DOUBLE *data, int n, int ldim, int d, int lfil, int **pS_i, int **pS_j)
{
   double ts, te;
   ts = Nfft4GPWtime();

   // create distance matrix
   size_t i, j;
   int *S_i, *S_j;

   NFFT4GP_MALLOC(S_i, n+1, int);
   NFFT4GP_MALLOC(S_j, (size_t)n*lfil, int);
   
   S_i[0] = 0;
   if (n <= lfil)
   {
      // in this case form a dense one
      for (i = 0 ; i < n ; i ++)
      {
         S_i[i + 1] = S_i[i];
         for (j = 0 ; j <= i ; j ++)
         {
            S_j[S_i[i + 1]++] = j;
         }
      }
   }
   else
   {

      // Note that A is symmetrix, we form A on the go
      // *
      // * *
      // * * *
      // * * * *
      for (i = 0 ; i < lfil ; i ++)
      {
         S_i[i + 1] = S_i[i];
         for (j = 0 ; j <= i ; j ++)
         {
            S_j[S_i[i + 1]++] = j;
         }
      }
      // Naive approach, setup this in parallel
      // first prepare the S_i array, we know that the remaining rows will have length of lfil
      S_i[lfil + 1] = S_i[lfil];
      for (i = lfil + 1 ; i < n ; i ++)
      {
         S_i[i + 1] = S_i[i] + lfil;
      }
      // next use OpenMP on this
      // fisrt create generate buffer for XX
      NFFT4GP_DOUBLE *A_YY = NULL;
      NFFT4GP_MALLOC(A_YY, n, NFFT4GP_DOUBLE);
      
      Nfft4GPDistanceEuclidSumXX(data, ldim, n, d, &A_YY);
#ifdef NFFT4GP_USING_OPENMP
      #pragma omp parallel
      {
         // create a local index marker
         int *marker = NULL;
         NFFT4GP_DOUBLE *dmarker = NULL;
         NFFT4GP_DOUBLE *A_XX = NULL;
         NFFT4GP_MALLOC(marker, n, int);
         NFFT4GP_MALLOC(dmarker, n, NFFT4GP_DOUBLE);
         NFFT4GP_MALLOC(A_XX, 1, NFFT4GP_DOUBLE);

         #pragma omp for
         for (i = lfil ; i < n ; i ++)
         {
            // we visit the 0 to i-1 of the i-th column of A, and select lfil-1 entries
            // combined with the diagonal to form the pattern
            if(i > 0)
            {

               // 1: K = XX = 2.0*X*X'
               Nfft4GPDistanceEuclidXY(data+i, data, ldim, ldim, 1, i, d, &dmarker);

               // 2: XX = sum(X.^2,2)
               Nfft4GPDistanceEuclidSumXX( data+i, ldim, 1, d, &A_XX);

               // 3: build kernel
               Nfft4GPDistanceEuclidMatrixAssemble( A_XX, 1, A_YY, i, -1.0, dmarker);
            }
   
            // compute KNN
            int k;
            for (k = 0 ; k < i ; k ++)
            {
               marker[k] = k;
            }
            // quick split
            Nfft4GPQsplitAscend( marker, dmarker, lfil - 1, 0, i - 1);
            
            for (k = 0 ; k < lfil - 1 ; k ++)
            {
               S_j[S_i[i + 1]++] = marker[k];
            }
            S_j[S_i[i + 1]++] = i;
         }
         free(marker);
         free(dmarker);
      }
#else
      {
         // create a local index marker
         int *marker = NULL;
         NFFT4GP_DOUBLE *dmarker = NULL;
         NFFT4GP_DOUBLE *A_XX = NULL;
         NFFT4GP_MALLOC(marker, n, int);
         NFFT4GP_MALLOC(dmarker, n, NFFT4GP_DOUBLE);
         NFFT4GP_MALLOC(A_XX, 1, NFFT4GP_DOUBLE);

         for (i = lfil ; i < n ; i ++)
         {
            // we visit the 0 to i-1 of the i-th column of A, and select lfil-1 entries
            // combined with the diagonal to form the pattern
            if(i > 0)
            {

               // 1: K = XX = 2.0*X*X'
               Nfft4GPDistanceEuclidXY(data+i, data, ldim, ldim, 1, i, d, &dmarker);

               // 2: XX = sum(X.^2,2)
               Nfft4GPDistanceEuclidSumXX( data+i, ldim, 1, d, &A_XX);

               // 3: build kernel
               Nfft4GPDistanceEuclidMatrixAssemble( A_XX, 1, A_YY, i, -1.0, dmarker);
            }
   
            // compute KNN
            int k;
            for (k = 0 ; k < i ; k ++)
            {
               marker[k] = k;
            }
            // quick split
            Nfft4GPQsplitAscend( marker, dmarker, lfil - 1, 0, i - 1);
            
            for (k = 0 ; k < lfil - 1 ; k ++)
            {
               S_j[S_i[i + 1]++] = marker[k];
            }
            S_j[S_i[i + 1]++] = i;
         }
         free(marker);
         free(dmarker);
      }
#endif

   }

   *pS_i = S_i;
   *pS_j = S_j;

   te = Nfft4GPWtime();
   printf("KNN time: %fs\n", te - ts);

}

void Nfft4GPDistanceEuclidMatrixKnn(NFFT4GP_DOUBLE *matrix, int n, int ldim, int lfil, int **pS_i, int **pS_j)
{
   double ts, te;
   ts = Nfft4GPWtime();

   // create distance matrix
   size_t i, j;
   int *S_i, *S_j;

   NFFT4GP_MALLOC(S_i, n+1, int);
   NFFT4GP_MALLOC(S_j, (size_t)n*lfil, int);
   
   S_i[0] = 0;
   if (n <= lfil)
   {
      // in this case form a dense one
      for (i = 0 ; i < n ; i ++)
      {
         S_i[i + 1] = S_i[i];
         for (j = 0 ; j <= i ; j ++)
         {
            S_j[S_i[i + 1]++] = j;
         }
      }
   }
   else
   {

      // Note that A is symmetrix, we form A on the go
      // *
      // * *
      // * * *
      // * * * *
      for (i = 0 ; i < lfil ; i ++)
      {
         S_i[i + 1] = S_i[i];
         for (j = 0 ; j <= i ; j ++)
         {
            S_j[S_i[i + 1]++] = j;
         }
      }
      // Naive approach, setup this in parallel
      // first prepare the S_i array, we know that the remaining rows will have length of lfil
      S_i[lfil + 1] = S_i[lfil];
      for (i = lfil + 1 ; i < n ; i ++)
      {
         S_i[i + 1] = S_i[i] + lfil;
      }
      // next use OpenMP on this
#ifdef NFFT4GP_USING_OPENMP
      #pragma omp parallel
      {
         // create a local index marker
         int *marker = NULL;
         NFFT4GP_DOUBLE *dmarker = NULL;
         NFFT4GP_MALLOC(marker, n, int);
         NFFT4GP_MALLOC(dmarker, n, NFFT4GP_DOUBLE);
         #pragma omp for
         for (i = lfil ; i < n ; i ++)
         {
            // we visit the 0 to i-1 of the i-th column of A, and select lfil-1 entries
            // combined with the diagonal to form the pattern
            // compute KNN
            int k;
            for (k = 0 ; k < i ; k ++)
            {
               marker[k] = k;
               dmarker[k] = -fabs(matrix[i*ldim+k]);
            }
            // quick split
            Nfft4GPQsplitAscend( marker, dmarker, lfil - 1, 0, i - 1);
            
            for (k = 0 ; k < lfil - 1 ; k ++)
            {
               S_j[S_i[i + 1]++] = marker[k];
            }
            S_j[S_i[i + 1]++] = i;
         }
         free(marker);
         free(dmarker);
      }
#else
      {
         // create a local index marker
         int *marker = NULL;
         NFFT4GP_DOUBLE *dmarker = NULL;
         NFFT4GP_MALLOC(marker, n, int);
         NFFT4GP_MALLOC(dmarker, n, NFFT4GP_DOUBLE);
         for (i = lfil ; i < n ; i ++)
         {
            // we visit the 0 to i-1 of the i-th column of A, and select lfil-1 entries
            // combined with the diagonal to form the pattern
            
            // compute KNN
            int k;
            for (k = 0 ; k < i ; k ++)
            {
               marker[k] = k;
               dmarker[k] = -fabs(matrix[i*ldim+k]);
            }
            // quick split
            Nfft4GPQsplitAscend( marker, dmarker, lfil - 1, 0, i - 1);
            
            for (k = 0 ; k < lfil - 1 ; k ++)
            {
               S_j[S_i[i + 1]++] = marker[k];
            }
            S_j[S_i[i + 1]++] = i;
         }
         free(marker);
         free(dmarker);
      }
#endif

   }

   *pS_i = S_i;
   *pS_j = S_j;

   te = Nfft4GPWtime();
   printf("KNN time: %fs\n", te - ts);
}

/* kernel parameters */

void* Nfft4GPKernelParamCreate( int max_n, int omp)
{
   pnfft4gp_kernel str;
   NFFT4GP_MALLOC(str, 1, nfft4gp_kernel);

   str->_max_n = max_n;
   str->_omp = omp;

#ifdef NFFT4GP_USING_OPENMP
   if(omp == 0)
   {
#endif
      str->_ldwork = max_n;
#ifdef NFFT4GP_USING_OPENMP
   }
   else
   {
      str->_ldwork = max_n * omp_get_max_threads();
   }
#endif
   NFFT4GP_MALLOC(str->_dwork, (size_t)str->_ldwork, NFFT4GP_DOUBLE);

   str->_own_buffer = 0;
   str->_buffer = NULL;
   str->_own_dbuffer = 0;
   str->_dbuffer = NULL;
   str->_ibufferp = NULL;
   str->_libufferp = NULL;

   str->_fkernel_buffer = NULL;

   str->_own_fkernel_buffer_params = 0;
   str->_fkernel_buffer_params = NULL; // remove

   str->_external = NULL;

   return (void*)str;
}

void Nfft4GPKernelParamFree(void *str)
{
   pnfft4gp_kernel pstr = (pnfft4gp_kernel) str;
   if(pstr)
   {
      NFFT4GP_FREE(pstr->_dwork);
      if(pstr->_own_buffer)
      {
         NFFT4GP_FREE(pstr->_buffer);
      }
      if(pstr->_own_dbuffer)
      {
         NFFT4GP_FREE(pstr->_dbuffer);
      }
      NFFT4GP_FREE(pstr->_ibufferp);
      if(pstr->_ldwork)
      {
         NFFT4GP_FREE(pstr->_dwork);
      }
      if(pstr->_own_fkernel_buffer_params)
      {
         Nfft4GPKernelParamFree(pstr->_fkernel_buffer_params);
      }

      NFFT4GP_FREE(str);
   }
}

void Nfft4GPKernelFree(void *str)
{
   NFFT4GP_FREE(str);
}

/* Gaussian kernel */

/**
 * @brief Given the matrix 2*X*X' and sum(X.^2, 2), compute the Gaussian kernel matrix.
 * @details Given the matrix 2*X*X' and sum(X.^2, 2), compute the Gaussian kernel matrix.
 * @param[in]     XX 2*X*X'.
 * @param[in]     nX Number of rows of XX.
 * @param[in,out] K Kernel matrix, when input should be the matrix 2*X*X'. If NULL, K will not be computed \n
 *                and K should be stored in 
 * @param[in,out] dK Gradient of the kernel matrix. If NULL, dK will not be computed
 * @param[in]     ff scale factor f.
 * @param[in]     kk lengthscale l.
 * @param[in]     noise Noise level.
 */
void Nfft4GPKernelGaussianPlusExpDiag( NFFT4GP_DOUBLE *XX, int nX, NFFT4GP_DOUBLE *K, NFFT4GP_DOUBLE *dK, NFFT4GP_DOUBLE ff, NFFT4GP_DOUBLE kk, NFFT4GP_DOUBLE noise)
{
   size_t i, j;
   NFFT4GP_DOUBLE *dK1, *dK2, *dK3, *dK1_j, *dK2_j, *dK3_j;
   NFFT4GP_DOUBLE *K_j;
   NFFT4GP_DOUBLE e2 = 2*kk*kk, e3 = kk*kk*kk, f1 = 2.0/ff, f12 = 2.0*ff, f2 = ff*ff, f2e3 = -f2/e3;
   //noise is already scaled
   
   if(dK)
   {
      /* set pointers for dK1, dK2, and dK3 */
      dK1 = dK;
      dK2 = dK1 + nX*nX;
      dK3 = dK2 + nX*nX;
      if(K)
      {
         /* the first case, compute both dK and K */
         for(j = 0 ; j < nX ; j ++)
         {
            K_j = K + j * nX;
            dK1_j = dK1 + j * nX;
            dK2_j = dK2 + j * nX;
            dK3_j = dK3 + j * nX;
#ifdef NFFT4GP_USING_OPENMP
            if(!omp_in_parallel())
            {
               #pragma omp parallel for private(i) schedule(static)
#endif
               for(i = 0 ; i < nX ; i ++)
               {
                  K_j[i] -= XX[i] + XX[j];
                  dK3_j[i] = 0.0;
                  dK2_j[i] = f2e3 * K_j[i] * exp(K_j[i]/e2);
                  K_j[i] = f2 * exp(K_j[i]/e2);
                  dK1_j[i] = f1 * K_j[i];
               }
#ifdef NFFT4GP_USING_OPENMP
            }
            else
            {
               for(i = 0 ; i < nX ; i ++)
               {
                  K_j[i] -= XX[i] + XX[j];
                  dK3_j[i] = 0.0;
                  dK2_j[i] = f2e3 * K_j[i] * exp(K_j[i]/e2);
                  K_j[i] = f2 * exp(K_j[i]/e2);
                  dK1_j[i] = f1 * K_j[i];
               }
            }
#endif
         }

#ifdef NFFT4GP_USING_OPENMP
         if(!omp_in_parallel())
         {
            #pragma omp parallel for private(i) schedule(static)
#endif
            for(i = 0 ; i < nX ; i ++)
            {
               K[i*nX+i] = f2 + noise;
               dK1[i*nX+i] = f1 * K[i*nX+i];
               dK2[i*nX+i] = 0.0;
               dK3[i*nX+i] = f2;
            }
#ifdef NFFT4GP_USING_OPENMP
         }
         else
         {
            for(i = 0 ; i < nX ; i ++)
            {
               K[i*nX+i] = f2 + noise;
               dK1[i*nX+i] = f1 * K[i*nX+i];
               dK2[i*nX+i] = 0.0;
               dK3[i*nX+i] = f2;
            }
         }
#endif
      }/* End of K and dK */
      else
      {
         /* the second case, compute only dK */
         for(j = 0 ; j < nX ; j ++)
         {
            dK1_j = dK1 + j * nX;
            dK2_j = dK2 + j * nX;
            dK3_j = dK3 + j * nX;
#ifdef NFFT4GP_USING_OPENMP
            if(!omp_in_parallel())
            {
               #pragma omp parallel for private(i) schedule(static)
#endif
               for(i = 0 ; i < nX ; i ++)
               {
                  dK1_j[i] -= XX[i] + XX[j];
                  dK3_j[i] = 0.0;
                  dK2_j[i] = f2e3 * dK1_j[i] * exp(dK1_j[i]/e2);
                  dK1_j[i] = f12 * exp(dK1_j[i]/e2);
               }
#ifdef NFFT4GP_USING_OPENMP
            }
            else
            {
               for(i = 0 ; i < nX ; i ++)
               {
                  dK1_j[i] -= XX[i] + XX[j];
                  dK3_j[i] = 0.0;
                  dK2_j[i] = f2e3 * dK1_j[i] * exp(dK1_j[i]/e2);
                  dK1_j[i] = f12 * exp(dK1_j[i]/e2);
               }
            }
#endif 
         }

#ifdef NFFT4GP_USING_OPENMP
         if(!omp_in_parallel())
         {
            #pragma omp parallel for private(i) schedule(static)
#endif
            for(i = 0 ; i < nX ; i ++)
            {
               dK1[i*nX+i] = f1 * (f2 + noise);
               dK2[i*nX+i] = 0.0;
               dK3[i*nX+i] = f2;
            }
#ifdef NFFT4GP_USING_OPENMP
         }
         else
         {
            for(i = 0 ; i < nX ; i ++)
            {
               dK1[i*nX+i] = f1 * (f2 + noise);
               dK2[i*nX+i] = 0.0;
               dK3[i*nX+i] = f2;
            }
         }
#endif

      }/* End of dK only */
   }/* End of dK */
   else
   {
      /* the third case, compute only K */
      for(j = 0 ; j < nX ; j ++)
      {
         K_j = K + j * nX;
#ifdef NFFT4GP_USING_OPENMP
         if(!omp_in_parallel())
         {
            #pragma omp parallel for private(i) schedule(static)
#endif
            for(i = 0 ; i < nX ; i ++)
            {
               K_j[i] -= XX[i] + XX[j];
               K_j[i] = f2 * exp(K_j[i]/e2);
            }
#ifdef NFFT4GP_USING_OPENMP
         }
         else
         {
            for(i = 0 ; i < nX ; i ++)
            {
               K_j[i] -= XX[i] + XX[j];
               K_j[i] = f2 * exp(K_j[i]/e2);
            }
         }
#endif
      }

#ifdef NFFT4GP_USING_OPENMP
      if(!omp_in_parallel())
      {
         #pragma omp parallel for private(i) schedule(static)
#endif
         for(i = 0 ; i < nX ; i ++)
         {
            K[i*nX+i] = f2 + noise;
         }
#ifdef NFFT4GP_USING_OPENMP
      }
      else
      {
         for(i = 0 ; i < nX ; i ++)
         {
            K[i*nX+i] = f2 + noise;
         }
      }
#endif
   }
}

int Nfft4GPKernelGaussianKernel(void *str, NFFT4GP_DOUBLE *data, int n, int ldim, int d, int *permr, int kr, int *permc, int kc, NFFT4GP_DOUBLE **Kp, NFFT4GP_DOUBLE **dKp)
{
   /* return immediately if no output is needed */
   if(Kp == NULL && dKp == NULL)
   {
      return 0;
   }

   size_t i, j, k, ii, jj;

   // the parameters are always in double
   pnfft4gp_kernel pstr = (pnfft4gp_kernel) str;
   NFFT4GP_DOUBLE kk = pstr->_params[1]; // lengthscale
   NFFT4GP_DOUBLE ff = pstr->_params[0]; // scale
   NFFT4GP_DOUBLE *K = NULL, *dK = NULL, *dK1 = NULL, *dK2 = NULL, *dK3 = NULL, dij, f1 = 2.0/ff, f12 = 2.0*ff, f2 = ff*ff, e2 = 2*kk*kk, e3 = kk*kk*kk, f2e3 = -f2/e3;
   NFFT4GP_DOUBLE noise_level = pstr->_noise_level * f2;

   if(permr)
   {
      if(permc)
      {
         /* Case I: part of a matrix K(permr, permc)
          * not a square matrix, create the whole mat 
          */
         if(dKp)
         {
            // NOTE: num grads is hand-written 
            // to be 3, hopefully we do not need to modify this
            if(*dKp == NULL)
            {
               NFFT4GP_CALLOC(dK, 3*(size_t)kr*kc, NFFT4GP_DOUBLE);
            }
            else
            {
               dK = *dKp;
               // fill to zero to avoid error
               Nfft4GPVecFill(dK, (size_t)3*kr*kc, 0.0);
            }
            /* Set pointers to each matrix */
            dK1 = dK;
            dK2 = dK1 + kr*kc;
            dK3 = dK2 + kr*kc;
         }

         if(Kp == NULL)
         {
            /* In this case, Kp is empty
             * we should store the distance matrix in
             * dKp
             */
            K = dK;
         }
         else
         {
            /* In this case, Kp is not empty
             * we can safely store the distance in Kp
             */
            if(*Kp == NULL)
            {
               NFFT4GP_CALLOC(K, (size_t)kr*kc, NFFT4GP_DOUBLE);
            }
            else
            {
               K = *Kp;
               Nfft4GPVecFill(K, (size_t)kr*kc, 0.0);
            }
         }
         
         /* Compute distance */
         for(k = 0 ; k < d ; k ++)
         {
            /* The parallel implementation */
#ifdef NFFT4GP_USING_OPENMP
            if(!omp_in_parallel())
            {
               #pragma omp parallel for private(ii,jj,i,j,dij) schedule(dynamic)
#endif
               for(ii = 0 ; ii < kc ; ii ++)
               {
                  i = permc[ii];
                  for(jj = 0 ; jj < kr ; jj ++)
                  {
                     j = permr[jj];
                     /* the value of K(jj, ii) is based on the distance between i and j */
                     if(i != j)
                     {
                        /* get the distance between the points with the permutated indeces */
                        dij = data[k*ldim+i] - data[k*ldim+j];
                        K[ii*kr+jj] -= dij*dij;
                     }
                  }
               }
#ifdef NFFT4GP_USING_OPENMP
            }
            else
            {
               /* The sequential imlementation */
               for(ii = 0 ; ii < kc ; ii ++)
               {
                  i = permc[ii];
                  for(jj = 0 ; jj < kr ; jj ++)
                  {
                     j = permr[jj];
                     /* the value of K(jj, ii) is based on the distance between i and j */
                     if(i != j)
                     {
                        /* get the distance between the points with the permutated indeces */
                        dij = data[k*ldim+i] - data[k*ldim+j];
                        K[ii*kr+jj] -= dij*dij;
                     }
                  }
               }
            }
#endif
         }/* End of computing distance */

         /* form the matrix */
         if(dKp)
         {
            /* With gradient, form gradient first 
             * form dK1 last since the distance might
             * be stored in it
             */
            
            /* we check Kp in the front
             * write duplicate code to avoid
             * if statement in the loop
             */

            if(Kp)
            {
               /* In this case both dKp and Kp */
#ifdef NFFT4GP_USING_OPENMP
               if(!omp_in_parallel())
               {
                  #pragma omp parallel for private(ii,jj,i,j,dij) schedule(dynamic)
#endif
                  for(ii = 0 ; ii < kc ; ii ++)
                  {
                     i = permc[ii];
                     for(jj = 0 ; jj < kr ; jj ++)
                     {
                        j = permr[jj];
                        /* when have permc we do not add diagonal */
                        dK2[ii*kr+jj] = f2e3 * K[ii*kr+jj]*exp(K[ii*kr+jj]/e2);
                        K[ii*kr+jj] = f2 * exp(K[ii*kr+jj]/e2);
                        dK1[ii*kr+jj] = f1 * K[ii*kr+jj];
                     }
                  }
#ifdef NFFT4GP_USING_OPENMP
               }
               else
               {
                  for(ii = 0 ; ii < kc ; ii ++)
                  {
                     i = permc[ii];
                     for(jj = 0 ; jj < kr ; jj ++)
                     {
                        j = permr[jj];
                        /* when have permc we do not add noise */
                        dK2[ii*kr+jj] = f2e3 * K[ii*kr+jj]*exp(K[ii*kr+jj]/e2);
                        K[ii*kr+jj] = f2 * exp(K[ii*kr+jj]/e2);
                        dK1[ii*kr+jj] = f1 * K[ii*kr+jj];
                     }
                  }
               }
#endif
            }/* End of loop for dKp and Kp */
            else
            {
               /* In this case dKp only */
#ifdef NFFT4GP_USING_OPENMP
               if(!omp_in_parallel())
               {
                  #pragma omp parallel for private(ii,jj,i,j,dij) schedule(dynamic)
#endif
                  for(ii = 0 ; ii < kc ; ii ++)
                  {
                     i = permc[ii];
                     for(jj = 0 ; jj < kr ; jj ++)
                     {
                        j = permr[jj];
                        /* when have permc we do not add noise */
                        dK2[ii*kr+jj] = f2e3 * K[ii*kr+jj]*exp(K[ii*kr+jj]/e2);
                        dK1[ii*kr+jj] = f12 * exp(K[ii*kr+jj]/e2);
                     }
                  }
#ifdef NFFT4GP_USING_OPENMP
               }
               else
               {
                  for(ii = 0 ; ii < kc ; ii ++)
                  {
                     i = permc[ii];
                     for(jj = 0 ; jj < kr ; jj ++)
                     {
                        j = permr[jj];
                        /* when have permc we do not add noise */
                        dK2[ii*kr+jj] = f2e3 * K[ii*kr+jj]*exp(K[ii*kr+jj]/e2);
                        dK1[ii*kr+jj] = f12 * exp(K[ii*kr+jj]/e2);
                     }
                  }
               }
#endif
            }/* End of loop for dKp only */
         }/* End of with dKp */
         else
         {
            /* In this case K only
             */
#ifdef NFFT4GP_USING_OPENMP
            if(!omp_in_parallel())
            {
               #pragma omp parallel for private(ii,jj,i,j,dij) schedule(dynamic)
#endif
               for(ii = 0 ; ii < kc ; ii ++)
               {
                  i = permc[ii];
                  for(jj = 0 ; jj < kr ; jj ++)
                  {
                     j = permr[jj];
                     /* when have permc we do not add noise */
                     K[ii*kr+jj] = f2 * exp(K[ii*kr+jj]/e2);
                  }
               }
#ifdef NFFT4GP_USING_OPENMP
            }
            else
            {
               for(ii = 0 ; ii < kc ; ii ++)
               {
                  i = permc[ii];
                  for(jj = 0 ; jj < kr ; jj ++)
                  {
                     j = permr[jj];
                     /* when have permc we do not add noise */
                     K[ii*kr+jj] = f2 * exp(K[ii*kr+jj]/e2);
                  }
               }
            }
#endif
         }/* End of Kp only */
      }/* End of the first case K(permr, permc) */
      else
      {
         /* Case II: part of a matrix K(permr, permr)
          * In this case the matrix is square, only 
          * the LOWER part is generated
          */
         if(dKp)
         {
            // NOTE: num grads is hand-written 
            // to be 3, hopefully we do not need to modify this
            if(*dKp == NULL)
            {
               NFFT4GP_CALLOC(dK, 3*(size_t)kr*kr, NFFT4GP_DOUBLE);
            }
            else
            {
               dK = *dKp;
               // fill to zero to avoid error
               Nfft4GPVecFill(dK, (size_t)3*kr*kr, 0.0);
            }
            /* Set pointers to each matrix */
            dK1 = dK;
            dK2 = dK1 + kr*kr;
            dK3 = dK2 + kr*kr;
         }

         if(Kp == NULL)
         {
            /* In this case, Kp is empty
             * we should store the distance matrix in
             * dKp
             */
            K = dK;
         }
         else
         {
            /* In this case, Kp is not empty
             * we can safely store the distance in Kp
             */
            if(*Kp == NULL)
            {
               NFFT4GP_CALLOC(K, (size_t)kr*kr, NFFT4GP_DOUBLE);
            }
            else
            {
               K = *Kp;
               Nfft4GPVecFill(K, (size_t)kr*kr, 0.0);
            }
         }

         /* Compute Distance 
          * The problem is symmetric, only fill the LOWER part!
          */
         for(k = 0 ; k < d ; k ++)
         {
            /* The parallel implementation */
#ifdef NFFT4GP_USING_OPENMP
            if(!omp_in_parallel())
            {
               #pragma omp parallel for private(i,j,dij) schedule(dynamic)
#endif
               for(i = 0 ; i < kr ; i ++)
               {
                  for(j = i+1 ; j < kr ; j ++)
                  {
                     dij = data[k*ldim+permr[i]] - data[k*ldim+permr[j]];
                     K[i*kr+j] -= dij*dij;
                  }
               }
#ifdef NFFT4GP_USING_OPENMP
            }
            else
            {
               /* The sequential imlementation */
               for(i = 0 ; i < kr ; i ++)
               {
                  for(j = i+1 ; j < kr ; j ++)
                  {
                     dij = data[k*ldim+permr[i]] - data[k*ldim+permr[j]];
                     K[i*kr+j] -= dij*dij;
                  }
               }
            }
#endif
         }

         /* form the matrix */
         if(dKp)
         {
            /* With gradient, form gradient first 
             * form dK1 last since the distance might
             * be stored in it
             */

            /* we check Kp in the front
             * write duplicate code to avoid
             * if statement in the loop
             */

            if(Kp)
            {
               /* In this case both dKp and Kp */
#ifdef NFFT4GP_USING_OPENMP
               if(!omp_in_parallel())
               {
                  #pragma omp parallel for private(i,j,dij) schedule(dynamic)
#endif
                  for(i = 0 ; i < kr ; i ++)
                  {
                     dK3[i*kr+i] = f2;
                     K[i*kr+i] = f2 + noise_level;
                     dK1[i*kr+i] = f1 * K[i*kr+i];
                     for(j = i+1 ; j < kr ; j ++)
                     {
                        dK2[i*kr+j] = f2e3 * K[i*kr+j]*exp(K[i*kr+j]/e2);
                        K[i*kr+j] = f2 * exp(K[i*kr+j]/e2);
                        dK1[i*kr+j] = f1 * K[i*kr+j];
                     }
                  }
#ifdef NFFT4GP_USING_OPENMP
               }
               else
               {
                  for(i = 0 ; i < kr ; i ++)
                  {
                     dK3[i*kr+i] = f2;
                     K[i*kr+i] = f2 + noise_level;
                     dK1[i*kr+i] = f1 * K[i*kr+i];
                     for(j = i+1 ; j < kr ; j ++)
                     {
                        dK2[i*kr+j] = f2e3 * K[i*kr+j]*exp(K[i*kr+j]/e2);
                        K[i*kr+j] = f2 * exp(K[i*kr+j]/e2);
                        dK1[i*kr+j] = f1 * K[i*kr+j];
                     }
                  }
               }
#endif
            }/* End of loop for dKp and Kp */
            else
            {
               /* In this case dKp only */
#ifdef NFFT4GP_USING_OPENMP
               if(!omp_in_parallel())
               {
                  #pragma omp parallel for private(i,j,dij) schedule(dynamic)
#endif
                  for(i = 0 ; i < kr ; i ++)
                  {
                     dK1[i*kr+i] = f1 * (f2 + noise_level);
                     dK3[i*kr+i] = f2;
                     for(j = i+1 ; j < kr ; j ++)
                     {
                        dK2[i*kr+j] = f2e3 * K[i*kr+j]*exp(K[i*kr+j]/e2);
                        dK1[i*kr+j] = f12 * exp(K[i*kr+j]/e2);
                     }
                  }
#ifdef NFFT4GP_USING_OPENMP
               }
               else
               {
                  for(i = 0 ; i < kr ; i ++)
                  {
                     dK1[i*kr+i] = f1 * (f2 + noise_level);
                     dK3[i*kr+i] = f2;
                     for(j = i+1 ; j < kr ; j ++)
                     {
                        dK2[i*kr+j] = f2e3 * K[i*kr+j]*exp(K[i*kr+j]/e2);
                        dK1[i*kr+j] = f12 * exp(K[i*kr+j]/e2);
                     }
                  }
               }
#endif
            }/* End of loop for dKp only */
         }/* End of with dKp */
         else
         {
#ifdef NFFT4GP_USING_OPENMP
            if(!omp_in_parallel())
            {
               #pragma omp parallel for private(i,j,dij) schedule(dynamic)
#endif
               for(i = 0 ; i < kr ; i ++)
               {
                  K[i*kr+i] = f2 + noise_level;
                  for(j = i+1 ; j < kr ; j ++)
                  {
                     K[i*kr+j] = f2 * exp(K[i*kr+j]/e2);
                  }
               }
#ifdef NFFT4GP_USING_OPENMP
            }
            else
            {
               for(i = 0 ; i < kr ; i ++)
               {
                  K[i*kr+i] = f2 + noise_level;
                  for(j = i+1 ; j < kr ; j ++)
                  {
                     K[i*kr+j] = f2 * exp(K[i*kr+j]/e2);
                  }
               }
            }
#endif
         }/* end of Kp only */
      }/* End of the second case K(permr, permr) */
   }
   else
   {
      /* Case III: The entire matrix K
       * In this case the matrix is square, only 
       * the LOWER part is generated
       */
#ifdef NFFT4GP_USING_OPENMP
      if(!omp_in_parallel())
      {
#endif
         /* This is the sequential version */
         if(pstr->_ldwork < (size_t)n)
         {
            printf("Nfft4GPKernelGaussianKernel: buffer size is not enough. Check your call to Nfft4GPKernelParamCreate.\n");
            printf("required buffer size is %d, but only %ld is provided.\n", n, pstr->_ldwork);
            return -1;
         }

         if(dKp)
         {
            // NOTE: num grads is hand-written 
            // to be 3, hopefully we do not need to modify this
            if(*dKp == NULL)
            {
               NFFT4GP_MALLOC(dK, 3*(size_t)n*n, NFFT4GP_DOUBLE);
            }
            else
            {
               dK = *dKp;
               // fill to zero to avoid error
               Nfft4GPVecFill(dK, (size_t)3*n*n, 0.0);
            }
         }
         else
         {
            dK = NULL;
         }

         if(Kp)
         {
            if(*Kp == NULL)
            {
               NFFT4GP_MALLOC(K, (size_t)n*n, NFFT4GP_DOUBLE);
            }
            else
            {
               K = *Kp;
               Nfft4GPVecFill(K, (size_t)n*n, 0.0);
            }
         }
         else
         {
            /* We do something different and set K to NULL */
            K = NULL;
         }

         /* 1: K = XX = 2.0*X*X'
          * 2: XX = sum(X.^2,2)
          * 3: build kernel
          */
         if(K)
         {
            Nfft4GPDistanceEuclidXY(data, data, ldim, ldim, n, n, d, &K);
         }
         else
         {
            Nfft4GPDistanceEuclidXY(data, data, ldim, ldim, n, n, d, &dK);
         }
         Nfft4GPDistanceEuclidSumXX(data, ldim, n, d, &(pstr->_dwork));
         Nfft4GPKernelGaussianPlusExpDiag( pstr->_dwork, n, K, dK, ff, kk, noise_level);

#ifdef NFFT4GP_USING_OPENMP
      }
      else
      {
         // This is in a OpenMP section, be careful
         int nthreads = omp_get_max_threads();
         if(pstr->_ldwork < (size_t)n*nthreads)
         {
            printf("Nfft4GPKernelGaussianKernel: buffer size is not enough. Check your call to Nfft4GPKernelParamCreate.\n");
            printf("required buffer size is %d, but only %ld is provided.\n", n*nthreads, pstr->_ldwork);
            return -1;
         }

         if(dKp)
         {
            // NOTE: num grads is hand-written 
            // to be 3, hopefully we do not need to modify this
            if(*dKp == NULL)
            {
               NFFT4GP_MALLOC(dK, 3*(size_t)n*n, NFFT4GP_DOUBLE);
            }
            else
            {
               dK = *dKp;
               // fill to zero to avoid error
               Nfft4GPVecFill(dK, (size_t)3*n*n, 0.0);
            }
         }
         else
         {
            dK = NULL;
         }

         if(Kp)
         {
            if(*Kp == NULL)
            {
               NFFT4GP_MALLOC(K, (size_t)n*n, NFFT4GP_DOUBLE);
            }
            else
            {
               K = *Kp;
               Nfft4GPVecFill(K, (size_t)n*n, 0.0);
            }
         }
         else
         {
            /* We do something different and set K to NULL */
            K = NULL;
         }

         if(*Kp == NULL)
         {
            NFFT4GP_MALLOC(K, (size_t)n*n, NFFT4GP_DOUBLE);
         }
         else
         {
            K = *Kp;
         }

         NFFT4GP_DOUBLE *dwork = pstr->_dwork + (size_t)pstr->_max_n*omp_get_thread_num();

         /* 1: K = XX = 2.0*X*X'
          * 2: XX = sum(X.^2,2)
          * 3: build kernel
          */
         if(K)
         {
            Nfft4GPDistanceEuclidXY(data, data, ldim, ldim, n, n, d, &K);
         }
         else
         {
            Nfft4GPDistanceEuclidXY(data, data, ldim, ldim, n, n, d, &dK);
         }
         Nfft4GPDistanceEuclidSumXX(data, ldim, n, d, &dwork);
         Nfft4GPKernelGaussianPlusExpDiag( dwork, n, K, dK, ff, kk, noise_level);
      }
#endif
   }/* End of the third case the entire K */
   
   if(Kp && *Kp == NULL)
   {
      *Kp = K;
   }
   if(dKp && *dKp == NULL)
   {
      *dKp = dK;
   }
   return 0;
}

NFFT4GP_DOUBLE Nfft4GPKernelGaussianKernelVal2Dist(void *str, NFFT4GP_DOUBLE tol)
{
   // create problem K(x,y) = f^2 * exp(-||x-y||^2/kk^2)
   // tol < f^2 * exp(-d^2/kk^2) => tol/f^2 < exp(-d^2/kk^2) => sqrt(-log(tol/(f^2))*kk^2) < d

   pnfft4gp_kernel pstr = (pnfft4gp_kernel) str;
   NFFT4GP_DOUBLE f2 = pstr->_params[0] * pstr->_params[0];
   NFFT4GP_DOUBLE kk = pstr->_params[1];
   return sqrt(-log(tol/f2)*kk*kk);
}

/* Matern32 kernel */

/**
 * @brief Given the matrix 2*X*X' and sum(X.^2, 2), compute the Gaussian kernel matrix.
 * @details Given the matrix 2*X*X' and sum(X.^2, 2), compute the Gaussian kernel matrix.
 * @param[in]     XX 2*X*X'.
 * @param[in]     nX Number of rows of XX.
 * @param[in,out] K Kernel matrix, when input should be the matrix 2*X*X'. If NULL, K will not be computed \n
 *                and K should be stored in 
 * @param[in,out] dK Gradient of the kernel matrix. If NULL, dK will not be computed
 * @param[in]     ff scale factor f.
 * @param[in]     kk lengthscale l.
 * @param[in]     noise Noise level.
 */
void Nfft4GPKernelMaternPlusExpDiag( NFFT4GP_DOUBLE *XX, int nX, NFFT4GP_DOUBLE *K, NFFT4GP_DOUBLE *dK, NFFT4GP_DOUBLE ff, NFFT4GP_DOUBLE kk, NFFT4GP_DOUBLE noise)
{
   size_t i, j;
   NFFT4GP_DOUBLE *dK1, *dK2, *dK3, *dK1_j, *dK2_j, *dK3_j;
   NFFT4GP_DOUBLE *K_j;
   NFFT4GP_DOUBLE me = -NFFT4GP_MATERN32_SQRT3/kk;
   NFFT4GP_DOUBLE l3 = kk*kk*kk;
   NFFT4GP_DOUBLE dk, f1 = 2.0/ff, f2 = ff*ff, f12 = 2.0*ff, f2e3 = - f2*3.0/l3;
   // noise is already scaled by f2
   
   if(dK)
   {
      /* set pointers for dK1, dK2, and dK3 */
      dK1 = dK;
      dK2 = dK1 + nX*nX;
      dK3 = dK2 + nX*nX;
      if(K)
      {
         /* the first case, compute both dK and K */
         for(j = 0 ; j < nX ; j ++)
         {
            K_j = K + j * nX;
            dK1_j = dK1 + j * nX;
            dK2_j = dK2 + j * nX;
            dK3_j = dK3 + j * nX;
#ifdef NFFT4GP_USING_OPENMP
            if(!omp_in_parallel())
            {
               #pragma omp parallel for private(i,dk) schedule(static)
#endif
               for(i = 0 ; i < nX ; i ++)
               {
                  K_j[i] -= XX[i] + XX[j];
                  K_j[i] = K_j[i] > 0.0 ? 0.0 : K_j[i];
                  dk = sqrt(-K_j[i]);
                  dK3_j[i] = 0.0;
                  dK2_j[i] = f2e3 * K_j[i] * exp(me * dk);
                  K_j[i] = f2 * (1.0 - me * dk) * exp(me * dk);
                  dK1_j[i] = f1 * K_j[i];
               }
#ifdef NFFT4GP_USING_OPENMP
            }
            else
            {
               for(i = 0 ; i < nX ; i ++)
               {
                  K_j[i] -= XX[i] + XX[j];
                  K_j[i] = K_j[i] > 0.0 ? 0.0 : K_j[i];
                  dk = sqrt(-K_j[i]);
                  dK3_j[i] = 0.0;
                  dK2_j[i] = f2e3 * K_j[i] * exp(me * dk);
                  K_j[i] = f2 * (1.0 - me * dk) * exp(me * dk);
                  dK1_j[i] = f1 * K_j[i];
               }
            }
#endif
         }

#ifdef NFFT4GP_USING_OPENMP
         if(!omp_in_parallel())
         {
            #pragma omp parallel for private(i) schedule(static)
#endif
            for(i = 0 ; i < nX ; i ++)
            {
               K[i*nX+i] = f2 + noise;
               dK1[i*nX+i] = f1 * (f2 + noise);
               dK2[i*nX+i] = 0.0;
               dK3[i*nX+i] = f2;
            }
#ifdef NFFT4GP_USING_OPENMP
         }
         else
         {
            for(i = 0 ; i < nX ; i ++)
            {
               K[i*nX+i] = f2 + noise;
               dK1[i*nX+i] = f1 * (f2 + noise);
               dK2[i*nX+i] = 0.0;
               dK3[i*nX+i] = f2;
            }
         }
#endif
      }/* End of K and dK */
      else
      {
         /* the second case, only compute dK */
         for(j = 0 ; j < nX ; j ++)
         {
            dK1_j = dK1 + j * nX;
            dK2_j = dK2 + j * nX;
            dK3_j = dK3 + j * nX;
#ifdef NFFT4GP_USING_OPENMP
            if(!omp_in_parallel())
            {
               #pragma omp parallel for private(i,dk) schedule(static)
#endif
               for(i = 0 ; i < nX ; i ++)
               {
                  dK1_j[i] -= XX[i] + XX[j];
                  dK1_j[i] = dK1_j[i] > 0.0 ? 0.0 : dK1_j[i];
                  dk = sqrt(-dK1_j[i]);
                  dK3_j[i] = 0.0;
                  dK2_j[i] = f2e3 * dK1_j[i] * exp(me * dk);
                  dK1_j[i] = f12 * (1.0 - me * dk) * exp(me * dk);
               }
#ifdef NFFT4GP_USING_OPENMP
            }
            else
            {
               for(i = 0 ; i < nX ; i ++)
               {
                  dK1_j[i] -= XX[i] + XX[j];
                  dK1_j[i] = dK1_j[i] > 0.0 ? 0.0 : dK1_j[i];
                  dk = sqrt(-dK1_j[i]);
                  dK3_j[i] = 0.0;
                  dK2_j[i] = f2e3 * dK1_j[i] * exp(me * dk);
                  dK1_j[i] = f12 * (1.0 - me * dk) * exp(me * dk);
               }
            }
#endif
         }
         
#ifdef NFFT4GP_USING_OPENMP
         if(!omp_in_parallel())
         {
            #pragma omp parallel for private(i) schedule(static)
#endif
            for(i = 0 ; i < nX ; i ++)
            {
               dK1[i*nX+i] = f1 * (f2 + noise);
               dK2[i*nX+i] = 0.0;
               dK3[i*nX+i] = f2;
            }
#ifdef NFFT4GP_USING_OPENMP
         }
         else
         {
            for(i = 0 ; i < nX ; i ++)
            {
               dK1[i*nX+i] = f1 * (f2 + noise);
               dK2[i*nX+i] = 0.0;
               dK3[i*nX+i] = f2;
            }
         }
#endif
      }/* End of dK only */
   }
   else
   {
      /* the third case, only compute K */
      for(j = 0 ; j < nX ; j ++)
      {
         K_j = K + j * nX;
#ifdef NFFT4GP_USING_OPENMP
         if(!omp_in_parallel())
         {
            #pragma omp parallel for private(i,dk) schedule(static)
#endif
            for(i = 0 ; i < nX ; i ++)
            {
               K_j[i] -= XX[i] + XX[j];
               K_j[i] = K_j[i] > 0.0 ? 0.0 : K_j[i];
               dk = sqrt(-K_j[i]);
               K_j[i] = f2 * (1.0 - me * dk) * exp(me * dk);
            }
#ifdef NFFT4GP_USING_OPENMP
         }
         else
         {
            for(i = 0 ; i < nX ; i ++)
            {
               K_j[i] -= XX[i] + XX[j];
               K_j[i] = K_j[i] > 0.0 ? 0.0 : K_j[i];
               dk = sqrt(-K_j[i]);
               K_j[i] = f2 * (1.0 - me * dk) * exp(me * dk);
            }
         }
#endif
      }
#ifdef NFFT4GP_USING_OPENMP
      if(!omp_in_parallel())
      {
         #pragma omp parallel for private(i) schedule(static)
#endif
         for(i = 0 ; i < nX ; i ++)
         {
            K[i*nX+i] = f2 + noise;
         }
#ifdef NFFT4GP_USING_OPENMP
      }
      else
      {
         for(i = 0 ; i < nX ; i ++)
         {
            K[i*nX+i] = f2 + noise;
         }
      }
#endif
   }/* End of K only */
}

int Nfft4GPKernelMatern32Kernel(void *str, NFFT4GP_DOUBLE *data, int n, int ldim, int d, int *permr, int kr, int *permc, int kc, NFFT4GP_DOUBLE **Kp, NFFT4GP_DOUBLE **dKp)
{
   /* return immediately if no output is needed */
   if(Kp == NULL && dKp == NULL)
   {
      return 0;
   }

   size_t i, j, k, ii, jj;

   pnfft4gp_kernel pstr = (pnfft4gp_kernel) str;
   NFFT4GP_DOUBLE ff = pstr->_params[0]; // scale
   NFFT4GP_DOUBLE me = -NFFT4GP_MATERN32_SQRT3/pstr->_params[1];
   NFFT4GP_DOUBLE l3 = pstr->_params[1]*pstr->_params[1]*pstr->_params[1];
   NFFT4GP_DOUBLE f1 = 2.0/ff, f2 = ff*ff, f12 = 2.0*ff, f2e3 = f2*3.0/l3;
   NFFT4GP_DOUBLE noise_level = pstr->_noise_level * f2;
   NFFT4GP_DOUBLE *K = NULL, *dK = NULL, *dK1 = NULL, *dK2 = NULL, *dK3 = NULL, dij, dk;

   if(permr)
   {
      if(permc)
      {
         /* Case I: part of a matrix K(permr, permc)
          * In this case the matrix is rectangular
          */
         if(dKp)
         {
            // NOTE: num grads is hand-written 
            // to be 3, hopefully we do not need to modify this
            if(*dKp == NULL)
            {
               NFFT4GP_CALLOC(dK, 3*(size_t)kr*kc, NFFT4GP_DOUBLE);
            }
            else
            {
               dK = *dKp;
               // fill to zero to avoid error
               Nfft4GPVecFill(dK, (size_t)3*kr*kc, 0.0);
            }
            /* Set pointers to each matrix */
            dK1 = dK;
            dK2 = dK1 + kr*kc;
            dK3 = dK2 + kr*kc;
         }

         if(Kp == NULL)
         {
            /* In this case, Kp is empty
             * we should store the distance matrix in
             * dKp
             */
            K = dK;
         }
         else
         {
            /* In this case, Kp is not empty
             * we can safely store the distance in Kp
             */
            if(*Kp == NULL)
            {
               NFFT4GP_CALLOC(K, (size_t)kr*kc, NFFT4GP_DOUBLE);
            }
            else
            {
               K = *Kp;
               Nfft4GPVecFill(K, (size_t)kr*kc, 0.0);
            }
         }

         /* Compute Distance */
         for(k = 0 ; k < d ; k ++)
         {
#ifdef NFFT4GP_USING_OPENMP
            if(!omp_in_parallel())
            {
               #pragma omp parallel for private(ii,jj,i,j,dij) schedule(dynamic)
#endif
               for(ii = 0 ; ii < kc ; ii ++)
               {
                  i = permc[ii];
                  for(jj = 0 ; jj < kr ; jj ++)
                  {
                     j = permr[jj];
                     /* the value of K(jj, ii) is based on the distance between i and j */
                     if(i != j)
                     {
                        /* get the distance between the points with the permutated indeces */
                        dij = data[k*ldim+i] - data[k*ldim+j];
                        K[ii*kr+jj] += dij*dij;
                     }
                  }
               }
#ifdef NFFT4GP_USING_OPENMP
            }
            else
            {
               for(ii = 0 ; ii < kc ; ii ++)
               {
                  i = permc[ii];
                  for(jj = 0 ; jj < kr ; jj ++)
                  {
                     j = permr[jj];
                     /* the value of K(jj, ii) is based on the distance between i and j */
                     if(i != j)
                     {
                        /* get the distance between the points with the permutated indeces */
                        dij = data[k*ldim+i] - data[k*ldim+j];
                        K[ii*kr+jj] += dij*dij;
                     }
                  }
               }
            }
#endif
         }

         /* form the matrix */
         if(dKp)
         {
            /* With gradient, form gradient first 
             * form dK1 last since the distance might
             * be stored in it
             */

            /* we check Kp in the front
             * write duplicate code to avoid
             * if statement in the loop
             */

            if(Kp)
            {
               /* In this case both dKp and Kp
                * note that what stored in K is D2 in this function
                * but me is negative
                */
#ifdef NFFT4GP_USING_OPENMP
               if(!omp_in_parallel())
               {
                  #pragma omp parallel for private(ii,jj,i,j,dij,dk) schedule(dynamic)
#endif
                  for(ii = 0 ; ii < kc ; ii ++)
                  {
                     i = permc[ii];
                     for(jj = 0 ; jj < kr ; jj ++)
                     {
                        j = permr[jj];
                        /* when have permc we do not add noise */
                        dk = sqrt(K[ii*kr+jj]);
                        dK2[ii*kr+jj] = f2e3 * dk * dk * exp(me * dk);
                        K[ii*kr+jj] = f2 * (1.0 - me * dk) * exp(me * dk);
                        dK1[ii*kr+jj] = f1 * K[ii*kr+jj];
                     }
                  }
#ifdef NFFT4GP_USING_OPENMP
               }
               else
               {
                  for(ii = 0 ; ii < kc ; ii ++)
                  {
                     i = permc[ii];
                     for(jj = 0 ; jj < kr ; jj ++)
                     {
                        j = permr[jj];
                        /* when have permc we do not add noise */
                        dk = sqrt(K[ii*kr+jj]);
                        dK2[ii*kr+jj] = f2e3 * dk * dk * exp(me * dk);
                        K[ii*kr+jj] = f2 * (1.0 - me * dk) * exp(me * dk);
                        dK1[ii*kr+jj] = f1 * K[ii*kr+jj];
                     }
                  }
               }
#endif
            }/* End of loop for dKp and Kp */
            else
            {
               /* In this case dKp only
                * note that what stored in K is D2 in this function
                * but me is negative
                */
#ifdef NFFT4GP_USING_OPENMP
               if(!omp_in_parallel())
               {
                  #pragma omp parallel for private(ii,jj,i,j,dij,dk) schedule(dynamic)
#endif
                  for(ii = 0 ; ii < kc ; ii ++)
                  {
                     i = permc[ii];
                     for(jj = 0 ; jj < kr ; jj ++)
                     {
                        j = permr[jj];
                        /* when have permc we do not add noise */
                        dk = sqrt(K[ii*kr+jj]);
                        dK2[ii*kr+jj] = f2e3 * dk * dk * exp(me * dk);
                        dK1[ii*kr+jj] = f12 * (1.0 - me * dk) * exp(me * dk);
                     }
                  }
#ifdef NFFT4GP_USING_OPENMP
               }
               else
               {
                  for(ii = 0 ; ii < kc ; ii ++)
                  {
                     i = permc[ii];
                     for(jj = 0 ; jj < kr ; jj ++)
                     {
                        j = permr[jj];
                        /* when have permc we do not add noise */
                        dk = sqrt(K[ii*kr+jj]);
                        dK2[ii*kr+jj] = f2e3 * dk * dk * exp(me * dk);
                        dK1[ii*kr+jj] = f12 * (1.0 - me * dk) * exp(me * dk);
                     }
                  }
               }
#endif
            }/* End of loop for dKp only */
         }/* End of with dKp */
         else
         {
            /* In this case K only
             * note that what stored in K is D2 in this function
             * but me is negative
             */
#ifdef NFFT4GP_USING_OPENMP
            if(!omp_in_parallel())
            {
               #pragma omp parallel for private(ii,jj,i,j,dij,dk) schedule(dynamic)
#endif
               for(ii = 0 ; ii < kc ; ii ++)
               {
                  i = permc[ii];
                  for(jj = 0 ; jj < kr ; jj ++)
                  {
                     j = permr[jj];
                     /* when have permc we do not add noise */
                     dk = sqrt(K[ii*kr+jj]);
                     K[ii*kr+jj] = f2 * (1.0 - me * dk) * exp(me * dk);
                  }
               }
#ifdef NFFT4GP_USING_OPENMP
            }
            else
            {
               for(ii = 0 ; ii < kc ; ii ++)
               {
                  i = permc[ii];
                  for(jj = 0 ; jj < kr ; jj ++)
                  {
                     j = permr[jj];
                     /* when have permc we do not add noise */
                     dk = sqrt(K[ii*kr+jj]);
                     K[ii*kr+jj] = f2 * (1.0 - me * dk) * exp(me * dk);
                  }
               }
            }
#endif
         }/* end of Kp only */
      }/* End of the first case K(permr, permc) */
      else
      {
         /* Case II: part of a matrix K(permr, permr)
          * In this case the matrix is square, only 
          * the LOWER part is generated
          */
         if(dKp)
         {
            // NOTE: num grads is hand-written 
            // to be 3, hopefully we do not need to modify this
            if(*dKp == NULL)
            {
               NFFT4GP_CALLOC(dK, 3*(size_t)kr*kr, NFFT4GP_DOUBLE);
            }
            else
            {
               dK = *dKp;
               // fill to zero to avoid error
               Nfft4GPVecFill(dK, (size_t)3*kr*kr, 0.0);
            }
            /* Set pointers to each matrix */
            dK1 = dK;
            dK2 = dK1 + kr*kr;
            dK3 = dK2 + kr*kr;
         }

         if(Kp == NULL)
         {
            /* In this case, Kp is empty
             * we should store the distance matrix in
             * dKp
             */
            K = dK;
         }
         else
         {
            /* In this case, Kp is not empty
             * we can safely store the distance in Kp
             */
            if(*Kp == NULL)
            {
               NFFT4GP_CALLOC(K, (size_t)kr*kr, NFFT4GP_DOUBLE);
            }
            else
            {
               K = *Kp;
               Nfft4GPVecFill(K, (size_t)kr*kr, 0.0);
            }
         }

         /* Compute Distance 
          * The problem is symmetric, only fill the LOWER part!
          */
         for(k = 0 ; k < d ; k ++)
         {
            /* The parallel implementation */
#ifdef NFFT4GP_USING_OPENMP
            if(!omp_in_parallel())
            {
               #pragma omp parallel for private(i,j,dij) schedule(dynamic)
#endif
               for(i = 0 ; i < kr ; i ++)
               {
                  for(j = i+1 ; j < kr ; j ++)
                  {
                     dij = data[k*ldim+permr[i]] - data[k*ldim+permr[j]];
                     K[i*kr+j] += dij*dij;
                  }
               }
#ifdef NFFT4GP_USING_OPENMP
            }
            else
            {
               /* The sequential imlementation */
               for(i = 0 ; i < kr ; i ++)
               {
                  for(j = i+1 ; j < kr ; j ++)
                  {
                     dij = data[k*ldim+permr[i]] - data[k*ldim+permr[j]];
                     K[i*kr+j] += dij*dij;
                  }
               }
            }
#endif
         }

         /* form the matrix */
         if(dKp)
         {
            /* With gradient, form gradient first 
             * form dK1 last since the distance might
             * be stored in it
             */

            /* we check Kp in the front
             * write duplicate code to avoid
             * if statement in the loop
             */

            if(Kp)
            {
               /* In this case both dKp and Kp
                * note that what stored in K is D2 in this function
                * but me is negative
                */
#ifdef NFFT4GP_USING_OPENMP
               if(!omp_in_parallel())
               {
                  #pragma omp parallel for private(i,j,dij,dk) schedule(dynamic)
#endif
                  for(i = 0 ; i < kr ; i ++)
                  {
                     dK1[i*kr+i] = f1 * (f2 + noise_level);
                     dK3[i*kr+i] = f2;
                     K[i*kr+i] = f2 + noise_level;
                     for(j = i+1 ; j < kr ; j ++)
                     {
                        dk = sqrt(K[i*kr+j]);
                        dK2[i*kr+j] = f2e3 * dk * dk * exp(me * dk);
                        K[i*kr+j] = f2 * (1.0 - me * dk) * exp(me * dk);
                        dK1[i*kr+j] = f1 * K[i*kr+j];
                     }
                  }
#ifdef NFFT4GP_USING_OPENMP
               }
               else
               {
                  for(i = 0 ; i < kr ; i ++)
                  {
                     dK1[i*kr+i] = f1 * (f2 + noise_level);
                     dK3[i*kr+i] = f2;
                     K[i*kr+i] = f2 + noise_level;
                     for(j = i+1 ; j < kr ; j ++)
                     {
                        dk = sqrt(K[i*kr+j]);
                        dK2[i*kr+j] = f2e3 * dk * dk * exp(me * dk);
                        K[i*kr+j] = f2 * (1.0 - me * dk) * exp(me * dk);
                        dK1[i*kr+j] = f1 * K[i*kr+j];
                     }
                  }
               }
#endif
            }/* End of loop for dKp and Kp */
            else
            {
               /* In this case dKp only
                * note that what stored in K is D2 in this function
                * but me is negative
                */
#ifdef NFFT4GP_USING_OPENMP
               if(!omp_in_parallel())
               {
                  #pragma omp parallel for private(i,j,dij,dk) schedule(dynamic)
#endif
                  for(i = 0 ; i < kr ; i ++)
                  {
                     dK1[i*kr+i] = f1 * (f2 + noise_level);
                     dK3[i*kr+i] = f2;
                     for(j = i+1 ; j < kr ; j ++)
                     {
                        dk = sqrt(K[i*kr+j]);
                        dK2[i*kr+j] = f2e3 * dk * dk * exp(me * dk);
                        dK1[i*kr+j] = f12 * (1.0 - me * dk) * exp(me * dk);
                     }
                  }
#ifdef NFFT4GP_USING_OPENMP
               }
               else
               {
                  for(i = 0 ; i < kr ; i ++)
                  {
                     dK1[i*kr+i] = f1 * (f2 + noise_level);
                     dK3[i*kr+i] = f2;
                     for(j = i+1 ; j < kr ; j ++)
                     {
                        dk = sqrt(K[i*kr+j]);
                        dK2[i*kr+j] = f2e3 * dk * dk * exp(me * dk);
                        dK1[i*kr+j] = f12 * (1.0 - me * dk) * exp(me * dk);
                     }
                  }
               }
#endif
            }/* End of loop for dKp only */
         }/* End of with dKp */
         else
         {
            /* In this case K only
             * note that what stored in K is D2 in this function
             * but me is negative
             */
#ifdef NFFT4GP_USING_OPENMP
            if(!omp_in_parallel())
            {
               #pragma omp parallel for private(i,j,dij,dk) schedule(dynamic)
#endif
               for(i = 0 ; i < kr ; i ++)
               {
                  K[i*kr+i] = f2 + noise_level;
                  for(j = i+1 ; j < kr ; j ++)
                  {
                     dk = sqrt(K[i*kr+j]);
                     K[i*kr+j] = f2 * (1.0 - me * dk) * exp(me * dk);
                  }
               }
#ifdef NFFT4GP_USING_OPENMP
            }
            else
            {
               for(i = 0 ; i < kr ; i ++)
               {
                  K[i*kr+i] = f2 + noise_level;
                  for(j = i+1 ; j < kr ; j ++)
                  {
                     dk = sqrt(K[i*kr+j]);
                     K[i*kr+j] = f2 * (1.0 - me * dk) * exp(me * dk);
                  }
               }
            }
#endif
         }/* end of Kp only */
      }/* End of the second case K(permr, permr) */
   }
   else
   {
      /* Case III: The entire matrix K
       * In this case the matrix is square, only 
       * the LOWER part is generated
       */
#ifdef NFFT4GP_USING_OPENMP
      if(!omp_in_parallel())
      {
#endif
         /* This is the sequential version */
         if(pstr->_ldwork < (size_t)n)
         {
            printf("Nfft4GPKernelMatern32Kernel buffer size is not enough. Check your call to Nfft4GPKernelParamCreate.\n");
            printf("required buffer size is %d, but only %ld is provided.\n", n, pstr->_ldwork);
            return -1;
         }

         if(dKp)
         {
            // NOTE: num grads is hand-written 
            // to be 3, hopefully we do not need to modify this
            if(*dKp == NULL)
            {
               NFFT4GP_MALLOC(dK, 3*(size_t)n*n, NFFT4GP_DOUBLE);
            }
            else
            {
               dK = *dKp;
               // fill to zero to avoid error
               Nfft4GPVecFill(dK, (size_t)3*n*n, 0.0);
            }
         }
         else
         {
            dK = NULL;
         }

         if(Kp)
         {
            if(*Kp == NULL)
            {
               NFFT4GP_MALLOC(K, (size_t)n*n, NFFT4GP_DOUBLE);
            }
            else
            {
               K = *Kp;
               Nfft4GPVecFill(K, (size_t)n*n, 0.0);
            }
         }
         else
         {
            /* We do something different and set K to NULL */
            K = NULL;
         }

         /* 1: K = XX = 2.0*X*X'
          * 2: XX = sum(X.^2,2)
          * 3: build kernel
          */
         if(K)
         {
            Nfft4GPDistanceEuclidXY(data, data, ldim, ldim, n, n, d, &K);
         }
         else
         {
            Nfft4GPDistanceEuclidXY(data, data, ldim, ldim, n, n, d, &dK);
         }
         Nfft4GPDistanceEuclidSumXX(data, ldim, n, d, &(pstr->_dwork));
         Nfft4GPKernelMaternPlusExpDiag( pstr->_dwork, n, K, dK, ff, pstr->_params[1], noise_level);
#ifdef NFFT4GP_USING_OPENMP
      }
      else
      {
         // This is in a OpenMP section, be careful
         int nthreads = omp_get_max_threads();
         if(pstr->_ldwork < (size_t)n*nthreads)
         {
            printf("Nfft4GPKernelMatern32Kernel buffer size is not enough. Check your call to Nfft4GPKernelParamCreate.\n");
            printf("required buffer size is %d, but only %ld is provided.\n", n*nthreads, pstr->_ldwork);
            return -1;
         }

         if(dKp)
         {
            // NOTE: num grads is hand-written 
            // to be 3, hopefully we do not need to modify this
            if(*dKp == NULL)
            {
               NFFT4GP_MALLOC(dK, 3*(size_t)n*n, NFFT4GP_DOUBLE);
            }
            else
            {
               dK = *dKp;
               // fill to zero to avoid error
               Nfft4GPVecFill(dK, (size_t)3*n*n, 0.0);
            }
         }
         else
         {
            dK = NULL;
         }

         if(Kp)
         {
            if(*Kp == NULL)
            {
               NFFT4GP_MALLOC(K, (size_t)n*n, NFFT4GP_DOUBLE);
            }
            else
            {
               K = *Kp;
               Nfft4GPVecFill(K, (size_t)n*n, 0.0);
            }
         }
         else
         {
            /* We do something different and set K to NULL */
            K = NULL;
         }

         if(*Kp == NULL)
         {
            NFFT4GP_MALLOC(K, (size_t)n*n, NFFT4GP_DOUBLE);
         }
         else
         {
            K = *Kp;
         }

         NFFT4GP_DOUBLE *dwork = pstr->_dwork + (size_t)pstr->_max_n*omp_get_thread_num();

         /* 1: K = XX = 2.0*X*X'
          * 2: XX = sum(X.^2,2)
          * 3: build kernel
          */
         if(K)
         {
            Nfft4GPDistanceEuclidXY(data, data, ldim, ldim, n, n, d, &K);
         }
         else
         {
            Nfft4GPDistanceEuclidXY(data, data, ldim, ldim, n, n, d, &dK);
         }
         Nfft4GPDistanceEuclidSumXX(data, ldim, n, d, &dwork);
         Nfft4GPKernelMaternPlusExpDiag( dwork, n, K, dK, ff, pstr->_params[1], noise_level);

      }
#endif

   }/* End of the third case the entire K */
   
   if(Kp && *Kp == NULL)
   {
      *Kp = K;
   }
   if(dKp && *dKp == NULL)
   {
      *dKp = dK;
   }
   return 0;
}

NFFT4GP_DOUBLE Nfft4GPKernelMatern32KernelVal2Dist(void *str, NFFT4GP_DOUBLE tol)
{
   // create problem K(x,y) = f^2 * (1+sqrt(3)*kk*||x-y||)exp(-sqrt(3)*kk*||x-y||)
   // tol < f^2 * (1+e*d)exp(-e*d) => -log(tol/(1+e)/f^2)/e < d

   pnfft4gp_kernel pstr = (pnfft4gp_kernel) str;
   NFFT4GP_DOUBLE f2 = pstr->_params[0] * pstr->_params[0];
   NFFT4GP_DOUBLE e = -NFFT4GP_MATERN32_SQRT3*pstr->_params[1];
   return -log(tol/(1.0+e)/f2)/e;
   
}

/* Matern12 kernel */

/**
 * @brief Given the matrix 2*X*X' and sum(X.^2, 2), compute the Gaussian kernel matrix.
 * @details Given the matrix 2*X*X' and sum(X.^2, 2), compute the Gaussian kernel matrix.
 * @param[in]     XX 2*X*X'.
 * @param[in]     nX Number of rows of XX.
 * @param[in,out] K Kernel matrix, when input should be the matrix 2*X*X'. If NULL, K will not be computed \n
 *                and K should be stored in 
 * @param[in,out] dK Gradient of the kernel matrix. If NULL, dK will not be computed
 * @param[in]     ff scale factor f.
 * @param[in]     kk lengthscale l.
 * @param[in]     noise Noise level.
 */
void Nfft4GPKernelMatern12PlusExpDiag( NFFT4GP_DOUBLE *XX, int nX, NFFT4GP_DOUBLE *K, NFFT4GP_DOUBLE *dK, NFFT4GP_DOUBLE ff, NFFT4GP_DOUBLE kk, NFFT4GP_DOUBLE noise)
{
   size_t i, j;
   NFFT4GP_DOUBLE *dK1, *dK2, *dK3, *dK1_j, *dK2_j, *dK3_j;
   NFFT4GP_DOUBLE *K_j;
   NFFT4GP_DOUBLE kk2 = kk*kk;
   NFFT4GP_DOUBLE dk, f1 = 2.0/ff, f2 = ff*ff, f2k2 = f2/kk2, f12 = 2.0 * ff;
   // noise is already scaled by f2
   
   if(dK)
   {
      /* set pointers for dK1, dK2, and dK3 */
      dK1 = dK;
      dK2 = dK1 + nX*nX;
      dK3 = dK2 + nX*nX;
      if(K)
      {
         /* the first case, compute both dK and K */
         for(j = 0 ; j < nX ; j ++)
         {
            K_j = K + j * nX;
            dK1_j = dK1 + j * nX;
            dK2_j = dK2 + j * nX;
            dK3_j = dK3 + j * nX;
#ifdef NFFT4GP_USING_OPENMP
            if(!omp_in_parallel())
            {
               #pragma omp parallel for private(i,dk) schedule(static)
#endif
               for(i = 0 ; i < nX ; i ++)
               {
                  K_j[i] -= XX[i] + XX[j];
                  K_j[i] = K_j[i] > 0.0 ? 0.0 : K_j[i];
                  K_j[i] = sqrt(-K_j[i]);
                  dk = exp(-K_j[i]/kk);
                  dK3_j[i] = 0.0;
                  dK2_j[i] = f2k2 * K_j[i] * dk;
                  K_j[i] = f2 * dk;
                  dK1_j[i] = f1 * K_j[i];
               }
#ifdef NFFT4GP_USING_OPENMP
            }
            else
            {
               for(i = 0 ; i < nX ; i ++)
               {
                  K_j[i] -= XX[i] + XX[j];
                  K_j[i] = K_j[i] > 0.0 ? 0.0 : K_j[i];
                  K_j[i] = sqrt(-K_j[i]);
                  dk = exp(-K_j[i]/kk);
                  dK3_j[i] = 0.0;
                  dK2_j[i] = f2k2 * K_j[i] * dk;
                  K_j[i] = f2 * dk;
                  dK1_j[i] = f1 * K_j[i];
               }
            }
#endif
         }

#ifdef NFFT4GP_USING_OPENMP
         if(!omp_in_parallel())
         {
            #pragma omp parallel for private(i) schedule(static)
#endif
            for(i = 0 ; i < nX ; i ++)
            {
               K[i*nX+i] = f2 + noise;
               dK1[i*nX+i] = f1 * (f2 + noise);
               dK2[i*nX+i] = 0.0;
               dK3[i*nX+i] = f2;
            }
#ifdef NFFT4GP_USING_OPENMP
         }
         else
         {
            for(i = 0 ; i < nX ; i ++)
            {
               K[i*nX+i] = f2 + noise;
               dK1[i*nX+i] = f1 * (f2 + noise);
               dK2[i*nX+i] = 0.0;
               dK3[i*nX+i] = f2;
            }
         }
#endif
      }/* End of K and dK */
      else
      {
         /* the second case, only compute dK */
         for(j = 0 ; j < nX ; j ++)
         {
            dK1_j = dK1 + j * nX;
            dK2_j = dK2 + j * nX;
            dK3_j = dK3 + j * nX;
#ifdef NFFT4GP_USING_OPENMP
            if(!omp_in_parallel())
            {
               #pragma omp parallel for private(i,dk) schedule(static)
#endif
               for(i = 0 ; i < nX ; i ++)
               {  
                  dK1_j[i] -= XX[i] + XX[j];
                  dK1_j[i] = dK1_j[i] > 0.0 ? 0.0 : dK1_j[i];
                  dK1_j[i] = sqrt(-dK1_j[i]);
                  dk = exp(-dK1_j[i]/kk);
                  dK3_j[i] = 0.0;
                  dK2_j[i] = f2k2 * dK1_j[i] * dk;
                  dK1_j[i] = f12 * dk;
               }
#ifdef NFFT4GP_USING_OPENMP
            }
            else
            {
               for(i = 0 ; i < nX ; i ++)
               {
                  dK1_j[i] -= XX[i] + XX[j];
                  dK1_j[i] = dK1_j[i] > 0.0 ? 0.0 : dK1_j[i];
                  dK1_j[i] = sqrt(-dK1_j[i]);
                  dk = exp(-dK1_j[i]/kk);
                  dK3_j[i] = 0.0;
                  dK2_j[i] = f2k2 * dK1_j[i] * dk;
                  dK1_j[i] = f12 * dk;
               }
            }
#endif
         }
         
#ifdef NFFT4GP_USING_OPENMP
         if(!omp_in_parallel())
         {
            #pragma omp parallel for private(i) schedule(static)
#endif
            for(i = 0 ; i < nX ; i ++)
            {
               dK1[i*nX+i] = f1 * (f2 + noise);
               dK2[i*nX+i] = 0.0;
               dK3[i*nX+i] = f2;
            }
#ifdef NFFT4GP_USING_OPENMP
         }
         else
         {
            for(i = 0 ; i < nX ; i ++)
            {
               dK1[i*nX+i] = f1 * (f2 + noise);
               dK2[i*nX+i] = 0.0;
               dK3[i*nX+i] = f2;
            }
         }
#endif
      }/* End of dK only */
   }
   else
   {
      /* the third case, only compute K */
      for(j = 0 ; j < nX ; j ++)
      {
         K_j = K + j * nX;
#ifdef NFFT4GP_USING_OPENMP
         if(!omp_in_parallel())
         {
            #pragma omp parallel for private(i,dk) schedule(static)
#endif
            for(i = 0 ; i < nX ; i ++)
            {
               K_j[i] -= XX[i] + XX[j];
               K_j[i] = K_j[i] > 0.0 ? 0.0 : K_j[i];
               K_j[i] = sqrt(-K_j[i]);
               K_j[i] = f2 * exp(-K_j[i] / kk);
            }
#ifdef NFFT4GP_USING_OPENMP
         }
         else
         {
            for(i = 0 ; i < nX ; i ++)
            {
               K_j[i] -= XX[i] + XX[j];
               dK1_j[i] = dK1_j[i] > 0.0 ? 0.0 : dK1_j[i];
               K_j[i] = sqrt(-K_j[i]);
               K_j[i] = f2 * exp(-K_j[i] / kk);
            }
         }
#endif
      }
#ifdef NFFT4GP_USING_OPENMP
      if(!omp_in_parallel())
      {
         #pragma omp parallel for private(i) schedule(static)
#endif
         for(i = 0 ; i < nX ; i ++)
         {
            K[i*nX+i] = f2 + noise;
         }
#ifdef NFFT4GP_USING_OPENMP
      }
      else
      {
         for(i = 0 ; i < nX ; i ++)
         {
            K[i*nX+i] = f2 + noise;
         }
      }
#endif
   }/* End of K only */
}

int Nfft4GPKernelMatern12Kernel(void *str, NFFT4GP_DOUBLE *data, int n, int ldim, int d, int *permr, int kr, int *permc, int kc, NFFT4GP_DOUBLE **Kp, NFFT4GP_DOUBLE **dKp)
{
   /* return immediately if no output is needed */
   if(Kp == NULL && dKp == NULL)
   {
      return 0;
   }

   size_t i, j, k, ii, jj;

   pnfft4gp_kernel pstr = (pnfft4gp_kernel) str;
   NFFT4GP_DOUBLE ff = pstr->_params[0]; // scale
   NFFT4GP_DOUBLE kk = pstr->_params[1];
   NFFT4GP_DOUBLE kk2 = kk * kk;
   NFFT4GP_DOUBLE f1 = 2.0/ff, f2 = ff*ff, f2k2 = f2/kk2;
   NFFT4GP_DOUBLE noise_level = pstr->_noise_level * f2;
   NFFT4GP_DOUBLE *K = NULL, *dK = NULL, *dK1 = NULL, *dK2 = NULL, *dK3 = NULL, dij, dk;

   if(permr)
   {
      if(permc)
      {
         /* Case I: part of a matrix K(permr, permc)
          * In this case the matrix is rectangular
          */
         if(dKp)
         {
            // NOTE: num grads is hand-written 
            // to be 3, hopefully we do not need to modify this
            if(*dKp == NULL)
            {
               NFFT4GP_CALLOC(dK, 3*(size_t)kr*kc, NFFT4GP_DOUBLE);
            }
            else
            {
               dK = *dKp;
               // fill to zero to avoid error
               Nfft4GPVecFill(dK, (size_t)3*kr*kc, 0.0);
            }
            /* Set pointers to each matrix */
            dK1 = dK;
            dK2 = dK1 + kr*kc;
            dK3 = dK2 + kr*kc;
         }

         if(Kp == NULL)
         {
            /* In this case, Kp is empty
             * we should store the distance matrix in
             * dKp
             */
            K = dK;
         }
         else
         {
            /* In this case, Kp is not empty
             * we can safely store the distance in Kp
             */
            if(*Kp == NULL)
            {
               NFFT4GP_CALLOC(K, (size_t)kr*kc, NFFT4GP_DOUBLE);
            }
            else
            {
               K = *Kp;
               Nfft4GPVecFill(K, (size_t)kr*kc, 0.0);
            }
         }

         /* Compute Distance */
         for(k = 0 ; k < d ; k ++)
         {
#ifdef NFFT4GP_USING_OPENMP
            if(!omp_in_parallel())
            {
               #pragma omp parallel for private(ii,jj,i,j,dij) schedule(dynamic)
#endif
               for(ii = 0 ; ii < kc ; ii ++)
               {
                  i = permc[ii];
                  for(jj = 0 ; jj < kr ; jj ++)
                  {
                     j = permr[jj];
                     /* the value of K(jj, ii) is based on the distance between i and j */
                     if(i != j)
                     {
                        /* get the distance between the points with the permutated indeces */
                        dij = data[k*ldim+i] - data[k*ldim+j];
                        K[ii*kr+jj] += dij*dij;
                     }
                  }
               }
#ifdef NFFT4GP_USING_OPENMP
            }
            else
            {
               for(ii = 0 ; ii < kc ; ii ++)
               {
                  i = permc[ii];
                  for(jj = 0 ; jj < kr ; jj ++)
                  {
                     j = permr[jj];
                     /* the value of K(jj, ii) is based on the distance between i and j */
                     if(i != j)
                     {
                        /* get the distance between the points with the permutated indeces */
                        dij = data[k*ldim+i] - data[k*ldim+j];
                        K[ii*kr+jj] += dij*dij;
                     }
                  }
               }
            }
#endif
         }

         /* form the matrix */
         if(dKp)
         {
            /* With gradient, form gradient first 
             * form dK1 last since the distance might
             * be stored in it
             */

            /* we check Kp in the front
             * write duplicate code to avoid
             * if statement in the loop
             */

            if(Kp)
            {
               /* In this case both dKp and Kp
                * note that what stored in K is D2 in this function
                * but me is negative
                */
#ifdef NFFT4GP_USING_OPENMP
               if(!omp_in_parallel())
               {
                  #pragma omp parallel for private(ii,jj,i,j,dij,dk) schedule(dynamic)
#endif
                  for(ii = 0 ; ii < kc ; ii ++)
                  {
                     i = permc[ii];
                     for(jj = 0 ; jj < kr ; jj ++)
                     {
                        j = permr[jj];
                        /* when have permc we do not add noise */
                        K[ii*kr+jj] = sqrt(K[ii*kr+jj]);
                        dk = exp(-K[ii*kr+jj]/kk);
                        dK2[ii*kr+jj] = f2k2 * K[ii*kr+jj] * dk;
                        K[ii*kr+jj] = f2 * dk;
                        dK1[ii*kr+jj] = f1 * K[ii*kr+jj];
                     }
                  }
#ifdef NFFT4GP_USING_OPENMP
               }
               else
               {
                  for(ii = 0 ; ii < kc ; ii ++)
                  {
                     i = permc[ii];
                     for(jj = 0 ; jj < kr ; jj ++)
                     {
                        j = permr[jj];
                        /* when have permc we do not add noise */
                        dk = exp(-K[ii*kr+jj]/kk);
                        dK2[ii*kr+jj] = f2k2 * K[ii*kr+jj] * dk;
                        K[ii*kr+jj] = f2 * dk;
                        dK1[ii*kr+jj] = f1 * K[ii*kr+jj];
                     }
                  }
               }
#endif
            }/* End of loop for dKp and Kp */
            else
            {
               /* In this case dKp only
                * note that what stored in K is D2 in this function
                * but me is negative
                */
#ifdef NFFT4GP_USING_OPENMP
               if(!omp_in_parallel())
               {
                  #pragma omp parallel for private(ii,jj,i,j,dij,dk) schedule(dynamic)
#endif
                  for(ii = 0 ; ii < kc ; ii ++)
                  {
                     i = permc[ii];
                     for(jj = 0 ; jj < kr ; jj ++)
                     {
                        j = permr[jj];
                        /* when have permc we do not add noise */
                        K[ii*kr+jj] = sqrt(K[ii*kr+jj]);
                        dk = exp(-K[ii*kr+jj]/kk);
                        dK2[ii*kr+jj] = f2k2 * K[ii*kr+jj] * dk;
                        dK1[ii*kr+jj] = f1 * f2 * dk;
                     }
                  }
#ifdef NFFT4GP_USING_OPENMP
               }
               else
               {
                  for(ii = 0 ; ii < kc ; ii ++)
                  {
                     i = permc[ii];
                     for(jj = 0 ; jj < kr ; jj ++)
                     {
                        j = permr[jj];
                        /* when have permc we do not add noise */
                        K[ii*kr+jj] = sqrt(K[ii*kr+jj]);
                        dk = exp(-K[ii*kr+jj]/kk);
                        dK2[ii*kr+jj] = f2k2 * K[ii*kr+jj] * dk;
                        dK1[ii*kr+jj] = f1 * f2 * dk;
                     }
                  }
               }
#endif
            }/* End of loop for dKp only */
         }/* End of with dKp */
         else
         {
            /* In this case K only
             * note that what stored in K is D2 in this function
             * but me is negative
             */
#ifdef NFFT4GP_USING_OPENMP
            if(!omp_in_parallel())
            {
               #pragma omp parallel for private(ii,jj,i,j,dij,dk) schedule(dynamic)
#endif
               for(ii = 0 ; ii < kc ; ii ++)
               {
                  i = permc[ii];
                  for(jj = 0 ; jj < kr ; jj ++)
                  {
                     j = permr[jj];
                     /* when have permc we do not add noise */
                     K[ii*kr+jj] = f2 * exp(-sqrt(K[ii*kr+jj])/kk);
                  }
               }
#ifdef NFFT4GP_USING_OPENMP
            }
            else
            {
               for(ii = 0 ; ii < kc ; ii ++)
               {
                  i = permc[ii];
                  for(jj = 0 ; jj < kr ; jj ++)
                  {
                     j = permr[jj];
                     /* when have permc we do not add noise */
                     K[ii*kr+jj] = f2 * exp(-sqrt(K[ii*kr+jj])/kk);
                  }
               }
            }
#endif
         }/* end of Kp only */
      }/* End of the first case K(permr, permc) */
      else
      {
         /* Case II: part of a matrix K(permr, permr)
          * In this case the matrix is square, only 
          * the LOWER part is generated
          */
         if(dKp)
         {
            // NOTE: num grads is hand-written 
            // to be 3, hopefully we do not need to modify this
            if(*dKp == NULL)
            {
               NFFT4GP_CALLOC(dK, 3*(size_t)kr*kr, NFFT4GP_DOUBLE);
            }
            else
            {
               dK = *dKp;
               // fill to zero to avoid error
               Nfft4GPVecFill(dK, (size_t)3*kr*kr, 0.0);
            }
            /* Set pointers to each matrix */
            dK1 = dK;
            dK2 = dK1 + kr*kr;
            dK3 = dK2 + kr*kr;
         }

         if(Kp == NULL)
         {
            /* In this case, Kp is empty
             * we should store the distance matrix in
             * dKp
             */
            K = dK;
         }
         else
         {
            /* In this case, Kp is not empty
             * we can safely store the distance in Kp
             */
            if(*Kp == NULL)
            {
               NFFT4GP_CALLOC(K, (size_t)kr*kr, NFFT4GP_DOUBLE);
            }
            else
            {
               K = *Kp;
               Nfft4GPVecFill(K, (size_t)kr*kr, 0.0);
            }
         }

         /* Compute Distance 
          * The problem is symmetric, only fill the LOWER part!
          */
         for(k = 0 ; k < d ; k ++)
         {
            /* The parallel implementation */
#ifdef NFFT4GP_USING_OPENMP
            if(!omp_in_parallel())
            {
               #pragma omp parallel for private(i,j,dij) schedule(dynamic)
#endif
               for(i = 0 ; i < kr ; i ++)
               {
                  for(j = i+1 ; j < kr ; j ++)
                  {
                     dij = data[k*ldim+permr[i]] - data[k*ldim+permr[j]];
                     K[i*kr+j] += dij*dij;
                  }
               }
#ifdef NFFT4GP_USING_OPENMP
            }
            else
            {
               /* The sequential imlementation */
               for(i = 0 ; i < kr ; i ++)
               {
                  for(j = i+1 ; j < kr ; j ++)
                  {
                     dij = data[k*ldim+permr[i]] - data[k*ldim+permr[j]];
                     K[i*kr+j] += dij*dij;
                  }
               }
            }
#endif
         }

         /* form the matrix */
         if(dKp)
         {
            /* With gradient, form gradient first 
             * form dK1 last since the distance might
             * be stored in it
             */

            /* we check Kp in the front
             * write duplicate code to avoid
             * if statement in the loop
             */

            if(Kp)
            {
               /* In this case both dKp and Kp
                * note that what stored in K is D2 in this function
                * but me is negative
                */
#ifdef NFFT4GP_USING_OPENMP
               if(!omp_in_parallel())
               {
                  #pragma omp parallel for private(i,j,dij,dk) schedule(dynamic)
#endif
                  for(i = 0 ; i < kr ; i ++)
                  {
                     dK1[i*kr+i] = f1 * (f2 + noise_level);
                     dK3[i*kr+i] = f2;
                     K[i*kr+i] = f2 + noise_level;
                     for(j = i+1 ; j < kr ; j ++)
                     {
                        K[i*kr+j] = sqrt(K[i*kr+j]);
                        dk = exp(-K[i*kr+j]/kk);
                        dK2[i*kr+j] = f2k2 * K[i*kr+j] * dk;
                        K[i*kr+j] = f2 * dk;
                        dK1[i*kr+j] = f1 * K[i*kr+j];
                     }
                  }
#ifdef NFFT4GP_USING_OPENMP
               }
               else
               {
                  for(i = 0 ; i < kr ; i ++)
                  {
                     dK1[i*kr+i] = f1 * (f2 + noise_level);
                     dK3[i*kr+i] = f2;
                     K[i*kr+i] = f2 + noise_level;
                     for(j = i+1 ; j < kr ; j ++)
                     {
                        K[i*kr+j] = sqrt(K[i*kr+j]);
                        dk = exp(-K[i*kr+j]/kk);
                        dK2[i*kr+j] = f2k2 * K[i*kr+j] * dk;
                        K[i*kr+j] = f2 * dk;
                        dK1[i*kr+j] = f1 * K[i*kr+j];
                     }
                  }
               }
#endif
            }/* End of loop for dKp and Kp */
            else
            {
               /* In this case dKp only
                * note that what stored in K is D2 in this function
                * but me is negative
                */
#ifdef NFFT4GP_USING_OPENMP
               if(!omp_in_parallel())
               {
                  #pragma omp parallel for private(i,j,dij,dk) schedule(dynamic)
#endif
                  for(i = 0 ; i < kr ; i ++)
                  {
                     dK1[i*kr+i] = f1 * (f2 + noise_level);
                     dK3[i*kr+i] = f2;
                     for(j = i+1 ; j < kr ; j ++)
                     {
                        K[i*kr+j] = sqrt(K[i*kr+j]);
                        dk = exp(-K[i*kr+j]/kk);
                        dK2[i*kr+j] = f2k2 * K[i*kr+j] * dk;
                        dK1[i*kr+j] = f1 * f2 * dk;
                     }
                  }
#ifdef NFFT4GP_USING_OPENMP
               }
               else
               {
                  for(i = 0 ; i < kr ; i ++)
                  {
                     dK1[i*kr+i] = f1 * (f2 + noise_level);
                     dK3[i*kr+i] = f2;
                     for(j = i+1 ; j < kr ; j ++)
                     {
                        K[i*kr+j] = sqrt(K[i*kr+j]);
                        dk = exp(-K[i*kr+j]/kk);
                        dK2[i*kr+j] = f2k2 * K[i*kr+j] * dk;
                        dK1[i*kr+j] = f1 * f2 * dk;
                     }
                  }
               }
#endif
            }/* End of loop for dKp only */
         }/* End of with dKp */
         else
         {
            /* In this case K only
             * note that what stored in K is D2 in this function
             * but me is negative
             */
#ifdef NFFT4GP_USING_OPENMP
            if(!omp_in_parallel())
            {
               #pragma omp parallel for private(i,j,dij,dk) schedule(dynamic)
#endif
               for(i = 0 ; i < kr ; i ++)
               {
                  K[i*kr+i] = f2 + noise_level;
                  for(j = i+1 ; j < kr ; j ++)
                  {
                     K[i*kr+j] = f2 * exp(-sqrt(K[i*kr+j])/kk);
                  }
               }
#ifdef NFFT4GP_USING_OPENMP
            }
            else
            {
               for(i = 0 ; i < kr ; i ++)
               {
                  K[i*kr+i] = f2 + noise_level;
                  for(j = i+1 ; j < kr ; j ++)
                  {
                     K[i*kr+j] = f2 * exp(-sqrt(K[i*kr+j])/kk);
                  }
               }
            }
#endif
         }/* end of Kp only */
      }/* End of the second case K(permr, permr) */
   }
   else
   {
      /* Case III: The entire matrix K
       * In this case the matrix is square, only 
       * the LOWER part is generated
       */
#ifdef NFFT4GP_USING_OPENMP
      if(!omp_in_parallel())
      {
#endif
         /* This is the sequential version */
         if(pstr->_ldwork < (size_t)n)
         {
            printf("Nfft4GPKernelMatern12Kernel buffer size is not enough. Check your call to Nfft4GPKernelParamCreate.\n");
            printf("required buffer size is %d, but only %ld is provided.\n", n, pstr->_ldwork);
            return -1;
         }

         if(dKp)
         {
            // NOTE: num grads is hand-written 
            // to be 3, hopefully we do not need to modify this
            if(*dKp == NULL)
            {
               NFFT4GP_MALLOC(dK, 3*(size_t)n*n, NFFT4GP_DOUBLE);
            }
            else
            {
               dK = *dKp;
               // fill to zero to avoid error
               Nfft4GPVecFill(dK, (size_t)3*n*n, 0.0);
            }
         }
         else
         {
            dK = NULL;
         }

         if(Kp)
         {
            if(*Kp == NULL)
            {
               NFFT4GP_MALLOC(K, (size_t)n*n, NFFT4GP_DOUBLE);
            }
            else
            {
               K = *Kp;
               Nfft4GPVecFill(K, (size_t)n*n, 0.0);
            }
         }
         else
         {
            /* We do something different and set K to NULL */
            K = NULL;
         }

         /* 1: K = XX = 2.0*X*X'
          * 2: XX = sum(X.^2,2)
          * 3: build kernel
          */
         if(K)
         {
            Nfft4GPDistanceEuclidXY(data, data, ldim, ldim, n, n, d, &K);
         }
         else
         {
            Nfft4GPDistanceEuclidXY(data, data, ldim, ldim, n, n, d, &dK);
         }
         Nfft4GPDistanceEuclidSumXX(data, ldim, n, d, &(pstr->_dwork));
         Nfft4GPKernelMatern12PlusExpDiag( pstr->_dwork, n, K, dK, ff, pstr->_params[1], noise_level);
#ifdef NFFT4GP_USING_OPENMP
      }
      else
      {
         // This is in a OpenMP section, be careful
         int nthreads = omp_get_max_threads();
         if(pstr->_ldwork < (size_t)n*nthreads)
         {
            printf("Nfft4GPKernelMatern32Kernel buffer size is not enough. Check your call to Nfft4GPKernelParamCreate.\n");
            printf("required buffer size is %d, but only %ld is provided.\n", n*nthreads, pstr->_ldwork);
            return -1;
         }

         if(dKp)
         {
            // NOTE: num grads is hand-written 
            // to be 3, hopefully we do not need to modify this
            if(*dKp == NULL)
            {
               NFFT4GP_MALLOC(dK, 3*(size_t)n*n, NFFT4GP_DOUBLE);
            }
            else
            {
               dK = *dKp;
               // fill to zero to avoid error
               Nfft4GPVecFill(dK, (size_t)3*n*n, 0.0);
            }
         }
         else
         {
            dK = NULL;
         }

         if(Kp)
         {
            if(*Kp == NULL)
            {
               NFFT4GP_MALLOC(K, (size_t)n*n, NFFT4GP_DOUBLE);
            }
            else
            {
               K = *Kp;
               Nfft4GPVecFill(K, (size_t)n*n, 0.0);
            }
         }
         else
         {
            /* We do something different and set K to NULL */
            K = NULL;
         }

         if(*Kp == NULL)
         {
            NFFT4GP_MALLOC(K, (size_t)n*n, NFFT4GP_DOUBLE);
         }
         else
         {
            K = *Kp;
         }

         NFFT4GP_DOUBLE *dwork = pstr->_dwork + (size_t)pstr->_max_n*omp_get_thread_num();

         /* 1: K = XX = 2.0*X*X'
          * 2: XX = sum(X.^2,2)
          * 3: build kernel
          */
         if(K)
         {
            Nfft4GPDistanceEuclidXY(data, data, ldim, ldim, n, n, d, &K);
         }
         else
         {
            Nfft4GPDistanceEuclidXY(data, data, ldim, ldim, n, n, d, &dK);
         }
         Nfft4GPDistanceEuclidSumXX(data, ldim, n, d, &dwork);
         Nfft4GPKernelMatern12PlusExpDiag( dwork, n, K, dK, ff, pstr->_params[1], noise_level);

      }
#endif

   }/* End of the third case the entire K */
   
   if(Kp && *Kp == NULL)
   {
      *Kp = K;
   }
   if(dKp && *dKp == NULL)
   {
      *dKp = dK;
   }
   return 0;
}

NFFT4GP_DOUBLE Nfft4GPKernelMatern12KernelVal2Dist(void *str, NFFT4GP_DOUBLE tol)
{
   // create problem K(x,y) = f^2 * exp(-||x-y||/kk)
   // tol < f^2 * exp(-d/kk) => tol/f^2 < exp(-d/kk) => -log(tol/(f^2))*kk < d

   pnfft4gp_kernel pstr = (pnfft4gp_kernel) str;
   NFFT4GP_DOUBLE f2 = pstr->_params[0] * pstr->_params[0];
   NFFT4GP_DOUBLE kk = pstr->_params[1];
   return -log(tol/f2)*kk;
   
}

void* Nfft4GPKernelAdditiveKernelParamCreate(NFFT4GP_DOUBLE *data, int n, int ldim, int d,
                                          int *windows, int nwindows, int dwindows, func_kernel fkernel)
{
   pnfft4gp_kernel pkernel = (pnfft4gp_kernel)Nfft4GPKernelParamCreate((size_t)n, 1);
   
   pkernel->_iparams[0] = nwindows;
   pkernel->_iparams[1] = dwindows;

   int skip_window = 1;
   pkernel->_iparams[2] = 0;
   while(skip_window < nwindows && windows[skip_window*dwindows-1] < 0)
   {
      skip_window++;
      pkernel->_iparams[2]++;
   }

   NFFT4GP_MALLOC( pkernel->_ibufferp, 1, int*);
   pkernel->_ibufferp[0] = windows;

   pkernel->_fkernel_buffer = fkernel;

   pkernel->_fkernel_buffer_params = Nfft4GPKernelParamCreate(n, 1);
   pkernel->_own_fkernel_buffer_params = 1;

   NFFT4GP_MALLOC( pkernel->_buffer, (size_t)n*nwindows*dwindows, NFFT4GP_DOUBLE);
   pkernel->_own_buffer = 1;

   int i, j;
   NFFT4GP_DOUBLE *data_window = pkernel->_buffer;
   int *feature_window = windows;
   for(i = 0 ; i < nwindows ; i ++)
   {
      for(j = 0 ; j < dwindows ; j ++)
      {
         if(*feature_window >= 0)
         {
            //printf("Reading feature %d\n", feature_window[0]);
            NFFT4GP_MEMCPY( data_window, data + feature_window[0]*ldim, n, NFFT4GP_DOUBLE);
            feature_window++;
            data_window += n;
         }
         else
         {
            //printf("Skip window.\n");
         }
      }
   }

   return pkernel;
}

// TODO: currently no dynamic memory allocation
// also the third paramemter needs more work
int Nfft4GPKernelAdditiveKernel(void *str, NFFT4GP_DOUBLE *data, int n, int ldim, int d, int *permr, int kr, int *permc, int kc, NFFT4GP_DOUBLE **Kp, NFFT4GP_DOUBLE **dKp)
{

   pnfft4gp_kernel pkernel = (pnfft4gp_kernel) str;
   pnfft4gp_kernel pkernel_window = (pnfft4gp_kernel) pkernel->_fkernel_buffer_params;
   int nwindows = pkernel->_iparams[0];
   int dwindows = pkernel->_iparams[1];
   int skip_window = pkernel->_iparams[2];
   NFFT4GP_DOUBLE *windows = pkernel->_buffer;
   NFFT4GP_DOUBLE scale_window = 1.0/(NFFT4GP_DOUBLE)nwindows;

   pkernel_window->_params[0] = pkernel->_params[0];
   pkernel_window->_params[1] = pkernel->_params[1];
   pkernel_window->_noise_level = pkernel->_noise_level;

   int i;
   NFFT4GP_DOUBLE *K = NULL, *dK = NULL;
   NFFT4GP_DOUBLE *K_work = NULL, *dK_work = NULL;
   // next, create the kernel matrix for each window
   if(permr)
   {
      if(permc)
      {
         if(dKp)
         {
            if(*dKp == NULL)
            {
               NFFT4GP_CALLOC(dK, 3*(size_t)kr*kc, NFFT4GP_DOUBLE);
            }
            else
            {
               dK = *dKp;
               Nfft4GPVecFill(dK, (size_t)3*kr*kc, 0.0);
            }
         }
         if(Kp == NULL)
         {
            K = dK;
         }
         else
         {
            if(*Kp == NULL)
            {
               NFFT4GP_CALLOC(K, (size_t)kr*kc, NFFT4GP_DOUBLE);
            }
            else
            {
               K = *Kp;
               Nfft4GPVecFill(K, (size_t)kr*kc, 0.0);
            }
         }
         if(dKp)
         {
            if(Kp)
            {
               NFFT4GP_CALLOC(K_work, (size_t)kr*kc, NFFT4GP_DOUBLE);
               NFFT4GP_CALLOC(dK_work, 3*(size_t)kr*kc, NFFT4GP_DOUBLE);
               if(skip_window)
               {
                  for(i = 0 ; i < nwindows-1 ; i ++)
                  {
                     NFFT4GP_DOUBLE *data_window = windows + i*n*dwindows;
                     pkernel->_fkernel_buffer( pkernel->_fkernel_buffer_params, data_window, n, n, dwindows, permr, kr, permc, kc, &K_work, &dK_work);
                     Nfft4GPVecAxpy( 1.0, K_work, (size_t)kr*kc, K);
                     Nfft4GPVecAxpy( 1.0, dK_work, (size_t)3*kr*kc, dK);
                     Nfft4GPVecFill(K_work, (size_t)kr*kc, 0.0);
                     Nfft4GPVecFill(dK_work, (size_t)3*kr*kc, 0.0);
                  }
                  i = nwindows-1;
                  NFFT4GP_DOUBLE *data_window = windows + i*n*dwindows;
                  pkernel->_fkernel_buffer(pkernel->_fkernel_buffer_params, data_window, n, n, dwindows - skip_window, permr, kr, permc, kc, &K_work, NULL);
                  Nfft4GPVecAxpy(1.0, K_work, (size_t)kr*kc, K);
                  Nfft4GPVecAxpy( 1.0, dK_work, (size_t)3*kr*kc, dK);
                  Nfft4GPVecFill(K_work, (size_t)kr*kc, 0.0);
                  Nfft4GPVecFill(dK_work, (size_t)3*kr*kc, 0.0);
               }
               else
               {
                  for(i = 0 ; i < nwindows ; i ++)
                  {
                     NFFT4GP_DOUBLE *data_window = windows + i*n*dwindows;
                     pkernel->_fkernel_buffer( pkernel->_fkernel_buffer_params, data_window, n, n, dwindows, permr, kr, permc, kc, &K_work, &dK_work);
                     Nfft4GPVecAxpy( 1.0, K_work, (size_t)kr*kc, K);
                     Nfft4GPVecAxpy( 1.0, dK_work, (size_t)3*kr*kc, dK);
                     Nfft4GPVecFill(K_work, (size_t)kr*kc, 0.0);
                     Nfft4GPVecFill(dK_work, (size_t)3*kr*kc, 0.0);
                  }
               }
               Nfft4GPVecScale(K, (size_t)kr*kc, scale_window);
               Nfft4GPVecScale(dK, (size_t)3*kr*kc, scale_window);
            }
            else
            {
               NFFT4GP_CALLOC(dK_work, 3*(size_t)kr*kc, NFFT4GP_DOUBLE);
               if(skip_window)
               {
                  for(i = 0 ; i < nwindows - 1 ; i ++)
                  {
                     NFFT4GP_DOUBLE *data_window = windows + i*n*dwindows;
                     pkernel->_fkernel_buffer( pkernel->_fkernel_buffer_params, data_window, n, n, dwindows, permr, kr, permc, kc, NULL, &dK_work);
                     Nfft4GPVecAxpy( 1.0, dK_work, (size_t)3*kr*kc, dK);
                     Nfft4GPVecFill(dK_work, (size_t)3*kr*kc, 0.0);
                  }
                  i = nwindows - 1;
                  NFFT4GP_DOUBLE *data_window = windows + i*n*dwindows;
                  pkernel->_fkernel_buffer( pkernel->_fkernel_buffer_params, data_window, n, n, dwindows - skip_window, permr, kr, permc, kc, NULL, &dK_work);
                  Nfft4GPVecAxpy( 1.0, dK_work, (size_t)3*kr*kc, dK);
                  Nfft4GPVecFill(dK_work, (size_t)3*kr*kc, 0.0);
               }
               else
               {
                  for(i = 0 ; i < nwindows ; i ++)
                  {
                     NFFT4GP_DOUBLE *data_window = windows + i*n*dwindows;
                     pkernel->_fkernel_buffer( pkernel->_fkernel_buffer_params, data_window, n, n, dwindows, permr, kr, permc, kc, NULL, &dK_work);
                     Nfft4GPVecAxpy( 1.0, dK_work, (size_t)3*kr*kc, dK);
                     Nfft4GPVecFill(dK_work, (size_t)3*kr*kc, 0.0);
                  }
               }
               Nfft4GPVecScale(dK, (size_t)3*kr*kc, scale_window);
            }/* End of loop for dKp only */
         }/* End of with dKp */
         else
         {
            NFFT4GP_CALLOC(K_work, (size_t)kr*kc, NFFT4GP_DOUBLE);
            if(skip_window)
            {
               for(i = 0 ; i < nwindows - 1 ; i ++)
               {
                  NFFT4GP_DOUBLE *data_window = windows + i*n*dwindows;
                  pkernel->_fkernel_buffer(pkernel->_fkernel_buffer_params, data_window, n, n, dwindows, permr, kr, permc, kc, &K_work, NULL);
                  Nfft4GPVecAxpy(1.0, K_work, (size_t)kr*kc, K);
                  Nfft4GPVecFill(K_work, (size_t)kr*kc, 0.0);
               }
               i = nwindows - 1;
               NFFT4GP_DOUBLE *data_window = windows + i*n*dwindows;
               pkernel->_fkernel_buffer(pkernel->_fkernel_buffer_params, data_window, n, n, dwindows - skip_window, permr, kr, permc, kc, &K_work, NULL);
               Nfft4GPVecAxpy(1.0, K_work, (size_t)kr*kc, K);
               Nfft4GPVecFill(K_work, (size_t)kr*kc, 0.0);
            }
            else
            {
               for(i = 0 ; i < nwindows ; i ++)
               {
                  NFFT4GP_DOUBLE *data_window = windows + i*n*dwindows;
                  pkernel->_fkernel_buffer( pkernel->_fkernel_buffer_params, data_window, n, n, dwindows, permr, kr, permc, kc, &K_work, NULL);
                  Nfft4GPVecAxpy( 1.0, K_work, (size_t)kr*kc, K);
                  Nfft4GPVecFill(K_work, (size_t)kr*kc, 0.0);
               }
            }
            Nfft4GPVecScale(K, (size_t)kr*kc, scale_window);
         }/* end of Kp only */
      }/* End of the first case K(permr, permc) */
      else
      {
         if(dKp)
         {
            if(*dKp == NULL)
            {
               NFFT4GP_CALLOC(dK, 3*(size_t)kr*kr, NFFT4GP_DOUBLE);
            }
            else
            {
               dK = *dKp;
               Nfft4GPVecFill(dK, (size_t)3*kr*kr, 0.0);
            }
         }
         if(Kp == NULL)
         {
            K = dK;
         }
         else
         {
            if(*Kp == NULL)
            {
               NFFT4GP_CALLOC(K, (size_t)kr*kr, NFFT4GP_DOUBLE);
            }
            else
            {
               K = *Kp;
               Nfft4GPVecFill(K, (size_t)kr*kr, 0.0);
            }
         }
         if(dKp)
         {
            if(Kp)
            {
               NFFT4GP_CALLOC(K_work, (size_t)kr*kr, NFFT4GP_DOUBLE);
               NFFT4GP_CALLOC(dK_work, 3*(size_t)kr*kr, NFFT4GP_DOUBLE);
               if(skip_window)
               {
                  for(i = 0 ; i < nwindows - 1 ; i ++)
                  {
                     NFFT4GP_DOUBLE *data_window = windows + i*n*dwindows;
                     pkernel->_fkernel_buffer( pkernel->_fkernel_buffer_params, data_window, n, n, dwindows, permr, kr, permc, kc, &K_work, &dK_work);
                     Nfft4GPVecAxpy( 1.0, K_work, (size_t)kr*kr, K);
                     Nfft4GPVecAxpy( 1.0, dK_work, (size_t)3*kr*kr, dK);
                     Nfft4GPVecFill(K_work, (size_t)kr*kr, 0.0);
                     Nfft4GPVecFill(dK_work, (size_t)3*kr*kr, 0.0);
                  }
                  i = nwindows - 1;
                  NFFT4GP_DOUBLE *data_window = windows + i*n*dwindows;
                  pkernel->_fkernel_buffer(pkernel->_fkernel_buffer_params, data_window, n, n, dwindows - skip_window, permr, kr, permc, kc, &K_work, &dK_work);
                  Nfft4GPVecAxpy(1.0, K_work, (size_t)kr*kr, K);
                  Nfft4GPVecAxpy(1.0, dK_work, (size_t)3*kr*kr, dK);
                  Nfft4GPVecFill(K_work, (size_t)kr*kr, 0.0);
                  Nfft4GPVecFill(dK_work, (size_t)3*kr*kr, 0.0);
               }
               else
               {
                  for(i = 0 ; i < nwindows ; i ++)
                  {
                     NFFT4GP_DOUBLE *data_window = windows + i*n*dwindows;
                     pkernel->_fkernel_buffer( pkernel->_fkernel_buffer_params, data_window, n, n, dwindows, permr, kr, permc, kc, &K_work, &dK_work);
                     Nfft4GPVecAxpy( 1.0, K_work, (size_t)kr*kr, K);
                     Nfft4GPVecAxpy( 1.0, dK_work, (size_t)3*kr*kr, dK);
                     Nfft4GPVecFill(K_work, (size_t)kr*kr, 0.0);
                     Nfft4GPVecFill(dK_work, (size_t)3*kr*kr, 0.0);
                  }
               }
               Nfft4GPVecScale(K, (size_t)kr*kr, scale_window);
               Nfft4GPVecScale(dK, (size_t)3*kr*kr, scale_window);
            }
            else
            {
               NFFT4GP_CALLOC(dK_work, 3*(size_t)kr*kr, NFFT4GP_DOUBLE);
               for(i = 0 ; i < nwindows ; i ++)
               {
                  NFFT4GP_DOUBLE *data_window = windows + i*n*dwindows;
                  pkernel->_fkernel_buffer( pkernel->_fkernel_buffer_params, data_window, n, n, dwindows, permr, kr, permc, kc, NULL, &dK_work);
                  Nfft4GPVecAxpy( 1.0, dK_work, (size_t)3*kr*kr, dK);
                  Nfft4GPVecFill(dK_work, (size_t)3*kr*kr, 0.0);
               }
               Nfft4GPVecScale(dK, (size_t)3*kr*kr, scale_window);
            }/* End of loop for dKp only */
         }/* End of with dKp */
         else
         {
            NFFT4GP_CALLOC(K_work, (size_t)kr*kr, NFFT4GP_DOUBLE);
            for(i = 0 ; i < nwindows ; i ++)
            {
               NFFT4GP_DOUBLE *data_window = windows + i*n*dwindows;
               pkernel->_fkernel_buffer( pkernel->_fkernel_buffer_params, data_window, n, n, dwindows, permr, kr, permc, kc, &K_work, NULL);
               Nfft4GPVecAxpy( 1.0, K_work, (size_t)kr*kr, K);
               Nfft4GPVecFill(K_work, (size_t)kr*kr, 0.0);
            }
            Nfft4GPVecScale(K, (size_t)kr*kr, scale_window);
         }/* end of Kp only */
      }
   }
   else
   {
      if(dKp)
      {
         if(*dKp == NULL)
         {
            NFFT4GP_CALLOC(dK, 3*(size_t)n*n, NFFT4GP_DOUBLE);
         }
         else
         {
            dK = *dKp;
            Nfft4GPVecFill(dK, (size_t)3*n*n, 0.0);
         }
      }
      if(Kp == NULL)
      {
         K = dK;
      }
      else
      {
         if(*Kp == NULL)
         {
            NFFT4GP_CALLOC(K, (size_t)n*n, NFFT4GP_DOUBLE);
         }
         else
         {
            K = *Kp;
            Nfft4GPVecFill(K, (size_t)n*n, 0.0);
         }
      }
      if(dKp)
      {
         if(Kp)
         {
            NFFT4GP_CALLOC(K_work, (size_t)n*n, NFFT4GP_DOUBLE);
            NFFT4GP_CALLOC(dK_work, 3*(size_t)n*n, NFFT4GP_DOUBLE);
            if(skip_window)
            {
               for(i = 0 ; i < nwindows - 1 ; i ++)
               {
                  NFFT4GP_DOUBLE *data_window = windows + i*n*dwindows;
                  pkernel->_fkernel_buffer( pkernel->_fkernel_buffer_params, data_window, n, n, dwindows, permr, kr, permc, kc, &K_work, &dK_work);
                  Nfft4GPVecAxpy( 1.0, K_work, (size_t)n*n, K);
                  Nfft4GPVecAxpy( 1.0, dK_work, (size_t)3*n*n, dK);
                  Nfft4GPVecFill(K_work, (size_t)n*n, 0.0);
                  Nfft4GPVecFill(dK_work, (size_t)3*n*n, 0.0);
               }
               i = nwindows - 1;
               NFFT4GP_DOUBLE *data_window = windows + i*n*dwindows;
               pkernel->_fkernel_buffer(pkernel->_fkernel_buffer_params, data_window, n, n, dwindows - skip_window, permr, kr, permc, kc, &K_work, &dK_work);
               Nfft4GPVecAxpy(1.0, K_work, (size_t)n*n, K);
               Nfft4GPVecAxpy(1.0, dK_work, (size_t)3*n*n, dK);
               Nfft4GPVecFill(K_work, (size_t)n*n, 0.0);
               Nfft4GPVecFill(dK_work, (size_t)3*n*n, 0.0);
            }
            else
            {
               for(i = 0 ; i < nwindows ; i ++)
               {
                  NFFT4GP_DOUBLE *data_window = windows + i*n*dwindows;
                  pkernel->_fkernel_buffer( pkernel->_fkernel_buffer_params, data_window, n, n, dwindows, permr, kr, permc, kc, &K_work, &dK_work);
                  Nfft4GPVecAxpy( 1.0, K_work, (size_t)n*n, K);
                  Nfft4GPVecAxpy( 1.0, dK_work, (size_t)3*n*n, dK);
                  Nfft4GPVecFill(K_work, (size_t)n*n, 0.0);
                  Nfft4GPVecFill(dK_work, (size_t)3*n*n, 0.0);
               }
            }
            Nfft4GPVecScale(K, (size_t)n*n, scale_window);
            Nfft4GPVecScale(dK, (size_t)3*n*n, scale_window);
         }
         else
         {
            NFFT4GP_CALLOC(dK_work, 3*(size_t)n*n, NFFT4GP_DOUBLE);
            if(skip_window)
            {
               for(i = 0 ; i < nwindows - 1 ; i ++)
               {
                  NFFT4GP_DOUBLE *data_window = windows + i*n*dwindows;
                  pkernel->_fkernel_buffer( pkernel->_fkernel_buffer_params, data_window, n, n, dwindows, permr, kr, permc, kc, NULL, &dK_work);
                  Nfft4GPVecAxpy( 1.0, dK_work, (size_t)3*n*n, dK);
                  Nfft4GPVecFill(dK_work, (size_t)3*n*n, 0.0);
               }
               i = nwindows - 1;
               NFFT4GP_DOUBLE *data_window = windows + i*n*dwindows;
               pkernel->_fkernel_buffer(pkernel->_fkernel_buffer_params, data_window, n, n, dwindows - skip_window, permr, kr, permc, kc, NULL, &dK_work);
               Nfft4GPVecAxpy(1.0, dK_work, (size_t)3*n*n, dK);
               Nfft4GPVecFill(dK_work, (size_t)3*n*n, 0.0);
            }
            else
            {
               for(i = 0 ; i < nwindows ; i ++)
               {
                  NFFT4GP_DOUBLE *data_window = windows + i*n*dwindows;
                  pkernel->_fkernel_buffer( pkernel->_fkernel_buffer_params, data_window, n, n, dwindows, permr, kr, permc, kc, NULL, &dK_work);
                  Nfft4GPVecAxpy( 1.0, dK_work, (size_t)3*n*n, dK);
                  Nfft4GPVecFill(dK_work, (size_t)3*n*n, 0.0);
               }
            }
            Nfft4GPVecScale(dK, (size_t)3*n*n, scale_window);
         }/* End of loop for dKp only */
      }/* End of with dKp */
      else
      {
         NFFT4GP_CALLOC(K_work, (size_t)n*n, NFFT4GP_DOUBLE);
         if(skip_window)
         {
            for(i = 0 ; i < nwindows - 1 ; i ++)
            {
               NFFT4GP_DOUBLE *data_window = windows + i*n*dwindows;
               pkernel->_fkernel_buffer(pkernel->_fkernel_buffer_params, data_window, n, n, dwindows, permr, kr, permc, kc, &K_work, NULL);
               Nfft4GPVecAxpy(1.0, K_work, (size_t)n*n, K);
               Nfft4GPVecFill(K_work, (size_t)n*n, 0.0);
            }
            i = nwindows - 1;
            NFFT4GP_DOUBLE *data_window = windows + i*n*dwindows;
            pkernel->_fkernel_buffer(pkernel->_fkernel_buffer_params, data_window, n, n, dwindows - skip_window, permr, kr, permc, kc, &K_work, NULL);
            Nfft4GPVecAxpy(1.0, K_work, (size_t)n*n, K);
            Nfft4GPVecFill(K_work, (size_t)n*n, 0.0);
         }
         else
         {
            for(i = 0 ; i < nwindows ; i ++)
            {
               NFFT4GP_DOUBLE *data_window = windows + i*n*dwindows;
               pkernel->_fkernel_buffer( pkernel->_fkernel_buffer_params, data_window, n, n, dwindows, permr, kr, permc, kc, &K_work, NULL);
               Nfft4GPVecAxpy( 1.0, K_work, (size_t)n*n, K);
               Nfft4GPVecFill(K_work, (size_t)n*n, 0.0);
            }
         }
         Nfft4GPVecScale(K, (size_t)n*n, scale_window);
      }/* end of Kp only */
   }/* End of the third case the entire K */
   NFFT4GP_FREE(K_work);
   NFFT4GP_FREE(dK_work);

   if(Kp && *Kp == NULL)
   {
      *Kp = K;
   }
   if(dKp && *dKp == NULL)
   {
      *dKp = dK;
   }

   return 0;
}

void* Nfft4GPKernelSchurCombineKernelParamCreate(NFFT4GP_DOUBLE *data, int n, int ldim, int d, 
                                             int *perm, int k, NFFT4GP_DOUBLE *chol_K11, NFFT4GP_DOUBLE *GdK11G,
                                             func_kernel fkernel, void *fkernel_params, int omp, int requires_grad)
{
   pnfft4gp_kernel pkernel;
   int n2 = n - k;

   if(!requires_grad)
   {
      // to form the Schur complement, first we need K12
      NFFT4GP_DOUBLE *K12 = NULL;
      NFFT4GP_MALLOC( K12, (size_t)n2*k, NFFT4GP_DOUBLE);

      // create K12
      fkernel( fkernel_params, data, n, ldim, d, perm, k, perm + k, n2, &K12, NULL);

      char uplo = 'L';
      char diag = 'N';
      char transn = 'N';
      char transt = 'T';
      int nknowns = k;
      int nrhs = n2;
      double one = 1.0;
      double zero = 0.0;
      int info;

      // apply G onto K12
      NFFT4GP_TRTRS( &uplo, &transn, &diag, &nknowns, &nrhs, chol_K11, &nknowns, K12, &nknowns, &info);

      // without gradient, we need n2*k buffer
      pkernel = (pnfft4gp_kernel)Nfft4GPKernelParamCreate((size_t)n2*k, omp);
      pkernel->_iparams[0] = k;
      pkernel->_iparams[1] = n2;
      pkernel->_buffer = K12;
      pkernel->_fkernel_buffer = fkernel;
      pkernel->_fkernel_buffer_params = fkernel_params;
   }
   else
   {
      // require gradient, we need more than K12
      NFFT4GP_DOUBLE *K12 = NULL;
      NFFT4GP_DOUBLE *dK12 = NULL;
      NFFT4GP_MALLOC( K12, (size_t)n2*k, NFFT4GP_DOUBLE);
      NFFT4GP_MALLOC( dK12, (size_t)6*n2*k, NFFT4GP_DOUBLE);
      NFFT4GP_DOUBLE *dK12_2 = dK12 + 3*n2*k;

      // create K12
      fkernel( fkernel_params, data, n, ldim, d, perm, k, perm + k, n2, &K12, &dK12);

      char uplo = 'L';
      char diag = 'N';
      char transn = 'N';
      char transt = 'T';
      int nknowns = k;
      int nrhs = n2;
      double one = 1.0;
      double zero = 0.0;
      int info;

      // apply G onto K12
      NFFT4GP_TRTRS( &uplo, &transn, &diag, &nknowns, &nrhs, chol_K11, &nknowns, K12, &nknowns, &info);

      // apply G onto dK12
      // WARNING: dK12{3} is zero so we do nothing to it
      nrhs = nrhs * 2;
      NFFT4GP_TRTRS( &uplo, &transn, &diag, &nknowns, &nrhs, chol_K11, &nknowns, dK12, &nknowns, &info);
      Nfft4GPVecFill( dK12+(size_t)2*n2*k, (size_t)n2*k, 0.0);

      // apply GdK11G to GK12
      NFFT4GP_DGEMM( &transt, &transn, &k, &n2, &k, &one, GdK11G + 0*k*k, &k, K12, &k, &zero, dK12_2+0*n2*k, &k);
      NFFT4GP_DGEMM( &transt, &transn, &k, &n2, &k, &one, GdK11G + 1*k*k, &k, K12, &k, &zero, dK12_2+1*n2*k, &k);
      NFFT4GP_DGEMM( &transt, &transn, &k, &n2, &k, &one, GdK11G + 2*k*k, &k, K12, &k, &zero, dK12_2+2*n2*k, &k);
      
      // with gradient, we need more buffer
      pkernel = (pnfft4gp_kernel)Nfft4GPKernelParamCreate((3*k+n2)*n2, omp);
      pkernel->_iparams[0] = k;
      pkernel->_iparams[1] = n2;
      pkernel->_buffer = K12;
      pkernel->_dbuffer = dK12;
      pkernel->_fkernel_buffer = fkernel;
      pkernel->_fkernel_buffer_params = fkernel_params;
   }
   // might need to set the number of threads
#ifdef NFFT4GP_USING_OPENMP
   if(!omp)
   {
#endif
      NFFT4GP_MALLOC( pkernel->_ibufferp, 1, int*);
      NFFT4GP_MALLOC( pkernel->_libufferp, 1, size_t);
#ifdef NFFT4GP_USING_OPENMP
   }
   else
   {
      int nthreads = omp_get_max_threads();
      NFFT4GP_MALLOC( pkernel->_ibufferp, nthreads, int*);
      NFFT4GP_MALLOC( pkernel->_libufferp, nthreads, size_t);
   }
#endif

   return (void*)pkernel;
}

int Nfft4GPKernelSchurCombineKernel(void *str, NFFT4GP_DOUBLE *data, int n, int ldim, int d, int *permr, int kr, int *permc, int kc, NFFT4GP_DOUBLE **Kp, NFFT4GP_DOUBLE **dKp)
{
   /* return immediately if no output is needed */
   if(Kp == NULL && dKp == NULL)
   {
      return 0;
   }

   pnfft4gp_kernel pkernel = (pnfft4gp_kernel)str;
   
   int k = pkernel->_iparams[0];
   int n2 = pkernel->_iparams[1];
   
   NFFT4GP_DOUBLE *A = pkernel->_buffer;
   NFFT4GP_DOUBLE *Alr, *K;
   NFFT4GP_DOUBLE *ddwork;
   NFFT4GP_DOUBLE *dAlr, *dA2lr, *dK;
   // dbuffer stores two groups of matrices
   NFFT4GP_DOUBLE *dA = pkernel->_dbuffer;
   NFFT4GP_DOUBLE *dA2 = dA + (size_t)k*n2*3;

   int *perm;
   int lperm;

#ifdef NFFT4GP_USING_OPENMP
   if(!omp_in_parallel())
   {
#endif
      perm = pkernel->_ibufferp[0];
      lperm = (int)pkernel->_libufferp[0];
#ifdef NFFT4GP_USING_OPENMP
   }
   else
   {
      perm = pkernel->_ibufferp[omp_get_thread_num()];
      lperm = (int)pkernel->_libufferp[omp_get_thread_num()];
   }
#endif


   int grad_num;
   size_t i, j;

   if(permr)
   {
      printf("The current version does not support permutation\n");
      return -1;
   }

   if(dKp)
   {
      /* we need both buffer and dbuffer */

      /* first check if we have enough memory */
#ifdef NFFT4GP_USING_OPENMP
      if(!omp_in_parallel())
      {
#endif
         // in this case not in parallel region
         if(pkernel->_ldwork < (size_t)(3*k+n)*n)
         {
            printf("Nfft4GPKernelSchurCombineKernel buffer size is not enough. Check your call to Nfft4GPKernelParamCreate.\n");
            printf("required buffer size is %ld, but only %ld is provided.\n", (size_t)(3*k+n)*n, pkernel->_ldwork);
            return -1;
         }
         Alr = pkernel->_dwork;
         dAlr = Alr + (size_t)k*n;
         dA2lr = dAlr + (size_t)k*n;
         ddwork = dA2lr + (size_t)k*n;
#ifdef NFFT4GP_USING_OPENMP
      }
      else
      {
         int nthreads = omp_get_max_threads();
         if( pkernel->_ldwork < (size_t)(3*k+n)*n*nthreads)
         {
            printf("Nfft4GPKernelSchurCombineKernel buffer size is not enough. Check your call to Nfft4GPKernelParamCreate.\n");
            printf("required buffer size is %ld, but only %ld is provided.\n", (size_t)(3*k+n)*n*nthreads, pkernel->_ldwork);
            return -1;
         }
         // this is nested in a OpenMP parallel region
         // we have checked memory earlier
         Alr = pkernel->_dwork + (size_t)pkernel->_max_n*omp_get_thread_num();
         dAlr = Alr + (size_t)k*n; // make dAlr close to Alr
         dA2lr = dAlr + (size_t)k*n;
         ddwork = dA2lr + (size_t)k*n;
      }
#endif

      /* fill Alr anyway */
#ifdef NFFT4GP_USING_OPENMP
      if(!omp_in_parallel())
      {
         #pragma omp parallel for private(i,j) schedule(dynamic)
#endif
         for(i = 0 ; i < (size_t)lperm ; i ++)
         {
            for(j = 0 ; j < (size_t)k ; j ++)
            {
               Alr[i*k+j] = A[(size_t)perm[i]*k+j];
            }
         }
#ifdef NFFT4GP_USING_OPENMP
      }
      else
      {
         for(i = 0 ; i < (size_t)lperm ; i ++)
         {
            for(j = 0 ; j < (size_t)k ; j ++)
            {
               Alr[i*k+j] = A[(size_t)perm[i]*k+j];
            }
         }
      }
#endif

      if(Kp)
      {
         /* In this case both dKp and Kp */
         if(*Kp == NULL)
         {
            NFFT4GP_MALLOC(K, (size_t)n*n, NFFT4GP_DOUBLE);
         }
         else
         {
            K = *Kp;
         }

         if(*dKp == NULL)
         {
            NFFT4GP_MALLOC(dK, 3*(size_t)n*n, NFFT4GP_DOUBLE);
         }
         else
         {
            dK = *dKp;
         }

         pkernel->_fkernel_buffer(pkernel->_fkernel_buffer_params, data, n, ldim, d, NULL, 0, NULL, 0, &K, &dK);

         // next apply the addition K = K + A^T*A
         double mone = -1.0;
         double zero = 0.0;
         double one = 1.0;
         char transn = 'N';
         char transt = 'T';
         
         NFFT4GP_DGEMM( &transt, &transn, &lperm, &lperm, &k, &mone, Alr, &k, Alr, &k, &one, K, &lperm);

         // now loop to setup the Schur complement
         // warning: we assume that dK{3} is zero so we skip it
         NFFT4GP_DOUBLE *dKi = dK; // just to simplify the code
         for(grad_num = 0 ; grad_num < 3 ; grad_num++)
         {

#ifdef NFFT4GP_USING_OPENMP
            if(!omp_in_parallel())
            {
               #pragma omp parallel for private(i,j) schedule(dynamic)
#endif
               for(i = 0 ; i < (size_t)lperm ; i ++)
               {
                  for(j = 0 ; j < (size_t)k ; j ++)
                  {
                     dAlr[i*k+j] = dA[(size_t)perm[i]*k+j];
                     dA2lr[i*k+j] = dA2[(size_t)perm[i]*k+j];
                  }
               }
#ifdef NFFT4GP_USING_OPENMP
            }
            else
            {
               for(i = 0 ; i < (size_t)lperm ; i ++)
               {
                  for(j = 0 ; j < (size_t)k ; j ++)
                  {
                     dAlr[i*k+j] = dA[(size_t)perm[i]*k+j];
                     dA2lr[i*k+j] = dA2[(size_t)perm[i]*k+j];
                  }
               }
            }
#endif
            // done, now we have all necessary matrices for this one
            // first compute matvec stroed in temp buffer
            // WARNING: skip this step since dK{3} is zero   
            NFFT4GP_DGEMM( &transt, &transn, &lperm, &lperm, &k, &mone, Alr, &k, dAlr, &k, &zero, ddwork, &lperm);

            // Next add this matrix with its transpose
#ifdef NFFT4GP_USING_OPENMP
            if(!omp_in_parallel())
            {
               #pragma omp parallel for private(i,j) schedule(dynamic)
               for(i = 0 ; i < (size_t)lperm ; i ++)
               {
                  for(j = 0 ; j < (size_t)lperm ; j ++)
                  {
                     dKi[i*lperm+j] += ddwork[i*lperm+j] + ddwork[j*lperm+i];
                  }
               }
            }
            else
            {
#endif
               for(i = 0 ; i < (size_t)lperm ; i ++)
               {
                  for(j = 0 ; j < (size_t)lperm ; j ++)
                  {
                     dKi[i*lperm+j] += ddwork[i*lperm+j] + ddwork[j*lperm+i];
                  }
               }
#ifdef NFFT4GP_USING_OPENMP
            }
#endif
            // and add the last component (G*K12)^T*(G*dK*G^T)*(G*K12)
            // the second temp buffer is of size k*kr
            NFFT4GP_DGEMM( &transt, &transn, &lperm, &lperm, &k, &one, Alr, &k, dA2lr, &k, &one, dKi, &lperm);
            
            // update pointers
            dA += (size_t)k*n2;
            dA2 += (size_t)k*n2;
            dKi += n*n;
         }// end of grad_num loop

      }// end of if Kp
      else
      {
         /* In this case dKp only */
         if(*dKp == NULL)
         {
            NFFT4GP_MALLOC(dK, 3*(size_t)n*n, NFFT4GP_DOUBLE);
         }
         else
         {
            dK = *dKp;
         }

         pkernel->_fkernel_buffer(pkernel->_fkernel_buffer_params, data, n, ldim, d, NULL, 0, NULL, 0, NULL, &dK);

         // next apply the addition K = K + A^T*A
         double mone = -1.0;
         double zero = 0.0;
         double one = 1.0;
         char transn = 'N';
         char transt = 'T';

         // now loop to setup the Schur complement
         NFFT4GP_DOUBLE *dKi = dK; // just to simplify the code
         for(grad_num = 0 ; grad_num < 3 ; grad_num++)
         {

#ifdef NFFT4GP_USING_OPENMP
            if(!omp_in_parallel())
            {
               #pragma omp parallel for private(i,j) schedule(dynamic)
#endif
               for(i = 0 ; i < (size_t)lperm ; i ++)
               {
                  for(j = 0 ; j < (size_t)k ; j ++)
                  {
                     dAlr[i*k+j] = dA[(size_t)perm[i]*k+j];
                     dA2lr[i*k+j] = dA2[(size_t)perm[i]*k+j];
                  }
               }
#ifdef NFFT4GP_USING_OPENMP
            }
            else
            {
               for(i = 0 ; i < (size_t)lperm ; i ++)
               {
                  for(j = 0 ; j < (size_t)k ; j ++)
                  {
                     dAlr[i*k+j] = dA[(size_t)perm[i]*k+j];
                     dA2lr[i*k+j] = dA2[(size_t)perm[i]*k+j];
                  }
               }
            }
#endif
            // done, now we have all necessary matrices for this one
            // first compute matvec stroed in temp buffer        
            NFFT4GP_DGEMM( &transt, &transn, &lperm, &lperm, &k, &mone, Alr, &k, dAlr, &k, &zero, ddwork, &lperm);

            // Next add this matrix with its transpose
#ifdef NFFT4GP_USING_OPENMP
            if(!omp_in_parallel())
            {
               #pragma omp parallel for private(i,j) schedule(dynamic)
               for(i = 0 ; i < (size_t)lperm ; i ++)
               {
                  for(j = 0 ; j < (size_t)lperm ; j ++)
                  {
                     dKi[i*lperm+j] += ddwork[i*lperm+j] + ddwork[j*lperm+i];
                  }
               }
            }
            else
            {
#endif
               for(i = 0 ; i < (size_t)lperm ; i ++)
               {
                  for(j = 0 ; j < (size_t)lperm ; j ++)
                  {
                     dKi[i*lperm+j] += ddwork[i*lperm+j] + ddwork[j*lperm+i];
                  }
               }
#ifdef NFFT4GP_USING_OPENMP
            }
#endif
            // and add the last component (G*K12)^T*(G*dK*G^T)*(G*K12)
            // the second temp buffer is of size k*kr
            NFFT4GP_DGEMM( &transt, &transn, &lperm, &lperm, &k, &one, Alr, &k, dA2lr, &k, &one, dKi, &lperm);
            
            // update pointers
            dA += (size_t)k*n2;
            dA2 += (size_t)k*n2;
            dKi += n*n;
         }// end of grad_num loop
      }// end of else Kp
   }
   else
   {
      /* In this case K only
       * only buffer is needed
       */
      
      /* first check if we have enough memory */
#ifdef NFFT4GP_USING_OPENMP
      if(!omp_in_parallel())
      {
#endif
         // in this case not in parallel region
         if(pkernel->_ldwork < (size_t)k*n)
         {
            printf("Nfft4GPKernelSchurCombineKernel buffer size is not enough. Check your call to Nfft4GPKernelParamCreate.\n");
            printf("required buffer size is %ld, but only %ld is provided.\n", (size_t)k*n, pkernel->_ldwork);
            return -1;
         }
         Alr = pkernel->_dwork;
#ifdef NFFT4GP_USING_OPENMP
      }
      else
      {
         int nthreads = omp_get_max_threads();
         if( pkernel->_ldwork < (size_t)pkernel->_max_n*nthreads )
         {
            printf("Nfft4GPKernelSchurCombineKernel buffer size is not enough. Check your call to Nfft4GPKernelParamCreate.\n");
            return -1;
         }
         // this is nested in a OpenMP parallel region
         // we have checked memory earlier
         Alr = pkernel->_dwork + (size_t)pkernel->_max_n*omp_get_thread_num();
      }
#endif

      if(*Kp == NULL)
      {
         NFFT4GP_MALLOC(K, (size_t)n*n, NFFT4GP_DOUBLE);
      }
      else
      {
         K = *Kp;
      }

      pkernel->_fkernel_buffer(pkernel->_fkernel_buffer_params, data, n, ldim, d, NULL, 0, NULL, 0, &K, NULL);

      /* now we have K22 available, starting to compute K21*K11^-1*K12 */
#ifdef NFFT4GP_USING_OPENMP
      if(!omp_in_parallel())
      {
         #pragma omp parallel for private(i,j) schedule(dynamic)
#endif
         for(i = 0 ; i < (size_t)lperm ; i ++)
         {
            for(j = 0 ; j < (size_t)k ; j ++)
            {
               Alr[i*k+j] = A[(size_t)perm[i]*k+j];
            }
         }
#ifdef NFFT4GP_USING_OPENMP
      }
      else
      {
         for(i = 0 ; i < (size_t)lperm ; i ++)
         {
            for(j = 0 ; j < (size_t)k ; j ++)
            {
               Alr[i*k+j] = A[(size_t)perm[i]*k+j];
            }
         }
      }
#endif

      // next apply the addition K = K + A^T*A
      double mone = -1.0;
      double one = 1.0;
      char transn = 'N';
      char transt = 'T';
      
      NFFT4GP_DGEMM( &transt, &transn, &lperm, &lperm, &k, &mone, Alr, &k, Alr, &k, &one, K, &lperm);

   }// end of K only region

   if(Kp && *Kp == NULL)
   {
      *Kp = K;
   }

   if(dKp && *dKp == NULL)
   {
      *dKp = dK;
   }

   return 0;

}