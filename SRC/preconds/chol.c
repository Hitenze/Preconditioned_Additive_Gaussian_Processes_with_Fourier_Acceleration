
#include "chol.h"

/*------------------------------------------
 * CHOL
 *------------------------------------------*/

void* Nfft4GPPrecondCholCreate()
{
   pprecond_chol str = NULL;
   NFFT4GP_MALLOC( str, 1, precond_chol);

   str->_stable = 0;

   str->_n = 0;
   str->_tits = 0;
   str->_titt = 0.0;
   str->_tset = 0.0;
   str->_tlogdet = 0.0;
   str->_tdvp = 0.0;

   str->_chol_data = NULL;
   str->_GdKG_data = NULL;
   str->_dchol_data = NULL;
   str->_dK_data = NULL;
   return (void*) str;
}

void Nfft4GPPrecondCholFree(void *str)
{
   pprecond_chol pstr = (pprecond_chol) str;

   if(pstr)
   {
      NFFT4GP_FREE(pstr->_chol_data);
      NFFT4GP_FREE(pstr->_GdKG_data);
      NFFT4GP_FREE(pstr->_dchol_data);
      NFFT4GP_FREE(pstr->_dK_data);
      /*
      printf("CHOL total setup time %fs\n",pstr->_tset);
      if(pstr->_tits > 0)
      {
         printf("CHOL average solve time %fs\n",pstr->_titt/pstr->_tits);
      }
      if(pstr->_tlogdet > 0)
      {
         printf("CHOL trace and logdet time %fs\n",pstr->_tlogdet);
      }
      if(pstr->_tdvp > 0)
      {
         printf("CHOL DVP time %fs\n",pstr->_tdvp);
      }
      */

      pstr->_n = 0;
      pstr->_tits = 0;
      pstr->_titt = 0.0;
      pstr->_tset = 0.0;
      pstr->_tlogdet = 0.0;
      pstr->_tdvp = 0.0;

      NFFT4GP_FREE(pstr);
   }
}

void Nfft4GPPrecondCholReset(void *str)
{
   pprecond_chol pstr = (pprecond_chol) str;

   if(pstr)
   {
      NFFT4GP_FREE(pstr->_chol_data);
      NFFT4GP_FREE(pstr->_GdKG_data);
      NFFT4GP_FREE(pstr->_dchol_data);
      NFFT4GP_FREE(pstr->_dK_data);
      printf("CHOL total setup time %fs\n",pstr->_tset);
      if(pstr->_tits > 0)
      {
         printf("CHOL average solve time %fs\n",pstr->_titt/pstr->_tits);
      }
      if(pstr->_tlogdet > 0)
      {
         printf("CHOL trace and logdet time %fs\n",pstr->_tlogdet);
      }
      if(pstr->_tdvp > 0)
      {
         printf("CHOL DVP time %fs\n",pstr->_tdvp);
      }
      
      pstr->_n = 0;
      pstr->_tits = 0;
      pstr->_titt = 0.0;
      pstr->_tset = 0.0;
      pstr->_tlogdet = 0.0;
      pstr->_tdvp = 0.0;

      // do not free pstr, remain for future work
   }
}

void Nfft4GPPrecondCholSetStable(void *str, int stable)
{
   pprecond_chol pstr = (pprecond_chol) str;

   if(pstr)
   {
      pstr->_stable = stable;
   }
}

int Nfft4GPPrecondCholSolve( void *vchol_mat, int n, NFFT4GP_DOUBLE *x, NFFT4GP_DOUBLE *rhs)
{

   double ts, te;
   ts = Nfft4GPWtime();

   pprecond_chol chol_mat = (pprecond_chol) vchol_mat;
   NFFT4GP_MEMCPY( x, rhs, chol_mat->_n, NFFT4GP_DOUBLE);

   chol_mat->_trans = 'N';
   NFFT4GP_TRTRS( &chol_mat->_uplo, &chol_mat->_trans, &chol_mat->_diag,
              &n, &chol_mat->_nrhs,
              chol_mat->_chol_data, &n,
              x, &n, &chol_mat->_info);
   chol_mat->_trans = 'T';
   NFFT4GP_TRTRS( &chol_mat->_uplo, &chol_mat->_trans, &chol_mat->_diag,
              &n, &chol_mat->_nrhs,
              chol_mat->_chol_data, &n,
              x, &n, &chol_mat->_info);
   
   te = Nfft4GPWtime();
   chol_mat->_titt += (te - ts);
   chol_mat->_tits ++;

   return 0;
}

int Nfft4GPPrecondCholDvp(void *vchol_mat, int n, int *mask, NFFT4GP_DOUBLE *x, NFFT4GP_DOUBLE **yp)
{
   double ts, te;
   ts = Nfft4GPWtime();

   pprecond_chol chol_mat = (pprecond_chol) vchol_mat;

   int *mask_l = NULL;
   if(!mask)
   {
      NFFT4GP_MALLOC( mask_l, 3, int);
      mask_l[0] = 1;mask_l[1] = 1;mask_l[2] = 1;
   }

   if(!chol_mat->_dchol_data)
   {
      printf("Setup chol without gradient, dvp not supported.\n");
      return -1;
   }

   if(!yp)
   {
      printf("output pointer cannot be NULL\n");
      return -1;
   }

   int i;
   NFFT4GP_DOUBLE *y;

   if(*yp == NULL)
   {
      NFFT4GP_CALLOC(y,3*n,NFFT4GP_DOUBLE);
   }
   else
   {
      y = *yp;
   }
   
   char transn = 'N';
   char transt = 'T';
   if(mask_l[0])
   {
      Nfft4GPDenseMatGemv(chol_mat->_dK_data, 'N', n, n, 1.0, x, 0.0, y);
      NFFT4GP_TRTRS( &chol_mat->_uplo, &transn, &chol_mat->_diag,
               &n, &chol_mat->_nrhs,
               chol_mat->_chol_data, &n,
               y, &n, &chol_mat->_info);
      NFFT4GP_TRTRS( &chol_mat->_uplo, &transt, &chol_mat->_diag,
               &n, &chol_mat->_nrhs,
               chol_mat->_chol_data, &n,
               y, &n, &chol_mat->_info);
   }
   if(mask_l[1])
   {
      Nfft4GPDenseMatGemv(chol_mat->_dK_data + (size_t)n*n, 'N', n, n, 1.0, x, 0.0, y + n);
      NFFT4GP_TRTRS( &chol_mat->_uplo, &transn, &chol_mat->_diag,
               &n, &chol_mat->_nrhs,
               chol_mat->_chol_data, &n,
               y+n, &n, &chol_mat->_info);
      NFFT4GP_TRTRS( &chol_mat->_uplo, &transt, &chol_mat->_diag,
               &n, &chol_mat->_nrhs,
               chol_mat->_chol_data, &n,
               y+n, &n, &chol_mat->_info);
   }
   if(mask_l[2])
   {
      Nfft4GPDenseMatGemv(chol_mat->_dK_data + 2*(size_t)n*n, 'N', n, n, 1.0, x, 0.0, y + 2*n);
      NFFT4GP_TRTRS( &chol_mat->_uplo, &transn, &chol_mat->_diag,
               &n, &chol_mat->_nrhs,
               chol_mat->_chol_data, &n,
               y+2*n, &n, &chol_mat->_info);
      NFFT4GP_TRTRS( &chol_mat->_uplo, &transt, &chol_mat->_diag,
               &n, &chol_mat->_nrhs,
               chol_mat->_chol_data, &n,
               y+2*n, &n, &chol_mat->_info);
   }

   if(*yp == NULL)
   {
      *yp = y;
   }

   te = Nfft4GPWtime();
   chol_mat->_tdvp += (te - ts);

   if(!mask)
   {
      NFFT4GP_FREE(mask_l);
   }

   return 0;
}

int Nfft4GPPrecondCholTrace(void *vchol_mat, NFFT4GP_DOUBLE **tracesp)
{
   double ts, te;
   ts = Nfft4GPWtime();

   pprecond_chol chol_mat = (pprecond_chol) vchol_mat;

   if(!chol_mat->_dchol_data)
   {
      printf("Setup chol without gradient, trace not supported.\n");
      return -1;
   }

   if(!tracesp)
   {
      printf("Trace pointer cannot be NULL\n");
      return -1;
   }

   int i;
   NFFT4GP_DOUBLE *traces;

   if(*tracesp == NULL)
   {
      NFFT4GP_CALLOC(traces,3,NFFT4GP_DOUBLE);
   }
   else
   {
      traces = *tracesp;
   }

   NFFT4GP_DOUBLE *pval = chol_mat->_chol_data;
   NFFT4GP_DOUBLE *dL1, *dL2, *dL3;
   dL1 = chol_mat->_dchol_data;
   dL2 = dL1 + chol_mat->_n * chol_mat->_n;
   dL3 = dL2 + chol_mat->_n * chol_mat->_n;
   for(i = 0 ; i < chol_mat->_n ; i ++)
   {
      traces[0] += dL1[0] / pval[0];
      traces[1] += dL2[0] / pval[0];
      traces[2] += dL3[0] / pval[0];
      pval += (chol_mat->_n + 1);
      dL1 += (chol_mat->_n + 1);
      dL2 += (chol_mat->_n + 1);
      dL3 += (chol_mat->_n + 1);
   }

   traces[0] *= 2;
   traces[1] *= 2;
   traces[2] *= 2;

   if(*tracesp == NULL)
   {
      *tracesp = traces;
   }

   te = Nfft4GPWtime();
   chol_mat->_tlogdet += (te - ts);

   return 0;
}

NFFT4GP_DOUBLE Nfft4GPPrecondCholLogdet(void *vchol_mat)
{
   double ts, te;
   ts = Nfft4GPWtime();

   pprecond_chol chol_mat = (pprecond_chol) vchol_mat;

   int i;
   NFFT4GP_DOUBLE val = 0.0;
   NFFT4GP_DOUBLE *pval = chol_mat->_chol_data;
   for(i = 0 ; i < chol_mat->_n ; i ++)
   {
      val += log(pval[0]);
      pval += (chol_mat->_n + 1);
   }

   val *= 2;

   te = Nfft4GPWtime();
   chol_mat->_tlogdet += (te - ts);
   
   return val;
}

/**
 * @brief      This is a helper function computing the matvec dL = L*(tril(GdKG,-1)+diag(diag(GdKG)/2)).
 * @details    This is a helper function computing the matvec dL = L*(tril(GdKG,-1)+diag(diag(GdKG)/2)).
 * @param[in]  vchol_mat Pointer to the solver (void).
 * @param[in]  dA Pointer to GdKG
 * @return     Return  if successful
 */
int Nfft4GPPrecondCholGdkgmv(void *vchol_mat, NFFT4GP_DOUBLE *dA)
{
   pprecond_chol chol_mat = (pprecond_chol) vchol_mat;

   size_t i, j, k;
   int n = chol_mat->_n;
   int n3 = n*3;
   NFFT4GP_DOUBLE temp_val1;
   NFFT4GP_DOUBLE temp_val2;
   NFFT4GP_DOUBLE temp_val3;
   NFFT4GP_DOUBLE *dA1, *dA2, *dA3;
   NFFT4GP_DOUBLE *dL1, *dL2, *dL3;
   NFFT4GP_DOUBLE *L_i, *dL1_i, *dL2_i, *dL3_i;

   dA1 = dA;
   dA2 = dA1 + n*n;
   dA3 = dA2 + n*n;

   NFFT4GP_CALLOC(chol_mat->_dchol_data, (size_t)n3*n, NFFT4GP_DOUBLE);
   dL1 = chol_mat->_dchol_data;
   dL2 = dL1 + n*n;
   dL3 = dL2 + n*n;
   
#ifdef NFFT4GP_USING_OPENMP
   if(!omp_in_parallel())
   {
      // parallel version
      #pragma omp parallel for private(i,j,k,temp_val1,temp_val2,temp_val3,L_i,dL1_i,dL2_i,dL3_i) NFFT4GP_DEFAULT_OPENMP_SCHEDULE
#endif
      for(j = 0 ; j < (size_t)n ; j++)
      {
         temp_val1 = dA1[j*n+j] / 2.0;
         temp_val2 = dA2[j*n+j] / 2.0;
         temp_val3 = dA3[j*n+j] / 2.0;
         L_i = chol_mat->_chol_data + j*n;
         dL1_i = dL1 + j*n;
         dL2_i = dL2 + j*n;
         dL3_i = dL3 + j*n;
         for(i = j ; i < (size_t)n ; i++)
         {
            dL1_i[i] += temp_val1 * L_i[i];
            dL2_i[i] += temp_val2 * L_i[i];
            dL3_i[i] += temp_val3 * L_i[i];
         }
         for(k = j+1 ; k < (size_t)n ; k++)
         {
            temp_val1 = dA1[j*n+k];
            temp_val2 = dA2[j*n+k];
            temp_val3 = dA3[j*n+k];
            L_i = chol_mat->_chol_data + k*n;
            dL1_i = dL1 + j*n;
            dL2_i = dL2 + j*n;
            dL3_i = dL3 + j*n;
            for(i = k ; i < (size_t)n ; i++)
            {
               dL1_i[i] += temp_val1 * L_i[i];
               dL2_i[i] += temp_val2 * L_i[i];
               dL3_i[i] += temp_val3 * L_i[i];
            }
         }
      }
#ifdef NFFT4GP_USING_OPENMP
   }
   else
   {
      // sequential version
      for(j = 0 ; j < (size_t)n ; j++)
      {
         temp_val1 = dA1[j*n+j] / 2.0;
         temp_val2 = dA2[j*n+j] / 2.0;
         temp_val3 = dA3[j*n+j] / 2.0;
         L_i = chol_mat->_chol_data + j*n;
         dL1_i = dL1 + j*n;
         dL2_i = dL2 + j*n;
         dL3_i = dL3 + j*n;
         for(i = j ; i < (size_t)n ; i++)
         {
            dL1_i[i] += temp_val1 * L_i[i];
            dL2_i[i] += temp_val2 * L_i[i];
            dL3_i[i] += temp_val3 * L_i[i];
         }
         for(k = j+1 ; k < (size_t)n ; k++)
         {
            temp_val1 = dA1[j*n+k];
            temp_val2 = dA2[j*n+k];
            temp_val3 = dA3[j*n+k];
            L_i = chol_mat->_chol_data + k*n;
            dL1_i = dL1 + j*n;
            dL2_i = dL2 + j*n;
            dL3_i = dL3 + j*n;
            for(i = k ; i < (size_t)n ; i++)
            {
               dL1_i[i] += temp_val1 * L_i[i];
               dL2_i[i] += temp_val2 * L_i[i];
               dL3_i[i] += temp_val3 * L_i[i];
            }
         }
      }
   }
#endif

   return 0;
}

int Nfft4GPPrecondCholSetupWithKernel(NFFT4GP_DOUBLE *data, int n, int ldim, int d,
                                    func_kernel fkernel, void *fkernel_params, int require_grad, void* vchol_mat)
{
   double ts, te;
   ts = Nfft4GPWtime();

   pprecond_chol chol_mat = (pprecond_chol) vchol_mat;
   chol_mat->_n = n;

   if(!require_grad)
   {
      if(chol_mat->_stable)
      {
         // stable version
         size_t i;
         chol_mat->_uplo = 'L';
         chol_mat->_diag = 'N';
         chol_mat->_nrhs = 1;
         fkernel( fkernel_params, data, n, ldim, d, NULL, 0, NULL, 0, &(chol_mat->_chol_data), NULL);
         
         // Add a small shift for stable
         char norm = 'F';
         char uplo = 'L';
         double A_fro = NFFT4GP_DLANSY( &norm, &uplo, &n, chol_mat->_chol_data, &n, NULL);
         double nu;
#ifdef NFFT4GP_USING_FLOAT32
         nu = sqrt((float)n)*(nextafter((float)A_fro,(float)(A_fro+1.0))-A_fro);
#else
         nu = sqrt((double)n)*(nextafter((double)A_fro,(double)(A_fro+1.0))-A_fro);
#endif
         // TODO: OpenMP this loop
         NFFT4GP_DOUBLE *A_ptr = chol_mat->_chol_data;
         for(i = 0 ; i < (size_t)n ; i ++)
         {
            (*A_ptr) += (NFFT4GP_DOUBLE)nu;
            A_ptr += (n+1);
         }
         
         NFFT4GP_DPOTRF( &chol_mat->_uplo, &n, chol_mat->_chol_data, &n, &chol_mat->_info);
      }
      else
      {
         // unstable version
         chol_mat->_uplo = 'L';
         chol_mat->_diag = 'N';
         chol_mat->_nrhs = 1;
         fkernel( fkernel_params, data, n, ldim, d, NULL, 0, NULL, 0, &(chol_mat->_chol_data), NULL);
         
         NFFT4GP_DPOTRF( &chol_mat->_uplo, &n, chol_mat->_chol_data, &n, &chol_mat->_info);
      }
   }
   else
   {
      // Gradient is required in this case
      if(chol_mat->_stable)
      {
         // stable version
         size_t i;
         chol_mat->_uplo = 'L';
         chol_mat->_diag = 'N';
         chol_mat->_nrhs = 1;
         fkernel( fkernel_params, data, n, ldim, d, NULL, 0, NULL, 0, &(chol_mat->_chol_data), &(chol_mat->_dK_data));
         NFFT4GP_MALLOC(chol_mat->_GdKG_data, n*n*3, NFFT4GP_DOUBLE);
         NFFT4GP_MEMCPY(chol_mat->_GdKG_data, chol_mat->_dK_data, n*n*3, NFFT4GP_DOUBLE);
         
         // Add a small shift for stable
         char norm = 'F';
         char uplo = 'L';
         NFFT4GP_DOUBLE A_fro = NFFT4GP_DLANSY( &norm, &uplo, &n, chol_mat->_chol_data, &n, NULL);
         NFFT4GP_DOUBLE nu;
#ifdef NFFT4GP_USING_FLOAT32
         nu = sqrt((float)n)*(nextafter((float)A_fro,(float)(A_fro+1.0))-A_fro);
#else
         nu = sqrt((double)n)*(nextafter((double)A_fro,(double)(A_fro+1.0))-A_fro);
#endif
         // TODO: OpenMP this loop
         NFFT4GP_DOUBLE *A_ptr = chol_mat->_chol_data;
         for(i = 0 ; i < (size_t)n ; i ++)
         {
            (*A_ptr) += (NFFT4GP_DOUBLE)nu;
            A_ptr += (n+1);
         }
         
         NFFT4GP_DPOTRF( &chol_mat->_uplo, &n, chol_mat->_chol_data, &n, &chol_mat->_info);
         // next, compute the matrix GdKG
         char transn = 'N';
         char transt = 'T';
         NFFT4GP_DOUBLE one = 1.0;
         char sider = 'R';
         int nrhs = 3*n;

         for(i = 0 ; i < 3 ; i ++)
         {
            NFFT4GP_DTRSM( &sider, &chol_mat->_uplo, &transt, &chol_mat->_diag, &n, &n, &one, chol_mat->_chol_data, &n, chol_mat->_GdKG_data+i*n*n, &n);
         }
         NFFT4GP_TRTRS( &chol_mat->_uplo, &transn, &chol_mat->_diag, &n, &nrhs, chol_mat->_chol_data, &n, chol_mat->_GdKG_data, &n, &chol_mat->_info);
         

         // next, compute dL
         Nfft4GPPrecondCholGdkgmv( (void*)chol_mat, chol_mat->_GdKG_data);

      }
      else
      {
         // unstable version
         size_t i;
         chol_mat->_uplo = 'L';
         chol_mat->_diag = 'N';
         chol_mat->_nrhs = 1;
         fkernel( fkernel_params, data, n, ldim, d, NULL, 0, NULL, 0, &(chol_mat->_chol_data), &chol_mat->_dK_data);
         NFFT4GP_MALLOC(chol_mat->_GdKG_data, n*n*3, NFFT4GP_DOUBLE);
         NFFT4GP_MEMCPY(chol_mat->_GdKG_data, chol_mat->_dK_data, n*n*3, NFFT4GP_DOUBLE);
         
         NFFT4GP_DPOTRF( &chol_mat->_uplo, &n, chol_mat->_chol_data, &n, &chol_mat->_info);

         // next, compute the matrix GdKG
         char transn = 'N';
         char transt = 'T';
         NFFT4GP_DOUBLE one = 1.0;
         char sider = 'R';
         int nrhs = 3*n;
         for(i = 0 ; i < 3 ; i ++)
         {
            NFFT4GP_DTRSM( &sider, &chol_mat->_uplo, &transt, &chol_mat->_diag, &n, &n, &one, chol_mat->_chol_data, &n, chol_mat->_GdKG_data+i*n*n, &n);
         }
         NFFT4GP_TRTRS( &chol_mat->_uplo, &transn, &chol_mat->_diag, &n, &nrhs, chol_mat->_chol_data, &n, chol_mat->_GdKG_data, &n, &chol_mat->_info);
         
         // next, compute dL
         Nfft4GPPrecondCholGdkgmv( (void*)chol_mat, chol_mat->_GdKG_data);
         
      }
   }
      
   te = Nfft4GPWtime();
   chol_mat->_tset = te - ts;
   
   return 0;
}