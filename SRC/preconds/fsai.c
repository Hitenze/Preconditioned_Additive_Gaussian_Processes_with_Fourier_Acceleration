#include "fsai.h"

/*------------------------------------------
 * FSAI
 *------------------------------------------*/

void* Nfft4GPPrecondFsaiCreate()
{
   pprecond_fsai str = NULL;
   NFFT4GP_MALLOC( str, 1, precond_fsai);

   str->_lfil = 50;

   str->_n = 0;
   str->_tits = 0;
   str->_titt = 0.0;
   str->_tset = 0.0;

   str->_L_i = NULL; // L, U without diagonal
   str->_L_j = NULL;
   str->_L_a = NULL;
   str->_dL_a = NULL;


   str->_work = NULL;

   return (void*)str;
}

void Nfft4GPPrecondFsaiFree(void *str)
{
   pprecond_fsai pstr = (pprecond_fsai)str;
   if(pstr)
   {
      pstr->_n = 0;
      NFFT4GP_FREE(pstr->_L_i);
      NFFT4GP_FREE(pstr->_L_j);
      NFFT4GP_FREE(pstr->_L_a);
      NFFT4GP_FREE(pstr->_dL_a);
      NFFT4GP_FREE(pstr->_work);
      /*
      printf("FSAI total setup time %fs\n",pstr->_tset);
      if(pstr->_tits > 0)
      {
         printf("FSAI average solve time %fs\n",pstr->_titt/pstr->_tits);
      }
      if(pstr->_tlogdet > 0)
      {
         printf("FSAI trace and logdet time %fs\n",pstr->_tlogdet);
      }
      if(pstr->_tdvp > 0)
      {
         printf("FSAI DVP time %fs\n",pstr->_tdvp);
      }
      */
         
      pstr->_n = 0;
      pstr->_tits = 0;
      pstr->_titt = 0.0;
      pstr->_tset = 0.0;

      NFFT4GP_FREE(pstr);
   }
}

void Nfft4GPPrecondFsaiReset(void *str)
{
   pprecond_fsai pstr = (pprecond_fsai)str;
   if(pstr)
   {
      pstr->_n = 0;
      NFFT4GP_FREE(pstr->_L_i);
      NFFT4GP_FREE(pstr->_L_j);
      NFFT4GP_FREE(pstr->_L_a);
      NFFT4GP_FREE(pstr->_dL_a);
      NFFT4GP_FREE(pstr->_work);
      printf("FSAI total setup time %fs\n",pstr->_tset);
      if(pstr->_tits > 0)
      {
         printf("FSAI average solve time %fs\n",pstr->_titt/pstr->_tits);
      }
      if(pstr->_tlogdet > 0)
      {
         printf("FSAI trace and logdet time %fs\n",pstr->_tlogdet);
      }
      if(pstr->_tdvp > 0)
      {
         printf("FSAI DVP time %fs\n",pstr->_tdvp);
      }
      
      pstr->_n = 0;
      pstr->_tits = 0;
      pstr->_titt = 0.0;
      pstr->_tset = 0.0;

      // do not free pstr, remain for future work
   }
}

void Nfft4GPPrecondFsaiSetLfil(void *str, int lfil)
{
   pprecond_fsai pstr = (pprecond_fsai)str;
   pstr->_lfil = lfil;
}

int Nfft4GPPrecondFsaiSolve( void *vfsai_mat, int n, NFFT4GP_DOUBLE *x, NFFT4GP_DOUBLE *rhs)
{
   double ts, te;
   ts = Nfft4GPWtime();

   pprecond_fsai fsai_mat = (pprecond_fsai)vfsai_mat;
   /* INV = L'*L */
   Nfft4GPCsrMv( fsai_mat->_L_i, fsai_mat->_L_j, fsai_mat->_L_a, fsai_mat->_n, fsai_mat->_n,
          'N', 1.0, rhs, 0.0, fsai_mat->_work);
   Nfft4GPCsrMv( fsai_mat->_L_i, fsai_mat->_L_j, fsai_mat->_L_a, fsai_mat->_n, fsai_mat->_n,
          'T', 1.0, fsai_mat->_work, 0.0, x);

   te = Nfft4GPWtime();
   fsai_mat->_titt += (te-ts);
   fsai_mat->_tits ++;

   return 0;
}

int Nfft4GPPrecondFsaiDvp(void *vfsai_mat, int n, int *mask, NFFT4GP_DOUBLE *x, NFFT4GP_DOUBLE **yp)
{
   double ts, te;
   ts = Nfft4GPWtime();

   pprecond_fsai fsai_mat = (pprecond_fsai)vfsai_mat;

   int *mask_l = NULL;
   if(!mask)
   {
      NFFT4GP_MALLOC( mask_l, 3, int);
      mask_l[0] = 1;mask_l[1] = 1;mask_l[2] = 1;
   }

   if(!fsai_mat->_dL_a)
   {
      printf("Setup FSAI without gradient, trace not supported.\n");
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
   
   if(mask_l[0])
   {
      Nfft4GPPrecondFsaiInvLT( vfsai_mat, n, fsai_mat->_work, x);
      Nfft4GPCsrMv( fsai_mat->_L_i, fsai_mat->_L_j, fsai_mat->_dL_a, fsai_mat->_n, fsai_mat->_n,
            'T', 1.0, fsai_mat->_work, 0.0, y);
      Nfft4GPPrecondFsaiInvLT( vfsai_mat, n, fsai_mat->_work + fsai_mat->_n, y);
      Nfft4GPPrecondFsaiInvL( vfsai_mat, n, y, fsai_mat->_work);
      Nfft4GPCsrMv( fsai_mat->_L_i, fsai_mat->_L_j, fsai_mat->_dL_a, fsai_mat->_n, fsai_mat->_n,
            'N', 1.0, y, 1.0, fsai_mat->_work + fsai_mat->_n);
      Nfft4GPCsrMv( fsai_mat->_L_i, fsai_mat->_L_j, fsai_mat->_L_a, fsai_mat->_n, fsai_mat->_n,
            'T', 1.0, fsai_mat->_work + fsai_mat->_n, 0.0, y);
   }

   if(mask_l[1])
   {
      Nfft4GPPrecondFsaiInvLT( vfsai_mat, n, fsai_mat->_work, x);
      Nfft4GPCsrMv( fsai_mat->_L_i, fsai_mat->_L_j, fsai_mat->_dL_a + fsai_mat->_L_i[fsai_mat->_n], fsai_mat->_n, fsai_mat->_n,
            'T', 1.0, fsai_mat->_work, 0.0, y + fsai_mat->_n);
      Nfft4GPPrecondFsaiInvLT( vfsai_mat, n, fsai_mat->_work + fsai_mat->_n, y + fsai_mat->_n);
      Nfft4GPPrecondFsaiInvL( vfsai_mat, n, y + fsai_mat->_n, fsai_mat->_work);
      Nfft4GPCsrMv( fsai_mat->_L_i, fsai_mat->_L_j, fsai_mat->_dL_a + fsai_mat->_L_i[fsai_mat->_n], fsai_mat->_n, fsai_mat->_n,
            'N', 1.0, y + fsai_mat->_n, 1.0, fsai_mat->_work + fsai_mat->_n);
      Nfft4GPCsrMv( fsai_mat->_L_i, fsai_mat->_L_j, fsai_mat->_L_a, fsai_mat->_n, fsai_mat->_n,
            'T', 1.0, fsai_mat->_work + fsai_mat->_n, 0.0, y + fsai_mat->_n);
   }

   if(mask_l[2])
   {
      Nfft4GPPrecondFsaiInvLT( vfsai_mat, n, fsai_mat->_work, x);
      Nfft4GPCsrMv( fsai_mat->_L_i, fsai_mat->_L_j, fsai_mat->_dL_a + 2*fsai_mat->_L_i[fsai_mat->_n], fsai_mat->_n, fsai_mat->_n,
            'T', 1.0, fsai_mat->_work, 0.0, y + 2*fsai_mat->_n);
      Nfft4GPPrecondFsaiInvLT( vfsai_mat, n, fsai_mat->_work + fsai_mat->_n, y + 2*fsai_mat->_n);
      Nfft4GPPrecondFsaiInvL( vfsai_mat, n, y + 2*fsai_mat->_n, fsai_mat->_work);
      Nfft4GPCsrMv( fsai_mat->_L_i, fsai_mat->_L_j, fsai_mat->_dL_a + 2*fsai_mat->_L_i[fsai_mat->_n], fsai_mat->_n, fsai_mat->_n,
            'N', 1.0, y + 2*fsai_mat->_n, 1.0, fsai_mat->_work + fsai_mat->_n);
      Nfft4GPCsrMv( fsai_mat->_L_i, fsai_mat->_L_j, fsai_mat->_L_a, fsai_mat->_n, fsai_mat->_n,
            'T', 1.0, fsai_mat->_work + fsai_mat->_n, 0.0, y + 2*fsai_mat->_n);
   }
   
   if(*yp == NULL)
   {
      *yp = y;
   }

   te = Nfft4GPWtime();
   fsai_mat->_tdvp += (te - ts);

   if(!mask)
   {
      NFFT4GP_FREE(mask_l);
   }

   return 0;
}

int Nfft4GPPrecondFsaiTrace(void *vfsai_mat, NFFT4GP_DOUBLE **tracesp)
{
   double ts, te;
   ts = Nfft4GPWtime();

   pprecond_fsai fsai_mat = (pprecond_fsai)vfsai_mat;

   if(!fsai_mat->_dL_a)
   {
      printf("Setup FSAI without gradient, trace not supported.\n");
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

   int n = fsai_mat->_n;
   NFFT4GP_DOUBLE *L = fsai_mat->_L_a;
   NFFT4GP_DOUBLE *dL1, *dL2, *dL3;
   dL1 = fsai_mat->_dL_a;
   dL2 = dL1 + fsai_mat->_L_i[n];
   dL3 = dL2 + fsai_mat->_L_i[n];

   for(i = 1 ; i <= n ; i ++)
   {
      traces[0] += dL1[fsai_mat->_L_i[i]-1] / L[fsai_mat->_L_i[i]-1];
      traces[1] += dL2[fsai_mat->_L_i[i]-1] / L[fsai_mat->_L_i[i]-1];
      traces[2] += dL3[fsai_mat->_L_i[i]-1] / L[fsai_mat->_L_i[i]-1];
   }

   traces[0] *= 2;
   traces[1] *= 2;
   traces[2] *= 2;

   if(*tracesp == NULL)
   {
      *tracesp = traces;
   }

   te = Nfft4GPWtime();
   fsai_mat->_tlogdet += (te - ts);

   return 0;
}

NFFT4GP_DOUBLE Nfft4GPPrecondFsaiLogdet(void *vfsai_mat)
{
   double ts, te;
   ts = Nfft4GPWtime();

   pprecond_fsai fsai_mat = (pprecond_fsai)vfsai_mat;

   int i;
   int n = fsai_mat->_n;
   NFFT4GP_DOUBLE val = 0.0;
   NFFT4GP_DOUBLE *L = fsai_mat->_L_a;
   for(i = 1 ; i <= n ; i ++)
   {
      val += log(1.0/L[fsai_mat->_L_i[i]-1]);
   }

   val *= 2;

   te = Nfft4GPWtime();
   fsai_mat->_tlogdet += (te - ts);
   
   return val;
}

int Nfft4GPPrecondFsaiSetupWithKernel(NFFT4GP_DOUBLE *data, int n, int ldim, int d,
                                    func_kernel fkernel, void *fkernel_params, int require_grad, void* vfsai_mat)
{
   pprecond_fsai fsai_mat = (pprecond_fsai)vfsai_mat;

   // first get the sparsity pattern
   int *A_i, *A_j;
   Nfft4GPDistanceEuclidKnn( data, n, ldim, d, fsai_mat->_lfil, &A_i, &A_j);

   return Nfft4GPPrecondFsaiSetupWithKernelPattern( A_i, A_j, data, n, ldim, d, fkernel, fkernel_params, require_grad, vfsai_mat);
}

int Nfft4GPPrecondFsaiSetupWithKernelPattern(int *A_i, int *A_j, NFFT4GP_DOUBLE *data, int n, int ldim, int d,
                                    func_kernel fkernel, void *fkernel_params, int require_grad, void* vfsai_mat)
{
   double ts, te;
   ts = Nfft4GPWtime();

   int i;
   pprecond_fsai fsai_mat = (pprecond_fsai)vfsai_mat;
   pnfft4gp_kernel pfkernel = (pnfft4gp_kernel)fkernel_params;

   fsai_mat->_n = n;
   fsai_mat->_L_i = A_i;
   fsai_mat->_L_j = A_j;

   NFFT4GP_DOUBLE *A_a = NULL;
   NFFT4GP_MALLOC( A_a, A_i[n], NFFT4GP_DOUBLE);

#ifdef NFFT4GP_USING_MKL
   /* disable MKL multithreading during FSAI */
   int n_threads = mkl_get_max_threads();
   mkl_set_num_threads(1);
#endif

   if(!require_grad)
   {
      /* compute col by col of U in csc format */
#ifdef NFFT4GP_USING_OPENMP
      if(!omp_in_parallel())
      {
         int nthreads = omp_get_max_threads();
         NFFT4GP_DOUBLE *K_datag = NULL, *K_ag = NULL;
         int lfil = 0;
         for(i = 0 ; i < n ; i ++)
         {
            NFFT4GP_MAX(lfil, A_i[i+1]-A_i[i], lfil);
         }
         NFFT4GP_MALLOC( K_datag, (size_t)lfil*d*nthreads, NFFT4GP_DOUBLE);
         NFFT4GP_MALLOC( K_ag, (size_t)lfil*lfil*nthreads, NFFT4GP_DOUBLE);
         
         #pragma omp parallel
         {
            int myid = omp_get_thread_num();
            #pragma omp for private(i) NFFT4GP_DEFAULT_OPENMP_SCHEDULE
            for(i = 0 ; i < n ; i ++)
            {
               int j1, j2, k, info, nrhs;
               j1 = A_i[i];
               j2 = A_i[i+1];
               k = j2 - j1;

               char uplo, diag, trans;
               NFFT4GP_DOUBLE *K_data = K_datag+lfil*d*myid;
               NFFT4GP_DOUBLE *K_a = K_ag+lfil*lfil*myid;
               NFFT4GP_DOUBLE *A_ai = A_a + j1;

               uplo = 'L';
               diag = 'N';
               nrhs = 1;
               
               /* get the data array */
               Nfft4GPSubData2( data, n, ldim, d, A_j+j1, k, K_data);

               /* get the submatrix */
               if(pfkernel->_ibufferp)
               {
                  pfkernel->_ibufferp[myid] = A_j+j1;
                  pfkernel->_libufferp[myid] = k;
               }
               fkernel( fkernel_params, K_data, k, k, d, NULL, 0, NULL, 0, &K_a, NULL);
               
               /* get the right-hand-side ei */
               Nfft4GPVecFill( A_ai, k, 0.0);
               A_ai[k-1] = 1.0;
               
               /* Solve iKe = K_a \ ei */
               NFFT4GP_DPOTRF( &uplo, &k, K_a, &k, &info);

               trans = 'N';
               NFFT4GP_TRTRS( &uplo, &trans, &diag, &k, &nrhs, K_a, &k, A_ai, &k, &info);
               trans = 'T';
               NFFT4GP_TRTRS( &uplo, &trans, &diag, &k, &nrhs, K_a, &k, A_ai, &k, &info);

               /* scale by 1.0/sqrt(e'*K^{-1}*e)  */
               Nfft4GPVecScale( A_ai, k, 1.0 / sqrt(A_ai[k-1]));
            }
         }
         
         NFFT4GP_FREE(K_datag);
         NFFT4GP_FREE(K_ag);
      }
      else
      {
#endif
         {
            NFFT4GP_DOUBLE *K_a = NULL, *K_data = NULL;
            int lfil = 0;
            for(i = 0 ; i < n ; i ++)
            {
               NFFT4GP_MAX(lfil, A_i[i+1]-A_i[i]-1, lfil);
            }
            NFFT4GP_MALLOC( K_a, (size_t)lfil*lfil, NFFT4GP_DOUBLE);
            NFFT4GP_MALLOC( K_data, (size_t)lfil*d, NFFT4GP_DOUBLE);

            for(i = 0 ; i < n ; i ++)
            {
               int j1, j2, k, info, nrhs;
               j1 = A_i[i];
               j2 = A_i[i+1];
               k = j2 - j1 - 1;

               char uplo, diag, trans;
               NFFT4GP_DOUBLE *A_ai = A_a + j1;

               uplo = 'L';
               diag = 'N';
               nrhs = 1;

               /* get the data array */
               Nfft4GPSubData2( data, n, ldim, d, A_j+j1, k, K_data);

               /* get the submatrix */
               if(pfkernel->_ibufferp)
               {
                  pfkernel->_ibufferp[0] = A_j+j1;
                  pfkernel->_libufferp[0] = k;
               }
               fkernel( fkernel_params, K_data, k, k, d, NULL, 0, NULL, 0, &K_a, NULL);

               /* get the right-hand-side ei */
               Nfft4GPVecFill( A_ai, k, 0.0);
               A_ai[k-1] = 1.0;
               
               /* Solve iKe = K_a \ ei */
               NFFT4GP_DPOTRF( &uplo, &k, K_a, &k, &info);

               trans = 'N';
               NFFT4GP_TRTRS( &uplo, &trans, &diag, &k, &nrhs, K_a, &k, A_ai, &k, &info);
               trans = 'T';
               NFFT4GP_TRTRS( &uplo, &trans, &diag, &k, &nrhs, K_a, &k, A_ai, &k, &info);

               /* scale by 1.0/sqrt(e'*K^{-1}*e)  */
               Nfft4GPVecScale( A_ai, k, 1.0 / sqrt(A_ai[k-1]));
               
            }
            NFFT4GP_FREE(K_a);
            NFFT4GP_FREE(K_data);
         }
#ifdef NFFT4GP_USING_OPENMP
      }
#endif

      fsai_mat->_L_a = A_a;
      
      NFFT4GP_MALLOC( fsai_mat->_work, fsai_mat->_n, NFFT4GP_DOUBLE);

   }
   else /* gradient is also required in this case */
   {
      NFFT4GP_DOUBLE *dA_a = NULL;
      NFFT4GP_MALLOC( dA_a, 3*A_i[n], NFFT4GP_DOUBLE);
      NFFT4GP_DOUBLE *dA_a1 = dA_a;
      NFFT4GP_DOUBLE *dA_a2 = dA_a1 + A_i[n];
      NFFT4GP_DOUBLE *dA_a3 = dA_a2 + A_i[n];

      /* compute col by col of U in csc format */
#ifdef NFFT4GP_USING_OPENMP
      if(!omp_in_parallel())
      {
         int nthreads = omp_get_max_threads();
         NFFT4GP_DOUBLE *K_datag = NULL, *K_ag = NULL, *dK_ag = NULL;
         int lfil = 0;
         for(i = 0 ; i < n ; i ++)
         {
            NFFT4GP_MAX(lfil, A_i[i+1]-A_i[i], lfil);
         }
         NFFT4GP_MALLOC( K_datag, (size_t)lfil*d*nthreads, NFFT4GP_DOUBLE);
         NFFT4GP_MALLOC( K_ag, (size_t)lfil*lfil*nthreads, NFFT4GP_DOUBLE);
         NFFT4GP_MALLOC( dK_ag, (size_t)3*lfil*lfil*nthreads, NFFT4GP_DOUBLE);
         
         #pragma omp parallel
         {
            int myid = omp_get_thread_num();
            #pragma omp for private(i) NFFT4GP_DEFAULT_OPENMP_SCHEDULE
            for(i = 0 ; i < n ; i ++)
            {
               int j1, j2, k, info, nrhs;
               j1 = A_i[i];
               j2 = A_i[i+1];
               k = j2 - j1;

               char uplo, diag, trans;
               NFFT4GP_DOUBLE dd;
               NFFT4GP_DOUBLE *K_data = K_datag+lfil*d*myid;
               NFFT4GP_DOUBLE *K_a = K_ag+lfil*lfil*myid;
               NFFT4GP_DOUBLE *dK_a = dK_ag+3*lfil*lfil*myid;
               NFFT4GP_DOUBLE *dK_a1 = dK_a;
               NFFT4GP_DOUBLE *dK_a2 = dK_a1 + k*k;
               NFFT4GP_DOUBLE *dK_a3 = dK_a2 + k*k;
               NFFT4GP_DOUBLE *A_ai = A_a + j1;
               NFFT4GP_DOUBLE *dA_ai1 = dA_a1 + j1;
               NFFT4GP_DOUBLE *dA_ai2 = dA_a2 + j1;
               NFFT4GP_DOUBLE *dA_ai3 = dA_a3 + j1;

               uplo = 'L';
               diag = 'N';
               nrhs = 1;
               
               /* get the data array */
               Nfft4GPSubData2( data, n, ldim, d, A_j+j1, k, K_data);

               /* get the submatrix */
               if(pfkernel->_ibufferp)
               {
                  pfkernel->_ibufferp[myid] = A_j+j1;
                  pfkernel->_libufferp[myid] = k;
               }
               fkernel( fkernel_params, K_data, k, k, d, NULL, 0, NULL, 0, &K_a, &dK_a);

               /* get the right-hand-side ei */
               Nfft4GPVecFill( A_ai, k, 0.0);
               A_ai[k-1] = 1.0;
               
               /* Solve iKe = K_a \ ei */
               NFFT4GP_DPOTRF( &uplo, &k, K_a, &k, &info);

               trans = 'N';
               NFFT4GP_TRTRS( &uplo, &trans, &diag, &k, &nrhs, K_a, &k, A_ai, &k, &info);
               trans = 'T';
               NFFT4GP_TRTRS( &uplo, &trans, &diag, &k, &nrhs, K_a, &k, A_ai, &k, &info);

               /* scale by 1.0/sqrt(e'*K^{-1}*e)  */
               dd = 1.0 / sqrt(A_ai[k-1]);
               Nfft4GPVecScale( A_ai, k, dd);

               /* next create gradient */
               Nfft4GPDenseMatGemv( dK_a1, 'N', k, k, -1.0, A_ai, 0.0, dA_ai1);
               Nfft4GPDenseMatGemv( dK_a2, 'N', k, k, -1.0, A_ai, 0.0, dA_ai2);
               Nfft4GPDenseMatGemv( dK_a3, 'N', k, k, -1.0, A_ai, 0.0, dA_ai3);
               trans = 'N';
               NFFT4GP_TRTRS( &uplo, &trans, &diag, &k, &nrhs, K_a, &k, dA_ai1, &k, &info);
               NFFT4GP_TRTRS( &uplo, &trans, &diag, &k, &nrhs, K_a, &k, dA_ai2, &k, &info);
               NFFT4GP_TRTRS( &uplo, &trans, &diag, &k, &nrhs, K_a, &k, dA_ai3, &k, &info);
               trans = 'T';
               NFFT4GP_TRTRS( &uplo, &trans, &diag, &k, &nrhs, K_a, &k, dA_ai1, &k, &info);
               NFFT4GP_TRTRS( &uplo, &trans, &diag, &k, &nrhs, K_a, &k, dA_ai2, &k, &info);
               NFFT4GP_TRTRS( &uplo, &trans, &diag, &k, &nrhs, K_a, &k, dA_ai3, &k, &info);
               
               Nfft4GPVecAxpy( -0.5*dA_ai1[k-1]*dd, A_ai, k, dA_ai1);
               Nfft4GPVecAxpy( -0.5*dA_ai2[k-1]*dd, A_ai, k, dA_ai2);
               Nfft4GPVecAxpy( -0.5*dA_ai3[k-1]*dd, A_ai, k, dA_ai3);
            }
         }
         NFFT4GP_FREE(K_datag);
         NFFT4GP_FREE(K_ag);
      }
      else
      {
#endif
         {
            NFFT4GP_DOUBLE *K_a = NULL, *K_data = NULL, *dK_a = NULL;
            NFFT4GP_DOUBLE *dK_a1 = NULL, *dK_a2 = NULL, *dK_a3 = NULL;
            int lfil = 0;
            for(i = 0 ; i < n ; i ++)
            {
               NFFT4GP_MAX(lfil, A_i[i+1]-A_i[i]-1, lfil);
            }
            NFFT4GP_MALLOC( K_a, (size_t)lfil*lfil, NFFT4GP_DOUBLE);
            NFFT4GP_MALLOC( dK_a, (size_t)3*lfil*lfil, NFFT4GP_DOUBLE);
            NFFT4GP_MALLOC( K_data, (size_t)lfil*d, NFFT4GP_DOUBLE);

            dK_a1 = dK_a;

            for(i = 0 ; i < n ; i ++)
            {
               int j1, j2, k, info, nrhs;
               j1 = A_i[i];
               j2 = A_i[i+1];
               k = j2 - j1 - 1;

               char uplo, diag, trans;
               NFFT4GP_DOUBLE dd;
               NFFT4GP_DOUBLE *A_ai = A_a + j1;
               NFFT4GP_DOUBLE *dA_ai1 = dA_a1 + j1;
               NFFT4GP_DOUBLE *dA_ai2 = dA_a2 + j1;
               NFFT4GP_DOUBLE *dA_ai3 = dA_a3 + j1;

               uplo = 'L';
               diag = 'N';
               nrhs = 1;

               /* get the data array */
               Nfft4GPSubData2( data, n, ldim, d, A_j+j1, k, K_data);

               /* get the submatrix */
               if(pfkernel->_ibufferp)
               {
                  pfkernel->_ibufferp[0] = A_j+j1;
                  pfkernel->_libufferp[0] = k;
               }
               fkernel( fkernel_params, K_data, k, k, d, NULL, 0, NULL, 0, &K_a, &dK_a);
               dK_a2 = dK_a1 + k*k;
               dK_a3 = dK_a2 + k*k;

               /* get the right-hand-side ei */
               Nfft4GPVecFill( A_ai, k, 0.0);
               A_ai[k-1] = 1.0;
               
               /* Solve iKe = K_a \ ei */
               NFFT4GP_DPOTRF( &uplo, &k, K_a, &k, &info);

               trans = 'N';
               NFFT4GP_TRTRS( &uplo, &trans, &diag, &k, &nrhs, K_a, &k, A_ai, &k, &info);
               trans = 'T';
               NFFT4GP_TRTRS( &uplo, &trans, &diag, &k, &nrhs, K_a, &k, A_ai, &k, &info);

               /* scale by 1.0/sqrt(e'*K^{-1}*e)  */
               dd = 1.0 / sqrt(A_ai[k-1]);
               Nfft4GPVecScale( A_ai, k, dd);

               /* next create gradient */
               Nfft4GPDenseMatGemv( dK_a1, 'N', k, k, -1.0, A_ai, 0.0, dA_ai1);
               Nfft4GPDenseMatGemv( dK_a2, 'N', k, k, -1.0, A_ai, 0.0, dA_ai2);
               Nfft4GPDenseMatGemv( dK_a3, 'N', k, k, -1.0, A_ai, 0.0, dA_ai3);
               trans = 'N';
               NFFT4GP_TRTRS( &uplo, &trans, &diag, &k, &nrhs, K_a, &k, dA_ai1, &k, &info);
               NFFT4GP_TRTRS( &uplo, &trans, &diag, &k, &nrhs, K_a, &k, dA_ai2, &k, &info);
               NFFT4GP_TRTRS( &uplo, &trans, &diag, &k, &nrhs, K_a, &k, dA_ai3, &k, &info);
               trans = 'T';
               NFFT4GP_TRTRS( &uplo, &trans, &diag, &k, &nrhs, K_a, &k, dA_ai1, &k, &info);
               NFFT4GP_TRTRS( &uplo, &trans, &diag, &k, &nrhs, K_a, &k, dA_ai2, &k, &info);
               NFFT4GP_TRTRS( &uplo, &trans, &diag, &k, &nrhs, K_a, &k, dA_ai3, &k, &info);
               Nfft4GPVecAxpy( -0.5*dA_ai1[k-1]*dd, A_ai, k, dA_ai1);
               Nfft4GPVecAxpy( -0.5*dA_ai2[k-1]*dd, A_ai, k, dA_ai2);
               Nfft4GPVecAxpy( -0.5*dA_ai3[k-1]*dd, A_ai, k, dA_ai3);
               
            }
            NFFT4GP_FREE(K_a);
            NFFT4GP_FREE(K_data);
         }
#ifdef NFFT4GP_USING_OPENMP
      }
#endif

      fsai_mat->_L_a = A_a;
      fsai_mat->_dL_a = dA_a;
      
      NFFT4GP_MALLOC( fsai_mat->_work, 2*fsai_mat->_n, NFFT4GP_DOUBLE);

   }

#ifdef NFFT4GP_USING_MKL
   /* reset */
   mkl_set_num_threads(n_threads);
#endif

   te = Nfft4GPWtime();
   fsai_mat->_tset += te - ts;

   return 0;
}

int Nfft4GPPrecondFsaiInvL( void *vfsai_mat, int n, NFFT4GP_DOUBLE *x, NFFT4GP_DOUBLE *rhs)
{

   pprecond_fsai fsai_mat = (pprecond_fsai)vfsai_mat;

   int   i, j, k1, k2;
   
   for( i = 0; i < n; i++ )
   {
      x[i] = rhs[i];
   }
   
   for( i = 0; i < n; i++ )
   {
      k1 = fsai_mat->_L_i[i]; 
      k2 = fsai_mat->_L_i[i+1];
      for(j = k1 ; j < k2-1; j++) 
      {
         x[i] -= fsai_mat->_L_a[j] * x[fsai_mat->_L_j[j]];
      }
      x[i] /= fsai_mat->_L_a[k2-1];
   }
   
   return 0;
}

int Nfft4GPPrecondFsaiInvLT( void *vfsai_mat, int n, NFFT4GP_DOUBLE *x, NFFT4GP_DOUBLE *rhs)
{

   pprecond_fsai fsai_mat = (pprecond_fsai)vfsai_mat;

   int   i, j, k1, k2;
   
   for( i = 0; i < n; i++ )
   {
      x[i] = rhs[i];
   }
   
   for( i = n-1; i >= 0; i-- ) 
   {
      k1 = fsai_mat->_L_i[i]; 
      k2 = fsai_mat->_L_i[i+1];
      x[i] /= fsai_mat->_L_a[k2-1];
      for( j = k1 ; j < k2-1; j++) 
      {
         x[fsai_mat->_L_j[j]] -= fsai_mat->_L_a[j] * x[i];
      }
   }
   
   return 0;
}

//TODO: use global variable is not elegant
int nfft4gp_fsai_plot = 1;

void Nfft4GPPrecondFsaiPlot(void *vfsai_mat)
{
   pprecond_fsai fsai_mat = (pprecond_fsai)vfsai_mat;
   if(nfft4gp_fsai_plot)
   {
      TestPlotCSRMatrix( fsai_mat->_L_i, fsai_mat->_L_j, fsai_mat->_L_a, fsai_mat->_n, fsai_mat->_n, "fsai");
      nfft4gp_fsai_plot = 0;
   }
}