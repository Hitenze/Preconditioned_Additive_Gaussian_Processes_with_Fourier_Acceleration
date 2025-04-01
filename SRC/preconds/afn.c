#include "afn.h"

// Major functions
void* Nfft4GPPrecondAFNCreate()
{
   precond_afn *str = NULL;
   NFFT4GP_MALLOC( str, 1, precond_afn);
   str->_n = 0;
   str->_k = 0;
   str->_own_perm = 0;
   str->_perm = NULL;

   str->_fA11_solve = &Nfft4GPPrecondCholSolve;
   str->_fA11_solve_free_data = &Nfft4GPPrecondCholFree;
   str->_fA11_solve_data = NULL;

   str->_fA12_matvec_own_data = 0;
   str->_fA12_matvec = NULL;
   str->_fA12_matvec_free_data = NULL;
   str->_fA12_matvec_data = NULL;

   str->_fS_solve_own_data = 0;
   str->_fS_solve = &Nfft4GPPrecondFsaiSolve;
   str->_fS_solve_free_data = &Nfft4GPPrecondFsaiFree;
   str->_fS_solve_data = NULL;

   return (void*) str;
}

void Nfft4GPPrecondAFNFree(void *str)
{
   precond_afn *pstr = (precond_afn*)str;
   if(pstr)
   {
      if(pstr->_own_perm)
      {
         NFFT4GP_FREE(pstr->_perm);
      }
      if(pstr->_fA11_solve_data != NULL)
      {
         if(pstr->_fA11_solve_free_data)
         {
            pstr->_fA11_solve_free_data(pstr->_fA11_solve_data);
         }
         else
         {
            NFFT4GP_FREE(pstr->_fA11_solve_data);
         }
      }
      if(pstr->_fA12_matvec_own_data)
      {
         if(pstr->_fA12_matvec_free_data)
         {
            pstr->_fA12_matvec_free_data(pstr->_fA12_matvec_data);
         }
         else
         {
            NFFT4GP_FREE(pstr->_fA12_matvec_data);
         }
      }
      if(pstr->_fS_solve_own_data)
      {
         if(pstr->_fS_solve_free_data)
         {
            pstr->_fS_solve_free_data(pstr->_fS_solve_data);
         }
         else
         {
            NFFT4GP_FREE(pstr->_fS_solve_data);
         }
      }
      printf("AFN total setup time %fs\n",pstr->_tset);
      if(pstr->_tits > 0)
      {
         printf("AFN average solve time %fs\n",pstr->_titt/pstr->_tits);
      }
      NFFT4GP_FREE(pstr);
   }
}


int Nfft4GPPrecondAFNSolve( void *vafn_mat, int n, NFFT4GP_DOUBLE *x, NFFT4GP_DOUBLE *rhs)
{
   NFFT4GP_DOUBLE ts, te;
   ts = Nfft4GPWtime();

   pprecond_afn pafn_mat = (pprecond_afn)vafn_mat;

   int i, n2;
   NFFT4GP_DOUBLE *rp, *rp2, *y, *y2;
   
   n2 = pafn_mat->_n - pafn_mat->_k;
   rp = pafn_mat->_dwork;
   rp2 = rp + pafn_mat->_k;
   y = pafn_mat->_dwork + pafn_mat->_n;
   y2 = y + pafn_mat->_k;

   /* handle two extrean cases */
   if(n2 == 0)
   {
      /* y = A11 \ rp */
      pafn_mat->_fA11_solve( pafn_mat->_fA11_solve_data, pafn_mat->_k, x, rhs);
   }
   else if(pafn_mat->_k == 0)
   {
      /* y = rp / noise_level */
      pafn_mat->_fS_solve( pafn_mat->_fS_solve_data, n2, x, rhs);
   }
   else
   {
      /* get permuted rhs [rp;rp2] = rhs(perm) */
      for(i = 0 ; i <  pafn_mat->_n ; i++)
      {
         rp[i] = rhs[ pafn_mat->_perm[i]];
      }

      /* y = A11 \ rp */
      pafn_mat->_fA11_solve( pafn_mat->_fA11_solve_data, pafn_mat->_k, y, rp);

      /* rp2 = rp2 - A21 * y */
      pafn_mat->_fA12_matvec( pafn_mat->_fA12_matvec_data, 'T', pafn_mat->_k, n2, -1.0, y, 1.0, rp2);

      /* y2 = precond(rp2) */
      pafn_mat->_fS_solve( pafn_mat->_fS_solve_data, n2, y2, rp2);
      
      /* rp = rp - A12 * y2 */
      pafn_mat->_fA12_matvec( pafn_mat->_fA12_matvec_data, 'N', pafn_mat->_k, n2, -1.0, y2, 1.0, rp);
      
      /* y = A11 \ rp */
      pafn_mat->_fA11_solve( pafn_mat->_fA11_solve_data, pafn_mat->_k, y, rp);

      /* reverse permutation */
      for(i = 0 ; i <  pafn_mat->_n ; i++)
      {
         x[pafn_mat->_perm[i]] = y[i];
      }
   }

   te = AfnWtime();
   pafn_mat->_tits ++;
   pafn_mat->_titt += te - ts;
   return 0;
}

NFFT4GP_DOUBLE Nfft4GPPreconddAFNTrace( void *vafn_mat)
{
   NFFT4GP_DOUBLE ts, te;
   ts = Nfft4GPWtime();

   pprecond_afn pafn_mat = (pprecond_afn)vafn_mat;

   NFFT4GP_DOUBLE trace = 0.0;

   te = Nfft4GPWtime();
   pafn_mat->_tits ++;
   pafn_mat->_titt += te - ts;
   // printf("The tracce is %f\n", trace);
   return trace;
}

void* Nfft4GPPrecondAFNSetup(NFFT4GP_DOUBLE *data, int n, int ldim, int d,
                           int max_k, int perm_opt, int schur_opt, int schur_lfil, int nsamples,
                           func_kernel fkernel, void *fkernel_params, int require_grad,
                           void* vafn_mat)
{
   NFFT4GP_DOUBLE ts, te, t1, t2;
   ts = Nfft4GPWtime();

   NFFT4GP_MIN(max_k,n,max_k);

   pprecond_afn afn_mat = (pprecond_afn) vafn_mat;

   afn_mat->_n = n;

   int n2;
   
   /***********************
    * 1. Rank estimation
    ***********************/
   t1 = Nfft4GPWtime();

   if(max_k > 0)
   {
      prankest rank_str = (prankest)Nfft4GPRankestStrCreate();
      rank_str->_kernel_func = fkernel;
      rank_str->_kernel_str = fkernel_params;
      rank_str->_max_rank = max_k;
      rank_str->_nsample = nsamples;

      // first apply the scaled rank estimation
      int rank = Nfft4GPRankestNysScaled( rank_str, data, n, ldim, d);

      if(rank >= max_k)
      {
         printf("This problem is not low-rank, skip nonscaled rank estimation\n");
         int *perm = NULL;
         afn_mat->_k = max_k;
         if(perm_opt == 1)
         {
            // FPS ordering
            pordering_fps pfps = (pordering_fps)Nfft4GPOrdFpsCreate();
            pfps->_algorithm = kFpsAlgorithmParallel1;
            pfps->_build_pattern = 0; // no pattern needed
            pfps->_tol = 0.0;
            Nfft4GPSortFps((void*)pfps, data, n, ldim, d, &(afn_mat->_k), &perm);
            Nfft4GPOrdFpsFree((void*)pfps);
         }
         else
         {
            // Random ordering
            void *fprand = Nfft4GPOrdRandCreate();
            Nfft4GPSortRand(fprand, data, n, ldim, d, &(afn_mat->_k), &perm);
            Nfft4GPOrdRandFree(fprand);
         }
         afn_mat->_perm = Nfft4GPExpandPerm( perm, afn_mat->_k, n);
         n2 = n - afn_mat->_k;
         afn_mat->_own_perm = 1;
         NFFT4GP_FREE(perm);
      }
      else
      {
         afn_mat->_k = Nfft4GPRankestDefault( rank_str, data, n, ldim, d);
         if( afn_mat->_k == max_k && perm_opt == 0 )
         {
            // Random ordering
            int *perm = NULL;
            afn_mat->_k = max_k;
            void *fprand = Nfft4GPOrdRandCreate();
            Nfft4GPSortRand(fprand, data, n, ldim, d, &(afn_mat->_k), &perm);
            Nfft4GPOrdRandFree(fprand);
            afn_mat->_perm = Nfft4GPExpandPerm( perm, afn_mat->_k, n);
            n2 = n - afn_mat->_k;
            afn_mat->_own_perm = 1;
            NFFT4GP_FREE(perm);
         }
         else
         {
            n2 = n - afn_mat->_k;
            afn_mat->_perm = Nfft4GPExpandPerm( rank_str->_perm, afn_mat->_k, n);
            afn_mat->_own_perm = 1;
         }
      }
      Nfft4GPRankestStrFree(rank_str);
   }
   else
   {
      /* In this case we skip the rank estimation
       * and use the predefined rank as the real rank value 
       */
      NFFT4GP_MIN(-max_k,n,max_k);
      afn_mat->_k = max_k;
      n2 = n-max_k;

      afn_mat->_perm = Nfft4GPExpandPerm( NULL, 0, n);
      afn_mat->_own_perm = 1;
   }

   t2 = Nfft4GPWtime();
   printf("---------------------------------------------------\n");
   printf("Rank estimation time: %fs wih rank %d \n", t2-t1, afn_mat->_k);
   printf("---------------------------------------------------\n");

   /*****************************
    * 2. Build preconditioner
    *****************************/
   
   t1 = AfnWtime();
   if(afn_mat->_k == 0)
   {
      // in this case we only need the Schur complement preconditioner
      // this option is disabled for now
      switch(schur_opt)
      {
         default:
         {
            // kernel FSAI
            afn_mat->_fS_solve = &AfnPrecondFsaiSolve;
            afn_mat->_fS_solve_free_data = &AfnPrecondFsaiFree;
            afn_mat->_fS_solve_data = AfnPrecondFsaiSetupWithKernel( data, n, ldim, d, schur_lfil, fkernel, fkernel_params);
            break;
         }
      }

   }
   else if(afn_mat->_k == n)
   {
      // in this case we only need the A11 preconditioner
      afn_mat->_fA11_solve = &AfnPrecondCholSolve;
      afn_mat->_fA11_solve_free_data = &AfnPrecondCholFree;
      afn_mat->_fA11_solve_data = AfnPrecondCholSetupWithKernel( data, n, ldim, d, 0, fkernel, fkernel_params);
   }
   else
   {
      if(afn_mat->_k < max_k)
      {
         // does not reach max_k, we use the NYS preconditioner
         afn_mat->_fA11_solve = &AfnPrecondNysSolve;
         afn_mat->_fA11_solve_free_data = &AfnPrecondNysFree;
         afn_mat->_fA11_solve_data = AfnPrecondNysSetupWithKernelandPerm2( data, n, ldim, d, afn_mat->_k, 1, afn_mat->_perm, fkernel, fkernel_params, require_grad);
         // the onewrship has been passed to NYS, we use the full NYS solver
         afn_mat->_k = n;
         printf("The rank is %d\n", afn_mat->_k);
         afn_mat->_perm = NULL;
      }
      else
      {
         // first build A11 chol
         AFN_DOUBLE *data1 = AfnSubData( data, n, ldim, d, afn_mat->_perm, afn_mat->_k);

         afn_mat->_fA11_solve = &AfnPrecondCholSolve;
         afn_mat->_fA11_solve_free_data = &AfnPrecondCholFree;
         afn_mat->_fA11_solve_data = AfnPrecondCholSetupWithKernel( data1, afn_mat->_k, afn_mat->_k, d, 0, fkernel, fkernel_params);

         AFN_FREE(data1);
         
         // build A12 for matvec
         AFN_DOUBLE *A12 = NULL;
         fkernel( fkernel_params, data, n, ldim, d, afn_mat->_perm, afn_mat->_k, afn_mat->_perm+afn_mat->_k, n2, &A12, NULL);

         afn_mat->_fA12_matvec_data = (void*)A12;
         afn_mat->_fA12_matvec = &AfnDenseMatGemv;

         // build Schur complement solve
         switch(schur_opt)
         {
            case 0:
            {
               // use the inverse noise level
               afn_mat->_fS_solve = &AfnPrecondScaleSolve;
               afn_mat->_fS_solve_free_data = &AfnPrecondScaleFree;
               pafn_kernel pkernel = (pafn_kernel)fkernel_params;
               afn_mat->_fS_solve_data = AfnPrecondScaleSetup( n2, 1.0/(pkernel->_noise_level));
               break;
            }
            case 1:
            {
               // matrix FSAI
               // In this case build the Schur complement preconditioner via explicit Schur complement matrix
               // get the S part of the data
               AFN_DOUBLE *data2 = AfnSubData( data, n, ldim, d, afn_mat->_perm + afn_mat->_k, n2);
               
               // create the matrix G*K12, K12 first
               AFN_DOUBLE *K12 = NULL;
               fkernel( fkernel_params, data, n, ldim, d, afn_mat->_perm, afn_mat->_k, afn_mat->_perm+afn_mat->_k, n2, &K12, NULL);
               
               char transn = 'N';
               int nknowns = afn_mat->_k;
               int nrhs = n2;
               pprecond_chol chol_mat = (pprecond_chol)afn_mat->_fA11_solve_data;
               
               // apply G onto K12
               AFN_TRTRS( &chol_mat->_uplo, &transn, &chol_mat->_diag, &nknowns, &nrhs, chol_mat->_chol_data, &nknowns, K12, &nknowns, &chol_mat->_info);

               // next, we create the Schur complement matrix via K22 - K12'*K12
               AFN_DOUBLE *K22 = NULL;
               fkernel( fkernel_params, data, n, ldim, d, afn_mat->_perm+afn_mat->_k, n2, afn_mat->_perm+afn_mat->_k, n2, &K22, NULL);
               
               // get pattern use K22
               //int *A_i, *A_j;
               //AfnDistanceEuclidMatrixKnn( K22, n2, n2, schur_lfil, &A_i, &A_j);

               //update K22 with K12'*K12
               char transt = 'T';
               AFN_DOUBLE mone = -1.0, one = 1.0;
               AFN_DGEMM( &transt, &transn, &n2, &n2, &afn_mat->_k, &mone, 
                  K12, &afn_mat->_k, K12, &afn_mat->_k, &one, K22, &n2);

               afn_mat->_fS_solve = &AfnPrecondFsaiSolve;
               afn_mat->_fS_solve_free_data = &AfnPrecondFsaiFree;
               afn_mat->_fS_solve_data = AfnPrecondFsaiSetupWithMatrix( K22, n2, n2, schur_lfil);
               //afn_mat->_fS_solve_data = AfnPrecondFsaiSetupWithMatrixPattern( A_i, A_j, K22, n2, n2);
               
               // after construction this buffer is nolonger needed
               AFN_FREE(K12);
               AFN_FREE(K22);
               free(data2);
               
               break;
            }
            case 2:
            {
               // matrix FSAI
               // In this case build the Schur complement preconditioner via explicit Schur complement matrix
               // get the S part of the data
               AFN_DOUBLE *data2 = AfnSubData( data, n, ldim, d, afn_mat->_perm + afn_mat->_k, n2);
               
               // create the matrix G*K12, K12 first
               AFN_DOUBLE *K12 = NULL;
               fkernel( fkernel_params, data, n, ldim, d, afn_mat->_perm, afn_mat->_k, afn_mat->_perm+afn_mat->_k, n2, &K12, NULL);
               
               char transn = 'N';
               int nknowns = afn_mat->_k;
               int nrhs = n2;
               pprecond_chol chol_mat = (pprecond_chol)afn_mat->_fA11_solve_data;
               
               // apply G onto K12
               AFN_TRTRS( &chol_mat->_uplo, &transn, &chol_mat->_diag, &nknowns, &nrhs, chol_mat->_chol_data, &nknowns, K12, &nknowns, &chol_mat->_info);

               // next, we create the Schur complement matrix via K22 - K12'*K12
               AFN_DOUBLE *K22 = NULL;
               fkernel( fkernel_params, data, n, ldim, d, afn_mat->_perm+afn_mat->_k, n2, afn_mat->_perm+afn_mat->_k, n2, &K22, NULL);
               
               // get pattern use K22
               int *A_i, *A_j;
               AfnDistanceEuclidMatrixKnn( K22, n2, n2, schur_lfil, &A_i, &A_j);

               //update K22 with K12'*K12
               char transt = 'T';
               AFN_DOUBLE mone = -1.0, one = 1.0;
               AFN_DGEMM( &transt, &transn, &n2, &n2, &afn_mat->_k, &mone, 
                  K12, &afn_mat->_k, K12, &afn_mat->_k, &one, K22, &n2);

               afn_mat->_fS_solve = &AfnPrecondFsaiSolve;
               afn_mat->_fSGT_matvec = &AfnPrecondFsaiLMv;
               afn_mat->_fSG_matvec =  &AfnPrecondFsaiLTMv;
               afn_mat->_dfSGT_matvec = &AfnPreconddFsaiLMv;
               afn_mat->_fS_solve_free_data = &AfnPrecondFsaiFree;
               //afn_mat->_fS_solve_data = AfnPrecondFsaiSetupWithMatrix( K22, n2, n2, schur_lfil);
               afn_mat->_fS_solve_data = AfnPrecondFsaiSetupWithMatrixPattern( A_i, A_j, K22, n2, n2);
               
               // after construction this buffer is nolonger needed
               AFN_FREE(K12);
               AFN_FREE(K22);
               free(data2);
               
               break;
            }
            case 3: default:
            {
               // kernel FSAI
               
               // In this case build the Schur complement preconditioner via Schur complement Kernel
               // get the S part of the data
               AFN_DOUBLE *data2 = AfnSubData( data, n, ldim, d, afn_mat->_perm + afn_mat->_k, n2);
               
               // create the matrix G*K12, K12 first
               AFN_DOUBLE *K12 = NULL;
               fkernel( fkernel_params, data, n, ldim, d, afn_mat->_perm, afn_mat->_k, afn_mat->_perm+afn_mat->_k, n2, &K12, NULL);
               char transn = 'N';
               int nknowns = afn_mat->_k;
               int nrhs = n2;
               pprecond_chol chol_mat = (pprecond_chol)afn_mat->_fA11_solve_data;
               
               // apply G onto K12
               AFN_TRTRS( &chol_mat->_uplo, &transn, &chol_mat->_diag, &nknowns, &nrhs, chol_mat->_chol_data, &nknowns, K12, &nknowns, &chol_mat->_info);

               pafn_kernel pkernel = (pafn_kernel)AfnKernelParamCreate(n,1);
               pkernel->_iparams[0] = afn_mat->_k;
               printf("The rank is %d\n", afn_mat->_k);
               pkernel->_buffer = K12;
               pkernel->_fkernel_buffer = fkernel;
               pkernel->_fkernel_buffer_params = fkernel_params;

#ifdef AFN_USING_OPENMP
               int nthreads = omp_get_max_threads();
               pkernel->_ldwork = (size_t)nthreads*afn_mat->_k*(schur_lfil+1);
               AFN_MALLOC( pkernel->_dwork, pkernel->_ldwork, AFN_DOUBLE);
               //pkernel->_ldwork2 = (size_t)nthreads*afn_mat->_k*schur_lfil;
               //AFN_MALLOC( pkernel->_dwork2, pkernel->_ldwork2, AFN_DOUBLE);
               //pkernel->_ldwork3 = (size_t)nthreads*afn_mat->_k*1;
               //AFN_MALLOC( pkernel->_dwork3, pkernel->_ldwork3, AFN_DOUBLE);
#else
               pkernel->_ldwork = (size_t)afn_mat->_k*(schur_lfil+1);
               AFN_MALLOC( pkernel->_dwork, pkernel->_ldwork, AFN_DOUBLE);
               //pkernel->_ldwork2 = (size_t)afn_mat->_k*schur_lfil;
               //AFN_MALLOC( pkernel->_dwork2, pkernel->_ldwork2, AFN_DOUBLE);
               //pkernel->_ldwork3 = (size_t)afn_mat->_k*1;
               //AFN_MALLOC( pkernel->_dwork3, pkernel->_ldwork3, AFN_DOUBLE);
#endif

               afn_mat->_fS_solve = &AfnPrecondFsaiSolve;
               afn_mat->_fS_solve_free_data = &AfnPrecondFsaiFree;
               afn_mat->_fS_solve_data = AfnPrecondFsaiSetupWithKernel( data2, n2, n2, d, schur_lfil, &AfnKernelSchurCombineKernel, pkernel);

               // after construction this buffer is nolonger needed
               free(K12);
               free(data2);
               AfnKernelParamFree(pkernel);

               break;
            }
         }
      }

   }

   t2 = AfnWtime();

   printf("---------------------------------------------------\n");
   printf("AFN Preconditioner (Exclude rankest) time: %fs\n", t2-t1);
   printf("---------------------------------------------------\n");

   /* create working buffer */
   AFN_CALLOC( afn_mat->_dwork, 4*(size_t)n, AFN_DOUBLE);

   te = AfnWtime();
   afn_mat->_tset = te - ts;

   return (void*) afn_mat;
}


/* set up of AFN gradient  */
void* AfnPreconddAFNSetup(AFN_DOUBLE *data, int n, int ldim, int d,
                           int max_k, int perm_opt, int schur_opt, int schur_lfil, int nsamples,
                           func_kernel fkernel, func_kernel fdkernel, void *fkernel_params)
{
   AFN_DOUBLE ts, te, t1, t2;
   ts = AfnWtime();

   AFN_MIN(max_k,n,max_k);

   pprecond_afn afn_mat = (pprecond_afn)AfnPrecondAFNCreate();

   afn_mat->_n = n;

   int n2;
   
   /***********************
    * 1. Rank estimation
    ***********************/
   t1 = AfnWtime();

   if(max_k > 0)
   {
      prankest rank_str = (prankest)AfnRankestStrCreate();
      rank_str->_kernel_func = fkernel;
      rank_str->_kernel_str = fkernel_params;
      rank_str->_max_rank = max_k;
      rank_str->_nsample = nsamples;

      // first apply the scaled rank estimation
      int rank = AfnRankestNysScaled( rank_str, data, n, ldim, d);

      if(rank >= max_k)
      {
         printf("This problem is not low-rank, skip nonscaled rank estimation\n");
         int *perm = NULL;
         afn_mat->_k = max_k;
         if(perm_opt == 1)
         {
            // FPS ordering
            pordering_fps pfps = (pordering_fps)AfnOrdFpsCreate();
            pfps->_algorithm = kFpsAlgorithmParallel1;
            pfps->_build_pattern = 0; // no pattern needed
            pfps->_tol = 0.0;
            AfnSortFps((void*)pfps, data, n, ldim, d, &(afn_mat->_k), &perm);
            AfnOrdFpsFree((void*)pfps);
         }
         else
         {
            // Random ordering
            void *fprand = AfnOrdRandCreate();
            AfnSortRand(fprand, data, n, ldim, d, &(afn_mat->_k), &perm);
            AfnOrdRandFree(fprand);
         }
         afn_mat->_perm = AfnExpandPerm( perm, afn_mat->_k, n);
         n2 = n - afn_mat->_k;
         afn_mat->_own_perm = 1;
         AFN_FREE(perm);

      }
      else
      {
         afn_mat->_k = AfnRankestDefault( rank_str, data, n, ldim, d);
         if( afn_mat->_k == max_k && perm_opt == 0 )
         {
            // Random ordering
            int *perm = NULL;
            afn_mat->_k = max_k;
            void *fprand = AfnOrdRandCreate();
            AfnSortRand(fprand, data, n, ldim, d, &(afn_mat->_k), &perm);
            AfnOrdRandFree(fprand);
            afn_mat->_perm = AfnExpandPerm( perm, afn_mat->_k, n);
            n2 = n - afn_mat->_k;
            afn_mat->_own_perm = 1;
            AFN_FREE(perm);
         }
         else
         {
            n2 = n - afn_mat->_k;
            afn_mat->_perm = AfnExpandPerm( rank_str->_perm, afn_mat->_k, n);
            afn_mat->_own_perm = 1;
         }
      }
      AfnRankestStrFree(rank_str);
   }
   else
   {
      afn_mat->_k = 0;
      n2 = n;

      afn_mat->_perm = AfnExpandPerm( NULL, 0, n);
      afn_mat->_own_perm = 1;
   }

   t2 = AfnWtime();
   printf("---------------------------------------------------\n");
   printf("Rank estimation time: %fs wih rank %d \n", t2-t1, afn_mat->_k);
   printf("---------------------------------------------------\n");

   /*****************************
    * 2. Build preconditioner
    *****************************/
   
   t1 = AfnWtime();
   if(afn_mat->_k == 0)
   {
      // in this case we only need the Schur complement preconditioner
      switch(schur_opt)
      {
         case 0:
         {
            // use the inverse noise level
            afn_mat->_fS_solve = &AfnPrecondScaleSolve;
            afn_mat->_fS_solve_free_data = &AfnPrecondScaleFree;
            pafn_kernel pkernel = (pafn_kernel)fkernel_params;
            afn_mat->_fS_solve_data = AfnPrecondScaleSetup( n, 1.0/(pkernel->_noise_level));
            break;
         }
         case 1:
         {
            // matrix FSAI
            afn_mat->_fS_solve = &AfnPrecondFsaiSolve;
            afn_mat->_fS_solve_free_data = &AfnPrecondFsaiFree;
            AFN_DOUBLE *K = NULL;
            fkernel( fkernel_params, data, n, ldim, d, afn_mat->_perm, afn_mat->_n, afn_mat->_perm, afn_mat->_n, &K, NULL);
            afn_mat->_fS_solve_data = AfnPrecondFsaiSetupWithMatrix( K, afn_mat->_n, afn_mat->_n, schur_lfil);
            AFN_FREE(K);
            break;
         }
         case 2: default:
         {
            // kernel FSAI
            afn_mat->_fS_solve = &AfnPrecondFsaiSolve;
            afn_mat->_fS_solve_free_data = &AfnPrecondFsaiFree;
            afn_mat->_fS_solve_data = AfnPrecondFsaiSetupWithKernel( data, n, ldim, d, schur_lfil, fkernel, fkernel_params);
            break;
         }
      }

   }
   else if(afn_mat->_k == n)
   {
      // in this case we only need the A11 preconditioner
      afn_mat->_fA11_solve = &AfnPrecondCholSolve;
      afn_mat->_fA11_solve_free_data = &AfnPrecondCholFree;
      afn_mat->_fA11_solve_data = AfnPreconddCholSetupWithKernel( data, n, ldim, d, 0, fkernel, fdkernel, fkernel_params);
   }
   else
   {
      if(afn_mat->_k < max_k)
      {
         // does not reach max_k, we use the RAN preconditioner
         afn_mat->_fA11_solve = &AfnPrecondNysSolve;
         afn_mat->_fA11_solve_free_data = &AfnPrecondNysFree;
         afn_mat->_fA11_solve_data = AfnPrecondNysSetupWithKernelandPerm( data, n, ldim, d, afn_mat->_k, 1, afn_mat->_perm, fkernel, fkernel_params);
         // the onewrship has been passed to RAN, we use the full RAN solver
         afn_mat->_k = n;
         printf("The rank is %d\n", afn_mat->_k);
         afn_mat->_perm = NULL;
      }
      else
      {
         // first build A11 chol
         AFN_DOUBLE *data1 = AfnSubData( data, n, ldim, d, afn_mat->_perm, afn_mat->_k);

         afn_mat->_fA11_solve = &AfnPrecondCholSolve;
         afn_mat->_fA11_solve_free_data = &AfnPrecondCholFree;
         afn_mat->_fA11_solveL = &AfnPrecondCholSolve; // solve
         afn_mat->_fA11_solveLT = &AfnPrecondCholSolve; // solveT
         afn_mat->_fdA11_matvec = &AfnDenseMatGemv;
         afn_mat->_fdA11T_matvec = &AfnDenseMatGemv;
         afn_mat->_fA11_solve_data = AfnPreconddCholSetupWithKernel( data1, afn_mat->_k, afn_mat->_k, d, 0, fkernel, fdkernel, fkernel_params);

         AFN_FREE(data1);
         
         // build A12 for matvec
         AFN_DOUBLE *A12 = NULL;
         fkernel( fkernel_params, data, n, ldim, d, afn_mat->_perm, afn_mat->_k, afn_mat->_perm+afn_mat->_k, n2, &A12, NULL);
         afn_mat->_fA12_matvec_data = (void*)A12;
         afn_mat->_fA12_matvec = &AfnDenseMatGemv;

         AFN_DOUBLE *dA12 = NULL;
         fdkernel(fkernel_params, data, n, ldim, d, afn_mat->_perm, afn_mat->_k, afn_mat->_perm+afn_mat->_k, n2, &dA12, NULL);
         afn_mat->_fdA12_matvec_data = (void*)dA12;
         afn_mat->_fdA12_matvec = &AfnDenseMatGemv;

         // build Schur complement solve
         switch(schur_opt)
         {
            case 0:
            {
               // use the inverse noise level
               afn_mat->_fS_solve = &AfnPrecondScaleSolve;
               afn_mat->_fS_solve_free_data = &AfnPrecondScaleFree;
               pafn_kernel pkernel = (pafn_kernel)fkernel_params;
               afn_mat->_fS_solve_data = AfnPrecondScaleSetup( n2, 1.0/(pkernel->_noise_level));
               break;
            }
            case 1:
            {
               // matrix FSAI
               // In this case build the Schur complement preconditioner via explicit Schur complement matrix
               // get the S part of the data
               AFN_DOUBLE *data2 = AfnSubData( data, n, ldim, d, afn_mat->_perm + afn_mat->_k, n2);
               
               // create the matrix G*K12, K12 first
               AFN_DOUBLE *K12 = NULL;
               fkernel( fkernel_params, data, n, ldim, d, afn_mat->_perm, afn_mat->_k, afn_mat->_perm+afn_mat->_k, n2, &K12, NULL);
               
               char transn = 'N';
               int nknowns = afn_mat->_k;
               int nrhs = n2;
               pprecond_chol chol_mat = (pprecond_chol)afn_mat->_fA11_solve_data;
               
               // apply G onto K12
               AFN_TRTRS( &chol_mat->_uplo, &transn, &chol_mat->_diag, &nknowns, &nrhs, chol_mat->_chol_data, &nknowns, K12, &nknowns, &chol_mat->_info);

               // next, we create the Schur complement matrix via K22 - K12'*K12
               AFN_DOUBLE *K22 = NULL;
               fkernel( fkernel_params, data, n, ldim, d, afn_mat->_perm+afn_mat->_k, n2, afn_mat->_perm+afn_mat->_k, n2, &K22, NULL);
               
               // get pattern use K22
               //int *A_i, *A_j;
               //AfnDistanceEuclidMatrixKnn( K22, n2, n2, schur_lfil, &A_i, &A_j);

               //update K22 with K12'*K12
               char transt = 'T';
               AFN_DOUBLE mone = -1.0, one = 1.0;
               AFN_DGEMM( &transt, &transn, &n2, &n2, &afn_mat->_k, &mone, 
                  K12, &afn_mat->_k, K12, &afn_mat->_k, &one, K22, &n2);

               afn_mat->_fS_solve = &AfnPrecondFsaiSolve;
               afn_mat->_fS_solve_free_data = &AfnPrecondFsaiFree;
               afn_mat->_fS_solve_data = AfnPrecondFsaiSetupWithMatrix( K22, n2, n2, schur_lfil);
               //afn_mat->_fS_solve_data = AfnPrecondFsaiSetupWithMatrixPattern( A_i, A_j, K22, n2, n2);
               
               // after construction this buffer is nolonger needed
               AFN_FREE(K12);
               AFN_FREE(K22);
               free(data2);
               
               break;
            }
            case 2:
            {
               printf("The permutation is:\n");
               for (int i = 0; i < n; i++)
                  printf("%d ", afn_mat->_perm[i]);
               // matrix FSAI
               // In this case build the Schur complement preconditioner via explicit Schur complement matrix
               // get the S part of the data
               AFN_DOUBLE *data2 = AfnSubData( data, n, ldim, d, afn_mat->_perm + afn_mat->_k, n2);
               
               // create the matrix G*K12, K12 first
               AFN_DOUBLE *K12 = NULL;
               fkernel( fkernel_params, data, n, ldim, d, afn_mat->_perm, afn_mat->_k, afn_mat->_perm+afn_mat->_k, n2, &K12, NULL);
               // printf("K12 is \n");
               // print_matrix(K12, afn_mat->_k, n2);
               
               char transn = 'N';
               int nknowns = afn_mat->_k;
               int nrhs = n2;
               pprecond_chol chol_mat = (pprecond_chol)afn_mat->_fA11_solve_data;
               
               // apply G onto K12
               AFN_TRTRS( &chol_mat->_uplo, &transn, &chol_mat->_diag, &nknowns, &nrhs, chol_mat->_chol_data, &nknowns, K12, &nknowns, &chol_mat->_info);

                // next, we create the Schur complement matrix via K22 - K12'*K12
               AFN_DOUBLE *K22 = NULL;
               fkernel( fkernel_params, data, n, ldim, d, afn_mat->_perm+afn_mat->_k, n2, afn_mat->_perm+afn_mat->_k, n2, &K22, NULL);
               // next, we create the Schur complement matrix via K22 - K12'*K12
               // get pattern use K22
               int *A_i, *A_j;
               AfnDistanceEuclidMatrixKnn( K22, n2, n2, schur_lfil, &A_i, &A_j);

               //update K22 with K12'*K12
               char transt = 'T';
               AFN_DOUBLE mone = -1.0, one = 1.0;
               AFN_DGEMM( &transt, &transn, &n2, &n2, &afn_mat->_k, &mone, 
                  K12, &afn_mat->_k, K12, &afn_mat->_k, &one, K22, &n2);

               afn_mat->_dfSGT_matvec = &AfnPreconddFsaiLTMv;
               afn_mat->_fS_solve = &AfnPrecondFsaiSolve;
               afn_mat->_fS_solve_free_data = &AfnPrecondFsaiFree;
               // printf("The Schur complement matrix is built\n");
               // print_matrix(K22, n2, n2);




               pafn_kernel pkernel = (pafn_kernel)AfnKernelParamCreate(n,1);
               pkernel->_iparams[0] = afn_mat->_k;
               printf("The rank is %d\n", afn_mat->_k);
               pkernel->_buffer = K12;
               pkernel->_fkernel_buffer = fkernel;
               pkernel->_fdkernel_buffer = fdkernel;
               pkernel->_fkernel_buffer_params = fkernel_params;
               pkernel->_buffer_A11chol = chol_mat->_chol_data;
              
           
               //afn_mat->_fS_solve_data = AfnPrecondFsaiSetupWithMatrix( K22, n2, n2, schur_lfil);
               // afn_mat->_fS_solve_data = AfnPrecondFsaiSetupWithMatrixPattern( A_i, A_j, K22, n2, n2);

               AFN_DOUBLE *KSchur = NULL;
               AFN_DOUBLE *dKSchur = NULL;
               AfnKernelSchurCombineKernel(pkernel, data2, n, n, d, NULL, 0, NULL, 0, &KSchur, NULL);
               // printf("The Schur complement matrix is built\n");
               // print_matrix(KSchur, n2, n2);
             
               AfnKerneldSchurCombineKernel(pkernel, data, n, n, d, afn_mat->_perm, afn_mat->_k, afn_mat->_perm + afn_mat->_k, n2, &dKSchur, NULL);
               // printf("The gradient Schur complement matrix is built\n");
               // print_matrix(dKSchur, n2, n2);
               afn_mat->_fSGT_matvec = &AfnPrecondFsaiLTMv;
               afn_mat->_fSG_matvec = &AfnPrecondFsaiLMv;
               afn_mat->_fSGT_solve = &AfnPrecondFsaiInvLT;
               afn_mat->_fS_solve_data = AfnPreconddFsaiSetupWithMatrixPattern( A_i, A_j, K22, dKSchur, n2, n2);
               
               
               // after construction this buffer is nolonger needed
               AFN_FREE(K12);
               AFN_FREE(K22);
               free(data2);
               
               break;
            }
            case 3: default:
            {
               // kernel FSAI
               
               // In this case build the Schur complement preconditioner via Schur complement Kernel
               // get the S part of the data
               AFN_DOUBLE *data2 = AfnSubData( data, n, ldim, d, afn_mat->_perm + afn_mat->_k, n2);
               
               // create the matrix G*K12, K12 first
               AFN_DOUBLE *K12 = NULL;
               fkernel( fkernel_params, data, n, ldim, d, afn_mat->_perm, afn_mat->_k, afn_mat->_perm+afn_mat->_k, n2, &K12, NULL);
               
               char transn = 'N';
               int nknowns = afn_mat->_k;
               int nrhs = n2;
               pprecond_chol chol_mat = (pprecond_chol)afn_mat->_fA11_solve_data;
               
               // apply G onto K12
               AFN_TRTRS( &chol_mat->_uplo, &transn, &chol_mat->_diag, &nknowns, &nrhs, chol_mat->_chol_data, &nknowns, K12, &nknowns, &chol_mat->_info);

               pafn_kernel pkernel = (pafn_kernel)AfnKernelParamCreate(n,1);
               pkernel->_iparams[0] = afn_mat->_k;
               printf("The rank is %d\n", afn_mat->_k);
               pkernel->_buffer = K12;
               pkernel->_fkernel_buffer = fkernel;
               pkernel->_fdkernel_buffer = fdkernel;
               pkernel->_fkernel_buffer_params = fkernel_params;
               pkernel->_buffer_A11chol = chol_mat->_chol_data;
               // pkernel->_buffer_A11chol_solve = afn_mat->_fA11_solve;
   

#ifdef AFN_USING_OPENMP
               int nthreads = omp_get_max_threads();
               pkernel->_ldwork = (size_t)nthreads*afn_mat->_k*(schur_lfil+1);
               AFN_MALLOC( pkernel->_dwork, pkernel->_ldwork, AFN_DOUBLE);
               //pkernel->_ldwork2 = (size_t)nthreads*afn_mat->_k*schur_lfil;
               //AFN_MALLOC( pkernel->_dwork2, pkernel->_ldwork2, AFN_DOUBLE);
               //pkernel->_ldwork3 = (size_t)nthreads*afn_mat->_k*1;
               //AFN_MALLOC( pkernel->_dwork3, pkernel->_ldwork3, AFN_DOUBLE);
#else
               pkernel->_ldwork = (size_t)afn_mat->_k*(schur_lfil+1);
               AFN_MALLOC( pkernel->_dwork, pkernel->_ldwork, AFN_DOUBLE);
               //pkernel->_ldwork2 = (size_t)afn_mat->_k*schur_lfil;
               //AFN_MALLOC( pkernel->_dwork2, pkernel->_ldwork2, AFN_DOUBLE);
               //pkernel->_ldwork3 = (size_t)afn_mat->_k*1;
               //AFN_MALLOC( pkernel->_dwork3, pkernel->_ldwork3, AFN_DOUBLE);
#endif         
               AFN_DOUBLE *KSchur = NULL;
               AFN_DOUBLE *dKSchur = NULL;
               // printf("The lengthscale is: %f\n", pkernel->_params[0]);
               pkernel->_params[0] = 1.0;
               pkernel->_noise_level = 0.01;
               // printf("The lengthscale is: %f\n", pkernel->_params[0]);
               AfnKernelSchurCombineKernel(pkernel, data2, n, n, d, NULL, 0, NULL, 0, &KSchur, NULL);
               printf("The Schur complement matrix is built\n");
               print_matrix(KSchur, n2, n2);
               AfnKerneldSchurCombineKernel(pkernel, data, n, n, d, afn_mat->_perm, afn_mat->_k, afn_mat->_perm + afn_mat->_k, n2, &dKSchur, NULL);
               printf("The gradient Schur complement matrix is built\n");
               print_matrix(dKSchur, n2, n2);
               afn_mat->_fS_solve = &AfnPrecondFsaiSolve;
               afn_mat->_fS_solve_free_data = &AfnPrecondFsaiFree;
               // afn_mat->_fS_solve_data = AfnPrecondFsaiSetupWithKernel( data2, n2, n2, d, schur_lfil, &AfnKerneldSchurCombineKernel, pkernel);
               afn_mat->_fS_solve_data = AfnPreconddFsaiSetupWithMatrix(KSchur, dKSchur, d, n2, schur_lfil);
               // afn_mat->_fS_solve_data = AfnPrecondFsaiSetupWithMatrix(KSchur, d, n2, n2);
               afn_mat->_fSGT_matvec = &AfnPrecondFsaiLTMv;
               // after construction this buffer is nolonger needed
               free(K12);
               free(data2);
               AfnKernelParamFree(pkernel);

               break;
            }
         }
      }

   }

   t2 = AfnWtime();

   printf("---------------------------------------------------\n");
   printf("AFN Preconditioner (Exclude rankest) time: %fs\n", t2-t1);
   printf("---------------------------------------------------\n");

   /* create working buffer */
   AFN_CALLOC( afn_mat->_dwork, 4*(size_t)n, AFN_DOUBLE);

   te = AfnWtime();
   afn_mat->_tset = te - ts;

   return (void*) afn_mat;

}


//TODO: use global variable is not elegant
int afn_afn_plot = 1;

void AfnPrecondAFNPlot(void *vafn_mat, AFN_DOUBLE *data, int n, int ldim, int d)
{
   pprecond_afn afn_mat = (pprecond_afn) vafn_mat;
   if(afn_afn_plot)
   {
      if(afn_mat->_perm)
      {
         TestPlotData( data, n, d, ldim, afn_mat->_perm, afn_mat->_k, "AFN");
      }
      else
      {
         pprecond_nys nys_mat = (pprecond_nys) afn_mat->_fA11_solve_data;
         TestPlotData( data, n, d, ldim, nys_mat->_perm, nys_mat->_k, "AFN");
      }
      afn_afn_plot = 0;
   }
}
