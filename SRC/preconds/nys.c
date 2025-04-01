#include "nys.h"

void* Nfft4GPPrecondNysCreate()
{
   pprecond_nys str = NULL;
   NFFT4GP_MALLOC( str, 1, precond_nys);

   str->_k_setup = 50;

   str->_n = 0;
   str->_tits = 0;
   str->_titt = 0.0;
   str->_tset = 0.0;

   str->_nys_opt = 0;

   str->_k = 0;
   str->_own_perm = 0;
   str->_perm = NULL;
   str->_eta = 1.0;

   str->_U = NULL;
   str->_s = NULL;
   str->_work = NULL;

   str->_K = NULL;
   str->_dK = NULL;
   str->_dU = NULL;
   str->_chol_K11 = NULL;
   str->_dvp_nosolve = 0;

   return (void*) str;
}

void Nfft4GPPrecondNysFree(void *str)
{
   pprecond_nys pstr = (pprecond_nys) str;
   
   if(pstr)
   {
      NFFT4GP_FREE(pstr->_U);
      NFFT4GP_FREE(pstr->_dU);
      NFFT4GP_FREE(pstr->_K);
      NFFT4GP_FREE(pstr->_dK)
      NFFT4GP_FREE(pstr->_s);
      NFFT4GP_FREE(pstr->_work);
      if(pstr->_chol_K11)
      {
         Nfft4GPPrecondCholFree(pstr->_chol_K11);
         pstr->_chol_K11 = NULL;
      }
      if(pstr->_own_perm)
      {
         NFFT4GP_FREE(pstr->_perm);
         pstr->_own_perm = 0;
      }
      else
      {
         pstr->_perm = NULL;
      }
      /*
      printf("RAN total setup time %fs\n",pstr->_tset);
      if(pstr->_tits > 0)
      {
         printf("RAN average solve time %fs\n",pstr->_titt/pstr->_tits);
      }
      */

      pstr->_n = 0;
      pstr->_tits = 0;
      pstr->_titt = 0.0;
      pstr->_tset = 0.0;

      NFFT4GP_FREE(pstr);
   }
}

void Nfft4GPPrecondNysReset(void *str)
{
   pprecond_nys pstr = (pprecond_nys) str;
   if(pstr)
   {
      NFFT4GP_FREE(pstr->_U);
      NFFT4GP_FREE(pstr->_dU);
      NFFT4GP_FREE(pstr->_K);
      NFFT4GP_FREE(pstr->_dK)
      NFFT4GP_FREE(pstr->_s);
      NFFT4GP_FREE(pstr->_work);
      if(pstr->_chol_K11)
      {
         Nfft4GPPrecondCholFree(pstr->_chol_K11);
         pstr->_chol_K11 = NULL;
      }
      // Keep permutation, do not reset time

      // do not free pstr, remain for future work
   }
}

void Nfft4GPPrecondNysSetRank(void *str, int k)
{
   pprecond_nys pstr = (pprecond_nys) str;

   pstr->_k_setup = k;
}

void Nfft4GPPrecondNysSetPerm(void *str, int *perm, int own_perm)
{
   pprecond_nys pstr = (pprecond_nys) str;

   pstr->_perm = perm;
   pstr->_own_perm = own_perm;
}

int Nfft4GPPrecondNysSolve( void *vnys_mat, int n, NFFT4GP_DOUBLE *x, NFFT4GP_DOUBLE *rhs)
{
   double ts, te;
   ts = Nfft4GPWtime();

   int i;
   pprecond_nys nys_mat = (pprecond_nys) vnys_mat;

   NFFT4GP_DOUBLE* z = nys_mat->_work;
   NFFT4GP_DOUBLE* y = nys_mat->_work + nys_mat->_k;
   
   /* get permuted rhs y = rhs(perm) */
   if(nys_mat->_perm)
   {
      for(i = 0 ; i <  nys_mat->_n ; i++)
      {
         y[i] = rhs[ nys_mat->_perm[i]];
      }
   }
   else
   {
      // in this case we do not need to permute outside, point y to the output
      y = x;
      NFFT4GP_MEMCPY( y, rhs, nys_mat->_n, NFFT4GP_DOUBLE);
   }

   // y = (U * (S * (U' * rp))) + (y - (U*U'*y))/eta
   
   // first z = U' * y
   Nfft4GPVecFill(z,nys_mat->_k,0.0);
   Nfft4GPDenseMatGemv( nys_mat->_U, 'T', nys_mat->_n, nys_mat->_k, 1.0, y, 0.0, z);

   // next y = (y - U * U' * y)/eta
   Nfft4GPDenseMatGemv( nys_mat->_U, 'N', nys_mat->_n, nys_mat->_k, -1.0/nys_mat->_eta, z, 1.0/nys_mat->_eta, y);

   // next z = S * z
   for(i = 0 ; i <  nys_mat->_k ; i++)
   {
      z[i] = z[i] * nys_mat->_s[i];
   }

   // next y = (U * (S * (U' * rp))) + (y - U * U' * y)/eta
   Nfft4GPDenseMatGemv( nys_mat->_U, 'N', nys_mat->_n, nys_mat->_k, 1.0, z, 1.0, y);

   /* reverse permutation */
   if(nys_mat->_perm)
   {
      for(i = 0 ; i <  nys_mat->_n ; i++)
      {
         x[nys_mat->_perm[i]] = y[i];
      }
   }

   te = Nfft4GPWtime();
   nys_mat->_titt += (te - ts);
   nys_mat->_tits ++;

   return 0;
}

int Nfft4GPPrecondNysDvp(void *vnys_mat, int n, int *mask, NFFT4GP_DOUBLE *x, NFFT4GP_DOUBLE **yp)
{
   double ts, te;
   ts = Nfft4GPWtime();

   pprecond_nys nys_mat = (pprecond_nys) vnys_mat;

   int *mask_l = NULL;
   if(!mask)
   {
      NFFT4GP_MALLOC( mask_l, 3, int);
      mask_l[0] = 1;mask_l[1] = 1;mask_l[2] = 1;
   }

   if(!nys_mat->_dK)
   {
      printf("Setup NYS without gradient, dvp not supported.\n");
      return -1;
   }

   if(!yp)
   {
      printf("output pointer cannot be NULL\n");
      return -1;
   }

   int i, grad_num;
   NFFT4GP_DOUBLE *y;

   if(*yp == NULL)
   {
      NFFT4GP_CALLOC(y,3*n,NFFT4GP_DOUBLE);
   }
   else
   {
      y = *yp;
   }
   
   if(*yp == NULL)
   {
      *yp = y;
   }

   NFFT4GP_DOUBLE* x_perm = nys_mat->_work + nys_mat->_n + nys_mat->_k;
   NFFT4GP_DOUBLE* dKx = x_perm + nys_mat->_n;
   NFFT4GP_DOUBLE* Kx = dKx + nys_mat->_k;
   NFFT4GP_DOUBLE* GdKGKx = Kx + nys_mat->_k;
   NFFT4GP_DOUBLE* y_perm = GdKGKx + nys_mat->_k;
   NFFT4GP_DOUBLE* y_sol = y_perm + nys_mat->_n;
   
   int nhs2 = 2;
   int nhs3 = 3;

   /* grad 1 and 2 */

   /* get permuted x_perm = x(perm) */
   if(nys_mat->_perm)
   {
      for(i = 0 ; i <  nys_mat->_n ; i++)
      {
         x_perm[i] = x[nys_mat->_perm[i]];
      }
   }
   else
   {
      // in this case we do not need to permute outside, point y to the output
      x_perm = x;
   }

   for(grad_num = 0 ; grad_num < 2 ; grad_num++)
   {
      if(!mask_l[grad_num])
      {
         continue;
      }
      /* get permuted x_perm = x(perm) */
      if(!nys_mat->_perm)
      {
         y_sol = y + grad_num*nys_mat->_n;
      }

      // K1x = K1*x_perm, note that K1 is the first block row so...
      Nfft4GPDenseMatGemv( nys_mat->_K, 'T', nys_mat->_n, nys_mat->_k, 1.0, x_perm, 0.0, Kx);
      // dK1x = dK1*x_perm
      Nfft4GPDenseMatGemv( nys_mat->_dK+(size_t)grad_num*nys_mat->_n*nys_mat->_k, 'T', nys_mat->_n, nys_mat->_k, 1.0, x_perm, 0.0, dKx);

      // then apply L11 solve on it since we are going to use it anyway
      // we put dKx and Kx together so they can be solved at the same time
      nys_mat->_chol_K11->_trans = 'N';
      NFFT4GP_TRTRS( &nys_mat->_chol_K11->_uplo, &nys_mat->_chol_K11->_trans, &nys_mat->_chol_K11->_diag,
               &nys_mat->_k, &nhs2,
               nys_mat->_chol_K11->_chol_data, &nys_mat->_k,
               dKx, &nys_mat->_k, &nys_mat->_chol_K11->_info);

      // next GdKGKx = GdGK*Kx
      Nfft4GPDenseMatGemv( nys_mat->_chol_K11->_GdKG_data+(size_t)grad_num*nys_mat->_k*nys_mat->_k, 'N', nys_mat->_k, nys_mat->_k, 1.0, Kx, 0.0, GdKGKx);

      // next we can solve three of them at the same time
      nys_mat->_chol_K11->_trans = 'T';
      NFFT4GP_TRTRS( &nys_mat->_chol_K11->_uplo, &nys_mat->_chol_K11->_trans, &nys_mat->_chol_K11->_diag,
               &nys_mat->_k, &nhs3,
               nys_mat->_chol_K11->_chol_data, &nys_mat->_k,
               dKx, &nys_mat->_k, &nys_mat->_chol_K11->_info);

      // finally puttinng them back to y
      Nfft4GPDenseMatGemv( nys_mat->_dK+(size_t)grad_num*nys_mat->_n*nys_mat->_k, 'N', nys_mat->_n, nys_mat->_k, 1.0, Kx, 0.0, y_perm);
      Nfft4GPDenseMatGemv( nys_mat->_K, 'N', nys_mat->_n, nys_mat->_k, -1.0, GdKGKx, 1.0, y_perm);
      Nfft4GPDenseMatGemv( nys_mat->_K, 'N', nys_mat->_n, nys_mat->_k, 1.0, dKx, 1.0, y_perm);

      // next apply Nystrom solve on it
      if(!nys_mat->_dvp_nosolve)
      {
         int *perm_backup = nys_mat->_perm;
         nys_mat->_perm = NULL;
         Nfft4GPPrecondNysSolve( vnys_mat, nys_mat->_n, y_sol, y_perm);
         nys_mat->_perm = perm_backup;
      }
      else
      {
         NFFT4GP_MEMCPY( y_sol, y_perm, nys_mat->_n, NFFT4GP_DOUBLE);
      }

      /* reverse permutation */
      if(nys_mat->_perm)
      {
         for(i = 0 ; i <  nys_mat->_n ; i++)
         {
            y[nys_mat->_perm[i]+grad_num*nys_mat->_n] = y_sol[i];
         }
      }
   }

   // the thrid one just need to be solved since the gradient is identity
   if(mask_l[2])
   {
      if(!nys_mat->_dvp_nosolve)
      {
         Nfft4GPPrecondNysSolve( vnys_mat, nys_mat->_n, y+2*nys_mat->_n, x);
      }
      else
      {
         NFFT4GP_MEMCPY( y+2*nys_mat->_n, x, nys_mat->_n, NFFT4GP_DOUBLE);
      }
      Nfft4GPVecScale( y+2*nys_mat->_n, nys_mat->_n, nys_mat->_f2);
   }

   te = Nfft4GPWtime();
   nys_mat->_tdvp += (te - ts);

   if(!mask)
   {
      NFFT4GP_FREE(mask_l);
   }

   return 0;
}

int Nfft4GPPrecondNysTrace(void *vnys_mat, NFFT4GP_DOUBLE **tracesp)
{
   double ts, te;
   ts = Nfft4GPWtime();

   pprecond_nys nys_mat = (pprecond_nys) vnys_mat;

   if(!nys_mat->_dK)
   {
      printf("Setup Nys without gradient, trace not supported.\n");
      return -1;
   }

   if(!tracesp)
   {
      printf("Trace pointer cannot be NULL\n");
      return -1;
   }

   int i,j;
   NFFT4GP_DOUBLE *traces;

   if(*tracesp == NULL)
   {
      NFFT4GP_CALLOC(traces,3,NFFT4GP_DOUBLE);
   }
   else
   {
      traces = *tracesp;
   }

   // we need first compute the matrix L / (mu I + L'*L)
   // we call L dU in our code, so we need two buffer
   NFFT4GP_DOUBLE *UU;
   NFFT4GP_DOUBLE *UUU;
   NFFT4GP_MALLOC(UU, (size_t)nys_mat->_k*nys_mat->_k, NFFT4GP_DOUBLE);
   NFFT4GP_MALLOC(UUU, (size_t)nys_mat->_n*nys_mat->_k, NFFT4GP_DOUBLE);
   NFFT4GP_MEMCPY(UUU, nys_mat->_dU, (size_t)nys_mat->_n*nys_mat->_k, NFFT4GP_DOUBLE);

   // first compute L'*L using matvec
   char charn = 'N';
   char chart = 'T';
   NFFT4GP_DOUBLE one = 1.0;
   NFFT4GP_DOUBLE zero = 0.0;
   NFFT4GP_DGEMM( &chart, &charn, &nys_mat->_k, &nys_mat->_k, &nys_mat->_n, 
      &one, nys_mat->_dU, &nys_mat->_n, nys_mat->_dU, &nys_mat->_n, &zero, UU, &nys_mat->_k);

   // add mu I to it, which is simply add mu to the diagonal
#ifdef NFFT4GP_USE_OPENMP
   if(!omp_in_parallel())
   {
      #pragma omp parallel for
#endif
      for(i = 0 ; i < nys_mat->_k ; i ++)
      {
         UU[i*(nys_mat->_k+1)] += nys_mat->_eta;
      }
#ifdef NFFT4GP_USE_OPENMP
   }
   else
   {
      for(i = 0 ; i < nys_mat->_k ; i ++)
      {
         UU[i*(nys_mat->_k+1)] += nys_mat->_eta;
      }
   }
#endif

   // then we solve for L / (mu I + L'*L) store in UUU
   char uplo = 'L';
   int info;
   NFFT4GP_DPOTRF( &uplo, &nys_mat->_k, UU, &nys_mat->_k, &info);

   char transn = 'N';
   char transt = 'T';
   char diag = 'N';
   NFFT4GP_DOUBLE two = 2.0;
   char sider = 'R';

   NFFT4GP_DTRSM( &sider, &uplo, &transt, &diag, &nys_mat->_n, &nys_mat->_k, 
               &one, UU, &nys_mat->_k, UUU, &nys_mat->_n);

   NFFT4GP_DTRSM( &sider, &uplo, &transn, &diag, &nys_mat->_n, &nys_mat->_k, 
               &one, UU, &nys_mat->_k, UUU, &nys_mat->_n);
   
   traces[0] = 0.0;
   traces[1] = 0.0;
   traces[2] = nys_mat->_n * nys_mat->_f2;
   
   // start accumulating values to traces 0 and 1
   NFFT4GP_DOUBLE *LdK;
   // only need to solve the first two
   NFFT4GP_MALLOC(LdK, (size_t)nys_mat->_n*nys_mat->_k*2, NFFT4GP_DOUBLE);
   NFFT4GP_MEMCPY(LdK, nys_mat->_dK, (size_t)nys_mat->_n*nys_mat->_k*2, NFFT4GP_DOUBLE);

   NFFT4GP_DTRSM( &sider, &uplo, &transt, &diag, &nys_mat->_n, &nys_mat->_k, 
               &two, nys_mat->_chol_K11->_chol_data, &nys_mat->_k, 
               LdK, &nys_mat->_n);
   NFFT4GP_DTRSM( &sider, &uplo, &transt, &diag, &nys_mat->_n, &nys_mat->_k, 
               &two, nys_mat->_chol_K11->_chol_data, &nys_mat->_k, 
               LdK+(size_t)nys_mat->_n*nys_mat->_k, &nys_mat->_n);

   NFFT4GP_DOUBLE mone = -1.0;
   NFFT4GP_DGEMM( &charn, &charn, &nys_mat->_n, &nys_mat->_k, &nys_mat->_k, 
      &mone, nys_mat->_dU, &nys_mat->_n, nys_mat->_chol_K11->_GdKG_data, &nys_mat->_k, 
      &one, LdK, &nys_mat->_n);
   NFFT4GP_DGEMM( &charn, &charn, &nys_mat->_n, &nys_mat->_k, &nys_mat->_k, 
      &mone, nys_mat->_dU, &nys_mat->_n, nys_mat->_chol_K11->_GdKG_data+(size_t)nys_mat->_k*nys_mat->_k, &nys_mat->_k, 
      &one, LdK+(size_t)nys_mat->_n*nys_mat->_k, &nys_mat->_n);

   size_t il;
   for(il = 0 ; il < (size_t)nys_mat->_n*nys_mat->_k ; il ++)
   {
      traces[0] += LdK[il]*nys_mat->_dU[il];
      traces[1] += LdK[il+(size_t)nys_mat->_n*nys_mat->_k]*nys_mat->_dU[il];
   }

   // lastly we need dM/dtheta * L
   nys_mat->_dvp_nosolve = 1;
   NFFT4GP_DOUBLE *DvP = NULL;
   int *perm_backup = nys_mat->_perm;
   nys_mat->_perm = NULL;
   for(il = 0 ; il < nys_mat->_k ; il ++)
   {
      Nfft4GPPrecondNysDvp( vnys_mat, nys_mat->_n, NULL, nys_mat->_dU+il*nys_mat->_n, &DvP);

      for(i = 0 ; i < nys_mat->_n ; i ++)
      {
         for(j = 0 ; j < 3 ; j ++)
         {
            traces[j] -= DvP[i+j*nys_mat->_n]*UUU[i+il*nys_mat->_n];
         }
      }
   }
   nys_mat->_perm = perm_backup;
   nys_mat->_dvp_nosolve = 0;

   traces[0] /= nys_mat->_eta;
   traces[1] /= nys_mat->_eta;
   traces[2] /= nys_mat->_eta;

   if(*tracesp == NULL)
   {
      *tracesp = traces;
   }

   te = Nfft4GPWtime();
   nys_mat->_tlogdet += (te - ts);

   NFFT4GP_FREE(UU);
   NFFT4GP_FREE(UUU);
   NFFT4GP_FREE(LdK);
   NFFT4GP_FREE(DvP);

   return 0;
}

NFFT4GP_DOUBLE Nfft4GPPrecondNysLogdet(void *vnys_mat)
{
   double ts, te;
   ts = Nfft4GPWtime();

   pprecond_nys nys_mat = (pprecond_nys) vnys_mat;

   int i;
   NFFT4GP_DOUBLE val0 = log(nys_mat->_eta);
   NFFT4GP_DOUBLE val = val0 * (nys_mat->_n - nys_mat->_k);
   
   for(i = 0 ; i < nys_mat->_k ; i ++)
   {
      if(nys_mat->_s[i] > 0)
      {
         val += log(1.0/nys_mat->_s[i]);
      }
      else
      {
         val += val0;
      }
   }

   te = Nfft4GPWtime();
   nys_mat->_tlogdet += (te - ts);
   
   return val;
}

int Nfft4GPPrecondNysSetupWithKernel(NFFT4GP_DOUBLE *data, int n, int ldim, int d,
                                    func_kernel fkernel, void *fkernel_params, int require_grad, void* vnys_mat)
{
   int i;
   //double t1, t2;
   double ts, te;
   ts = Nfft4GPWtime();

   /*****************************
    * 1. Reordering
    *****************************/

   pprecond_nys nys_mat = (pprecond_nys) vnys_mat;
   int k = nys_mat->_k_setup;
   
   nys_mat->_n = n;
   nys_mat->_k = k;
   if(require_grad)
   {
      NFFT4GP_MALLOC( nys_mat->_work, (size_t)5*n+6*k, NFFT4GP_DOUBLE);
   }
   else
   {   
      NFFT4GP_MALLOC( nys_mat->_work, (size_t)n+k, NFFT4GP_DOUBLE);
   }

   /*****************************
    * 2. Extract the submatrix
    *****************************/
   //t1 = Nfft4GPWtime();
   NFFT4GP_DOUBLE* data1 = Nfft4GPSubData( data, n, ldim, d, nys_mat->_perm, nys_mat->_k);
   //t2 = Nfft4GPWtime();
   //printf("---------------------------------------------------\n");
   //printf("Extract data time: %fs\n", t2-t1);
   //printf("Rank: %d \n", nys_mat->_k);
   //printf("---------------------------------------------------\n");
   
   /*****************************
    * 3. Compute G
    *****************************/
   
   // set noise level to 0 and reset later
   pnfft4gp_kernel pkernel = (pnfft4gp_kernel) fkernel_params;
   NFFT4GP_DOUBLE noise_level = pkernel->_noise_level;
   nys_mat->_f2 = pkernel->_params[0]*pkernel->_params[0]; // the first param is the scale
   pkernel->_noise_level = 0.0;

   //t1 = Nfft4GPWtime();
   nys_mat->_chol_K11 = (pprecond_chol) Nfft4GPPrecondCholCreate();
   nys_mat->_chol_K11->_stable = 1;
   Nfft4GPPrecondCholSetupWithKernel( data1, nys_mat->_k, nys_mat->_k, d,
                                          fkernel, (void*)fkernel_params, require_grad, (void*)nys_mat->_chol_K11);
   NFFT4GP_DOUBLE *G;
   NFFT4GP_MALLOC( G, (size_t)nys_mat->_k*nys_mat->_k, NFFT4GP_DOUBLE);
   NFFT4GP_MEMCPY( G, nys_mat->_chol_K11->_chol_data, (size_t)nys_mat->_k*nys_mat->_k, NFFT4GP_DOUBLE);
   NFFT4GP_FREE(data1);
   //t2 = Nfft4GPWtime();
   //printf("---------------------------------------------------\n");
   //printf("Exact Cholesky time: %fs\n", t2-t1);
   //printf("---------------------------------------------------\n");

   /*****************************
    * 4. Compute G^{-1}
    *****************************/
   
   //t1 = Nfft4GPWtime();
   AFnTrilNystromInv( (void*)G, nys_mat->_k, nys_mat->_k);
   //t2 = Nfft4GPWtime();

   //printf("---------------------------------------------------\n");
   //printf("Inv of Cholesky time: %fs\n", t2-t1);
   //printf("---------------------------------------------------\n");

   //t1 = Nfft4GPWtime();
   if(require_grad)
   {
      fkernel( fkernel_params, data, n, ldim, d, nys_mat->_perm, nys_mat->_n, nys_mat->_perm, nys_mat->_k, &(nys_mat->_U), &(nys_mat->_dK));
      NFFT4GP_MALLOC( nys_mat->_K, nys_mat->_n*nys_mat->_k, NFFT4GP_DOUBLE);
      NFFT4GP_MEMCPY( nys_mat->_K, nys_mat->_U, (size_t)nys_mat->_n*nys_mat->_k, NFFT4GP_DOUBLE); // store a backup of K
   }
   else
   {
      fkernel( fkernel_params, data, n, ldim, d, nys_mat->_perm, nys_mat->_n, nys_mat->_perm, nys_mat->_k, &(nys_mat->_U), NULL);
   }

   // compute A / chol(A11), note that chol(A11) is the upper triangular so is the transpose
   Nfft4GPTrilNystromMm( G, nys_mat->_k, nys_mat->_U, 
               nys_mat->_n, nys_mat->_n, nys_mat->_k, 1.0);

   NFFT4GP_FREE(G);
   
   if(require_grad)
   {
      NFFT4GP_MALLOC( nys_mat->_dU, (size_t)nys_mat->_n*nys_mat->_k, NFFT4GP_DOUBLE);
      NFFT4GP_MEMCPY( nys_mat->_dU, nys_mat->_U, (size_t)nys_mat->_n*nys_mat->_k, NFFT4GP_DOUBLE); // store a backup
   }

   if(!require_grad)
   {
      Nfft4GPPrecondCholFree(nys_mat->_chol_K11);
      nys_mat->_chol_K11 = NULL;
   }
   //t2 = Nfft4GPWtime();
   //printf("---------------------------------------------------\n");
   //printf("A1/chol(A11) time: %fs\n", t2-t1);
   //printf("---------------------------------------------------\n");

   /*****************************
    * 5. Compute SVD
    *****************************/
   
   //t1 = Nfft4GPWtime();
   Nfft4GPTrilNystromSvd( nys_mat->_U, nys_mat->_n, nys_mat->_n, nys_mat->_k, (void*)&(nys_mat->_s));
   //t2 = Nfft4GPWtime();
   //printf("---------------------------------------------------\n");
   //printf("SVD time: %fs\n", t2-t1);
   //printf("---------------------------------------------------\n");

   /*****************************
    * 6. Update S
    *****************************/
   
   //t1 = Nfft4GPWtime();
   nys_mat->_eta = noise_level * nys_mat->_f2; // note: now our noise level has been chanced, use this different noise level!

   for(i = 0 ; i < nys_mat->_k ; i ++)
   {
      nys_mat->_s[i] = 1.0/(nys_mat->_s[i]*nys_mat->_s[i]+nys_mat->_eta);
      NFFT4GP_MAX( nys_mat->_s[i], 0.0, nys_mat->_s[i]); // do not allow negative values
   }
   //t2 = Nfft4GPWtime();
   //printf("---------------------------------------------------\n");
   //printf("Compute S time: %fs\n", t2-t1);
   //printf("---------------------------------------------------\n");

   // reset noise level
   pkernel->_noise_level = noise_level;

   te = Nfft4GPWtime();
   nys_mat->_tset = te-ts;

   return 0;
}

//TODO: use global variable is not elegant
int nfft4gp_nys_plot = 1;

void Nfft4GPPrecondNysPlot(void *vnys_mat, NFFT4GP_DOUBLE *data, int n, int ldim, int d)
{
   pprecond_nys nys_mat = (pprecond_nys) vnys_mat;
   if(nfft4gp_nys_plot)
   {
      TestPlotData( data, n, d, ldim, nys_mat->_perm, nys_mat->_k, "nystrom");
      nfft4gp_nys_plot = 0;
   }
}
