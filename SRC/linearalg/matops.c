#include "matops.h"

int Nfft4GPDenseMatSymv(void *data, int n, NFFT4GP_DOUBLE alpha, NFFT4GP_DOUBLE *x, NFFT4GP_DOUBLE beta, NFFT4GP_DOUBLE *y)
{
   int one = 1;
   char uplo = 'L';

   NFFT4GP_DOUBLE *ddata = (NFFT4GP_DOUBLE*) data;

   NFFT4GP_DSYMV( &uplo, &n, &alpha, ddata, &n, x, &one, &beta, y, &one);

   return 0;
}

int Nfft4GPDenseGradMatSymv(void *data, int n, NFFT4GP_DOUBLE alpha, NFFT4GP_DOUBLE *x, NFFT4GP_DOUBLE beta, NFFT4GP_DOUBLE *y)
{
   int one = 1;
   char uplo = 'L';

   NFFT4GP_DOUBLE *ddata = (NFFT4GP_DOUBLE*) data;

   NFFT4GP_DSYMV( &uplo, &n, &alpha, ddata, &n, x, &one, &beta, y, &one);
   NFFT4GP_DSYMV( &uplo, &n, &alpha, ddata+(size_t)n*n, &n, x, &one, &beta, y+n, &one);
   // TODO: the third dAz is a scale
   // we should simplify this later
   NFFT4GP_DSYMV( &uplo, &n, &alpha, ddata+(size_t)2*n*n, &n, x, &one, &beta, y+2*n, &one);

   return 0;
}

int Nfft4GPDenseMatGemv(void *data, char trans, int m, int n, NFFT4GP_DOUBLE alpha, NFFT4GP_DOUBLE *x, NFFT4GP_DOUBLE beta, NFFT4GP_DOUBLE *y)
{
   int one = 1;

   NFFT4GP_DOUBLE *ddata = (NFFT4GP_DOUBLE*) data;
   NFFT4GP_DGEMV( &trans, &m, &n, &alpha, ddata, &m, x, &one, &beta, y, &one);

   return 0;
}

int AFnTrilNystromInv(void *data, int lda, int n)
{
   char uplo = 'L';
   char diag = 'N';
   int info;
   NFFT4GP_DOUBLE *ddata = (NFFT4GP_DOUBLE*) data;
   NFFT4GP_DTRTRI( &uplo, &diag, &n, ddata, &lda, &info);
   return info;
}

int Nfft4GPTrilNystromMm( void *a, int lda, void *b, int ldb, int m, int n, NFFT4GP_DOUBLE alpha)
{
   char side = 'R';
   char uplo = 'L';
   char transa = 'T';
   char diag = 'N';
   NFFT4GP_DOUBLE *da = (NFFT4GP_DOUBLE*) a;
   NFFT4GP_DOUBLE *db = (NFFT4GP_DOUBLE*) b;
   NFFT4GP_DTRMM( &side, &uplo, &transa, &diag, &m, &n, &alpha, da, &lda, db, &ldb);
   return 0;
}

// TODO: the ARPACK part will not work without modification to the precond.c.
// Be mindful in the order of S! eta uses min(s), for svd this is s[end] and for eig this is s[0].
int Nfft4GPTrilNystromSvd( void *a, int lda, int m, int n, void **s)
{
   // A = USV => U = A*inv(V')*inv(S)
   int i,j;
   NFFT4GP_DOUBLE *AA = NULL;
   NFFT4GP_DOUBLE *AV = NULL;
   NFFT4GP_MALLOC(AA, (size_t)n*n, NFFT4GP_DOUBLE);
   NFFT4GP_MALLOC(AV, (size_t)m*n, NFFT4GP_DOUBLE);
   char charn = 'N';
   char chart = 'T';
   NFFT4GP_DOUBLE one = 1.0;
   NFFT4GP_DOUBLE zero = 0.0;
   NFFT4GP_DOUBLE *da = (NFFT4GP_DOUBLE*) a;
   NFFT4GP_DGEMM( &chart, &charn, &n, &n, &m, &one, da, &lda, da, &lda, &zero, AA, &n);

   // next compute the eigendecomposition of AA
   char charv = 'V';
   char charl = 'L';
   NFFT4GP_DOUBLE *w = NULL;
   NFFT4GP_DOUBLE *w1 = NULL;
   NFFT4GP_MALLOC(w, n, NFFT4GP_DOUBLE);
   NFFT4GP_MALLOC(w1, n, NFFT4GP_DOUBLE);
   int lwork = 3*n+1;
   NFFT4GP_DOUBLE *work = NULL;
   NFFT4GP_MALLOC(work, lwork, NFFT4GP_DOUBLE);
   int info;

#ifdef NFFT4GP_USING_OPENMP
   int nthreads = omp_get_max_threads();
   omp_set_num_threads(NFFT4GP_OPENMP_REDUCED_THREADS);
#endif

   NFFT4GP_DSYEV( &charv, &charl, &n, AA, &n, w1, work, &lwork, &info);

   // now we need (A/V')/sqrt(w)
   int *ipiv = NULL;
   NFFT4GP_MALLOC(ipiv, n, int);
   NFFT4GP_DGETRF( &n, &n, AA, &n, ipiv, &info);
   NFFT4GP_DGETRI( &n, AA, &n, ipiv, work, &lwork, &info);

#ifdef NFFT4GP_USING_OPENMP
   omp_set_num_threads(nthreads);
#endif

   NFFT4GP_DGEMM( &charn, &chart, &m, &n, &n, &one, da, &lda, AA, &n, &zero, AV, &m);

   // scale by 1.0/sqrt(w)
   NFFT4GP_DOUBLE *Amat = da;
   for(i = n-1 ; i >=0 ; i --)
   {
      NFFT4GP_DOUBLE *Vmat = AV + (size_t)i*m;
      NFFT4GP_DOUBLE wi = sqrt(w1[i]);
      w[n-1-i] = wi;
      if(wi < 1e-12)
      {
         (*Amat++) = (*Vmat++) * 1e12;
      }
      else
      {
         for(j = 0 ; j < m ; j ++)
         {
            (*Amat++) = (*Vmat++) / wi;
         }
      }
   }
   *s = (void*)w;
   NFFT4GP_FREE(ipiv);
   NFFT4GP_FREE(AA);
   NFFT4GP_FREE(work);
   NFFT4GP_FREE(AV);
   NFFT4GP_FREE(w1);
   return info;
}

int Nfft4GPCsrMv( int *ia, int *ja, NFFT4GP_DOUBLE *aa, int nrows, int ncols, char trans, NFFT4GP_DOUBLE alpha, NFFT4GP_DOUBLE *x, NFFT4GP_DOUBLE beta, NFFT4GP_DOUBLE *y)
{

#ifdef NFFT4GP_USING_MKL
   mkl_dcsrmv( &trans, &nrows, &ncols, &alpha, "GXXCXX", aa, ja, ia, ia+1, x, &beta, y);
   return 0;
#endif

   int      i, j, j1, j2;
   NFFT4GP_DOUBLE   r, xi, *x_temp = NULL;

   if( (x == y) && (alpha != 0.0) )
   {
      if (trans == 'N')
      {
         NFFT4GP_MALLOC(x_temp, nrows, NFFT4GP_DOUBLE);
         NFFT4GP_MEMCPY(x_temp, y, nrows, NFFT4GP_DOUBLE);
         x = x_temp;
      }
      else
      {
         NFFT4GP_MALLOC(x_temp, ncols, NFFT4GP_DOUBLE);
         NFFT4GP_MEMCPY(x_temp, y, ncols, NFFT4GP_DOUBLE);
         x = x_temp;
      }
   }

   if(beta != 1.0)
   {
      if(beta != 0.0)
      {
         if (trans == 'N')
         {
            for (i = 0; i < nrows; i++)
            {
               y[i] *= beta;
            }
         }
         else
         {
            /* if x == y need to create new x */
            for (i = 0; i < ncols; i++)
            {
               y[i] *= beta;
            }
         }
      }
      else
      {
         if (trans == 'N')
         {
            for (i = 0; i < nrows; i++)
            {
               y[i] = 0.0;
            }
         }
         else
         {
            for (i = 0; i < ncols; i++)
            {
               y[i] = 0.0;
            }
         }
      }
   }

   if(alpha != 0.0)
   {
      if(alpha != 1.0)
      {
         if (trans == 'N')
         {
            for (i = 0; i < nrows; i++)
            {
               r = 0.0;
               j1 = ia[i];
               j2 = ia[i+1];
               for (j = j1; j < j2; j++)
               {
                  r += aa[j] * x[ja[j]];
               }
               y[i] += alpha*r;
            }
         }
         else
         {
            for (i = 0; i < nrows; i++)
            {
               xi = alpha * x[i];
               j1 = ia[i];
               j2 = ia[i+1];
               for (j = j1; j < j2; j++)
               {
                  y[ja[j]] += aa[j] * xi;
               }
            }
         }
      }
      else
      {
         if (trans == 'N')
         {
            for (i = 0; i < nrows; i++)
            {
               j1 = ia[i];
               j2 = ia[i+1];
               for (j = j1; j < j2; j++)
               {
                  y[i] += aa[j] * x[ja[j]];
               }
            }
         }
         else
         {
            for (i = 0; i < nrows; i++)
            {
               j1 = ia[i];
               j2 = ia[i+1];
               for (j = j1; j < j2; j++)
               {
                  y[ja[j]] += aa[j] * x[i];
               }
            }
         }
      }
   }

   if(x_temp)
   {
      NFFT4GP_FREE( x_temp);
   }

   return 0;
}

int Nfft4GPModifiedGS( NFFT4GP_DOUBLE *w, int n, int kdim, NFFT4GP_DOUBLE *V, NFFT4GP_DOUBLE *H, NFFT4GP_DOUBLE *t, int k, NFFT4GP_DOUBLE tol_orth, NFFT4GP_DOUBLE tol_reorth)
{

   if(k < 0 || n == 0)
   {
      /* in this case, we don't have any previous vectors, return immediatly */
      return 0;
   }

   /*------------------------
    * 1: Modified Gram-Schmidt
    *------------------------*/

   int i;
   NFFT4GP_DOUBLE t1, normw;

   NFFT4GP_DOUBLE *v;

   /* compute initial ||w|| if we need to reorth */
   if(tol_reorth > 0.0)
   {
      normw = Nfft4GPVecNorm2(w, n);
   }
   else
   {
      normw = 0.0;
   }

   for( i = 0 ; i <= k ; i ++)
   {
      /* inner produce and update H, w */
      v = V + i * n;

      t1 = Nfft4GPVecDdot(w, n, v);
      H[k*(kdim+1)+i] = t1;

      Nfft4GPVecAxpy( -t1, v, n, w);
   }

   /* Compute ||w|| */
   *t = Nfft4GPVecNorm2( w, n);

   /*------------------------
    * 2: Re-orth step
    *------------------------*/

   /* t < tol_orth is considered be lucky breakdown */
   while( *t < normw * tol_reorth && *t >= tol_orth)
   {
      normw = *t;
      /* Re-orth */
      for (i = 0; i <= k; i++)
      {
         v = V + i * n;

         t1 = Nfft4GPVecDdot(w, n, v);

         H[k*(kdim+1)+i] += t1;

         Nfft4GPVecAxpy( -t1, v, n, w);
      }
      /* Compute ||w|| */
      *t = Nfft4GPVecNorm2( w, n);

   }
   H[k*(kdim+1)+k+1] = *t;

   /* scale w in this function */
   Nfft4GPVecScale( w, n, 1.0/(*t));

   return 0;

}

int Nfft4GPModifiedGS2( NFFT4GP_DOUBLE *w, int n, NFFT4GP_DOUBLE *V, NFFT4GP_DOUBLE *Z, NFFT4GP_DOUBLE *TD, NFFT4GP_DOUBLE *TE, NFFT4GP_DOUBLE *t, int k, NFFT4GP_DOUBLE tol_orth, NFFT4GP_DOUBLE tol_reorth)
{

   if(k < 0 || n == 0)
   {
      /* in this case, we don't have any previous vectors, return immediatly */
      return 0;
   }

   /*------------------------
    * 1: Modified Gram-Schmidt
    *------------------------*/

   int i;
   NFFT4GP_DOUBLE t1, normw;

   NFFT4GP_DOUBLE *v, *z;

   /* compute initial ||w|| if we need to reorth */
   if(tol_reorth > 0.0)
   {
      normw = Nfft4GPVecNorm2(w, n);
   }
   else
   {
      normw = 0.0;
   }
   
   for( i = 0 ; i <= k ; i ++)
   {
      /* inner product and update H, w */
      v = V + i * n;
      z = Z + i * n;

      t1 = Nfft4GPVecDdot(w, n, v);
      
      if(i == k-1)
      {
         TE[0] = t1;
      }
      if(i == k)
      {
         TD[0] = t1;
      }

      /* use z to update */
      Nfft4GPVecAxpy( -t1, z, n, w);
   }

   /* Compute ||w|| */
   *t = Nfft4GPVecNorm2( w, n);
   
   /*------------------------
    * 2: Re-orth step
    *------------------------*/

   /* t < tol_orth is considered be lucky breakdown */
   while( *t < normw * tol_reorth && *t >= tol_orth)
   {
      normw = *t;
      /* Re-orth */
      for (i = 0; i <= k; i++)
      {
         v = V + i * n;
         z = Z + i * n;

         t1 = Nfft4GPVecDdot(w, n, v);
         if(i == k-1)
         {
            TE[0] += t1;
         }
         if(i == k)
         {
            TD[0] += t1;
         }

         /* use z to update */
         Nfft4GPVecAxpy( -t1, z, n, w);
      }
      /* Compute ||w|| */
      *t = Nfft4GPVecNorm2( w, n);

   }

   return 0;

}