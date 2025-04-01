#include "fgmres.h"

int Nfft4GPSolverLanczos( void *mat_data,
                     int n,
                     func_symmatvec matvec,
                     void *prec_data,
                     func_solve precondfunc,
                     NFFT4GP_DOUBLE *x,
                     NFFT4GP_DOUBLE *rhs,
                     int wsize,
                     int maxits,
                     int atol,
                     NFFT4GP_DOUBLE tol,
                     NFFT4GP_DOUBLE *prel_res,
                     NFFT4GP_DOUBLE **prel_res_v,
                     int *piter,
                     int *tsize,
                     NFFT4GP_DOUBLE **TDp,
                     NFFT4GP_DOUBLE **TEp,
                     int print_level)
{

   /* Declare variables */
   int           	iter, chol_size, i, j, k, wsizei;
   NFFT4GP_DOUBLE		normb, EPSILON, normr, normz, beta, le, ls = 0.0, tolr, t, gam;
   NFFT4GP_DOUBLE     dotvz = 0.0;
   NFFT4GP_DOUBLE		idotvz, hii, hii1;

   NFFT4GP_DOUBLE		*v = NULL, *z = NULL, *w = NULL;
   NFFT4GP_DOUBLE		*V, *Z;
   NFFT4GP_DOUBLE     *TD, *TE; // diagonal and subdiagonal of T
   NFFT4GP_DOUBLE     td, te;
   NFFT4GP_DOUBLE     *TLD, *TLE; // diagonal and subdiagonal of the Cholesky factor of T
   NFFT4GP_DOUBLE		*rel_res_v;
   NFFT4GP_DOUBLE     *y = NULL;

   EPSILON			= DBL_EPSILON;

   //printf("Starting Lanczos\n");

   if(wsize <= 0)
   {
      wsize = maxits;
   }
   NFFT4GP_MIN(wsize, maxits, wsize);

   if(n == 0)
   {
      *prel_res = 0.0;
      *piter = 0;
      NFFT4GP_CALLOC(rel_res_v, 1, NFFT4GP_DOUBLE);
      *prel_res_v = rel_res_v;

      return 0;
   }

   /* Exit if RHS is zero */
   normb = Nfft4GPVecNorm2(rhs, n);

   /* the solution is direct */
   if (normb < EPSILON)
   {
      Nfft4GPVecFill( x,  n, 0.0);
      *prel_res = 0.0;
      *piter = 0;
      NFFT4GP_CALLOC(rel_res_v, 1, NFFT4GP_DOUBLE);
      *prel_res_v = rel_res_v;

      return 0;
   }

   /* Working matrices to hold the Krylov basis and upper-Hessenberg matrices */
   /* TODO: we should reduce memory size by rewriting this using wsize */
   NFFT4GP_MALLOC(V, (size_t)n*(maxits+1), NFFT4GP_DOUBLE);
   NFFT4GP_MALLOC(Z, (size_t)n*(maxits+1), NFFT4GP_DOUBLE);
   NFFT4GP_MALLOC(TLD, (size_t)(maxits+1), NFFT4GP_DOUBLE);
   NFFT4GP_MALLOC(TLE, (size_t)(maxits+1), NFFT4GP_DOUBLE);
   NFFT4GP_MALLOC(y, (size_t)(maxits+1), NFFT4GP_DOUBLE);

   if(*TDp)
   {
      TD = *TDp;
   }
   else
   {
      NFFT4GP_CALLOC(TD, (size_t)(maxits+1), NFFT4GP_DOUBLE);
   }

   if(*TEp)
   {
      TE = *TEp;
   }
   else
   {
      NFFT4GP_CALLOC(TE, (size_t)maxits, NFFT4GP_DOUBLE);
   }

   /* Make b the first vector of the basis (will normalize later) */
   z = Z;
   v = V;

   /* Compute the residual norm z = rhs - A*x */
   NFFT4GP_MEMCPY( z, rhs, n, NFFT4GP_DOUBLE);
   matvec( mat_data, n, -1.0, x, 1.0, z);

   /* v = M^{-1} * z -- apply the preconditioner */
   if(prec_data)
   {
      precondfunc( prec_data, n, v, z);
   }
   else
   {
      NFFT4GP_MEMCPY( v, z, n, NFFT4GP_DOUBLE);
   }

   /* Compute the 2-norm and M-norm of residual */
   normr = Nfft4GPVecNorm2(z, n);
   beta = sqrt(Nfft4GPVecDdot( v, n, z));

   if(beta < EPSILON)
   {
      *prel_res = 0.0;
      *piter = 0;
      NFFT4GP_CALLOC(rel_res_v, 1, NFFT4GP_DOUBLE);
      *prel_res_v = rel_res_v;

      NFFT4GP_FREE(V);
      NFFT4GP_FREE(Z);
      NFFT4GP_FREE(TD);
      NFFT4GP_FREE(TE);
      NFFT4GP_FREE(TLD);
      NFFT4GP_FREE(TLE);

      return 0;
   }

   /* use absolute tol? */
   if(atol)
   {
      tolr = tol/beta;
   }
   else
   {
      tolr = tol;
   }

   /* maintain a Cholesky factorization of H for convergence check */
   if(tolr > 0.0)
   {
      Nfft4GPVecFill( TLD, maxits, 0.0);
      Nfft4GPVecFill( TLE, maxits, 0.0);
   }

   /* A few variables used to keep track of the loop's state */
   NFFT4GP_CALLOC( rel_res_v, (size_t)maxits+1, NFFT4GP_DOUBLE);

   //this->_rel_res_vector[0] = normr;
   rel_res_v[0] = normr/normb;
   iter = 0;
   chol_size = 0;

   if( print_level > 0 )
   {
      printf("--------------------------------------------------------------------------------\n");
      printf("Start Lanczos(%d)\n",maxits);
      printf("Residual Tol: %e\nMax number of inner iterations: %d\n", tolr, maxits);
      printf("--------------------------------------------------------------------------------\n");
      printf("Step    Residual norm  Relative res.  Convergence Rate\n");
      printf("%5d   %8e   %8e   N/A\n", 0, normr, rel_res_v[0]);
   }

   /* Scale the starting vector */
   idotvz = 1.0/beta;
   Nfft4GPVecScale(v, n, idotvz);
   Nfft4GPVecScale(z, n, idotvz);

   // Main loop
   while (iter < maxits)
   {
      iter++;

      z = Z+iter*n;
      w = V+(iter-1)*n;
      v = V+iter*n;

      /* matvec */
      matvec( mat_data, n, 1.0, w, 0.0, z);
      
      /* Modified Gram-schmidt without re-orth */
      NFFT4GP_MIN(iter-1, wsize, wsizei);
      
      // here the return value t is the norm of the final z
      Nfft4GPModifiedGS2( z, n, V, Z, TD+iter-1, TE+iter-2, &t, wsizei, EPSILON, 0.7071);

      // first break condition
      // stop if the norm of z is too small
      if(t < EPSILON)
      {
         break;
      }

      /* v = M^{-1} * z -- apply the preconditioner */
      if(prec_data)
      {
         precondfunc( prec_data, n, v, z);
      }
      else
      {
         NFFT4GP_MEMCPY( v, z, n, NFFT4GP_DOUBLE);
      }

      dotvz = sqrt(Nfft4GPVecDdot( v, n, z));

      // second break condition
      // stop if v'*z is too small
      if(dotvz < EPSILON)
      {
         break;
      }

      idotvz = 1.0/dotvz;
      Nfft4GPVecScale(v, n, idotvz);
      Nfft4GPVecScale(z, n, idotvz);

      /* update the Cholesky factorization of Hm */
      normz = Nfft4GPVecNorm2(z, n);
      if(iter != 1)
      {
         TLE[iter-2] = TE[iter-2]/TLD[iter-2];
         TLD[iter-1] = sqrt(TD[iter-1] - TLE[iter-2]*TLE[iter-2]);
         le = 1.0 / TLD[iter-1];
         ls = -ls * TLE[iter-2] * le;
         normr = fabs(le * ls) * dotvz * beta * normz;
      }
      else
      {
         TLD[0] = sqrt(TD[0]);
         ls = 1.0 / TLD[0];
         normr = dotvz / TD[0] * beta * normz;
      }
      chol_size++;

      rel_res_v[iter] = normr/normb;
      if( print_level > 0)
      {
         printf("%5d   %8e   %8e   %8.6f\n", iter, normr, rel_res_v[iter], rel_res_v[iter] / rel_res_v[iter-1]);
      }
      
      if (normr <= tolr)
      {
         break;
      }
   } // End of inner loop

   /* Print residual norm at the end of each cycle */
   if ( print_level == 0)
   {
      printf("Rel. residual at the end of the iteration (# of its: %d): %e \n", iter, rel_res_v[iter]);
   }

   /* Solve the triangular systems */
   y[0] = beta / TLD[0];
   for(k = 1 ; k < chol_size ; k ++)
   {
      y[k] = (- y[k-1] * TLE[k-1] ) / TLD[k];
   }
   y[chol_size-1] /= TLD[chol_size-1];
   for(k = chol_size - 2 ; k >= 0 ; k--)
   {
      y[k] = ( y[k] - TLE[k] * y[k+1] ) / TLD[k];
   }
   
   /* compute the final solution */
   for(k = 0 ; k < chol_size ; k ++)
   {
      Nfft4GPVecAxpy( y[k], V+k*n, n, x);
   }

   // update those values here
   // since after this step
   // noting will be stored
   //this->_rel_res = normr / this->_rel_res_vector[0];
   *prel_res = normr / normb;
   *piter = iter;
   *prel_res_v = rel_res_v;

   /* note that we are not done yet 
    * the solution might be accurate, but 
    * we still need to have a more accurate estimation
    * of the tridiagonal matrix T
    */

   /* second loop for updating T 
    * the second loop makes sense only if we
    * have enough memory to store the whole basis
    */
   while(iter < wsize)
   {
      if(t < EPSILON || dotvz < EPSILON)
      {
         /* In this case, we cannot proceed anymore, restart */
         z = Z+iter*n;
         v = V+iter*n;
         Nfft4GPVecRand(z, n);
         
         // here the return value t is the norm of the final z
         // here we always assume that the basis is full
         // it does not make sense if we throw away some vectors and restart
         Nfft4GPModifiedGS2( z, n, V, Z, &td, &te, &t, iter-1, EPSILON, 0.7071);

         if(t < EPSILON)
         {
            /* restart failed, done */
            break;
         }

         /* v = M^{-1} * z -- apply the preconditioner */
         if(prec_data)
         {
            precondfunc( prec_data, n, v, z);
         }
         else
         {
            NFFT4GP_MEMCPY( v, z, n, NFFT4GP_DOUBLE);
         }

         dotvz = sqrt(Nfft4GPVecDdot( v, n, z));

         if(dotvz < EPSILON)
         {
            /* restart failed, done */
            break;
         }

         /* reach here means restart succeeded */
         idotvz = 1.0/dotvz;
         Nfft4GPVecScale(v, n, idotvz);
         Nfft4GPVecScale(z, n, idotvz);
      }// end of new restart loop

      // Main loop
      while (iter < wsize)
      {
         iter++;

         z = Z+iter*n;
         w = V+(iter-1)*n;
         v = V+iter*n;
         
         /* matvec */
         matvec( mat_data, n, 1.0, w, 0.0, z);
         
         /* Modified Gram-schmidt without re-orth */
         NFFT4GP_MIN(iter-1, wsize, wsizei);
         
         // here the return value t is the norm of the final z
         Nfft4GPModifiedGS2( z, n, V, Z, TD+iter-1, TE+iter-2, &t, iter-1, EPSILON, 0.7071);

         // first break condition
         // stop if the norm of z is too small
         if(t < EPSILON)
         {
            break;
         }

         /* v = M^{-1} * z -- apply the preconditioner */
         if(prec_data)
         {
            precondfunc( prec_data, n, v, z);
         }
         else
         {
            NFFT4GP_MEMCPY( v, z, n, NFFT4GP_DOUBLE);
         }

         dotvz = sqrt(Nfft4GPVecDdot( v, n, z));

         // second break condition
         // stop if v'*z is too small
         if(dotvz < EPSILON)
         {
            break;
         }

         idotvz = 1.0/dotvz;
         Nfft4GPVecScale(v, n, idotvz);
         Nfft4GPVecScale(z, n, idotvz);

         /* no need to update the Cholesky factorization of Hm */
            
         if( print_level > 0)
         {
            printf("%5d   Building T\n", iter);
         }

      } // End of inner loop

   }// End of second loop

   *tsize = iter;

   if(*TDp == NULL)
   {
      *TDp = TD;
   }
   if(*TEp == NULL)
   {
      *TEp = TE;
   }

   /* De-allocate */
   NFFT4GP_FREE(V);
   NFFT4GP_FREE(Z);
   NFFT4GP_FREE(TLD);
   NFFT4GP_FREE(TLE);
   NFFT4GP_FREE(y);

   return 0;
}

int Nfft4GPLanczosQuadratureLogdet( void *mat_data,
                                 void *dmat_data,
                                 int n,
                                 func_symmatvec matvec,
                                 func_symmatvec dmatvec,
                                 void *prec_data,
                                 func_solve precondfunc,
                                 func_trace tracefunc,
                                 func_logdet logdetfunc,
                                 func_dvp dvpfunc,
                                 int maxits,
                                 int nvecs,
                                 NFFT4GP_DOUBLE *radamacher,
                                 int print_level,
                                 NFFT4GP_DOUBLE *logdet,
                                 NFFT4GP_DOUBLE **dlogdetp)
{
   int i, j;
   NFFT4GP_DOUBLE val, *dval;
   NFFT4GP_DOUBLE logdet_precond = 0.0, *traces_precond = NULL;
   NFFT4GP_DOUBLE *z, *dAz, *px, *x;

   NFFT4GP_DOUBLE EPSILON = DBL_EPSILON;

   if(*dlogdetp)
   {
      dval = *dlogdetp;
   }
   else
   {
      NFFT4GP_CALLOC(dval, 3, NFFT4GP_DOUBLE);
   }

   NFFT4GP_CALLOC(traces_precond, 3, NFFT4GP_DOUBLE);

   if(prec_data)
   {
      // if we have a preconditioner, first compute the trace and logdet of the preconditioner
      tracefunc( prec_data, &traces_precond);
      for(i = 0 ; i < 3 ; i ++)
      {
         traces_precond[i] /= (NFFT4GP_DOUBLE)n;
      }
      logdet_precond = logdetfunc( prec_data) / (NFFT4GP_DOUBLE)n;
      NFFT4GP_MALLOC(px, (size_t)n*3, NFFT4GP_DOUBLE);
   }

   // TODO: use pre-assigned memory
   NFFT4GP_MALLOC(z, (size_t)n, NFFT4GP_DOUBLE);
   NFFT4GP_MALLOC(x, (size_t)n, NFFT4GP_DOUBLE);
   NFFT4GP_MALLOC(dAz, (size_t)n*3, NFFT4GP_DOUBLE);

   char jobz = 'V';
   NFFT4GP_DOUBLE *TV = NULL;
   NFFT4GP_DOUBLE *work = NULL;
   int lwork;
   NFFT4GP_MAX(1, 2*maxits-2, lwork);
   int info;

   NFFT4GP_MALLOC(TV, (size_t)maxits*maxits, NFFT4GP_DOUBLE);
   NFFT4GP_MALLOC(work, (size_t)lwork, NFFT4GP_DOUBLE);

   // main loop
   val = 0.0;
   Nfft4GPVecFill(dval, 3, 0.0);

   // TODO: for test with fixed random seed!
   // srand(222);

   for(i = 0 ; i < nvecs ; i ++)
   {
      // create random radamacher vector
      if(radamacher)
      {
         // copy the column
         NFFT4GP_MEMCPY( z, radamacher + (size_t)i*n, n, NFFT4GP_DOUBLE);
      }
      else
      {
         Nfft4GPVecRadamacher( z, n);
      }
      // TestPrintMatrix(z,1,n,1); // TODO: for test
      Nfft4GPVecFill( x, n, 0.0);

      // compute dA*z
      NFFT4GP_DOUBLE* dmat_data_double = (NFFT4GP_DOUBLE*)dmat_data;
      
      dmatvec( dmat_data_double, n, 1.0, z, 0.0, dAz);
      
      // TODO: improve memory reuse
      NFFT4GP_DOUBLE rel_res, *rel_res_v = NULL;
      NFFT4GP_DOUBLE *TD = NULL;
      NFFT4GP_DOUBLE *TE = NULL;
      int niter;
      int tsize;

      //printf("Starting LOGDET number %d/%d\n",i,nvecs);

      // solve with Lanczos
      Nfft4GPSolverLanczos( mat_data, n, matvec, prec_data, precondfunc, x, z, maxits,
                           maxits, 0, EPSILON, &rel_res, &rel_res_v, &niter, &tsize, &TD, &TE, print_level);
      
      NFFT4GP_FREE(rel_res_v);

      // avoid having nan in return
      while(tsize > 0 && isnan(TD[tsize - 1]))
      {
         tsize --;
      }

      // run the estimation
      if(tsize == 0)
      {
         printf("Warning: empty tridiagonal matrix\n");
         return 0;
      }

      // first compute the eigendecomposition of T
      NFFT4GP_DSTEV( &jobz, &tsize, TD, TE, TV, &tsize, work, &info);
      if(info != 0)
      {
         printf("Warning: DSTEV failed at iteration %d/%d\n", i, nvecs);
         printf("TD\n");
         TestPrintMatrix(TD, 1, tsize, 1);
         printf("TE\n");
         TestPrintMatrix(TE, 1, tsize, 1);
         return -1;
      }

      // logdet
      for(j = 0 ; j < tsize ; j ++)
      {
         val += TV[j*tsize] * TV[j*tsize] * log(fabs(TD[j]));
      }

      if(prec_data)
      {
         dvpfunc( prec_data, n, NULL, z, &px);
      }

      for(j = 0 ; j < 3 ; j ++)
      {
         dval[j] += Nfft4GPVecDdot( dAz + j*n, n, x);
         if(prec_data)
         {
            dval[j] -= Nfft4GPVecDdot( px + j*n, n, z);
         }
      }

      //printf("Current val: %e | current grad: %e %e %e\n", val, dval[0], dval[1], dval[2]);

      NFFT4GP_FREE(TD);
      NFFT4GP_FREE(TE);
   }

   //printf("Loop over\n");

   NFFT4GP_FREE(z);
   NFFT4GP_FREE(x);
   NFFT4GP_FREE(dAz);
   NFFT4GP_FREE(TV);
   NFFT4GP_FREE(work);
   if(prec_data)
   {
      NFFT4GP_FREE(px);
   }

   //NFFT4GP_DOUBLE scale = 1.0 / (NFFT4GP_DOUBLE)nvecs / (NFFT4GP_DOUBLE)n;
   NFFT4GP_DOUBLE scale = 1.0 / (NFFT4GP_DOUBLE)nvecs;
   val *= scale;
   scale /= (NFFT4GP_DOUBLE)n;
   Nfft4GPVecScale( dval, 3, scale);

   // add the contribution from the preconditioner
   val += logdet_precond;
   Nfft4GPVecAxpy( 1.0, traces_precond, 3, dval);
   
   NFFT4GP_FREE(traces_precond);

   // ready to return
   *logdet = val;
   if(*dlogdetp == NULL)
   {
      *dlogdetp = dval;
   }

   //printf("Loss: %e, grad: %e %e %e\n", val, dval[0], dval[1], dval[2]);

   return 0;
}
