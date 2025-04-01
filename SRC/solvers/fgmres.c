#include "fgmres.h"

int Nfft4GPSolverFgmres( void *mat_data,
                     int n,
                     func_symmatvec matvec,
                     void *prec_data,
                     func_solve precondfunc,
                     NFFT4GP_DOUBLE *x,
                     NFFT4GP_DOUBLE *rhs,
                     int kdim,
                     int maxits,
                     int atol,
                     NFFT4GP_DOUBLE tol,
                     NFFT4GP_DOUBLE *prel_res,
                     NFFT4GP_DOUBLE **prel_res_v,
                     int *piter,
                     int print_level)
{

   /* Declare variables */
   int           	iter, i, j, k;
   NFFT4GP_DOUBLE		normb, EPSILON, normr, tolr, t, gam;
   NFFT4GP_DOUBLE		inormr, hii, hii1;

   NFFT4GP_DOUBLE		*c, *s, *rs, *v = NULL, *z = NULL, *w = NULL;
   NFFT4GP_DOUBLE		*V, *Z, *H;
   NFFT4GP_DOUBLE		*rel_res_v;

   EPSILON			= DBL_EPSILON;

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
   NFFT4GP_MALLOC(V, (size_t)n*(kdim+1), NFFT4GP_DOUBLE);
   NFFT4GP_MALLOC(Z, (size_t)n*(kdim+1), NFFT4GP_DOUBLE);
   NFFT4GP_MALLOC(H, (size_t)kdim*(kdim+1), NFFT4GP_DOUBLE);

   /* Working vectors for Givens rotations */
   NFFT4GP_MALLOC(c, (size_t)kdim, NFFT4GP_DOUBLE);
   NFFT4GP_MALLOC(s, kdim, NFFT4GP_DOUBLE);
   NFFT4GP_MALLOC(rs, (size_t)kdim+1, NFFT4GP_DOUBLE);

   /* Make b the first vector of the basis (will normalize later) */
   v = V;

   /* Compute the residual norm v = rhs - A*x */
   NFFT4GP_MEMCPY( v, rhs, n, NFFT4GP_DOUBLE);
   matvec( mat_data, n, -1.0, x, 1.0, v);

   /* Compute the 2-norm of residual */
   normr = Nfft4GPVecNorm2(v, n);

   /* the solution is direct */
   if (normr < EPSILON)
   {
      *prel_res = 0.0;
      *piter = 0;
      NFFT4GP_CALLOC(rel_res_v, 1, NFFT4GP_DOUBLE);
      *prel_res_v = rel_res_v;

      NFFT4GP_FREE(c);
      NFFT4GP_FREE(s);
      NFFT4GP_FREE(rs);
      NFFT4GP_FREE(V);
      NFFT4GP_FREE(Z);
      NFFT4GP_FREE(H);

      return 0;
   }

   /* use absolute tol? */
   if(atol)
   {
      tolr = tol;
   }
   else
   {
      tolr = tol*normb;
   }

   /* A few variables used to keep track of the loop's state */
   NFFT4GP_CALLOC( rel_res_v, (size_t)maxits+1, NFFT4GP_DOUBLE);

   //this->_rel_res_vector[0] = normr;
   rel_res_v[0] = normr/normb;
   iter = 0;

   if( print_level > 0 )
   {
      printf("--------------------------------------------------------------------------------\n");
      printf("Start FlexGMRES(%d)\n",kdim);
      printf("Residual Tol: %e\nMax number of inner iterations: %d\n", tolr, maxits);
      printf("--------------------------------------------------------------------------------\n");
      printf("Step    Residual norm  Relative res.  Convergence Rate\n");
      printf("%5d   %8e   %8e   N/A\n", 0, normr, rel_res_v[0]);
   }

   /* Outer loop */
   while ( iter < maxits)
   {
      /* Scale the starting vector */
      rs[0] = normr;
      inormr = 1.0/normr;

      Nfft4GPVecScale( v, n, inormr);

      // Inner loop
      i = 0;

      while (i < kdim && iter < maxits)
      {
         i++;
         iter++;

         v = V+(i-1)*n;
         z = Z+(i-1)*n;
         w = V+i*n;
         
         /* zi = M^{-1} * vi -- apply the preconditioner */
         if(prec_data)
         {
            precondfunc( prec_data, n, z, v);

            matvec( mat_data, n, 1.0, z, 0.0, w);
         }
         else
         {
            NFFT4GP_MEMCPY( z, v, n, NFFT4GP_DOUBLE);
            matvec( mat_data, n, 1.0, v, 0.0, w);
         }
         
         /* Modified Gram-schmidt without re-orth */
         Nfft4GPModifiedGS( w, n, kdim, V, H, &t, i-1, 1e-12, -1.0);

         /* Least squares problem of H */
         for (j = 1; j < i; j++)
         {
            hii = H[(i-1)*(kdim+1)+j-1];

            H[(i-1)*(kdim+1)+j-1] = c[j-1]*hii + s[j-1]*H[(i-1)*(kdim+1)+j];

            H[(i-1)*(kdim+1)+j] = -s[j-1]*hii + c[j-1]*H[(i-1)*(kdim+1)+j];
         }

         hii = H[(i-1)*(kdim+1)+i-1];
         hii1 = H[(i-1)*(kdim+1)+i];

         gam = sqrt(hii*hii + hii1*hii1);

         if (fabs(gam) < EPSILON)
         {
            goto label;
         }
         c[i-1] = hii / gam;
         s[i-1] = hii1 / gam;
         rs[i] = -s[i-1] * rs[i-1];
         rs[i-1] = c[i-1] * rs[i-1];
         /* Check residual norm */
         H[(i-1)*(kdim+1)+i-1] = c[i-1]*hii + s[i-1]*hii1;
         normr = fabs(rs[i]);
         //this->_rel_res_vector[this->_iter] = normr;
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
         printf("Rel. residual at the end of current cycle (# of steps per cycle/total its: %d/%d): %e \n", kdim, iter, rel_res_v[iter]);
      }

      /* Solve the upper triangular system */
      rs[i-1] /= H[(i-1)*(kdim+1)+i-1];
      for ( k = i-2; k >= 0; k--)
      {
         for ( j = k+1; j < i; j++)
         {
            rs[k] -= H[j*(kdim+1)+k]*rs[j];
         }
         rs[k] /= H[k*(kdim+1)+k];
      }

      /* Get current approximate solution */
      for ( j = 0; j < i; j++)
      {
         z = Z + j*n;
         Nfft4GPVecAxpy( rs[j], z, n, x);
      }

      /* Test convergence */
      if (normr <= tolr)
      {
         *prel_res = normr;
         break;
      }

      /* Restart */
      v = V;

      NFFT4GP_MEMCPY( v, rhs, n, NFFT4GP_DOUBLE);
      matvec( mat_data, n, 1.0, x, 0.0, w);

      Nfft4GPVecAxpy( -1.0, w, n, v);

   } // End of outer loop

label:
   //this->_rel_res = normr / this->_rel_res_vector[0];
   *prel_res = normr / normb;
   *piter = iter;
   *prel_res_v = rel_res_v;

   /* De-allocate */
   NFFT4GP_FREE(c);
   NFFT4GP_FREE(s);
   NFFT4GP_FREE(rs);
   NFFT4GP_FREE(V);
   NFFT4GP_FREE(Z);
   NFFT4GP_FREE(H);

   return 0;
}
