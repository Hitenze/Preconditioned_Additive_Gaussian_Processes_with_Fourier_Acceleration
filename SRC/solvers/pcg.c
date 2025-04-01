#include "pcg.h"

int Nfft4GPSolverPcg( void *mat_data,
         int n,
         func_symmatvec matvec,
         void *prec_data,
         func_solve precondfunc,
         NFFT4GP_DOUBLE *x,
         NFFT4GP_DOUBLE *rhs,
         int maxits,
         int atol,
         NFFT4GP_DOUBLE tol,
         NFFT4GP_DOUBLE *prel_res,
         NFFT4GP_DOUBLE **prel_res_v,
         int *piter,
         int print_level)
{
   /* Declare variables */
   int				iter = 0, ii;
   NFFT4GP_DOUBLE		rho = 1.0, alpha, beta;
   NFFT4GP_DOUBLE		normb, EPSILON, normr, normr2, tolb;

   NFFT4GP_DOUBLE		*r = NULL, *z = NULL, *p = NULL, *q = NULL;
   NFFT4GP_DOUBLE		*rel_res_v;

   EPSILON			= DBL_EPSILON;

   /* Exit if RHS is zero */
   normb = Nfft4GPVecNorm2(rhs, n);

   /* the solution is direct */
   if (normb < EPSILON)
   {
      Nfft4GPVecFill( x, n, 0.0);
      *prel_res = 0.0;
      *piter = 0;
      NFFT4GP_CALLOC(rel_res_v, 1, NFFT4GP_DOUBLE);
      *prel_res_v = rel_res_v;

      return 0;
   }

   /* use absolute tol? */
   if(atol)
   {
      tolb = tol;
   }
   else
   {
      tolb = tol*normb;
   }

   // do not exceed size of matrix
   NFFT4GP_MIN( maxits, n, maxits);

   /* Working vectors */
   NFFT4GP_MALLOC(r, n, NFFT4GP_DOUBLE);
   NFFT4GP_MALLOC(z, n, NFFT4GP_DOUBLE);
   NFFT4GP_MALLOC(p, n, NFFT4GP_DOUBLE);
   NFFT4GP_MALLOC(q, n, NFFT4GP_DOUBLE);

   /* Compute the residual norm r = rhs - A*x */
   NFFT4GP_MEMCPY( r, rhs, n, NFFT4GP_DOUBLE);
   matvec( mat_data, n, -1.0, x, 1.0, r);

   /* Compute the 2-norm of residual */
   normr = Nfft4GPVecNorm2(r, n);

   /* the solution is direct */
   if (normr < tolb)
   {
      *prel_res = normr/normb;
      *piter = 0;
      NFFT4GP_MALLOC(rel_res_v, 1, NFFT4GP_DOUBLE);
      rel_res_v[0] = *prel_res;
      *prel_res_v = rel_res_v;

      NFFT4GP_FREE(r);
      NFFT4GP_FREE(z);
      NFFT4GP_FREE(p);
      NFFT4GP_FREE(q);

      return 0;
   }
   normr2 = normr;

   /* A few variables used to keep track of the loop's state */
   NFFT4GP_CALLOC( rel_res_v, (size_t)maxits+1, NFFT4GP_DOUBLE);
   rel_res_v[0] = normr/normb;

   if( print_level > 0 )
   {
      printf("--------------------------------------------------------------------------------\n");
      printf("Start PCG\n");
      printf("Residual Tol: %e\nMax number of iterations: %d\n", tolb, maxits);
      printf("--------------------------------------------------------------------------------\n");
      printf("Step    Residual norm  Relative res.  Convergence Rate\n");
      printf("%5d   %8e   %8e   N/A\n", 0, normr, rel_res_v[0]);
   }

   for ( ii = 1; ii <= maxits; ii++)
   {
      // z = M^{-1} * r
      if(prec_data)
      {
         precondfunc( prec_data, n, z, r);
      }
      // z = r
      else
      {
         NFFT4GP_MEMCPY( z, r, n, NFFT4GP_DOUBLE);
      }

      NFFT4GP_DOUBLE rho1 = rho;

      // rho = (r_{k+1},z_{k+1})
      rho = Nfft4GPVecDdot( z, n, r);

      if (rho == 0.0)
      {
         if (print_level > 1.0)
         {
            printf("rho = %.16e\n", rho);
         }
         break;
      }

      if (ii == 1)
      {
         NFFT4GP_MEMCPY( p, z, n, NFFT4GP_DOUBLE);
      }
      else
      {
         // beta = (r_{k+1},z_{k+1}) / (r_k,z_k)
         beta = rho / rho1;
         if (beta == 0.0)
         {
            if (print_level > 1.0)
            {
               printf("beta = %.16e\n", beta);
            }
            break;
         }
         // p_{k+1} = z_{k+1} + beta*p_k
         Nfft4GPVecScale( p, n, beta);
         Nfft4GPVecAxpy( 1.0, z, n, p);

      }

      // q_{k+1} = A*p_{k+1}
      matvec( mat_data, n, 1.0, p, 0.0, q);

      // alpha = (r_{k+1},z_{k+1}) / (p_{k+1},q_{k+1})
      NFFT4GP_DOUBLE pq = Nfft4GPVecDdot(q, n, p);
      if (pq <= 0)
      {
         if (print_level > 1.0)
         {
            printf("pq = %.16e\n", pq);
         }
         break;
      }
      else
      {
         alpha = rho / pq;
      }
      // x_{k+1} = x_k + alpha*p_{k+1}
      Nfft4GPVecAxpy(alpha, p, n, x);
      // r_{k+1} = r_k - alpha*q_{k+1}
      Nfft4GPVecAxpy(-alpha, q, n, r);
      // normr = ||r_{k+1}||
      normr = Nfft4GPVecNorm2(r, n);
      normr2 = normr;
      rel_res_v[ii] = normr/normb;
      if( print_level > 0)
      {
         printf("%5d   %8e   %8e   %8.6f\n", ii, normr, rel_res_v[ii], rel_res_v[ii] / rel_res_v[ii-1]);
      }

      // check for convergence
      if (normr <= tolb)
      {
         NFFT4GP_MEMCPY( r, rhs, n, NFFT4GP_DOUBLE);
         // r = r - A*x
         matvec( mat_data, n, -1.0, x, 1.0, r);
         normr2 = Nfft4GPVecNorm2(r, n);
         rel_res_v[ii] = normr2;
         if (normr2 <= tolb)
         {
            iter = ii;
            break;
         }
      }
   } // main loop

   *prel_res = normr2 / normb;
   *piter = iter;
   *prel_res_v = rel_res_v;

   NFFT4GP_FREE(r);
   NFFT4GP_FREE(z);
   NFFT4GP_FREE(p);
   NFFT4GP_FREE(q);

   return 0;
}