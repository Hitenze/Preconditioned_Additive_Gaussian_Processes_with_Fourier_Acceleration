#include "vecops.h"

NFFT4GP_DOUBLE Nfft4GPVecNorm2(NFFT4GP_DOUBLE *x, int n)
{
   int one = 1;
   return sqrt(NFFT4GP_DDOT( &n, x, &one, x, &one));
}

NFFT4GP_DOUBLE Nfft4GPVecDdot(NFFT4GP_DOUBLE *x, int n, NFFT4GP_DOUBLE *y)
{
   int one = 1;
   return NFFT4GP_DDOT( &n, x, &one, y, &one);
}

void Nfft4GPVecRand(NFFT4GP_DOUBLE *x, int n)
{
   int i;
   /* Note: no parallel implementation in order to make random vector consistant 
    * TODO: get some thread-safe random number generator
    */
   for(i = 0 ; i < n ; i ++)
   {
      x[i] = (NFFT4GP_DOUBLE)rand() / (NFFT4GP_DOUBLE)RAND_MAX;
   }
}

void Nfft4GPVecRadamacher(NFFT4GP_DOUBLE *x, int n)
{
   int i;
   /* Note: no parallel implementation in order to make random vector consistant
    * TODO: get some thread-safe random number generator
    */
   for(i = 0 ; i < n ; i ++)
   {
      x[i] = (NFFT4GP_DOUBLE)rand() / (NFFT4GP_DOUBLE)RAND_MAX;
      if(x[i] < 0.5)
      {
         x[i] = -1.0;
      }
      else
      {
         x[i] = 1.0;
      }
   }
}

void Nfft4GPVecFill(NFFT4GP_DOUBLE *x, size_t n, NFFT4GP_DOUBLE val)
{
   size_t i;
#ifdef NFFT4GP_USING_OPENMP
   if(!omp_in_parallel())
   {
      #pragma omp parallel for private(i) NFFT4GP_DEFAULT_OPENMP_SCHEDULE
#endif
      for(i = 0 ; i < n ; i ++)
      {
         x[i] = val;
      }
#ifdef NFFT4GP_USING_OPENMP
   }
   else
   {
      for(i = 0 ; i < n ; i ++)
      {
         x[i] = val;
      }
   }
#endif
}

void Nfft4GPVecScale(NFFT4GP_DOUBLE *x, size_t n, NFFT4GP_DOUBLE scale)
{
   size_t i;
   if(scale == 0.0)
   {
      Nfft4GPVecFill( x, n, 0.0);
   }
   else
   {
#ifdef NFFT4GP_USING_OPENMP
      if(!omp_in_parallel())
      {
         #pragma omp parallel for private(i) NFFT4GP_DEFAULT_OPENMP_SCHEDULE
#endif
         for(i = 0 ; i < n ; i ++)
         {
            x[i] *= scale;
         }
#ifdef NFFT4GP_USING_OPENMP
      }
      else
      {
         for(i = 0 ; i < n ; i ++)
         {
            x[i] *= scale;
         }
      }
#endif
   }
}

void Nfft4GPVecAxpy(NFFT4GP_DOUBLE alpha, NFFT4GP_DOUBLE *x, size_t n, NFFT4GP_DOUBLE *y)
{
   //int one = 1;
   //NFFT4GP_DAXPY(&n, &alpha, x, &one, y, &one);
   size_t i;
   if(alpha == 0.0)
   {
      return;
   }
   else if(alpha == 1.0)
   {
#ifdef NFFT4GP_USING_OPENMP
      if(!omp_in_parallel())
      {
         #pragma omp parallel for private(i) NFFT4GP_DEFAULT_OPENMP_SCHEDULE
#endif
         for(i = 0 ; i < n ; i ++)
         {
            y[i] += x[i];
         }
#ifdef NFFT4GP_USING_OPENMP
      }
      else
      {
         for(i = 0 ; i < n ; i ++)
         {
            y[i] += x[i];
         }
      }
#endif
   }
   else
   {
#ifdef NFFT4GP_USING_OPENMP
      if(!omp_in_parallel())
      {
         #pragma omp parallel for private(i) NFFT4GP_DEFAULT_OPENMP_SCHEDULE
#endif
         for(i = 0 ; i < n ; i ++)
         {
            y[i] += alpha * x[i];
         }
#ifdef NFFT4GP_USING_OPENMP
      }
      else
      {
         for(i = 0 ; i < n ; i ++)
         {
            y[i] += alpha * x[i];
         }
      }
#endif
   }
}

void Nfft4GPIVecFill(int *x, int n, int val)
{
   int i;
#ifdef NFFT4GP_USING_OPENMP
   if(!omp_in_parallel())
   {
      #pragma omp parallel for private(i) NFFT4GP_DEFAULT_OPENMP_SCHEDULE
#endif
      for(i = 0 ; i < n ; i ++)
      {
         x[i] = val;
      }
#ifdef NFFT4GP_USING_OPENMP
   }
   else
   {
      for(i = 0 ; i < n ; i ++)
      {
         x[i] = val;
      }
   }
#endif
}
