#include "rankest.h"

void* Nfft4GPRankestStrCreate()
{
   prankest str;
   NFFT4GP_MALLOC( str, 1, rankest);

   str->_max_rank = 2000;
   str->_full_tol = 0.9;
   str->_nsample = 500;
   str->_nsample_r = 5;
   str->_perm = NULL;
   str->_kernel_func = &Nfft4GPKernelGaussianKernel;
   str->_kernel_str = NULL;

   return (void*)str;
}

void Nfft4GPRankestStrFree(void* str)
{
   prankest pstr = (prankest)str;
   if(pstr)
   {
      NFFT4GP_FREE(pstr->_perm);
      NFFT4GP_FREE(pstr);
   }
}

// tolerance estimation using kernel function
NFFT4GP_DOUBLE Nfft4GPRankestDefaultToleranceEstimation(void *str, NFFT4GP_DOUBLE *data, int n, int ldim, int d, int nsamples, int *pk)
{
   prankest pstr = (prankest)str;
   if(pstr->_kernel_str == NULL)
   {
      printf("Error: kernel parameters are not set.\n");
      exit(1);
   }

   int i, k, n1, *randrows, rank, rank2;
   int *perm;
   NFFT4GP_DOUBLE *eigs, *small_data = NULL, *K = NULL, *dist, tol, tol2, tol3, h;

   pnfft4gp_kernel pkstr = (pnfft4gp_kernel)pstr->_kernel_str;
   NFFT4GP_DOUBLE noise_level = pkstr->_noise_level;

   tol = 0.41;
   tol2 = 0.2;
   tol3 = 1.1*noise_level;

   // generate random samples
   NFFT4GP_MIN(nsamples, n, n1);
   randrows = Nfft4GPRandPerm(n, n1);
   small_data = Nfft4GPSubData( data, n, ldim, d, randrows, n1);
   
   // FPS sort
   pordering_fps pfps = (pordering_fps)Nfft4GPOrdFpsCreate();
   pfps->_algorithm = kFpsAlgorithmSequential1;
   pfps->_build_pattern = 0; // no pattern needed
   pfps->_tol = 0.0;
   pfps->_rho = 2.1;

   k = 0;
   Nfft4GPSortFps((void*)pfps, small_data, n1, n1, d, &k, &perm);
   dist = pfps->_dist;

   // svd of kernel
   {
      pstr->_kernel_func( pstr->_kernel_str, small_data, n1, n1, d, NULL, 0, NULL, 0, &K, NULL);
      
      NFFT4GP_MALLOC( eigs, n1, NFFT4GP_DOUBLE);

      char jobz = 'N';
      char uplo = 'L';
      NFFT4GP_DOUBLE *work;
      int lwork = 3*n1-1;
      int info;
      NFFT4GP_MALLOC( work, lwork, NFFT4GP_DOUBLE);
#ifdef NFFT4GP_USING_OPENMP
      int nthreads = omp_get_max_threads();
      omp_set_num_threads(NFFT4GP_OPENMP_REDUCED_THREADS);
#endif
      NFFT4GP_DSYEV(&jobz, &uplo, &n1, K, &n1, eigs, work, &lwork, &info);
#ifdef NFFT4GP_USING_OPENMP
      omp_set_num_threads(nthreads);
#endif
      
      NFFT4GP_FREE(work);
   }

   rank = 0;
   for(i = n1-1 ; i >= 0 ; i --)
   {
      if(eigs[i] < tol3)
      {
         break;
      }
      rank++;
   }

   /* now rank is the number of entries that are larter than tol3
    * shift back by 1 to use C index
    */
   rank2 = rank-1;
   while(rank > 1)
   {
      rank --;
      if( ((dist[rank-1] - dist[rank])/ dist[rank] > tol) || (dist[rank] <= (1.0+tol2)*dist[rank2]) )
      {
         break;
      }
   }

   h = dist[rank];
   //te = FpsWtime();printf("Remaining time %fs\n",te-ts);
   printf("Estimate rank: %d/%d\n",rank+1,n1);

   if(pk)
   {
      *pk = rank+1;
   }

   NFFT4GP_FREE(randrows);
   NFFT4GP_FREE(small_data);
   NFFT4GP_FREE(K);
   NFFT4GP_FREE(eigs);
   NFFT4GP_FREE(perm);
   Nfft4GPOrdFpsFree((void*)pfps);
   
   return h;
}

int Nfft4GPRankestDefault(void *str, double *data, int n, int ldim, int d)
{
   prankest pstr = (prankest)str;

   int i;
   int rank, est_rank, est_rank_total = 0;

   // apply first fill-distance estimation
   NFFT4GP_DOUBLE tol = Nfft4GPRankestDefaultToleranceEstimation( str, data, n, ldim, d, pstr->_nsample, &est_rank);
   est_rank_total += est_rank;
   
   // add remaining
   for(i = 1 ; i < pstr->_nsample_r ; i ++)
   {
      NFFT4GP_DOUBLE tol1 = Nfft4GPRankestDefaultToleranceEstimation( str, data, n, ldim, d, pstr->_nsample, &est_rank);
      //tol = tol1 < tol? tol1 : tol;
      tol += tol1;
      est_rank_total += est_rank;
   }
   tol = tol / pstr->_nsample_r;

   if( est_rank_total / (NFFT4GP_DOUBLE)(pstr->_nsample*pstr->_nsample_r) > pstr->_full_tol)
   {
      // estimate rank on the full dataset
      pordering_fps pfps = (pordering_fps)Nfft4GPOrdFpsCreate();
      pfps->_algorithm = kFpsAlgorithmParallel1;
      pfps->_build_pattern = 0; // no pattern needed
      pfps->_tol = 0.0; // use max rank

      rank = pstr->_max_rank;
      Nfft4GPSortFps((void*)pfps, data, n, ldim, d, &rank, &(pstr->_perm));
      Nfft4GPOrdFpsFree((void*)pfps);
   }
   else
   {
      // estimate rank on the full dataset
      pordering_fps pfps = (pordering_fps)Nfft4GPOrdFpsCreate();
      pfps->_algorithm = kFpsAlgorithmParallel1;
      pfps->_build_pattern = 0; // no pattern needed
      pfps->_tol = tol;

      rank = pstr->_max_rank;
      Nfft4GPSortFps((void*)pfps, data, n, ldim, d, &rank, &(pstr->_perm));
      Nfft4GPOrdFpsFree((void*)pfps);
   }
   
   return rank;
}

// compute the Rank K Nystrom approximation error
// working array of size (n+k)*k + n*n, we can easily use 3n^2
NFFT4GP_DOUBLE Nfft4GPRankestNysError(void *str, double *data, int n, int ldim, int d, int k, NFFT4GP_DOUBLE *matrix, int *perm, NFFT4GP_DOUBLE A_fro, NFFT4GP_DOUBLE *dwork)
{
   if(k == n)
   {
      return 0.0;
   }
   if(k == 0)
   {
      return 1.0;
   }

   prankest pstr = (prankest)str;

   NFFT4GP_DOUBLE *K1 = dwork;
   NFFT4GP_DOUBLE *K11 = K1 + (size_t)n*k;
   NFFT4GP_DOUBLE *Knys = K11 + (size_t)k*k;

   // Get K1
   pstr->_kernel_func( pstr->_kernel_str, data, n, ldim, d, perm, k, perm, n, &K1, NULL);

   // Get the Chol of K11
   char uplo = 'L';
   char diag = 'N';
   int info = 0;
   
   NFFT4GP_MEMCPY(K11, K1, (size_t)k*k, NFFT4GP_DOUBLE);


#ifdef NFFT4GP_USING_OPENMP
   int nthreads = omp_get_max_threads();
   omp_set_num_threads(NFFT4GP_OPENMP_REDUCED_THREADS);
#endif
   NFFT4GP_DPOTRF( &uplo, &k, K11, &k, &info);

   // apply G to K1
   char transn = 'N';
   int nrhs = n;
   
   NFFT4GP_TRTRS( &uplo, &transn, &diag, &k, &nrhs, K11, &k, K1, &k, &info);

#ifdef NFFT4GP_USING_OPENMP
   omp_set_num_threads(nthreads);
#endif

   // get Knys
   char transt = 'T';
   NFFT4GP_DOUBLE one = 1.0;
   NFFT4GP_DOUBLE zero = 0.0;
   NFFT4GP_DGEMM( &transt, &transn, &n, &n, &k, &one, K1, &k, K1, &k, &zero, Knys, &n);
   
   // compute error
   NFFT4GP_DOUBLE mone = -1.0;
   Nfft4GPVecAxpy( mone, matrix, (size_t)n*n, Knys);

   // get norm 
   char norm = 'F';

   return NFFT4GP_DLANSY( &norm, &uplo, &n, Knys, &n, NULL)/A_fro;

}

#ifndef NFFT4GP_RANKEST_NPOINTS
#define NFFT4GP_RANKEST_NPOINTS 50
#endif

int Nfft4GPRankestNysScaledEstimateRank(void *str, double *data, int n, int ldim, int d)
{
   prankest pstr = (prankest)str;

   // first sample a scaled dataset
   if(pstr->_kernel_str == NULL)
   {
      printf("Error: kernel parameters are not set.\n");
      exit(1);
   }

   size_t i;
   int k, n1;
   int rank = 0;

   NFFT4GP_DOUBLE *K = NULL;
   NFFT4GP_DOUBLE *dwork = NULL;
   int *randrows = NULL;
   int *perm = NULL;
   NFFT4GP_DOUBLE *small_data = NULL, tol;

   pnfft4gp_kernel pkstr = (pnfft4gp_kernel)pstr->_kernel_str;
   NFFT4GP_DOUBLE noise_level = pkstr->_noise_level;

   //tol = 10*noise_level;
   tol = 0.1;

   // generate random samples
   NFFT4GP_MIN(pstr->_nsample, n, n1);
   randrows = Nfft4GPRandPerm(n, n1);
   small_data = Nfft4GPSubData( data, n, ldim, d, randrows, n1);
   // scale by so that the average distance of small data is semilar to original data
   // original data: n^(1/d)
   // small data: n1^(1/d)
   // need to scale by (n1/n)^(1/d)
   Nfft4GPVecScale(small_data, (size_t)n1*d, pow((NFFT4GP_DOUBLE)n1/n, 1.0/d));
   
   // FPS sort
   pordering_fps pfps = (pordering_fps)Nfft4GPOrdFpsCreate();
   pfps->_algorithm = kFpsAlgorithmSequential1;
   pfps->_build_pattern = 0; // no pattern needed
   pfps->_tol = 0.0;
   pfps->_rho = 2.1;

   k = 0;
   Nfft4GPSortFps((void*)pfps, small_data, n1, n1, d, &k, &perm);

   // create K without noise level
   pkstr->_noise_level = 0.0;
   pstr->_kernel_func( pstr->_kernel_str, small_data, n1, n1, d, perm, n1, NULL, 0, &K, NULL);

   // Add a small shift
   char uplo = 'L';
   char norm = 'F';
   NFFT4GP_DOUBLE A_fro = NFFT4GP_DLANSY( &norm, &uplo, &n1, K, &n1, NULL);
   
   NFFT4GP_DOUBLE nu;
#ifdef NFFT4GP_USING_FLOAT32
   nu = sqrt((float)n)*(nextafter((float)A_fro,(float)(A_fro+1.0))-A_fro);
#else
   nu = sqrt((double)n)*(nextafter((double)A_fro,(double)(A_fro+1.0))-A_fro);
#endif
   NFFT4GP_DOUBLE* A_ptr = K;
   for(i = 0 ; i < (size_t)n1 ; i ++)
   {
      (*A_ptr) += (NFFT4GP_DOUBLE)nu;
      A_ptr += (n1+1);
   }
   pkstr->_noise_level = nu;
   A_fro = NFFT4GP_DLANSY( &norm, &uplo, &n1, K, &n1, NULL);

   NFFT4GP_MALLOC( dwork, (size_t)n1*n1*3, NFFT4GP_DOUBLE);

   // next, compute the array of Nystrom error
   //int ntests = NFFT4GP_RANKEST_NPOINTS;
   int ngap = n1/NFFT4GP_RANKEST_NPOINTS;
   NFFT4GP_MAX(ngap,1,ngap);
   //NFFT4GP_DOUBLE nyserr[NFFT4GP_RANKEST_NPOINTS];

   rank = n1;

   for(i = 0 ; i < NFFT4GP_RANKEST_NPOINTS ; i ++)
   {
      if(i*ngap >= n1 || (int)floor(((i-1)*ngap)*(double)n/n1) > 2*pstr->_max_rank)
      //if(i*ngap >= n1)
      {
         rank = (i-1)*ngap;
         break;
      }
      if(Nfft4GPRankestNysError( str, small_data, n1, n1, d, i*ngap, K, perm, A_fro, dwork) < tol)
      {
         rank = i*ngap;
         break;
      }
   }

   rank = (int)floor(rank*(double)n/n1);

   pkstr->_noise_level = noise_level;

   NFFT4GP_FREE(dwork);
   NFFT4GP_FREE(K);
   Nfft4GPOrdFpsFree((void*)pfps);
   return rank;
}

int Nfft4GPRankestNysScaled(void *str, double *data, int n, int ldim, int d)
{
   prankest pstr = (prankest)str;

   // first sample a scaled dataset
   if(pstr->_kernel_str == NULL)
   {
      printf("Error: kernel parameters are not set.\n");
      exit(1);
   }

   size_t i;
   int rank, est_rank_total = 0;
   
   
   // add remaining
   for(i = 0 ; i < pstr->_nsample_r ; i ++)
   {
      est_rank_total += Nfft4GPRankestNysScaledEstimateRank( str, data, n, ldim, d);
   }

   rank = (int)floor((double)est_rank_total/pstr->_nsample_r);

   int n1;
   NFFT4GP_MIN(pstr->_nsample, n, n1);
   int ngap = n1/NFFT4GP_RANKEST_NPOINTS;
   NFFT4GP_MAX(ngap,1,ngap);
   if(rank <= (int)floor(ngap*(double)n/n1))
   {
      // if estimated as lower bound, need extra check
      printf("Extra cehck needed, perform global FPS\n");
      rank = 0;
   }

   //te = FpsWtime();printf("Remaining time %fs\n",te-ts);
   printf("Estimate rank(scaled): %d/%d\n",rank,n);

   return rank;
}