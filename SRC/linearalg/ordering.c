#include "ordering.h"

/* FPS ordering */

void* Nfft4GPOrdFpsCreate()
{
   pordering_fps str = NULL;
   NFFT4GP_MALLOC(str, 1, ordering_fps);
   str->_algorithm = kFpsAlgorithmParallel1;
   str->_tol = 1e-6;
   str->_rho = 1.0;
   str->_fdist = &Nfft4GPDistanceEuclid;
   str->_fdist_params = NULL;
   str->_dist = NULL;
   str->_build_pattern = 0;
   str->_pattern_lfil = 0;
   str->_pattern_opt = kFpsPatternDefault;
   str->_S_i = NULL;
   str->_S_j = NULL;
   return (void*)str;
}

void Nfft4GPOrdFpsFree(void *str)
{
   pordering_fps pstr = (pordering_fps)str;
   if(pstr)
   {
      NFFT4GP_FREE(pstr->_dist);
      NFFT4GP_FREE(pstr->_S_i);
      NFFT4GP_FREE(pstr->_S_j);
      NFFT4GP_FREE(pstr);
   }
}

// helper function for sorting, O(nlogn) algorithm
int Nfft4GPSortFpsSeq1(NFFT4GP_DOUBLE *data, int n, int ldim, int d, int *lfil, NFFT4GP_DOUBLE tol, NFFT4GP_DOUBLE rho,
            func_dist fdist, void *fdist_params, int **pperm, NFFT4GP_DOUBLE **dist,
            int pattern_lfil, int pattern_uonly, int **pS_i, int **pS_j)
{
   // only the first k?
   int k = (*lfil > 0 ) ? *lfil : n;
   printf("Using sequential FPS algorithm 1, max k = %d\n", k);

   // do we return distance?
   int return_dist = (dist != NULL);
   int return_pattern = (pS_i != NULL && pS_j != NULL);
   pattern_lfil = (pattern_lfil > 0) ? pattern_lfil : n;

   *pperm = NULL;
   if (return_dist)
   {
      *dist = NULL;
   }
   if (return_pattern)
   {
      *pS_i = NULL;
      *pS_j = NULL;
   }
   *lfil = 0;

   if (n == 0 || d == 0)
   {
      return 0;
   }

   int i, j, i1;
   int perm_idx;

   int *perm_v;
   NFFT4GP_DOUBLE *l_v, *lf_v, *x_i1, *x_i;

   pvec_int *pi_v2, *ci_v2;
   pvec_double *cd_v2;

   pheap l_h;

   /******************
    * 0: Init
    ******************/

   NFFT4GP_MALLOC(perm_v, n, int); // create array perm

   NFFT4GP_MALLOC(l_v, n, NFFT4GP_DOUBLE); // create array l of distance
   NFFT4GP_MALLOC(lf_v, n, NFFT4GP_DOUBLE); // create array of final distance

   l_h = Nfft4GPHeapCreate(); // create heap l of heap, set length to n
   Nfft4GPHeapInit( l_h, n);

   NFFT4GP_MALLOC(pi_v2, n, pvec_int); // create parent vector

   NFFT4GP_MALLOC(ci_v2, n, pvec_int); // create child
   NFFT4GP_MALLOC(cd_v2, n, pvec_double);

   // Init
   for (i = 0 ; i < n ; i ++)
   {
      pi_v2[i] = Nfft4GPVecIntCreate();
      ci_v2[i] = Nfft4GPVecIntCreate();
      cd_v2[i] = Nfft4GPVecDoubleCreate();
   }

   /******************
    * 1: First node
    ******************/

   /* First node is the center node i1
    * set its cover circle to (rho+1)*max dist(i1,j)
    */

   // get i1
   {
      NFFT4GP_DOUBLE *x_mean, dmin, di;
      NFFT4GP_CALLOC(x_mean, d, NFFT4GP_DOUBLE);

      /* compute the mean
       * TODO: add OpenMP
       */
      for (j = 0 ; j < d ; j ++)
      {
         for (i = 0 ; i < n ; i ++)
         {
            x_mean[j] += data[ldim * j + i] / n;
         }
      }

      /* compute the one that is most close to the center */
      dmin = fdist(fdist_params, data, ldim, x_mean, 1, d);
      i1 = 0;

      for (i = 1 ; i < n ; i ++)
      {
         di = fdist(fdist_params, data + i, ldim, x_mean, 1, d);
         if (di < dmin)
         {
            dmin = di;
            i1 = i;
         }
      }

      x_i1 = data + i1;

      NFFT4GP_FREE(x_mean);
   }

   // add to permutation
   perm_idx = 0;
   perm_v[perm_idx++] = i1;

   // Create heap, also update parent and child
   {
      NFFT4GP_DOUBLE di, dmax = 0.0;

      Nfft4GPVecIntInit(ci_v2[i1], n - 1);
      Nfft4GPVecDoubleInit(cd_v2[i1], n - 1);

      for (i = 0 ; i < n ; i ++)
      {
         if (i != i1)
         {
            // push to heap
            di = fdist(fdist_params, data + i, ldim, x_i1, ldim, d);
            l_v[i] = di;
            Nfft4GPHeapAdd(l_h, di, i);

            // update p and c
            Nfft4GPVecIntPushback(pi_v2[i], i1);

            Nfft4GPVecIntPushback(ci_v2[i1], i);
            Nfft4GPVecDoublePushback(cd_v2[i1], di);

            if (di > dmax)
            {
               dmax = di;
            }

         }
      }

      l_v[i1] = 2.0 * rho * dmax;
      lf_v[0] = dmax;

      if (dmax < tol || perm_idx >= k)
      {
         // can directly return
         goto label_clear;
      }

      // Sort ci and cd
      Nfft4GPQsortAscend(ci_v2[i1]->_data, cd_v2[i1]->_data, 0, ci_v2[i1]->_len - 1);
   }

   /******************
    * 2: Remaining
    ******************/

   /* handeling remaining nodes */

   // main loop
   while (perm_idx < k)
   {
      int jj, kk, kkk;
      NFFT4GP_DOUBLE l, dij, dik, djk, *x_kkk;

      // Get the root of the heap, remove it, and restore the heap propert
      Nfft4GPHeapPop(l_h, &l, &i);

      if (l < tol)
      {
         // done
         goto label_clear;
      }

      // update the permutation
      perm_v[perm_idx] = i;
      lf_v[perm_idx++] = l;

      x_i = data + i;
      l_v[i] = rho * l;

      // Select the parent node that has all possible
      //  children of i amongst its children, and is closest to i
      kk = -1;
      dik = l_v[i1] + 1.0;

      for (jj = 0 ; jj < pi_v2[i]->_len ; jj ++)
      {
         // search within j \in p(i)
         // such that dist(i,j)+rho*l(i)\leq rho*l(j)
         // find one that min dist(i,j)

         j = pi_v2[i]->_data[jj];
         dij = fdist(fdist_params, data + j, ldim, x_i, ldim, d);

         if ( (dij + l_v[i]) <= l_v[j])
         {
            // candidate
            if (dij < dik)
            {
               kk = jj;
               dik = dij;
            }
         }
      }

      // now kkk = pi_v2[i]->_data[kk] is the parent
      if (kk >= 0)
      {
         kkk = pi_v2[i]->_data[kk];
         x_kkk = data + kkk;

         for (jj = 0 ; jj < ci_v2[kkk]->_len ; jj ++)
         {
            j = ci_v2[kkk]->_data[jj];
            djk = fdist(fdist_params, data + j, ldim, x_kkk, ldim, d);
            if (djk > dik + l_v[i])
            {
               break;
            }

            // decrease (H,j,dist(i,j))
            dij = fdist(fdist_params, data + j, ldim, x_i, ldim, d);
            Nfft4GPHeapDecrease(l_h, dij, j);

            // update c and p
            if (dij < l_v[i])
            {
               Nfft4GPVecIntPushback(ci_v2[i], j);
               Nfft4GPVecDoublePushback(cd_v2[i], dij);

               Nfft4GPVecIntPushback(pi_v2[j], i);
            }
         }
      }

      // sort ci and cd
      Nfft4GPQsortAscend(ci_v2[i]->_data, cd_v2[i]->_data, 0, ci_v2[i]->_len - 1);

   }// end of main while loop while(perm_idx < k)

label_clear:

   /******************
    * 3: Build pattern
    ******************/

   if (return_pattern)
   {
      int ii, jj;
      int *S_i, *S_j;
      /* standard approach, add ci into S_i */
      int child_len, child_maxlen;
      int *rperm;
      NFFT4GP_MALLOC( rperm, n, int);
      for (i = 0 ; i < n ; i ++)
      {
         rperm[i] = -1;
      }
      for (i = 0 ; i < perm_idx ; i ++)
      {
         rperm[perm_v[i]] = i;
      }

      NFFT4GP_CALLOC(S_i, (size_t)perm_idx + 1, int);

      for (ii = 0 ; ii < perm_idx ; ii ++)
      {
         /* diag */
         S_i[ii + 1]++;

         /* offd */
         i = perm_v[ii];

         child_len = 0;
         NFFT4GP_MIN( pattern_lfil, ci_v2[i]->_len, child_maxlen );

         for (jj = 0 ; jj < ci_v2[i]->_len ; jj ++)
         {
            j = rperm[ci_v2[i]->_data[jj]];
            if (j < perm_idx)
            {
               if (j > ii)
               {
                  /* (j, ii) and (ii, j) */
                  S_i[j + 1]++;
                  if (!pattern_uonly)
                  {
                     S_i[ii + 1]++;
                  }
               }
               child_len++;
               if (child_len >= child_maxlen)
               {
                  break;
               }
            }
         }
      }

      for (i = 2 ; i <= perm_idx ; i ++)
      {
         S_i[i] += S_i[i - 1];
      }

      /* TODO: lazy implementation
       * Second loop to insert value
       */
      NFFT4GP_MALLOC(S_j, S_i[perm_idx], int);
      for (ii = 0 ; ii < perm_idx ; ii ++)
      {
         /* diag */
         S_j[S_i[ii]++] = ii;

         /* offd */
         i = perm_v[ii];

         child_len = 0;
         NFFT4GP_MIN( pattern_lfil, ci_v2[i]->_len, child_maxlen );

         for (jj = 0 ; jj < ci_v2[i]->_len ; jj ++)
         {
            j = rperm[ci_v2[i]->_data[jj]];
            if (j < perm_idx)
            {
               if (j > ii)
               {
                  /* (j, ii) and (ii, j) */
                  S_j[S_i[j]++] = ii;
                  if (!pattern_uonly)
                  {
                     S_j[S_i[ii]++] = j;
                  }
               }
               child_len++;
               if (child_len >= child_maxlen)
               {
                  break;
               }
            }
         }
      }
      for (i = perm_idx - 1 ; i > 0 ; i --)
      {
         S_i[i] = S_i[i - 1];
      }
      S_i[0] = 0;

      NFFT4GP_FREE(rperm);

      *pS_i = S_i;
      *pS_j = S_j;
   }

   /******************
    * -1: Return and finalize
    ******************/

   *lfil = perm_idx;
   *pperm = perm_v;
   if (return_dist)
   {
      *dist = lf_v;
   }

   Nfft4GPHeapClear(l_h);
   NFFT4GP_FREE(l_v);

   for (i = 0 ; i < n ; i ++)
   {
      Nfft4GPVecIntFree(pi_v2[i]);
      Nfft4GPVecIntFree(ci_v2[i]);
      Nfft4GPVecDoubleFree(cd_v2[i]);
   }
   NFFT4GP_FREE(pi_v2);
   NFFT4GP_FREE(ci_v2);
   NFFT4GP_FREE(cd_v2);

   return 0;
}


// helper function for sorting, O(n^2) algorithm
int Nfft4GPSortFpsPar1(NFFT4GP_DOUBLE *data, int n, int ldim, int d, int *lfil, NFFT4GP_DOUBLE tol,
                  func_dist fdist, void *fdist_params, int **pperm, NFFT4GP_DOUBLE **dist)
{
   // only the first k?
   int k = (*lfil > 0 ) ? *lfil : n;
   printf("Using parallel FPS algorithm 1, max k = %d\n", k);

   // do we return distance?
   int return_dist = (dist != NULL);

   *pperm = NULL;
   if (return_dist)
   {
      *dist = NULL;
   }

   int i, j, i1, perm_idx;
   int *perm_v, *marker_v;
   NFFT4GP_DOUBLE *dist_v, *dist_current;
#ifdef NFFT4GP_USING_OPENMP
   int nthreads = omp_get_max_threads();
#endif

   NFFT4GP_MALLOC( perm_v, k, int);
   NFFT4GP_MALLOC( marker_v, n, int);
   NFFT4GP_MALLOC( dist_v, k, NFFT4GP_DOUBLE);
   NFFT4GP_MALLOC( dist_current, n, NFFT4GP_DOUBLE);
#ifdef NFFT4GP_USING_OPENMP
   #pragma omp parallel for
#endif
   for (i = 0 ; i < n ; i ++)
   {
      marker_v[i] = -1;
   }

   /******************
    * 1: First node
    ******************/

   /* First node is the center node i1
    * set its cover circle to (rho+1)*max dist(i1,j)
    */

   // get i1
   {
      NFFT4GP_DOUBLE *x_mean, dmin, di;

#ifdef NFFT4GP_USING_OPENMP
      NFFT4GP_CALLOC(x_mean, (size_t)d * nthreads, NFFT4GP_DOUBLE);

      /* compute the mean
       */
      for (j = 0 ; j < d ; j ++)
      {
         #pragma omp parallel
         {
            int myid = omp_get_thread_num();
            #pragma omp for
            for (i = 0 ; i < n ; i ++)
            {
               x_mean[j + myid * d] += data[ldim * j + i] / n;
            }
         }// end of omp parallel
      }

      #pragma omp parallel for
      for (j = 0 ; j < d ; j ++)
      {
         for (i = 1 ; i < nthreads ; i ++)
         {
            x_mean[j] += x_mean[j + i * d];
         }
      }
#else
      NFFT4GP_CALLOC(x_mean, d, NFFT4GP_DOUBLE);

      /* compute the mean
       */
      for (j = 0 ; j < d ; j ++)
      {
         for (i = 0 ; i < n ; i ++)
         {
            x_mean[j] += data[ldim * j + i] / n;
         }
      }
#endif

      /* compute the one that is most close to the center */
      dmin = fdist(fdist_params, data, ldim, x_mean, 1, d);
      i1 = 0;
#ifdef NFFT4GP_USING_OPENMP
      #pragma omp parallel
      {
         NFFT4GP_DOUBLE dmin_l = dmin;
         int i1_l = i1;
         #pragma omp for nowait
         for (i = 1 ; i < n ; i ++)
         {
            di = fdist(fdist_params, data + i, ldim, x_mean, 1, d);
            if (di < dmin_l)
            {
               dmin_l = di;
               i1_l = i;
            }
         }
         #pragma omp critical
         {
            if (dmin_l < dmin)
            {
               dmin = dmin_l;
               i1 = i1_l;
            }
         }
      }
#else
      {
         for (i = 1 ; i < n ; i ++)
         {
            di = fdist(fdist_params, data + i, ldim, x_mean, 1, d);
            if (di < dmin)
            {
               dmin = di;
               i1 = i;
            }
         }
      }
#endif

      NFFT4GP_FREE(x_mean);
   }

   /* Create the initial distance
    */
   {
      int i2;
      NFFT4GP_DOUBLE *x_i1 = data + i1, dmax, di;

      /* compute the one that is most close to the center */
      dmax = 0;
      i2 = 0;
#ifdef NFFT4GP_USING_OPENMP
      #pragma omp parallel
      {
         NFFT4GP_DOUBLE dmax_l = dmax;
         int i2_l = i2;
         #pragma omp for nowait
         for (i = 0 ; i < n ; i ++)
         {
            di = fdist(fdist_params, data + i, ldim, x_i1, ldim, d);
            dist_current[i] = di;
            if (di > dmax_l)
            {
               dmax_l = di;
               i2_l = i;
            }
         }
         #pragma omp critical
         {
            if (dmax_l > dmax)
            {
               dmax = dmax_l;
               i2 = i2_l;
            }
         }
      }
#else
      {
         for (i = 0 ; i < n ; i ++)
         {
            di = fdist(fdist_params, data + i, ldim, x_i1, ldim, d);
            dist_current[i] = di;
            if (di > dmax)
            {
               dmax = di;
               i2 = i;
            }
         }
      }
#endif

      // add to permutation
      perm_idx = 0;
      dist_current[i1] = dmax;
      marker_v[i1] = perm_idx;
      dist_v[perm_idx] = dmax;
      perm_v[perm_idx++] = i1;

      if ( dmax < tol || perm_idx >= k )
      {
         // we are ready to go
         goto label_naive_clear;
      }

      i1 = i2;
      marker_v[i1] = perm_idx;
      dist_v[perm_idx] = dmax;
      perm_v[perm_idx++] = i1;

   }

   /************************************
    * 1: Proceed the remainint nodes
    ************************************/

   /* Now dist_current holds all the current indices
    *
    */
   while (perm_idx < k && dist_current[i1] >= tol)
   {
      /* first update the distance to i1 */
      int i2;
      NFFT4GP_DOUBLE *x_i1 = data + i1, dmax, di;

      /* compute the one that is most close to the center */
      dmax = 0;
      i2 = 0;
#ifdef NFFT4GP_USING_OPENMP
      #pragma omp parallel
      {
         NFFT4GP_DOUBLE dmax_l = dmax;
         int i2_l = i2;
         #pragma omp for nowait
         for (i = 0 ; i < n ; i ++)
         {
            if (marker_v[i] < 0)
            {
               di = fdist(fdist_params, data + i, ldim, x_i1, ldim, d);
               NFFT4GP_MIN( dist_current[i], di, dist_current[i]);
               if ( dist_current[i] > dmax_l)
               {
                  dmax_l = dist_current[i];
                  i2_l = i;
               }
            }
         }
         #pragma omp critical
         {
            if (dmax_l > dmax)
            {
               dmax = dmax_l;
               i2 = i2_l;
            }
         }
      }
#else
      {
         for (i = 0 ; i < n ; i ++)
         {
            if (marker_v[i] < 0)
            {
               di = fdist(fdist_params, data + i, ldim, x_i1, ldim, d);
               NFFT4GP_MIN( dist_current[i], di, dist_current[i]);
               if ( dist_current[i] > dmax)
               {
                  dmax = dist_current[i];
                  i2 = i;
               }
            }
         }
      }
#endif

      i1 = i2;

      // add to permutation

      marker_v[i1] = perm_idx;
      dist_v[perm_idx] = dmax;
      perm_v[perm_idx++] = i1;

   }

label_naive_clear:

   *lfil = perm_idx;
   *pperm = perm_v;
   if (return_dist)
   {
      *dist = dist_v;
   }
   else
   {
      NFFT4GP_FREE(dist_v);
   }

   NFFT4GP_FREE( marker_v);
   NFFT4GP_FREE( dist_current);

   return 0;
}

int Nfft4GPSortFps(void *str, NFFT4GP_DOUBLE *data, int n, int ldim, int d, int *k, int **pperm)
{
   pordering_fps fps = (pordering_fps) str;

   switch (fps->_algorithm)
   {
      case kFpsAlgorithmSequential1:
      {
         NFFT4GP_DOUBLE tol = fps->_tol;
         NFFT4GP_DOUBLE rho = fps->_rho;
         int pattern_uonly = fps->_pattern_opt == kFpsPatternUonly ? 1 : 0;

         return Nfft4GPSortFpsSeq1(data, n, ldim, d, k, tol, rho, &Nfft4GPDistanceEuclid, NULL, 
               pperm, &(fps->_dist), fps->_pattern_lfil, pattern_uonly, &(fps->_S_i), &(fps->_S_j));

         break;
      }
      case kFpsAlgorithmParallel1: default:
      {
         NFFT4GP_DOUBLE tol = fps->_tol;
         return Nfft4GPSortFpsPar1(data, n, ldim, d, k, tol, &Nfft4GPDistanceEuclid, NULL, 
               pperm, &(fps->_dist));
      }
   }

   return 0;
}

void* Nfft4GPOrdRandCreate()
{
   pordering_rand str = NULL;
   NFFT4GP_MALLOC(str, 1, ordering_rand);
   str->_reset_seed = kRandordDefault;
   str->_seed = 123;
   return str;
}

void Nfft4GPOrdRandFree(void *str)
{
   NFFT4GP_FREE(str);
}

int Nfft4GPSortRand(void *str, NFFT4GP_DOUBLE *data, int n, int ldim, int d, int *k, int **pperm)
{
   pordering_rand ord = (pordering_rand) str;
   if(ord->_reset_seed == kRandprdResetSeed)
   {
      srand(ord->_seed);
   }

   int rank = (k[0] > n) ? n : k[0];
   printf("Using random reordering algorithm, max k = %d\n", k[0]);

   int *perm = Nfft4GPRandPerm(n, rank);
   *pperm = perm;
   
   return 0;
}
