#include "vector.h"

pvec_int Nfft4GPVecIntCreate()
{
   pvec_int vec;
   NFFT4GP_MALLOC( vec, 1, vec_int);

   /* set value */
   vec->_len = 0;
   vec->_max_len = 0;
   vec->_data = NULL;

   return vec;
}

void Nfft4GPVecIntInit(pvec_int vec, int n_max)
{
   NFFT4GP_MALLOC(vec->_data, n_max, int);
   vec->_max_len = n_max;
}

void Nfft4GPVecIntPushback(pvec_int vec, int val)
{
   if( vec->_len == vec->_max_len)
   {
      NFFT4GP_MAX( vec->_max_len + 1, (int)(vec->_max_len * NFFT4GP_EXPAND_FACT), vec->_max_len);

      NFFT4GP_REALLOC( vec->_data, vec->_max_len, int);
   }
   vec->_data[vec->_len++] = val;
}

void Nfft4GPVecIntFree(pvec_int vec)
{
   if(vec)
   {
      NFFT4GP_FREE(vec->_data);
   }
   NFFT4GP_FREE(vec);
}

pvec_double Nfft4GPVecDoubleCreate()
{
   pvec_double vec;
   NFFT4GP_MALLOC( vec, 1, vec_double);

   /* set value */
   vec->_len = 0;
   vec->_max_len = 0;
   vec->_data = NULL;

   return vec;
}

void Nfft4GPVecDoubleInit(pvec_double vec, int n_max)
{
   NFFT4GP_MALLOC(vec->_data, n_max, NFFT4GP_DOUBLE);
   vec->_max_len = n_max;
}

void Nfft4GPVecDoublePushback(pvec_double vec, NFFT4GP_DOUBLE val)
{
   if( vec->_len == vec->_max_len)
   {
      NFFT4GP_MAX( vec->_max_len + 1, (int)(vec->_max_len * NFFT4GP_EXPAND_FACT), vec->_max_len);

      NFFT4GP_REALLOC( vec->_data, vec->_max_len, NFFT4GP_DOUBLE);
   }
   vec->_data[vec->_len++] = val;
}

void Nfft4GPVecDoubleFree(pvec_double vec)
{
   if(vec)
   {
      NFFT4GP_FREE(vec->_data);
   }
   NFFT4GP_FREE(vec);
}

pheap Nfft4GPHeapCreate()
{
   /* create empty heap */
   pheap fpsheap;
   NFFT4GP_MALLOC( fpsheap, 1, heap);

   /* set value */
   fpsheap->_max_len = 0;
   fpsheap->_len = 0;
   fpsheap->_dist_v = NULL;
   fpsheap->_index_v = NULL;
   fpsheap->_rindex_v = NULL;

   return fpsheap;
}

void Nfft4GPHeapInit(pheap fpsheap, int n)
{
   int i;

   /* create data slots */
   fpsheap->_max_len = n;
   NFFT4GP_MALLOC(fpsheap->_dist_v, n, NFFT4GP_DOUBLE);
   NFFT4GP_MALLOC(fpsheap->_index_v, n, int);
   NFFT4GP_MALLOC(fpsheap->_rindex_v, n, int);

   /* init reverse index to -1 */
   for(i = 0 ; i < n ; i ++)
   {
      fpsheap->_rindex_v[i] = -1;
   }

}

void Nfft4GPHeapAdd(pheap fpsheap, NFFT4GP_DOUBLE dist, int idx)
{

   int len = fpsheap->_len;
   int p;

   int temp_i;
   NFFT4GP_DOUBLE temp_d;

   /* put to the end */
   fpsheap->_dist_v[fpsheap->_len] = dist;
   fpsheap->_index_v[fpsheap->_len] = idx;
   fpsheap->_rindex_v[idx] = fpsheap->_len++;

   /* move it upwoard */
   while(len > 0)
   {
      /* get the parent index */
      p = (len-1)/2;
      if(fpsheap->_dist_v[p] < fpsheap->_dist_v[len])
      {
         /* this is larger, swap with the parent */
         temp_d = fpsheap->_dist_v[p];
         fpsheap->_dist_v[p] = fpsheap->_dist_v[len];
         fpsheap->_dist_v[len] = temp_d;

         temp_i = fpsheap->_index_v[p];
         fpsheap->_index_v[p] = fpsheap->_index_v[len];
         fpsheap->_index_v[len] = temp_i;

         fpsheap->_rindex_v[fpsheap->_index_v[p]] = p;
         fpsheap->_rindex_v[fpsheap->_index_v[len]] = len;

         /* repeat on the parent node */
         len = p;
      }
      else
      {
         break;
      }
   }
}

void Nfft4GPHeapPop(pheap fpsheap, NFFT4GP_DOUBLE *dist, int *idx)
{
   /* parent, left, right */
   int p,l,r;

   int temp_i;
   NFFT4GP_DOUBLE temp_d;

   fpsheap->_len--;

   *dist = fpsheap->_dist_v[0];
   *idx = fpsheap->_index_v[0];

   /* swap the first element to last and reset reverse index */
   if(fpsheap->_len > 0)
   {
      fpsheap->_rindex_v[fpsheap->_index_v[0]] = -1;

      fpsheap->_dist_v[0] = fpsheap->_dist_v[fpsheap->_len];
      fpsheap->_index_v[0] = fpsheap->_index_v[fpsheap->_len];
      fpsheap->_rindex_v[fpsheap->_index_v[0]] = 0;

   }
   else
   {
      fpsheap->_rindex_v[fpsheap->_index_v[0]] = -1;
   }

   p = 0;
   l = 1;

   /* while still in the heap */
   while(l < fpsheap->_len)
   {
      r = 2*p+2;
      /* two childs, pick the larger one */
      l = (r >= fpsheap->_len || fpsheap->_dist_v[l] > fpsheap->_dist_v[r]) ? l : r;
      if(fpsheap->_dist_v[l]>fpsheap->_dist_v[p])
      {
         /* this is smaller, swap with the child */
         temp_d = fpsheap->_dist_v[p];
         fpsheap->_dist_v[p] = fpsheap->_dist_v[l];
         fpsheap->_dist_v[l] = temp_d;

         temp_i = fpsheap->_index_v[p];
         fpsheap->_index_v[p] = fpsheap->_index_v[l];
         fpsheap->_index_v[l] = temp_i;

         fpsheap->_rindex_v[fpsheap->_index_v[p]] = p;
         fpsheap->_rindex_v[fpsheap->_index_v[l]] = l;

         p = l;
         l = 2*p+1;
      }
      else
      {
         break;
      }
   }
}

void Nfft4GPHeapDecrease(pheap fpsheap, NFFT4GP_DOUBLE dist, int idx)
{
   /* parent, left, right */
   int p,l,r;

   int temp_i;
   NFFT4GP_DOUBLE temp_d;

   p = fpsheap->_rindex_v[idx];
   if(p < 0 || fpsheap->_dist_v[p] <= dist)
   {
      /* non-decreasing, skip */
      return;
   }
   fpsheap->_dist_v[p] = dist;

   l = 2*p+1;

   /* while still in the heap */
   while(l < fpsheap->_len)
   {
      r = 2*p+2;
      /* two childs, pick the larger one */
      l = r >= fpsheap->_len || fpsheap->_dist_v[l] > fpsheap->_dist_v[r] ? l : r;
      if(fpsheap->_dist_v[l]>fpsheap->_dist_v[p])
      {
         /* this is smaller, swap with the child */
         temp_d = fpsheap->_dist_v[p];
         fpsheap->_dist_v[p] = fpsheap->_dist_v[l];
         fpsheap->_dist_v[l] = temp_d;

         temp_i = fpsheap->_index_v[p];
         fpsheap->_index_v[p] = fpsheap->_index_v[l];
         fpsheap->_index_v[l] = temp_i;

         fpsheap->_rindex_v[fpsheap->_index_v[p]] = p;
         fpsheap->_rindex_v[fpsheap->_index_v[l]] = l;

         p = l;
         l = 2*p+1;
      }
      else
      {
         break;
      }
   }
}

void Nfft4GPHeapClear(pheap fpsheap)
{

   if(fpsheap)
   {

      fpsheap->_max_len = 0;
      fpsheap->_len = 0;

      NFFT4GP_FREE(fpsheap->_dist_v);
      NFFT4GP_FREE(fpsheap->_index_v);
      NFFT4GP_FREE(fpsheap->_rindex_v);

      NFFT4GP_FREE(fpsheap);
   }
}
