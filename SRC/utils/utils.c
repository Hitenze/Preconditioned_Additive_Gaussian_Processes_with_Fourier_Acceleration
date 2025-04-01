#include "utils.h"


double Nfft4GPWtime()
{
   struct timeval ctime;
   gettimeofday(&ctime, NULL);
   return (double)ctime.tv_sec + (double)0.000001*ctime.tv_usec;
}

void Nfft4GPSwap( int *v_i, NFFT4GP_DOUBLE *v_d, int i, int j)
{
   int  temp_i;
   NFFT4GP_DOUBLE temp_d;

   temp_i = v_i[i];
   v_i[i] = v_i[j];
   v_i[j] = temp_i;
   temp_d = v_d[i];
   v_d[i] = v_d[j];
   v_d[j] = temp_d;
}


void Nfft4GPQsortAscend( int *v_i, NFFT4GP_DOUBLE *v_d, int l, int r)
{
   int i, last;

   if (l >= r)
   {
      return;
   }
   Nfft4GPSwap( v_i, v_d, l, (l + r) / 2);
   last = l;
   for (i = l + 1; i <= r; i++)
   {
      if (v_d[i] < v_d[l])
      {
         Nfft4GPSwap(v_i, v_d, ++last, i);
      }
   }
   Nfft4GPSwap(v_i, v_d, l, last);
   Nfft4GPQsortAscend(v_i, v_d, l, last - 1);
   Nfft4GPQsortAscend(v_i, v_d, last + 1, r);
}

void Nfft4GPQsplitAscend( int *v_i, NFFT4GP_DOUBLE *v_d, int k, int l, int r)
{
   int i, last;

   if (l >= r)
   {
      return;
   }
   Nfft4GPSwap( v_i, v_d, l, (l + r) / 2);
   last = l;
   for (i = l + 1; i <= r; i++)
   {
      if (v_d[i] < v_d[l])
      {
         Nfft4GPSwap(v_i, v_d, ++last, i);
      }
   }
   Nfft4GPSwap(v_i, v_d, l, last);
   Nfft4GPQsplitAscend(v_i, v_d, k, l, last - 1);
   if(last + 1 <= k)
   {
      Nfft4GPQsplitAscend(v_i, v_d, k, last + 1, r);
   }
}

int* Nfft4GPRandPerm( int n, int k)
{
   int i;
   int *v_i, *perm;
   NFFT4GP_DOUBLE *v_d;

   NFFT4GP_MALLOC(v_i, n, int);
   NFFT4GP_MALLOC(perm, k, int);
   NFFT4GP_MALLOC(v_d, n, NFFT4GP_DOUBLE);
// TODO: use thread safe random number generator
//#ifdef NFFT4GP_USING_OPENMP
//   #pragma omp parallel for private(i) NFFT4GP_DEFAULT_OPENMP_SCHEDULE
//#endif
   for(i = 0 ; i < n ; i ++)
   {
      v_i[i] = i;
      v_d[i] = (NFFT4GP_DOUBLE)rand();
   }

   Nfft4GPQsplitAscend( v_i, v_d, k-1, 0, n-1);

#ifdef NFFT4GP_USING_OPENMP
   #pragma omp parallel for private(i) NFFT4GP_DEFAULT_OPENMP_SCHEDULE
#endif
   for(i = 0 ; i < k ; i ++)
   {
      perm[i] = v_i[i];
   }

   NFFT4GP_FREE(v_i);
   NFFT4GP_FREE(v_d);

   return perm;

}

NFFT4GP_DOUBLE* Nfft4GPSubData( NFFT4GP_DOUBLE *data, int n, int ldim, int d, int *perm, int k)
{
   int i, j;
   NFFT4GP_DOUBLE *data1;
   NFFT4GP_MALLOC( data1, (size_t)k*d, NFFT4GP_DOUBLE);

   // one dimension at a time
   for(j = 0 ; j < d ; j ++)
   {
      NFFT4GP_DOUBLE *dj = data + j * ldim;
      NFFT4GP_DOUBLE *d1j = data1 + j * k;
#ifdef NFFT4GP_USING_OPENMP
      if (!omp_in_parallel())
      {
         #pragma omp parallel for private(i) NFFT4GP_DEFAULT_OPENMP_SCHEDULE
#endif
         for(i = 0 ; i < k ; i ++)
         {
            d1j[i] = dj[perm[i]];
         }
#ifdef NFFT4GP_USING_OPENMP
      }
      else
      {
         for(i = 0 ; i < k ; i ++)
         {
            d1j[i] = dj[perm[i]];
         }
      }
#endif
   }
   return data1;
}

int Nfft4GPSubData2( NFFT4GP_DOUBLE *data, int n, int ldim, int d, int *perm, int k, NFFT4GP_DOUBLE *subdata)
{
   int i, j;

   // one dimension at a time
   for(j = 0 ; j < d ; j ++)
   {
      NFFT4GP_DOUBLE *dj = data + j * ldim;
      NFFT4GP_DOUBLE *d1j = subdata + j * k;
#ifdef NFFT4GP_USING_OPENMP
      if (!omp_in_parallel())
      {
         #pragma omp parallel for private(i) NFFT4GP_DEFAULT_OPENMP_SCHEDULE
#endif
         for(i = 0 ; i < k ; i ++)
         {
            d1j[i] = dj[perm[i]];
         }
#ifdef NFFT4GP_USING_OPENMP
      }
      else
      {
         for(i = 0 ; i < k ; i ++)
         {
            d1j[i] = dj[perm[i]];
         }
      }
#endif
   }
   return 0;
}

NFFT4GP_DOUBLE* AnfSubMatrix( NFFT4GP_DOUBLE *K, int n, int ldim, int *perm, int k)
{
   int i, j;
   NFFT4GP_DOUBLE *K1;
   NFFT4GP_MALLOC( K1, (size_t)k*k, NFFT4GP_DOUBLE);

   // one dimension at a time
   for(j = 0 ; j < k ; j ++)
   {
      NFFT4GP_DOUBLE *dj = K + perm[j] * ldim;
      NFFT4GP_DOUBLE *d1j = K1 + j * k;
#ifdef NFFT4GP_USING_OPENMP
      if(!omp_in_parallel())
      {
         #pragma omp parallel for private(i) NFFT4GP_DEFAULT_OPENMP_SCHEDULE
#endif
         for(i = 0 ; i < k ; i ++)
         {
            d1j[i] = dj[perm[i]];
         }
#ifdef NFFT4GP_USING_OPENMP
      }
      else
      {
         for(i = 0 ; i < k ; i ++)
         {
            d1j[i] = dj[perm[i]];
         }
      }
#endif
   }
   return K1;
}

int* Nfft4GPExpandPerm( int *perm, int k, int n)
{
   int i, idx;
   int *new_perm, *marker = NULL;

   if(k >= n)
   {
      NFFT4GP_MALLOC(new_perm, k, int);
      NFFT4GP_MEMCPY(new_perm, perm, k, int);
      return new_perm;
   }

   NFFT4GP_MALLOC(new_perm, n, int);
   NFFT4GP_MEMCPY(new_perm, perm, k, int);

   NFFT4GP_CALLOC(marker, n, int);
   for(i = 0 ; i < k ; i ++)
   {
      marker[perm[i]] = 1;
   }

// TODO: add openmp
   idx = k;
   for(i = 0 ; i < n  ; i ++)
   {
      if(marker[i] == 0)
      {
         /* not used */
         new_perm[idx++] = i;
      }
   }

   NFFT4GP_FREE(marker);
   return new_perm;
}

void* Nfft4GPDatasetRegular2D(int nx, int ny)
{
   int n, i, j, idx;
   NFFT4GP_DOUBLE *data;
   
   n = nx * ny;
   
   NFFT4GP_MALLOC(data, (size_t)n*2, NFFT4GP_DOUBLE);
   
   idx = 0;
   for (i = 0 ; i < ny ; i ++)
   {
      for ( j = 0 ; j < nx ; j ++)
      {
         data[idx] = j;
         data[n+idx++] = i;
      }
   }

   return (void*)data;
}

void* Nfft4GPDatasetUniformRandom(int n, int d)
{
   int i, j, idx;
   NFFT4GP_DOUBLE scale, *data;
   
   NFFT4GP_MALLOC(data, (size_t)n*d, NFFT4GP_DOUBLE);

   scale = pow( n, 1.0/d);
   
   idx = 0;
   for (i = 0 ; i < n ; i ++)
   {
      for ( j = 0 ; j < d ; j ++)
      {
         data[j*n+idx] = scale * (NFFT4GP_DOUBLE)rand() / (NFFT4GP_DOUBLE)RAND_MAX;
      }
      idx++;
   }
   
   return (void*)data;
}


// print a matrix to terminal
void TestPrintMatrix(NFFT4GP_DOUBLE *matrix, int m, int n, int ldim)
{
   int i, j;
   for(i=0; i<m; i++)
   {
      for(j=0; j<n; j++)
      {
         printf("%24.20f ", matrix[i+j*ldim]);
      }
      printf("\n");
   }
}

void TestPrintMatrixToFile(FILE *file, NFFT4GP_DOUBLE *matrix, int m, int n, int ldim)
{
   int i, j;
   for(i=0; i<m; i++)
   {
      for(j=0; j<n; j++)
      {
         fprintf(file, "%16.6f ", matrix[i+j*ldim]);
      }
      fprintf(file, "\n");
   }
}

void TestPrintTrilMatrixToFile(FILE *file, NFFT4GP_DOUBLE *matrix, int m, int n, int ldim)
{
   int i, j;
   for(i=0; i<m; i++)
   {
      for(j=0; j<=i; j++)
      {
         fprintf(file, "%16.6f ", matrix[i+j*ldim]);
      }
      for(j=i+1; j<n; j++)
      {
         fprintf(file, "%16.6f ", 0.0);
      }
      fprintf(file, "\n");
   }
}

// print a CSR matrix to terminal
void TestPrintCSRMatrixPattern(int *A_i, int *A_j, int m, int n)
{
   int *temp_matrix = NULL;
   NFFT4GP_CALLOC(temp_matrix, (size_t)m*n, int);

   int i, j1, j2, j;
   for(i=0; i<m; i++)
   {
      j1 = A_i[i];
      j2 = A_i[i+1];
      for(j=j1; j<j2; j++)
      {
         temp_matrix[i+A_j[j]*m] = 1;
      }
   }

   for(i=0; i<m; i++)
   {
      for(j=0; j<n; j++)
      {
         printf("%d ", temp_matrix[i+j*m]);
      }
      printf("\n");
   }

   NFFT4GP_FREE(temp_matrix);
}

void TestPrintCSRMatrixToFile(FILE *file, int *A_i, int *A_j, NFFT4GP_DOUBLE *A_a, int m, int n)
{
   NFFT4GP_DOUBLE *temp_matrix = NULL;
   NFFT4GP_CALLOC(temp_matrix, (size_t)m*n, NFFT4GP_DOUBLE);

   int i, j1, j2, j;
   for(i=0; i<m; i++)
   {
      j1 = A_i[i];
      j2 = A_i[i+1];
      for(j=j1; j<j2; j++)
      {
         temp_matrix[i+A_j[j]*m] = A_a[j];
      }
   }

   for(i=0; i<m; i++)
   {
      for(j=0; j<n; j++)
      {
         fprintf(file, "%16.6f", temp_matrix[i+j*m]);
      }
      fprintf(file, "\n");
   }

   NFFT4GP_FREE(temp_matrix);
}

void TestPrintCSRMatrixVal(int *A_i, int *A_j, int m, int n, NFFT4GP_DOUBLE *A_a)
{
   int field_width = 10; // Adjust this as needed
   int i, j, k = 0;
   for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            if (j == A_j[k] && k < A_i[i+1]) {
                printf("%*f ", field_width, A_a[k]);
                k++;
            } else {
                printf("%*d ", field_width, 0);
            }
        }
        printf("\n");
   }
}

#ifndef NFFT4GP_MAX_PLOT_DATA_COUNT
#define NFFT4GP_MAX_PLOT_DATA_COUNT 300000
#endif

int nfft4gp_plot_data_count;

void TestPlotCSRMatrix(int *A_i, int *A_j, NFFT4GP_DOUBLE *A_a, int m, int n, const char *datafilename)
{
   int i, j1, j2, j;
   // we print only the first nnz
   mkdir( "./TempData", 0777 );

   FILE *fdata, *pgnuplot;
   
   char tempfilename[1024];
   snprintf( tempfilename, 1024, "./TempData/%s%05d", datafilename, nfft4gp_plot_data_count);

   nfft4gp_plot_data_count++;

   if ((fdata = fopen(tempfilename, "w")) == NULL)
   {
      printf("Can't open file.\n");
      exit(1);
   }
   
   if ((pgnuplot = popen("gnuplot -persistent", "w")) == NULL)
   {
      printf("Can't open gnuplot file.\n");
      exit(1);
   }
   
   if(A_a)
   {
      /*
      NFFT4GP_DOUBLE maxA_a = 1e-12;

      for(i=0; i<m; i++)
      {
         j1 = A_i[i];
         j2 = A_i[i+1];
         for(j=j1; j<j2; j++)
         {
            if(j > NFFT4GP_MAX_PLOT_DATA_COUNT)
            {
               break;
            }
            NFFT4GP_MAX( fabs(A_a[j]), maxA_a, maxA_a );
         }
      }
      */
      
      for(i=0; i<m; i++)
      {
         j1 = A_i[i];
         j2 = A_i[i+1];
         for(j=j1; j<j2; j++)
         {
            if(j > NFFT4GP_MAX_PLOT_DATA_COUNT)
            {
               break;
            }
            //fprintf(fdata, "%d %d %d \n", i, n-A_j[j], (int)(floor(fabs(A_a[j])/maxA_a*255)) );
            fprintf(fdata, "%d %d %24.20f \n", i, n-A_j[j], fabs(A_a[j]) );
         }
      }
   }
   else
   {
      for(i=0; i<m; i++)
      {
         j1 = A_i[i];
         j2 = A_i[i+1];
         for(j=j1; j<j2; j++)
         {
            if(j > NFFT4GP_MAX_PLOT_DATA_COUNT)
            {
               break;
            }
            fprintf(fdata, "%d %d \n", m-i, A_j[j]);
         }
      }
   }

   fclose(fdata);

   fprintf(pgnuplot, "set title \"spy fsai\"\n");
   if(A_a)
   {
      //f65536*$3+256*$3+$3
      fprintf(pgnuplot, "plot '%s' u 1:2:($3) w d lc palette\n", tempfilename);
   }
   else
   {
      fprintf(pgnuplot, "plot '%s' pt 0\n", tempfilename);
   }
   
   pclose(pgnuplot);

}

void TestPlotData(NFFT4GP_DOUBLE *data, int n, int d, int ldim, int *perm, int k, const char *datafilename)
{
   int i, ii;
   if(d < 2)
   {
      printf("Error: data dimension is less than 2.\n");
      exit(1);
   }

   // we print at most 10000 data points 
   n = (n > NFFT4GP_MAX_PLOT_DATA_COUNT) ? NFFT4GP_MAX_PLOT_DATA_COUNT : n;
   k = (k > NFFT4GP_MAX_PLOT_DATA_COUNT) ? NFFT4GP_MAX_PLOT_DATA_COUNT : k;

   mkdir( "./TempData", 0777 );

   FILE *fdata, *pgnuplot;
   
   char tempfilename[1024];
   snprintf( tempfilename, 1024, "./TempData/%s%05ddata", datafilename, nfft4gp_plot_data_count);

   char tempfilename2[1024];
   snprintf( tempfilename2, 1024, "./TempData/%s%05dselected", datafilename, nfft4gp_plot_data_count);
   
   nfft4gp_plot_data_count++;

   if ((fdata = fopen(tempfilename, "w")) == NULL)
   {
      printf("Can't open file.\n");
      exit(1);
   }
   
   if ((pgnuplot = popen("gnuplot -persistent", "w")) == NULL)
   {
      printf("Can't open gnuplot file.\n");
      exit(1);
   }
   
   // first print the base data
   if(d == 2)
   {
      for(i = 0 ; i < n ; i ++)
      {
         fprintf(fdata, "%24.20f %24.20f \n", data[i], data[i+ldim]);
      }
   }
   else
   {
      for(i = 0 ; i < n ; i ++)
      {
         fprintf(fdata, "%24.20f %24.20f %24.20f \n", data[i], data[i+ldim], data[i+2*ldim]);
      }
   }

   fclose(fdata);
   
   // now print the selected data
   if(k > 0 && perm != NULL)
   {
      
      if ((fdata = fopen(tempfilename2, "w")) == NULL)
      {
         printf("Can't open file.\n");
         exit(1);
      }
      
      if(d == 2)
      {
         for(i = 0 ; i < k ; i ++)
         {
            ii = perm[i];
            fprintf(fdata, "%24.20f %24.20f \n", data[ii], data[ii+ldim]);
         }
      }
      else
      {
         for(i = 0 ; i < k ; i ++)
         {
            ii = perm[i];
            fprintf(fdata, "%24.20f %24.20f %24.20f \n", data[ii], data[ii+ldim], data[ii+2*ldim]);
         }
      }
      
      fclose(fdata);
   }

   fprintf(pgnuplot, "set title \"view data\"\n");
   if(d == 2)
   {
      fprintf(pgnuplot, "plot '%s' pt 0", tempfilename);
      if(k > 0 && perm != NULL)
      {
         fprintf(pgnuplot, ", '%s' pt 7", tempfilename2);
      }
      fprintf(pgnuplot, "\n");
   }
   else
   {
      fprintf(pgnuplot, "splot '%s' pt 0", tempfilename);
      if(k > 0 && perm != NULL)
      {
         fprintf(pgnuplot, ", '%s' pt 7", tempfilename2);
      }
      fprintf(pgnuplot, "\n");
   }
   
   pclose(pgnuplot);

}