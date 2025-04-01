#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cassert>

#include "nfft4gp_headers.h"

double* ReadDataFromFile(const char* filename, int *n, int *d)
{
   int np, dim;
   size_t idx;
   double *data;
   
   FILE * fp;
   
   printf("Reading data from %s\n", filename);

   fp = fopen(filename, "r");
   
   if(fscanf(fp, "%d %d\n", &np, &dim) != 2)
   {
      printf("Error reading data file\n");
      exit(1);
   }
   
   data = (double*)malloc(sizeof(double)*np*dim);
   
   idx = 0;
   for(int i = 0 ; i < dim ; i ++)
   {
      for(int j = 0 ; j < np ; j ++)
      {
         if(fscanf(fp, "%lf ", data+idx++) != 1)
         {
            printf("Error reading data file at entry %ld\n", idx);
            exit(1);
         }
      }
   }
   
   fclose(fp);
   *n = np;
   *d = dim;
   
   return data;
}

double* ReadLabelFromFile(const char* filename, int *n)
{
   int np;
   size_t idx;
   double *label;
   
   FILE * fp;
   
   fp = fopen (filename, "r");
   
   if(fscanf(fp, "%d\n", &np) != 1)
   {
      printf("Error reading label file\n");
      exit(1);
   }
   
   label = (double*)malloc(sizeof(double)*np);
   
   idx = 0;
   for(int i = 0 ; i < np ; i ++)
   {
      if(fscanf(fp, "%lf", label+idx++) != 1)
      {
         printf("Error reading label file\n");
         exit(1);
      }
   }
   
   fclose(fp);
   *n = np;
   
   return label;
}

int* ReadWindowFromFile(const char* filename, int *n, int *d)
{
   int nwindow, dwindow, idx;
   int *window;
   
   FILE * fp;
   
   printf("Reading window from %s\n", filename);

   fp = fopen(filename, "r");
   
   if(fscanf(fp, "%d %d\n", &nwindow, &dwindow) != 2)
   {
      printf("Error reading data file\n");
      exit(1);
   }
   
   window = (int*)malloc(sizeof(int)*nwindow*dwindow);
   
   idx = 0;
   for(int i = 0 ; i < dwindow ; i ++)
   {
      for(int j = 0 ; j < nwindow ; j ++)
      {
         if(fscanf(fp, "%d ", window+idx++) != 1)
         {
            printf("Error reading window file at entry %d\n", idx);
            exit(1);
         }
      }
   }
   
   fclose(fp);
   *n = nwindow;
   *d = dwindow;
   
   return window;
}

int main(int argc, char **argv)
{
   int i;
   int n1, ntemp;
   int d;
   int kernel;
   int nwindows, dwindows;
   int n_sample;
   size_t seed = 906;
   NFFT4GP_DOUBLE x00, dx00, x01, dx01, x02, dx02;
   NFFT4GP_DOUBLE u1, u2, z1, z2;
   NFFT4GP_DOUBLE x0[3];

   if(argc < 8 || argc > 10)
   {
      printf("Usage: %s <data> <window> <n> <kernel> <f> <l> <mu> [<seed>] [<threads>]\n", argv[0]);
      return 1;
   }

   if(argc == 9)
   {
      seed = atoi(argv[8]);
   }

   if(argc == 10)
   {
#ifdef NFFT4GP_USING_OPENMP
      omp_set_num_threads(atoi(argv[9]));
      printf("Using %d threads\n", atoi(argv[9]));
#endif
   }

   n_sample = atoi(argv[3]);
   kernel = atoi(argv[4]);

   x0[0] = atof(argv[5]);
   x0[1] = atof(argv[6]);
   x0[2] = atof(argv[7]);
   
   srand(seed);

   char *train_feature_name = (char*)malloc(sizeof(char)*(strlen(argv[1])+strlen(".train.feature")+1));
   char *train_label_name = (char*)malloc(sizeof(char)*(strlen(argv[1])+strlen(".train.label")+1));
   strcpy(train_feature_name, argv[1]);
   strcat(train_feature_name, ".train.feature");
   strcpy(train_label_name, argv[1]);
   strcat(train_label_name, ".train.label");

   printf("Loading data file:\n");
   printf("  Train X: %s\n", train_feature_name);
   printf("  Train Y: %s\n", train_label_name);

   NFFT4GP_DOUBLE *X1_tmp = ReadDataFromFile(train_feature_name, &n1, &d);
   NFFT4GP_DOUBLE *Y1_tmp = ReadLabelFromFile(train_label_name, &ntemp);
   if(n1 != ntemp)
   {
      printf("Error: number of points in X1 and Y1 are not the same\n");
      exit(1);
   }
   
   int *windows = ReadWindowFromFile(argv[2], &nwindows, &dwindows);
   
   printf("n_sample = %d, kernel = %d, f = %10.10f, l = %10.10f, mu = %10.10f\n", n1, kernel, x0[0], x0[1], x0[2]);
   printf("nwindows = %d, dwindows = %d\n", nwindows, dwindows);
   printf("Random seed: %ld\n",seed);

   n_sample = n_sample <= 0 ? n1 : n_sample;
   int *train_sample = Nfft4GPRandPerm(n1, n_sample);
   NFFT4GP_DOUBLE *X1 = (NFFT4GP_DOUBLE*)malloc(sizeof(NFFT4GP_DOUBLE)*n_sample*d);
   NFFT4GP_DOUBLE *Y1 = (NFFT4GP_DOUBLE*)malloc(sizeof(NFFT4GP_DOUBLE)*n_sample);

   for(int i = 0 ; i < n_sample ; i ++)
   {
      for(int j = 0 ; j < d ; j ++)
      {
         X1[(size_t)j*n_sample+i] = X1_tmp[(size_t)j*n1+train_sample[i]];
      }
      Y1[i] = Y1_tmp[train_sample[i]];
   }
   free(X1_tmp);
   free(Y1_tmp);
   free(train_sample);

   printf("n_sample = %d\n", n_sample);

   n1 = n_sample;

   func_kernel additivekernel = &Nfft4GPKernelAdditiveKernel;
   pnfft4gp_kernel additivekernel_data = kernel == 0 ? (pnfft4gp_kernel)Nfft4GPKernelAdditiveKernelParamCreate( X1, n1, n1, d, 
                                             windows, nwindows, dwindows, &Nfft4GPKernelGaussianKernel) : 
                                             (pnfft4gp_kernel)Nfft4GPKernelAdditiveKernelParamCreate( X1, n1, n1, d, 
                                             windows, nwindows, dwindows, &Nfft4GPKernelMatern12Kernel);
   func_kernel nfft_additivekernel = kernel == 0 ? &Nfft4GPNFFTAdditiveKernelGaussianKernel : &Nfft4GPNFFTAdditiveKernelMatern12Kernel;
   pnfft4gp_kernel nfft_additivekernel_data = (pnfft4gp_kernel)Nfft4GPNFFTAdditiveKernelParamCreate( X1, n1, n1, d, 
                                             windows, nwindows, dwindows);
   
   additivekernel_data->_params[0] = x0[0];
   additivekernel_data->_params[1] = x0[1];
   additivekernel_data->_noise_level = x0[2];

   nfft_additivekernel_data->_params[0] = x0[0];
   nfft_additivekernel_data->_params[1] = x0[1];
   nfft_additivekernel_data->_noise_level = x0[2];

   NFFT4GP_DOUBLE *additive_mat = NULL;
   NFFT4GP_DOUBLE *additive_mat_grad = NULL;

   void *nfft_additive_mat = NULL;
   void *nfft_additive_mat_grad = NULL;

   additivekernel( (void*)additivekernel_data, X1, n_sample, n_sample, d, NULL, 0, NULL, 0, &additive_mat, &additive_mat_grad);
   nfft_additivekernel( (void*)nfft_additivekernel_data, X1, n_sample, n_sample, d, NULL, 0, NULL, 0, (NFFT4GP_DOUBLE**)&nfft_additive_mat, (NFFT4GP_DOUBLE**)&nfft_additive_mat_grad);

   NFFT4GP_DOUBLE *x_vec = NULL, *y_nfft = NULL, *dy_nfft = NULL;
   NFFT4GP_DOUBLE *y_exact = NULL, *dy_exact = NULL;
   NFFT4GP_MALLOC(x_vec, n_sample, NFFT4GP_DOUBLE);
   NFFT4GP_CALLOC(y_nfft, n_sample, NFFT4GP_DOUBLE);
   NFFT4GP_CALLOC(dy_nfft, 3*n_sample, NFFT4GP_DOUBLE);
   NFFT4GP_CALLOC(y_exact, n_sample, NFFT4GP_DOUBLE);
   NFFT4GP_CALLOC(dy_exact, 3*n_sample, NFFT4GP_DOUBLE);

   Nfft4GPVecRand(x_vec, n_sample);
   for(int i = 0 ; i < n_sample ; i ++)
   {
      x_vec[i] -= 0.5;
   }
   

   Nfft4GPAdditiveNFFTMatSymv( nfft_additive_mat, n_sample, 1.0, x_vec, 0.0, y_nfft);
   Nfft4GPAdditiveNFFTGradMatSymv( nfft_additive_mat_grad, n_sample, 1.0, x_vec, 0.0, dy_nfft);

   Nfft4GPDenseMatSymv(additive_mat, n_sample, 1.0, x_vec, 0.0, y_exact);
   Nfft4GPDenseGradMatSymv(additive_mat_grad, n_sample, 1.0, x_vec, 0.0, dy_exact);
   
   NFFT4GP_DOUBLE err = 0.0, derr[3] = {0.0, 0.0, 0.0};
   NFFT4GP_DOUBLE err1 = 0.0, derr1[3] = {0.0, 0.0, 0.0};
   NFFT4GP_DOUBLE nrmy = Nfft4GPVecNorm2(y_exact, n_sample);
   NFFT4GP_DOUBLE nrmdy[3];
   nrmdy[0] = Nfft4GPVecNorm2(dy_exact, n_sample);
   nrmdy[1] = Nfft4GPVecNorm2(dy_exact+n_sample, n_sample);
   nrmdy[2] = Nfft4GPVecNorm2(dy_exact+2*n_sample, n_sample);

   for(int i = 0 ; i < n_sample ; i ++)
   {
      NFFT4GP_MAX(err1, fabs(y_nfft[i] - y_exact[i]), err1);
      NFFT4GP_DOUBLE diff = y_nfft[i] - y_exact[i];
      err += diff*diff;
   }
   for(int i = 0 ; i < n_sample ; i ++)
   {
      for(int j = 0 ; j < 3 ; j ++)
      {
         NFFT4GP_MAX(derr1[j], fabs(dy_nfft[j*n_sample+i] - dy_exact[j*n_sample+i]), derr1[j]);
         NFFT4GP_DOUBLE diff = dy_nfft[j*n_sample+i] - dy_exact[j*n_sample+i];
         derr[j] += diff*diff;
      }
   }

   NFFT4GP_DOUBLE derr1_total = derr1[0] > derr1[1] ? derr1[0] : derr1[1];
   derr1_total = derr1_total > derr1[2] ? derr1_total : derr1[2];
   NFFT4GP_DOUBLE nrmdy_total = sqrt(nrmdy[0]*nrmdy[0] + nrmdy[1]*nrmdy[1] + nrmdy[2]*nrmdy[2]);

   printf("Linf Error. Abs: %24.20e ; Rel: %24.20e\n",err1, err1/nrmy);
   printf("Linf Gradient Error 1. Abs: %24.20e ; Rel: %24.20e\n", derr1[0], derr1[0]/nrmdy[0]);
   printf("Linf Gradient Error 2. Abs: %24.20e ; Rel: %24.20e\n", derr1[1], derr1[1]/nrmdy[1]);
   printf("Linf Gradient Error 3. Abs: %24.20e ; Rel: %24.20e\n", derr1[2], derr1[2]/nrmdy[2]);
   printf("Linf Gradient Total. Abs: %24.20e ; Rel: %24.20e\n", derr1_total, derr1_total/nrmdy_total);
   printf("L2 Error. Abs: %24.20e ; Rel: %24.20e\n", sqrt(err), sqrt(err)/nrmy);
   printf("L2 Gradient Error 1. Abs: %24.20e ; Rel: %24.20e\n", sqrt(derr[0]), sqrt(derr[0])/nrmdy[0]);
   printf("L2 Gradient Error 2. Abs: %24.20e ; Rel: %24.20e\n", sqrt(derr[1]), sqrt(derr[1])/nrmdy[1]);
   printf("L2 Gradient Error 3. Abs: %24.20e ; Rel: %24.20e\n", sqrt(derr[2]), sqrt(derr[2])/nrmdy[2]);
   printf("L2 Gradient Total. Abs: %24.20e ; Rel: %24.20e\n", sqrt(derr[0] + derr[1] + derr[2]), sqrt(derr[0] + derr[1] + derr[2])/nrmdy_total);

   NFFT4GP_FREE(Y1);
   NFFT4GP_FREE(X1);
   NFFT4GP_FREE(windows);
   NFFT4GP_FREE(x_vec);
   NFFT4GP_FREE(y_nfft);
   NFFT4GP_FREE(dy_nfft);
   NFFT4GP_FREE(y_exact);
   NFFT4GP_FREE(dy_exact);
   NFFT4GP_FREE(additive_mat);
   NFFT4GP_FREE(additive_mat_grad);
   Nfft4GPKernelParamFree(additivekernel_data);
   Nfft4GPAdditiveNFFTKernelFree(nfft_additivekernel_data);
   free(train_feature_name);
   free(train_label_name);

   return 0;
}