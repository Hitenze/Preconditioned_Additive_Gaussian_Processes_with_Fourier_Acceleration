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
   int n1, n2, ntemp;
   int d, dtemp;
   int k;
   int kernel;
   int nwindows, dwindows;
   int n_train, n_test;
   int compute_std;
   NFFT4GP_DOUBLE x00, dx00, x01, dx01, x02, dx02;
   int adam_maxits, learn_maxits, learn_nvecs;
   NFFT4GP_DOUBLE u1, u2, z1, z2;
   NFFT4GP_DOUBLE x0[3];

   if(argc < 15 || argc > 16)
   {
      printf("Usage: %s <name> <data> <window> <ntrain> <ntest> <kernel> <f> <l> <mu> <adam_maxits> <learn_maxits> <learn_nvecs> <rank> <compute_std> [<threads>]\n", argv[0]);
      return 1;
   }

   if(argc == 16)
   {
#ifdef NFFT4GP_USING_OPENMP
      omp_set_num_threads(atoi(argv[15]));
      printf("Using %d threads\n", atoi(argv[15]));
#endif
   }

   n_train = atoi(argv[4]);
   n_test = atoi(argv[5]);
   kernel = atoi(argv[6]);

   x0[0] = atof(argv[7]);
   x0[1] = atof(argv[8]);
   x0[2] = atof(argv[9]);
   adam_maxits = atoi(argv[10]);
   learn_maxits = atoi(argv[11]);
   learn_nvecs = atoi(argv[12]);
   k = atoi(argv[13]);
   compute_std = atoi(argv[14]);
   
   srand(906);

   char *data_name = (char*)malloc(sizeof(char)*(strlen(argv[1])+1));
   char *train_feature_name = (char*)malloc(sizeof(char)*(strlen(argv[2])+strlen(".train.feature")+1));
   char *train_label_name = (char*)malloc(sizeof(char)*(strlen(argv[2])+strlen(".train.label")+1));
   char *test_feature_name = (char*)malloc(sizeof(char)*(strlen(argv[2])+strlen(".test.feature")+1));
   char *test_label_name = (char*)malloc(sizeof(char)*(strlen(argv[2])+strlen(".test.label")+1));
   strcpy(data_name, argv[1]);
   strcpy(train_feature_name, argv[2]);
   strcat(train_feature_name, ".train.feature");
   strcpy(train_label_name, argv[2]);
   strcat(train_label_name, ".train.label");
   strcpy(test_feature_name, argv[2]);
   strcat(test_feature_name, ".test.feature");
   strcpy(test_label_name, argv[2]);
   strcat(test_label_name, ".test.label");

   printf("Loading data file:\n");
   printf("  Train X: %s\n", train_feature_name);
   printf("  Train Y: %s\n", train_label_name);
   printf("  Test X: %s\n", test_feature_name);
   printf("  Test Y: %s\n", test_label_name);

   NFFT4GP_DOUBLE *X1_tmp = ReadDataFromFile(train_feature_name, &n1, &d);
   NFFT4GP_DOUBLE *X2_tmp = ReadDataFromFile(test_feature_name, &n2, &dtemp);
   if(d != dtemp)
   {
      printf("Error: dimension of X1 and X2 are not the same\n");
      exit(1);
   }
   NFFT4GP_DOUBLE *Y1_tmp = ReadLabelFromFile(train_label_name, &ntemp);
   if(n1 != ntemp)
   {
      printf("Error: number of points in X1 and Y1 are not the same\n");
      exit(1);
   }
   NFFT4GP_DOUBLE *Y2_tmp = ReadLabelFromFile(test_label_name, &ntemp);
   if(n2 != ntemp)
   {
      printf("Error: number of points in X2 and Y2 are not the same\n");
      exit(1);
   }
   
   int *windows = ReadWindowFromFile(argv[3], &nwindows, &dwindows);
   
   printf("n1 = %d, n2 = %d, kernel = %d, f = %10.10f, l = %10.10f, mu = %10.10f\n", n1, n2, kernel, x0[0], x0[1], x0[2]);
   printf("adam_maxits = %d, learn_maxits = %d, learn_nvecs = %d, rank = %d\n", adam_maxits, learn_maxits, learn_nvecs, k);
   printf("nwindows = %d, dwindows = %d\n", nwindows, dwindows);

   n_train = n_train <= 0 ? n1 : n_train;
   n_test = n_test <= 0 ? n2 : n_test;
   int *train_sample = Nfft4GPRandPerm(n1, n_train);
   int *test_sample = Nfft4GPRandPerm(n2, n_test);
   NFFT4GP_DOUBLE *X1 = (NFFT4GP_DOUBLE*)malloc(sizeof(NFFT4GP_DOUBLE)*n_train*d);
   NFFT4GP_DOUBLE *X2 = (NFFT4GP_DOUBLE*)malloc(sizeof(NFFT4GP_DOUBLE)*n_test*d);
   NFFT4GP_DOUBLE *Y1 = (NFFT4GP_DOUBLE*)malloc(sizeof(NFFT4GP_DOUBLE)*n_train);
   NFFT4GP_DOUBLE *Y2 = (NFFT4GP_DOUBLE*)malloc(sizeof(NFFT4GP_DOUBLE)*n_test);

   for(int i = 0 ; i < n_train ; i ++)
   {
      for(int j = 0 ; j < d ; j ++)
      {
         X1[(size_t)j*n_train+i] = X1_tmp[(size_t)j*n1+train_sample[i]];
      }
      Y1[i] = Y1_tmp[train_sample[i]];
   }
   for(int i = 0 ; i < n_test ; i ++)
   {
      for(int j = 0 ; j < d ; j ++)
      {
         X2[(size_t)j*n_test+i] = X2_tmp[(size_t)j*n2+test_sample[i]];
      }
      Y2[i] = Y2_tmp[test_sample[i]];
   }
   free(X1_tmp);
   free(X2_tmp);
   free(Y1_tmp);
   free(Y2_tmp);
   free(train_sample);
   free(test_sample);

   printf("n_train = %d, n_test = %d\n", n_train, n_test);

   n1 = n_train;
   n2 = n_test;

   NFFT4GP_DOUBLE *data_all = Nfft4GPNFFTAppendData(X1, n1, n1, d, X2, n2, n2);
   
   func_kernel additivekernel = &Nfft4GPKernelAdditiveKernel;
   pnfft4gp_kernel additivekernel_data = kernel == 0 ? (pnfft4gp_kernel)Nfft4GPKernelAdditiveKernelParamCreate( X1, n1, n1, d, 
                                             windows, nwindows, dwindows, &Nfft4GPKernelGaussianKernel) : 
                                             (pnfft4gp_kernel)Nfft4GPKernelAdditiveKernelParamCreate( X1, n1, n1, d, 
                                             windows, nwindows, dwindows, &Nfft4GPKernelMatern12Kernel);
   func_kernel nfft_additivekernel = kernel == 0 ? &Nfft4GPNFFTAdditiveKernelGaussianKernel : &Nfft4GPNFFTAdditiveKernelMatern12Kernel;
   pnfft4gp_kernel nfft_additivekernel_data = (pnfft4gp_kernel)Nfft4GPNFFTAdditiveKernelParamCreate( X1, n1, n1, d, 
                                             windows, nwindows, dwindows);
   pnfft4gp_kernel nfft_additivekernel_data_l = (pnfft4gp_kernel)Nfft4GPNFFTAdditiveKernelParamCreate( data_all, n1+n2, n1+n2, d, 
                                             windows, nwindows, dwindows);

   srand(807);

   int *permn = Nfft4GPRandPerm( n1, n1);
   pprecond_nys nys_mat = (pprecond_nys) Nfft4GPPrecondNysCreate();
   Nfft4GPPrecondNysSetPerm(nys_mat, permn, 0);
   Nfft4GPPrecondNysSetRank(nys_mat, k);
   
   void* adam = Nfft4GPOptimizationAdamCreate();
   
   void* gp_problem = Nfft4GPGPProblemCreate();
   int mask[3] = {1, 1, 1};
   Nfft4GPGpProblemSetup(gp_problem,
                        NFFT4GP_OPTIMIZER_ADAM,
                        Nfft4GPOptimizationAdamStep,
                        adam,
                        X1,
                        n1,
                        n1,
                        d,
                        Y1,
                        nfft_additivekernel,
                        nfft_additivekernel_data,
                        &Nfft4GPNFFTKernelFree,
                        &Nfft4GPAdditiveNFFTMatSymv,
                        &Nfft4GPAdditiveNFFTGradMatSymv,
                        additivekernel,
                        additivekernel_data,
                        &Nfft4GPKernelFree,
                        &Nfft4GPPrecondNysSetupWithKernel,
                        &Nfft4GPPrecondNysSolve,
                        &Nfft4GPPrecondNysTrace,
                        &Nfft4GPPrecondNysLogdet,
                        &Nfft4GPPrecondNysDvp,
                        &Nfft4GPPrecondNysReset,
                        nys_mat,
                        0,
                        1e-04,
                        0,
                        learn_maxits,
                        learn_nvecs,
                        NULL,
                        NFFT4GP_TRANSFORM_SOFTPLUS,
                        mask,
                        -1,
                        X2,
                        n2,
                        n2,
                        d,
                        Y2,
                        &Nfft4GPDenseMatGemv,
                        0,
                        1e-06,
                        0,
                        50,
                        -1,
                        1);
   
   Nfft4GPOptimizationAdamSetProblem( adam, 
                                 &Nfft4GPGpProblemLoss,
                                 gp_problem,
                                 3);

   NFFT4GP_DOUBLE adam_tol = 1e-06;
   NFFT4GP_DOUBLE adam_beta1 = 0.9;
   NFFT4GP_DOUBLE adam_beta2 = 0.999;
   NFFT4GP_DOUBLE adam_epsilon = 1e-08;
   NFFT4GP_DOUBLE adam_alpha = 0.01;

   Nfft4GPOptimizationAdamSetOptions( adam, 
                                    adam_maxits,
                                    adam_tol,
                                    adam_beta1,
                                    adam_beta2,
                                    adam_epsilon,
                                    adam_alpha);

   Nfft4GPOptimizationAdamInit( adam, x0);

   int flag = 0;
   for(i = 0 ; i < adam_maxits ; i ++)
   {
      flag = Nfft4GPOptimizationAdamStep( adam);
      if(flag != 0)
      {
         break;
      }
   }

   NFFT4GP_DOUBLE *label_predict = NULL;
   NFFT4GP_DOUBLE *std_predict = NULL;
   
   poptimizer_adam adamp = (poptimizer_adam) adam;

   if(flag == -1)
   {
      adamp->_nits -= 1;
   }

   printf("Training over with flag %d\n", flag);
   printf("Total iterations: %d\n", adamp->_nits);
   printf("Final parameters: %10.10f, %10.10f, %10.10f\n", adamp->_x_history[adamp->_nits*3], adamp->_x_history[adamp->_nits*3+1], adamp->_x_history[adamp->_nits*3+2]);
   
   Nfft4GPTransform( NFFT4GP_TRANSFORM_SOFTPLUS, adamp->_x_history[adamp->_nits*3], 0, &x00, &dx00);
   Nfft4GPTransform( NFFT4GP_TRANSFORM_SOFTPLUS, adamp->_x_history[adamp->_nits*3+1], 0, &x01, &dx01);
   Nfft4GPTransform( NFFT4GP_TRANSFORM_SOFTPLUS, adamp->_x_history[adamp->_nits*3+2], 0, &x02, &dx02);
   printf("Final parameters (after transform): %10.10f, %10.10f, %10.10f\n", x00, x01, x02);

   Nfft4GPAdditiveNFFTKernelFree(nfft_additivekernel_data);
   nfft_additivekernel_data = (pnfft4gp_kernel)Nfft4GPNFFTAdditiveKernelParamCreate( X1, n1, n1, d,
                                             windows, nwindows, dwindows);

   if(compute_std)
   {
      Nfft4GPAdditiveNFFTGpPredict(adamp->_x_history+adamp->_nits*3,
                                 X1,
                                 Y1,
                                 n1,
                                 n1,
                                 d,
                                 X2,
                                 n2,
                                 n2,
                                 data_all,
                                 nfft_additivekernel,
                                 nfft_additivekernel_data,
                                 nfft_additivekernel_data_l,
                                 &Nfft4GPNFFTKernelFree,
                                 &Nfft4GPAdditiveNFFTMatSymv,
                                 additivekernel,
                                 additivekernel_data,
                                 &Nfft4GPKernelFree,
                                 &Nfft4GPPrecondNysSetupWithKernel,
                                 &Nfft4GPPrecondNysSolve,
                                 nys_mat,
                                 0,
                                 1e-08,
                                 50,
                                 NFFT4GP_TRANSFORM_SOFTPLUS,
                                 -1,
                                 NULL,
                                 &label_predict,
                                 &std_predict);
   }
   else
   {
      Nfft4GPAdditiveNFFTGpPredict(adamp->_x_history+adamp->_nits*3,
                                 X1,
                                 Y1,
                                 n1,
                                 n1,
                                 d,
                                 X2,
                                 n2,
                                 n2,
                                 data_all,
                                 nfft_additivekernel,
                                 nfft_additivekernel_data,
                                 nfft_additivekernel_data_l,
                                 &Nfft4GPNFFTKernelFree,
                                 &Nfft4GPAdditiveNFFTMatSymv,
                                 additivekernel,
                                 additivekernel_data,
                                 &Nfft4GPKernelFree,
                                 &Nfft4GPPrecondNysSetupWithKernel,
                                 &Nfft4GPPrecondNysSolve,
                                 nys_mat,
                                 0,
                                 1e-08,
                                 50,
                                 NFFT4GP_TRANSFORM_SOFTPLUS,
                                 -1,
                                 NULL,
                                 &label_predict,
                                 NULL);
   }

   NFFT4GP_DOUBLE error = 0.0;

   for(i = 0; i < n2; i++)
   {
      error += (label_predict[i] - Y2[i])*(label_predict[i] - Y2[i]);
   }

   error = sqrt(error/n2);

   printf("Prediction error: %10.10f\n", error);

   char outfile_name[2048];
   sprintf(outfile_name, "%s_%d_nfft.txt", data_name, kernel);
   FILE *fp = fopen( outfile_name, "w");
   
   if(compute_std)
   {
      fprintf(fp, "Label        |  Predict        |  Std\n");
      for(i = 0; i < n2; i++)
      {
         fprintf(fp, "%10.10f %10.10f %10.10f\n", Y2[i], label_predict[i], std_predict[i]);
      }
   }
   else
   {
      fprintf(fp, "Label        |  Predict        \n");
      for(i = 0; i < n2; i++)
      {
         fprintf(fp, "%10.10f %10.10f\n", Y2[i], label_predict[i]);
      }
   }

   fclose(fp);
   
   char lossoutfile_name[2048];
   sprintf(lossoutfile_name, "%s_%d_loss_nfft.txt", data_name, kernel);
   fp = fopen( lossoutfile_name, "w");

   for(i = 0; i <= adamp->_nits; i++)
   {
      fprintf(fp, "%10.10f\n", adamp->_loss_history[i]);
   }

   fclose(fp);

   NFFT4GP_FREE(permn);
   NFFT4GP_FREE(Y1);
   NFFT4GP_FREE(X1);
   NFFT4GP_FREE(Y2);
   NFFT4GP_FREE(X2);
   NFFT4GP_FREE(data_all);
   NFFT4GP_FREE(windows);
   Nfft4GPGPProblemFree( gp_problem);
   Nfft4GPOptimizationAdamFree( adam );
   Nfft4GPPrecondNysFree(nys_mat);
   Nfft4GPKernelParamFree(additivekernel_data);
   Nfft4GPAdditiveNFFTKernelFree(nfft_additivekernel_data);
   Nfft4GPAdditiveNFFTKernelFree(nfft_additivekernel_data_l);
   free(data_name);
   free(train_feature_name);
   free(test_feature_name);
   free(train_label_name);
   free(test_label_name);
   NFFT4GP_FREE(label_predict);
   NFFT4GP_FREE(std_predict);

   return 0;
}