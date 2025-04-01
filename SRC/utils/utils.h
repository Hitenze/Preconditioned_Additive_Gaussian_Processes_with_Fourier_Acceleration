#ifndef NFFT4GP_UTIL_H
#define NFFT4GP_UTIL_H

/**
 * @file util.h
 * @brief Basic functions, and defines, should depend only on memory.h
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
//#include <time.h>

#include <sys/stat.h>

#include "memory.h"

// Expand factor for data allocation
#define NFFT4GP_EXPAND_FACT 1.3

// Some rountines are not good with too many threads
// change to #define NFFT4GP_OPENMP_REDUCED_THREADS nthreads to use all
#define NFFT4GP_OPENMP_REDUCED_THREADS 1

// Double or Float
#ifdef NFFT4GP_USING_FLOAT32
#define NFFT4GP_DOUBLE float
#else
#define NFFT4GP_DOUBLE double
#endif

/**
 * @brief   Get time in second. Always in double precision.
 * @details Get time in second. Always in double precision.
 * @return           Return time in second.
 */
double Nfft4GPWtime();

/**
 * @brief   Swap two entries in two arrays (an int array and a real array).
 * @details Swap two entries in two arrays (an int array and a real array).
 * @param [in,out]   v_i:     int array.
 * @param [in,out]   v_d:     real array.
 * @param [in]       i:       first index.
 * @param [in]       j:       second index, swap v_i[i] with v_i[j], v_d[i] with v_d[j].
 * @return           No return.
 */
void Nfft4GPSwap(int *v_i, NFFT4GP_DOUBLE *v_d, int i, int j);

/**
 * @brief   Quick sort based on v_d in ascending order between v_d[l] and v_d[r]. Also swap v_i.
 * @details Quick sort based on v_d in ascending order between v_d[l] and v_d[r]. Also swap v_i.
 * @param [in,out]   v_i:     int array.
 * @param [in,out]   v_d:     real array, sort based on its value.
 * @param [in]       l:       start index.
 * @param [in]       r:       end index, sort between v_d[l] and v_d[r], includes v_d[r].
 * @return           No return.
 */
void Nfft4GPQsortAscend(int *v_i, NFFT4GP_DOUBLE *v_d, int l, int r);

/**
 * @brief   Quick split. Similar to quick sort, but stop when v_d[l] till v_d[k] are sorted.
 * @details Quick split. Similar to quick sort, but stop when v_d[l] till v_d[k] are sorted.
 * @param [in,out]   v_i:     int array.
 * @param [in,out]   v_d:     real array, sort based on its value.
 * @param [in]       k:       stop when v_d[l] till v_d[k] are sorted.
 * @param [in]       l:       start index.
 * @param [in]       r:       end index, sort between v_d[l] and v_d[r], includes v_d[r].
 * @return           No return.
 */
void Nfft4GPQsplitAscend(int *v_i, NFFT4GP_DOUBLE *v_d, int k, int l, int r);

/**
 * @brief   Generate a random permutation of integers from 0 to n-1.
 * @details Generate a random permutation of integers from 0 to n-1.
 * @param [in]       n:       upper bound of the permutation.
 * @param [in]       k:       length of the permutation, if k < n, return a subset.
 * @return           Return the permutation array.
 */
int* Nfft4GPRandPerm(int n, int k);

/**
 * @brief   Extract a subset of data based on permutation.
 * @details Extract a subset of data based on permutation.
 * @param [in]       data:    data matrix (ldim by d).
 * @param [in]       n:       number of points.
 * @param [in]       ldim:    leading dimension of the data matrix.
 * @param [in]       d:       dimension of data.
 * @param [in]       perm:    permutation array.
 * @param [in]       k:       length of the permutation.
 * @return           Return the subset of data.
 */
NFFT4GP_DOUBLE* Nfft4GPSubData(
   NFFT4GP_DOUBLE *data, 
   int n, 
   int ldim, 
   int d, 
   int *perm, 
   int k);

/**
 * @brief   Extract a subset of data based on permutation, with preallocated memory.
 * @details Extract a subset of data based on permutation, with preallocated memory.
 * @param [in]       data:     data matrix (ldim by d).
 * @param [in]       n:        number of points.
 * @param [in]       ldim:     leading dimension of the data matrix.
 * @param [in]       d:        dimension of data.
 * @param [in]       perm:     permutation array.
 * @param [in]       k:        length of the permutation.
 * @param [out]      subdata:  preallocated memory for the subset of data.
 * @return           Return error code.
 */
int Nfft4GPSubData2(
   NFFT4GP_DOUBLE *data, 
   int n, 
   int ldim, 
   int d, 
   int *perm, 
   int k, 
   NFFT4GP_DOUBLE *subdata);

/**
 * @brief   Extract a subset of matrix based on permutation.
 * @details Extract a subset of matrix based on permutation.
 * @param [in]       K:       matrix (ldim by n).
 * @param [in]       n:       number of columns.
 * @param [in]       ldim:    leading dimension of the matrix.
 * @param [in]       perm:    permutation array.
 * @param [in]       k:       length of the permutation.
 * @return           Return the subset of matrix.
 */
NFFT4GP_DOUBLE* Nfft4GPSubMatrix(
   NFFT4GP_DOUBLE *K, 
   int n, 
   int ldim, 
   int *perm, 
   int k);

/**
 * @brief   Expand a permutation array to full size.
 * @details Expand a permutation array to full size.
 * @param [in]       perm:    permutation array.
 * @param [in]       k:       length of the permutation.
 * @param [in]       n:       full size.
 * @return           Return the expanded permutation array.
 */
int* Nfft4GPExpandPerm(int *perm, int k, int n);

/**
 * @brief   Generate a regular 2D grid dataset.
 * @details Generate a regular 2D grid dataset.
 * @param [in]       nx:      number of points in x direction.
 * @param [in]       ny:      number of points in y direction.
 * @return           Return the dataset.
 */
void* Nfft4GPDatasetRegular2D(int nx, int ny);

/**
 * @brief   Generate a uniform random dataset.
 * @details Generate a uniform random dataset.
 * @param [in]       n:       number of points.
 * @param [in]       d:       dimension of data.
 * @return           Return the dataset.
 */
void* Nfft4GPDatasetUniformRandom(int n, int d);

/**
 * @brief   Print a matrix to stdout.
 * @details Print a matrix to stdout.
 * @param [in]       matrix:  matrix to print.
 * @param [in]       m:       number of rows.
 * @param [in]       n:       number of columns.
 * @param [in]       ldim:    leading dimension of the matrix.
 */
void TestPrintMatrix(NFFT4GP_DOUBLE *matrix, int m, int n, int ldim);

/**
 * @brief   Print a matrix to a file.
 * @details Print a matrix to a file.
 * @param [in]       file:    file pointer.
 * @param [in]       matrix:  matrix to print.
 * @param [in]       m:       number of rows.
 * @param [in]       n:       number of columns.
 * @param [in]       ldim:    leading dimension of the matrix.
 */
void TestPrintMatrixToFile(FILE *file, NFFT4GP_DOUBLE *matrix, int m, int n, int ldim);

/**
 * @brief   Print a lower triangular matrix to a file.
 * @details Print a lower triangular matrix to a file.
 * @param [in]       file:    file pointer.
 * @param [in]       matrix:  matrix to print.
 * @param [in]       m:       number of rows.
 * @param [in]       n:       number of columns.
 * @param [in]       ldim:    leading dimension of the matrix.
 */
void TestPrintTrilMatrixToFile(FILE *file, NFFT4GP_DOUBLE *matrix, int m, int n, int ldim);

/**
 * @brief   Print the pattern of a CSR matrix.
 * @details Print the pattern of a CSR matrix.
 * @param [in]       A_i:     row pointers of the CSR matrix.
 * @param [in]       A_j:     column indices of the CSR matrix.
 * @param [in]       m:       number of rows.
 * @param [in]       n:       number of columns.
 */
void TestPrintCSRMatrixPattern(int *A_i, int *A_j, int m, int n);

/**
 * @brief   Print a CSR matrix to a file.
 * @details Print a CSR matrix to a file.
 * @param [in]       file:    file pointer.
 * @param [in]       A_i:     row pointers of the CSR matrix.
 * @param [in]       A_j:     column indices of the CSR matrix.
 * @param [in]       A_a:     values of the CSR matrix.
 * @param [in]       m:       number of rows.
 * @param [in]       n:       number of columns.
 */
void TestPrintCSRMatrixToFile(
   FILE *file, 
   int *A_i, 
   int *A_j, 
   NFFT4GP_DOUBLE *A_a, 
   int m, 
   int n);

/**
 * @brief   Print the values of a CSR matrix.
 * @details Print the values of a CSR matrix.
 * @param [in]       A_i:     row pointers of the CSR matrix.
 * @param [in]       A_j:     column indices of the CSR matrix.
 * @param [in]       m:       number of rows.
 * @param [in]       n:       number of columns.
 * @param [in]       A_a:     values of the CSR matrix.
 */
void TestPrintCSRMatrixVal(
   int *A_i, 
   int *A_j, 
   int m, 
   int n, 
   NFFT4GP_DOUBLE *A_a);

/**
 * @brief   Plot a CSR matrix to a file for visualization.
 * @details Plot a CSR matrix to a file for visualization.
 * @param [in]       A_i:           row pointers of the CSR matrix.
 * @param [in]       A_j:           column indices of the CSR matrix.
 * @param [in]       A_a:           values of the CSR matrix.
 * @param [in]       m:             number of rows.
 * @param [in]       n:             number of columns.
 * @param [in]       datafilename:  name of the output file.
 */
void TestPlotCSRMatrix(
   int *A_i, 
   int *A_j, 
   NFFT4GP_DOUBLE *A_a, 
   int m, 
   int n, 
   const char *datafilename);

/**
 * @brief   Plot data points to a file for visualization.
 * @details Plot data points to a file for visualization.
 * @param [in]       data:          data matrix (ldim by d).
 * @param [in]       n:             number of points.
 * @param [in]       d:             dimension of data.
 * @param [in]       ldim:          leading dimension of the data matrix.
 * @param [in]       perm:          permutation array, can be NULL.
 * @param [in]       k:             length of the permutation.
 * @param [in]       datafilename:  name of the output file.
 */
void TestPlotData(
   NFFT4GP_DOUBLE *data, 
   int n, 
   int d, 
   int ldim, 
   int *perm, 
   int k, 
   const char *datafilename);

#endif
