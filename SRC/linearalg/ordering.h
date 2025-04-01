#ifndef NFFT4GP_ORDERING_H
#define NFFT4GP_ORDERING_H

/**
 * @file ordering.h
 * @brief Give order to a dataset.
 */

#include "../utils/utils.h"
#include "../utils/protos.h"
#include "kernels.h"
#include "vector.h"
#include "vecops.h"

/**
 * @brief   General ordering function. Get a ordering of length k.
 * @details Geteral ordering function. Get a ordering of length k.
 * @param [in]       str: helper data structure
 * @param [in]       data: data matrix (ldim by d).
 * @param [in]       n: number of points.
 * @param [in]       ldim: leading dimension of the data matrix
 * @param [in]       d: dimension of data
 * @param [in,out]   k: number of selected points, out put the nember of selected points.
 * @param [out]      pperm: pointer to the permutation can not be NULL.
 */
typedef int (*func_ordering)(void *str, NFFT4GP_DOUBLE *data, int n, int ldim, int d, int *k, int **pperm);

/* FPS ordering */

typedef enum NFFT4GP_FPS_PATTERN_ENUM 
{
   kFpsPatternDefault = 0, // build both pattern
   kFpsPatternUonly // build only U pattern
}fps_pattern_enum;

typedef enum NFFT4GP_FPS_ALGORITHM_ENUM 
{
   kFpsAlgorithmParallel1 = 0, // O(nlogn) sequential algorithm
   kFpsAlgorithmSequential1 // O(n^2) parallel algorithm
}fps_algorithm_enum;

typedef struct NFFT4GP_FPS_STRUCT 
{
   /* FPS paramters */
   fps_algorithm_enum _algorithm; // algorithm to use
   double _tol; // tolerance to stop the FPS. The value is used only if lfil is not set.
   double _rho; // radius of the FPS search for the O(nlogn) algorithm
   func_dist _fdist; // fill distance function, by default is the euclidean distance
   void *_fdist_params; // parameter for the distance function
   NFFT4GP_DOUBLE *_dist; // distance array

   /* pattern paramters */
   int _build_pattern; // should we build pattern?
   int _pattern_lfil; // nnz for the nonzero pattern in the upper triangular part per column
   fps_pattern_enum _pattern_opt;

   /* data structures holding the pattern in CSC format */
   int *_S_i; // row pointer
   int *_S_j; // column index

}ordering_fps,*pordering_fps;

/**
 * @brief   Create a FPS ordering structure, and set the default parameters.
 * @details Create a FPS ordering structure, and set the default parameters.
 * @return  Pointer to the created structure.
 */
void* Nfft4GPOrdFpsCreate();

/**
 * @brief   Free the memory of the FPS ordering structure.
 * @details Free the memory of the FPS ordering structure.
 * @param [in]       str: pointer to the structure.
 */
void Nfft4GPOrdFpsFree(void *str);

/**
 * @brief   Compute the FPS of data points. This is a general interface.
 * @details Compute the FPS of data points. This is a general interface.
 * @param [in]       data: data values. n times d.
 * @param [in]       n: number of data points.
 * @param [in]       ldim: data leading dimension.
 * @param [in]       d: data dimension.
 * @param [in,out]   lfil: number of selected points, out put the nember of selected points. \n
 *                         If the input <= 0 then only tol is used.
 * @return           Return error message.
 */
int Nfft4GPSortFps(void *str, NFFT4GP_DOUBLE *data, int n, int ldim, int d, int *k, int **pperm);

/* random ordering */

typedef enum NFFT4GP_RANDORD_SEED_ENUM 
{
   kRandordDefault = 0, // do not reset seed
   kRandprdResetSeed // reset random seed
}fps_randord_seed_enum;

typedef struct NFFT4GP_RAND_STRUCT 
{
   /* FPS paramters */
   fps_randord_seed_enum _reset_seed; // should we reset the seed?
   int _seed; // seed for the random number generator
}ordering_rand,*pordering_rand;

/**
 * @brief   Create a random ordering structure, and set the default parameters.
 * @details Create a random ordering structure, and set the default parameters.
 * @return  Pointer to the created structure.
 */
void* Nfft4GPOrdRandCreate();

/**
 * @brief   Free the memory of the random ordering structure.
 * @details Free the memory of the random ordering structure.
 * @param [in]       str: pointer to the structure.
 */
void Nfft4GPOrdRandFree(void *str);

/**
 * @brief   Compute a random permutation of data points.
 * @details Compute a random permutation of data points.
 * @param [in]       data: data values. n times d.
 * @param [in]       n: number of data points.
 * @param [in]       ldim: data leading dimension.
 * @param [in]       d: data dimension.
 * @param [in,out]   lfil: number of selected points, out put the nember of selected points. \n
 *                         If the input <= 0 then only tol is used.
 * @return           Return error message.
 */
int Nfft4GPSortRand(void *str, NFFT4GP_DOUBLE *data, int n, int ldim, int d, int *k, int **pperm);

#endif