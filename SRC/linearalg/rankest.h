#ifndef NFFT4GP_RANKEST_H
#define NFFT4GP_RANKEST_H

/**
 * @file rankest.h
 * @brief Estimate the rank of a matrix
 */

#include "../utils/utils.h"
#include "kernels.h"
#include "ordering.h"

typedef struct NFFT4GP_RANKEST_STRUCT 
{
   int _nsample; // size of subsampled dataset
   int _nsample_r; // numbers we redo the subsampling
   int _max_rank; // maximum rank we consider
   NFFT4GP_DOUBLE _full_tol; // tolerance for skipping the rank estimation
   func_kernel _kernel_func; // pointer to the kernel function
   void* _kernel_str; // data for the kernel function we do not free this pointer here
   void* _ordering_str; // we do not free this pointer here
   int* _perm; // permutation of the selected data, might stored here for some algorithms
}rankest,*prankest;

/**
 * @brief   Create a helper data structure for rank estimation.
 * @details Create a helper data structure for rank estimation.
 * @return  Pointer to the created data structure.
 */
void* Nfft4GPRankestStrCreate();

/**
 * @brief   Free the memory of a helper data structure for rank estimation.
 * @details Free the memory of a helper data structure for rank estimation.
 * @param [in,out]   str: helper data structure
 */
void Nfft4GPRankestStrFree(void* str);

/**
 * @brief   Rank estimation of a given kernel. Return the estimated rank.
 * @details Rank estimation of a given kernel. Return the estimated rank. \n
 *          Note that this rank is not the numerical rank of the kernel matrix. This is a near
 *          optimal rank for the NFFT4GP preconditioner.
 * @param [in]       str: helper data structure
 * @param [in]       data: data matrix (ldim by d).
 * @param [in]       n: number of points.
 * @param [in]       ldim: leading dimension of the data matrix
 * @param [in]       d: dimension of data
 * @return           Return the tolerance of distance.
 */
typedef int (*func_rank_esimate)(void *str, double *data, int n, int ldim, int d);

/**
 * @brief   Rank estimation of a given kernel. Return the estimated rank.
 * @details Rank estimation of a given kernel. Return the estimated rank. \n
 *          Note that this rank is not the numerical rank of the kernel matrix. This is a near
 *          optimal rank for the NFFT4GP preconditioner.
 * @param [in]       str: helper data structure
 * @param [in]       data: data matrix (ldim by d).
 * @param [in]       n: number of points.
 * @param [in]       ldim: leading dimension of the data matrix
 * @param [in]       d: dimension of data
 * @return           Return the tolerance of distance.
 */
int Nfft4GPRankestDefault(void *str, double *data, int n, int ldim, int d);

/**
 * @brief   Rank estimation of a given kernel. Return the estimated rank. This version is a comprehensive version using Nystrom approximation.
 * @details Rank estimation of a given kernel. Return the estimated rank. This version is a comprehensive version using Nystrom approximation. \n
 *          Note that this rank is not the numerical rank of the kernel matrix. This is a near
 *          optimal rank for the NFFT4GP preconditioner.
 * @param [in]       str: helper data structure
 * @param [in]       data: data matrix (ldim by d).
 * @param [in]       n: number of points.
 * @param [in]       ldim: leading dimension of the data matrix
 * @param [in]       d: dimension of data
 * @return           Return the tolerance of distance.
 */
int Nfft4GPRankestNysScaled(void *str, double *data, int n, int ldim, int d);

#endif