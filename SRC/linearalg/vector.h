#ifndef NFFT4GP_VECTOR_H
#define NFFT4GP_VECTOR_H

/**
 * @file vector.h
 * @brief Array based data structures.
 * @details In order to have a clean interface, we do not open those data structures to the user.
 */

#include "stdio.h"
#include "stdlib.h"
#include "../utils/utils.h"

/*------------------------------------------
 * other datastrs
 *------------------------------------------*/

/**
 * @brief   Simple integer vector.
 * @details Simple integer vector.
 */
typedef struct NFFT4GP_INTVEC_STRUCT
{
   int   _len;
   int   _max_len;
   int   *_data;
} vec_int,*pvec_int;

/**
 * @brief   Create an integer vector.
 * @details Create an integer vector.
 * @return  Pointer to the created vector.
 */
pvec_int Nfft4GPVecIntCreate();

/**
 * @brief   Initialize an integer vector.
 * @details Initialize an integer vector.
 * @param[in,out]   vec     Pointer to the vector.
 * @param[in]       n_max   Maximum length of the vector.
 */
void Nfft4GPVecIntInit(pvec_int vec, int n_max);

/**
 * @brief   Add an integer to the end of the vector.
 * @details Add an integer to the end of the vector.
 * @param[in,out]   vec     Pointer to the vector.
 * @param[in]       val     Value to be added to the vector.
 */
void Nfft4GPVecIntPushback(pvec_int vec, int val);

/**
 * @brief   Free the memory of an integer vector.
 * @details Free the memory of an integer vector.
 * @param[in,out]   vec     Pointer to the vector.
 */
void Nfft4GPVecIntFree(pvec_int vec);

/**
 * @brief   Simple double vector.
 * @details Simple double vector.
 */
typedef struct NFFT4GP_DOUBLEVEC_STRUCT
{
   int _len;
   int _max_len;
   NFFT4GP_DOUBLE *_data;
} vec_double,*pvec_double;

/**
 * @brief   Create a double vector.
 * @details Create a double vector.
 * @return  Pointer to the created vector.
 */
pvec_double Nfft4GPVecDoubleCreate();

/**
 * @brief   Initialize a double vector.
 * @details Initialize a double vector.
 * @param[in,out]   vec     Pointer to the vector.
 * @param[in]       n_max   Maximum length of the vector.
 */
void Nfft4GPVecDoubleInit(pvec_double vec, int n_max);

/**
 * @brief   Add a double value to the end of the vector.
 * @details Add a double value to the end of the vector.
 * @param[in,out]   vec     Pointer to the vector.
 * @param[in]       val     Value to be added to the vector.
 */
void Nfft4GPVecDoublePushback(pvec_double vec, NFFT4GP_DOUBLE val);

/**
 * @brief   Free the memory of a double vector.
 * @details Free the memory of a double vector.
 * @param[in,out]   vec     Pointer to the vector.
 */
void Nfft4GPVecDoubleFree(pvec_double vec);

/**
 * @brief   Data structure of a max-heap.
 * @details Data structure of a max-heap. Containing points from 0 to _max_len - 1. \n
 *          _max_len: length of arrays \n
 *          _len: current length \n
 *          _dist_v: distance array \n
 *          _index_v: index array, index[i] is the point number of dist[i] \n
 *          _rindex_v: reverse index array, rindex[i] is the location of point i in the heap
 * @note    This data structure is used in the FPS algorithm.
 */
typedef struct NFFT4GP_FPS_HEAP_STRUCT
{
   int _max_len;// max length of the heap
   int _len;

   NFFT4GP_DOUBLE *_dist_v; // dist, index, and rindex
   int *_index_v;
   int *_rindex_v;
} heap,*pheap;

/**
 * @brief   Create a max-heap.
 * @details Create a max-heap.
 * @return  Pointer to the created heap.
 */
pheap Nfft4GPHeapCreate();

/**
 * @brief   Initialize a max-heap.
 * @details Initialize a max-heap.
 * @param[in,out]   fpsheap     Pointer to the heap.
 * @param[in]       n           Maximum length of the heap.
 */
void Nfft4GPHeapInit(pheap fpsheap, int n);

/**
 * @brief   Add a (idx, dist) pair to the heap.
 * @details Add a (idx, dist) pair to the heap.
 * @param[in,out]   fpsheap     Pointer to the heap.
 * @param[in]       dist        Distance.
 * @param[in]       idx         Index.
 */
void Nfft4GPHeapAdd(pheap fpsheap, NFFT4GP_DOUBLE dist, int idx);

/**
 * @brief   Get the largest entry from the heap and remove it.
 * @details Get the largest entry from the heap and remove it.
 * @param[in,out]   fpsheap     Pointer to the heap.
 * @param[out]      dist        Distance.
 * @param[out]      idx         Index.
 */
void Nfft4GPHeapPop(pheap fpsheap, NFFT4GP_DOUBLE *dist, int *idx);

/**
 * @brief   Decrease the (idx, dist) pair of the heap if already inside.
 * @details Decrease the (idx, dist) pair of the heap if already inside.
 * @param[in,out]   fpsheap     Pointer to the heap.
 * @param[in]       dist        Distance.
 * @param[in]       idx         Index.
 */
void Nfft4GPHeapDecrease(pheap fpsheap, NFFT4GP_DOUBLE dist, int idx);

/**
 * @brief   Clean the heap.
 * @details Clean the heap.
 * @param[in,out]   fpsheap     Pointer to the heap.
 */
void Nfft4GPHeapClear(pheap fpsheap);

#endif
