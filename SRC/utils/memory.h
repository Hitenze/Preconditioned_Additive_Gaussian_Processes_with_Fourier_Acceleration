#ifndef NFFT4GP_MEMORY_H
#define NFFT4GP_MEMORY_H

/**
 * @file memory.h
 * @brief Memory management, does not depend on any other files
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef NFFT4GP_USING_OPENMP
#include "omp.h"
// note: NFFT4GP_DEFAULT_OPENMP_SCHEDULE is also defined in util.h
#ifndef NFFT4GP_DEFAULT_OPENMP_SCHEDULE
#define NFFT4GP_DEFAULT_OPENMP_SCHEDULE schedule(static)
#endif
#endif

/**
 * @brief   Macro to find the minimum of two values.
 * @details Macro to find the minimum of two values.
 */
#define NFFT4GP_MIN(a, b, c) {\
   (c) = (a) <= (b) ? (a) : (b);\
}

/**
 * @brief   Macro to find the maximum of two values.
 * @details Macro to find the maximum of two values.
 */
#define NFFT4GP_MAX(a, b, c) {\
   (c) = (a) >= (b) ? (a) : (b);\
}

/**
 * @brief   Macro to allocate memory.
 * @details Macro to allocate memory using Nfft4GPMalloc.
 */
#define NFFT4GP_MALLOC(ptr, length, ...) {\
   (ptr) = (__VA_ARGS__*) Nfft4GPMalloc((size_t)(length)*sizeof(__VA_ARGS__));\
}

/**
 * @brief   Macro to allocate and initialize memory to zero.
 * @details Macro to allocate and initialize memory to zero using Nfft4GPCalloc.
 */
#define NFFT4GP_CALLOC(ptr, length, ...) {\
   (ptr) = (__VA_ARGS__*) Nfft4GPCalloc((size_t)(length)*sizeof(__VA_ARGS__), 1);\
}

/**
 * @brief   Macro to reallocate memory.
 * @details Macro to reallocate memory using Nfft4GPRealloc.
 */
#define NFFT4GP_REALLOC(ptr, length, ...) {\
   (ptr) = (__VA_ARGS__*) Nfft4GPRealloc((void*)(ptr), (size_t)(length)*sizeof(__VA_ARGS__));\
}

/**
 * @brief   Macro to copy memory.
 * @details Macro to copy memory using Nfft4GPMemcpy.
 */
#define NFFT4GP_MEMCPY(ptr_to, ptr_from, length, ...) {\
   Nfft4GPMemcpy((void*)(ptr_to), (void*)(ptr_from), (size_t)(length)*sizeof(__VA_ARGS__));\
}

/**
 * @brief   Macro to free memory.
 * @details Macro to free memory using Nfft4GPFreeHost and set pointer to NULL.
 */
#define NFFT4GP_FREE(ptr) {\
   if(ptr){Nfft4GPFreeHost((void*)(ptr));}\
   (ptr) = NULL;\
}

/**
 * @brief   Allocate memory.
 * @details Allocate memory of specified size.
 * @param [in]       size           Size of memory to allocate in bytes.
 * @return           Return pointer to allocated memory.
 */
static inline void* Nfft4GPMalloc(size_t size)
{
   void *ptr;
   ptr = malloc(size);
   if(!ptr)
   {
      printf("Error in malloc\n");
      exit(EXIT_FAILURE);
   }
   return ptr;
}

/**
 * @brief   Allocate and initialize memory to zero.
 * @details Allocate and initialize memory to zero.
 * @param [in]       length         Length of memory to allocate.
 * @param [in]       unitsize       Size of each unit in bytes.
 * @return           Return pointer to allocated memory.
 */
static inline void* Nfft4GPCalloc(size_t length, int unitsize)
{
   void *ptr;
   ptr = calloc(length, unitsize);
   if(!ptr)
   {
      printf("Error in calloc\n");
      exit(EXIT_FAILURE);
   }
   return ptr;
}

/**
 * @brief   Reallocate memory.
 * @details Reallocate memory of specified size.
 * @param [in]       ptr            Pointer to memory to reallocate.
 * @param [in]       size           New size of memory in bytes.
 * @return           Return pointer to reallocated memory.
 */
static inline void* Nfft4GPRealloc(void *ptr, size_t size)
{
   void *newptr;
   newptr = realloc(ptr, size);
   if(!newptr)
   {
      printf("Error in realloc\n");
      exit(EXIT_FAILURE);
   }
   return newptr;
}

/**
 * @brief   Copy memory.
 * @details Copy memory from source to destination.
 * @param [out]      ptr_to         Destination pointer.
 * @param [in]       ptr_from       Source pointer.
 * @param [in]       size           Size of memory to copy in bytes.
 * @return           No return.
 */
static inline void Nfft4GPMemcpy(void *ptr_to, void *ptr_from, size_t size)
{
   if(size <= 0)
   {
      return;
   }
   
   if(!ptr_to)
   {
      printf("Error in memcpy: destination is NULL\n");
      exit(EXIT_FAILURE);
   }
   
   if(!ptr_from)
   {
      printf("Error in memcpy: source is NULL\n");
      exit(EXIT_FAILURE);
   }
   
   memcpy(ptr_to, ptr_from, size);
   return;
}

/**
 * @brief   Free memory.
 * @details Free memory allocated with Nfft4GPMalloc, Nfft4GPCalloc, or Nfft4GPRealloc.
 * @param [in]       ptr            Pointer to memory to free.
 * @return           No return.
 */
static inline void Nfft4GPFreeHost(void *ptr)
{
   free(ptr);
   return;
}

#endif
