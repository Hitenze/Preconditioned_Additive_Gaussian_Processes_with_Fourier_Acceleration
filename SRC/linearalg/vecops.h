#ifndef NFFT4GP_VECOPS_H
#define NFFT4GP_VECOPS_H

/**
 * @file vecops.h
 * @brief Vector operations
 */

#include "../utils/utils.h"
#include "../utils/protos.h"
#include "vector.h"

/**
 * @brief   Compute the 2-norm of a vector.
 * @details Compute the 2-norm of a vector.
 * @param [in]       x              Pointer to the vector.
 * @param [in]       n              Length of the vector.
 * @return           Return the 2-norm of the vector.
 */
NFFT4GP_DOUBLE Nfft4GPVecNorm2(
   NFFT4GP_DOUBLE *x, 
   int n);

/**
 * @brief   Compute the dot product of two vectors.
 * @details Compute the dot product of two vectors.
 * @param [in]       x              Pointer to the first vector.
 * @param [in]       n              Length of the vectors.
 * @param [in]       y              Pointer to the second vector.
 * @return           Return the dot product of the two vectors.
 */
NFFT4GP_DOUBLE Nfft4GPVecDdot(
   NFFT4GP_DOUBLE *x, 
   int n, 
   NFFT4GP_DOUBLE *y);

/**
 * @brief   Compute the AXPY operation: y = alpha*x + y.
 * @details Compute the AXPY operation: y = alpha*x + y.
 * @param [in]       alpha          Scaling factor for the first vector.
 * @param [in]       x              Pointer to the first vector.
 * @param [in]       n              Length of the vectors.
 * @param [in,out]   y              Pointer to the second vector.
 * @return           No return.
 */
void Nfft4GPVecAxpy(
   NFFT4GP_DOUBLE alpha, 
   NFFT4GP_DOUBLE *x, 
   size_t n, 
   NFFT4GP_DOUBLE *y);

/**
 * @brief   Fill a vector with random values between 0 and 1.
 * @details Fill a vector with random values between 0 and 1.
 * @param [out]      x              Pointer to the vector.
 * @param [in]       n              Length of the vector.
 * @return           No return.
 */
void Nfft4GPVecRand(
   NFFT4GP_DOUBLE *x, 
   int n);

/**
 * @brief   Fill a vector with random values from the Rademacher distribution.
 * @details Fill a vector with random values from the Rademacher distribution (random ±1 values).
 * @param [out]      x              Pointer to the vector.
 * @param [in]       n              Length of the vector.
 * @return           No return.
 */
void Nfft4GPVecRadamacher(
   NFFT4GP_DOUBLE *x, 
   int n);

/**
 * @brief   Fill a vector with a constant value.
 * @details Fill a vector with a constant value.
 * @param [out]      x              Pointer to the vector.
 * @param [in]       n              Length of the vector.
 * @param [in]       val            Value to fill the vector with.
 * @return           No return.
 */
void Nfft4GPVecFill(
   NFFT4GP_DOUBLE *x, 
   size_t n, 
   NFFT4GP_DOUBLE val);

/**
 * @brief   Scale a vector by a constant value.
 * @details Scale a vector by a constant value.
 * @param [in,out]   x              Pointer to the vector.
 * @param [in]       n              Length of the vector.
 * @param [in]       scale          Scaling factor.
 * @return           No return.
 */
void Nfft4GPVecScale(
   NFFT4GP_DOUBLE *x, 
   size_t n, 
   NFFT4GP_DOUBLE scale);

/**
 * @brief   Fill an integer vector with a constant value.
 * @details Fill an integer vector with a constant value.
 * @param [out]      x              Pointer to the vector.
 * @param [in]       n              Length of the vector.
 * @param [in]       val            Value to fill the vector with.
 * @return           No return.
 */
void Nfft4GPIVecFill(
   int *x, 
   int n, 
   int val);

#endif
