#ifndef NFFT4GP_TRANSFORM_H
#define NFFT4GP_TRANSFORM_H

/**
 * @file transform.h
 * @brief Transformations for the optimization.
 */

#include "../utils/utils.h"

/**
 * @brief   Enumeration of the possible transformations.
 * @details Enumeration of the possible transformations.
 */
typedef enum {
   NFFT4GP_TRANSFORM_SOFTPLUS = 0,
   NFFT4GP_TRANSFORM_SIGMOID,
   NFFT4GP_TRANSFORM_EXP,
   NFFT4GP_TRANSFORM_IDENTITY
} nfft4gp_transform_type;

/**
 * @brief   Transform a value.
 * @details Transform a value.
 *          The transformation is done according to the nfft4gp_transform_type. Supported transformations are:
 *          - NFFT4GP_TRANSFORM_SOFTPLUS: softplus transformation.
 *          - NFFT4GP_TRANSFORM_SIGMOID: sigmoid transformation.
 *          - NFFT4GP_TRANSFORM_EXP: exponential transformation.
 *          - NFFT4GP_TRANSFORM_IDENTITY: identity transformation.
 * @param [in]       type           Type of transformation, see nfft4gp_transform_type.
 * @param [in]       val            Value to transform.
 * @param [in]       inverse        Transform or inverse transform.
 * @param [out]      tvalp          Pointer to the transformed value.
 * @param [out]      dtvalp         Pointer to the derivative of the transformed value. Only used if inverse is false. \n
 *                                  Returns a NFFT4GP_DOUBLE array of length 1.
 * @return           Return 0 if successful.
 */
int Nfft4GPTransform(
   nfft4gp_transform_type type, 
   NFFT4GP_DOUBLE val, 
   int inverse, 
   NFFT4GP_DOUBLE *tvalp, 
   NFFT4GP_DOUBLE *dtvalp);

#endif