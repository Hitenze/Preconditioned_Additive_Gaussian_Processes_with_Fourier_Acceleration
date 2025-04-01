#include "transform.h"


int Nfft4GPTransform(nfft4gp_transform_type type, NFFT4GP_DOUBLE val, int inverse, NFFT4GP_DOUBLE *tvalp, NFFT4GP_DOUBLE *dtvalp)
{
   switch(type)
   {
      case NFFT4GP_TRANSFORM_SIGMOID:
      {
         if(!inverse)
         {
            *tvalp = 1.0 / (exp(-val) + 1.0);
            *dtvalp = *tvalp*(1-*tvalp);
         }
         else
         {
            *tvalp = log(val/(1.0-val));
         }
         break;
      }
      case NFFT4GP_TRANSFORM_SOFTPLUS:
      {
         if(!inverse)
         {
            if(val > 20.0) // this is the default threshold in pytorch
            {
               *tvalp = val;
               *dtvalp = 1.0;
            }
            else if(val < -20.0)
            {
               *tvalp = exp(val);
               *dtvalp = exp(val);
            }
            else
            {
               *tvalp = log(1.0 + exp(val));
               *dtvalp = exp(val)/(1.0 + exp(val));
            }
         }
         else
         {
            if(val > 20.0)
            {
               *tvalp = val;
            }
            else if(val < 2.06115362243856e-09)
            {
               *tvalp = log(val);
            }
            else
            {
               *tvalp = log(exp(val) - 1.0);
            }
         }
         break;
      }
      case NFFT4GP_TRANSFORM_EXP:
      {
         if(!inverse)
         {
            *tvalp = exp(val);
            *dtvalp = exp(val);
         }
         else
         {
            *tvalp = log(val);
         }
         break;
      case NFFT4GP_TRANSFORM_IDENTITY:
         if(!inverse)
         {
            *tvalp = val;
            *dtvalp = 1.0;
         }
         else
         {
            *tvalp = val;
         }
         break;
      }
      default:
      {
         printf("Error: unknown transform type.\n");
         return -1;
      }
   }

   return 0;
}