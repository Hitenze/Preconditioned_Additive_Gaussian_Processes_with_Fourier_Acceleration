function [tval, dval] = transform(val, type, inverse)
%% [tval, dval] = transform(val, type, inverse)
%  author: Tianshi Xu <xuxx1180@umn.edu>
%  date: 11/08/23
%  brief:   1. Transform a val, return both value and gradient.
%           2. Inverse transform a val, return the original value.
%              Note that in this case the gradient will be empty.
%
%  input:
%           val:        value
%           type:       transform type.
%                       "sigmoid": sigmoid function
%                       "softplus": softplus function
%                       "exp": exp function
%                       "identity": identity function
%           inverse:    Optional. Default value is false.
%                       If true, apply the inverse transform.
%
%  output:
%           tval:       output value after transform/inverse transform
%           dval:       gradient/empty

if(nargin < 3)
   inverse = false;
end

   switch type
      case "sigmoid"
         [tval, dval] = sigmoid(val, inverse);
      case "softplus"
         [tval, dval] = softplus(val, inverse);
      case "exp"
         [tval, dval] = myexp(val, inverse);
      case "identity"
         [tval, dval] = identity(val, inverse);
   end
end

function [tval, dval] = sigmoid(val, inverse)
%% [tval, dval] = sigmoid(val, inverse)

   if(~inverse)
      tval = 1 / (exp(-val) + 1);
      dval = tval*(1-tval);
   else
      tval = log(val/(1-val));
      dval = [];
   end

end

function [tval, dval] = softplus(val, inverse)
%% [tval, dval] = softplus(val, inverse)

   if(~inverse)
      if(val > 20) % this is the default threshold in pytorch
         tval = val;
         dval = 1;
      elseif(val < -20)
         tval = exp(val);
         dval = exp(val);
      else
         tval = log(1 + exp(val));
         dval = exp(val)/(1 + exp(val));
      end
   else
      if(val > 20)
         tval = val;
      elseif(val < exp(-20))
         tval = log(val);
      else
         tval = log(exp(val) - 1);
      end
      dval = [];
   end

end

function [tval, dval] = myexp(val, inverse)
%% [tval, dval] = myexp(val, inverse)

   if(~inverse)
      tval = exp(val);
      dval = exp(val);
   else
      tval = log(val);
      dval = [];
   end

end

function [tval, dval] = identity(val, inverse)
%% [tval, dval] = identity(val, inverse)

   if(~inverse)
      tval = val;
      dval = 1;
   else
      tval = val;
      dval = [];
   end

end