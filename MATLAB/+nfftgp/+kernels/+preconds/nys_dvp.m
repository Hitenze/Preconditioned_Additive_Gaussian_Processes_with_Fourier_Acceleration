function [y] = nys_dvp(PRE, x, nonperm)
%% [y] = nys_dvp(PRE, x, nonperm)
%  author: Tianshi Xu <xuxx1180@umn.edu>
%  date: 09/23/23
%  brief:   Apply the multiplication dM/dtheta * x
%
%  input:
%           PRE:     RAN preconditioner returned by ran_setup
%           x:       the right-hand-side
%           nonperm: do we apply permutation (optional)
%
%  output:
%           y:       the solution of the multiplication
%
%  example:
%           y = nfftgp.kernels.preconds.ran_dvp(ran, x);
%           Compute the DVP

   % RAN add a small shift to compute the Nystrom approximation
   % we approximate it by ignoring the small shift
   % the shift is very small, bias here but is negligible

   if nargin < 3
      nonperm = false;
   end

   num_grads = numel(PRE.dK11);
   
   n = length(x);
   permed_x = zeros(n,1);
         
   perm = PRE.perm;

   if(nonperm)
      permed_x = x;
   else
      for i = 1:n
         permed_x(i) = x(perm(i));
      end
   end
   
   K1x = PRE.K1*permed_x;
   K11K1x = PRE.K11\K1x;

   y = cell(num_grads, 1);
   % TODO: better if we store Chol factorization for this
   % MATLAB will handle this
   for i = 1:num_grads-1
      permed_yi = PRE.dK1{i}'*K11K1x;
      permed_yi = permed_yi - PRE.K1'*(PRE.K11\(PRE.dK11{i}*K11K1x));
      permed_yi = permed_yi + PRE.K1'*(PRE.K11\(PRE.dK1{i}*permed_x));
      
      yi = zeros(n,1);
      
      if(nonperm)
         yi = permed_yi;
      else
         for j = 1:n
            yi(perm(j)) = permed_yi(j);
         end
      end
      y{i} = yi;

   end
   % the last element is the noise level, thus dKs are zero, cannot solve with it!
   y{3} = PRE.f2*x;

end