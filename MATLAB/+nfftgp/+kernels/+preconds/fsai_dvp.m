function [y] = fsai_dvp(PRE, x)
%% [y] = fsai_dvp(PRE, x)
%  author: Tianshi Xu <xuxx1180@umn.edu>
%  date: 09/23/23
%  brief:   Apply the derivative vector multiplication dM/dtheta * x
%
%  input:
%           PRE:     FSAI struct returned by fsai_setup
%           x:       the vector to be multiplied
%
%   output:
%           y:       the result of the multiplication
%
%  example:
%           y = nfftgp.kernels.preconds.fsai_dvp(fsai, x);
%           Compute the DVP

   % we know that the chol factorization is exact, and thus we can directly use
   % the derivative of the kernel matrix to do the matvec

   num_grads = numel(PRE.dG);

   y = cell(num_grads, 1);
   for i = 1:num_grads
      zi = PRE.G'\x;
      z1i = PRE.dG{i}*( PRE.G \ zi );
      z2i = z1i + PRE.G'\( PRE.dG{i}'*zi );
      y{i} = PRE.G\z2i;
   end

end