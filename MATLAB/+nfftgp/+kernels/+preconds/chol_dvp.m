function [y] = chol_dvp(PRE, x)
%% [y] = chol_dvp(PRE, x)
%  author: Tianshi Xu <xuxx1180@umn.edu>
%  date: 09/23/23
%  brief:   Apply the derivative vector multiplication dM/dtheta * x where M is the preconditioner
%
%  input:
%           PRE:     CHOL preconditioner returned by chol_setup
%           x:       the vector to be multiplied
%
%  output:
%           y:       the result of the multiplication
%
%  example:
%           y = nfftgp.kernels.preconds.chol_dvp(chol, x);
%           Compute the DVP

   % we know that the chol factorization is exact, and thus we can directly use
   % the derivative of the kernel matrix to do the matvec

   num_grads = numel(PRE.dK);

   y = cell(num_grads, 1);
   for i = 1:num_grads
      y{i} = PRE.dK{i}*x;
   end

end