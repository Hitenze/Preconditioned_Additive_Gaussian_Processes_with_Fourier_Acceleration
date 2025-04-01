function [val] = chol_trace(PRE)
%% [val] = chol_trace(PRE)
%  author: Tianshi Xu <xuxx1180@umn.edu>
%  date: 09/23/23
%  brief:   Get the trace of M^{-1}dM/dtheta
%
%  input:
%           PRE:     CHOL preconditioner returned by chol_setup.m
%
%  output:
%           val:     the trace value
%
%  example:
%           val = nfftgp.kernels.preconds.chol_trace(chol);
%           Compute the trace of M^{-1}dM/dtheta

   % the trace of the M^{-1}dM/dtheta
   
   num_grads = numel(PRE.dL);
   val = zeros(num_grads, 1);

   for i = 1:num_grads
      val(i) = 2*sum( diag(PRE.dL{i})./diag(PRE.L));
   end

end