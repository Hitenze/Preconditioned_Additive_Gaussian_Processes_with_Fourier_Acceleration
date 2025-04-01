function [val] = fsai_trace(PRE)
%% [val] = fsai_trace(PRE)
%  author: Tianshi Xu <xuxx1180@umn.edu>
%  date: 09/23/23
%  brief:   Get the trace of M^{-1}dM/dtheta
%
%  input:
%           PRE:     FSAI struct returned by fsai_setup
%
%  output:
%           val:     the trace value
%
%  example:
%           val = nfftgp.kernels.preconds.fsai_trace(fsai);
%           Compute the trace of M^{-1}dM/dtheta

   % the trace of the M^{-1}dM/dtheta
   
   num_grads = numel(PRE.dG);
   val = zeros(num_grads, 1);

   for i = 1:num_grads
      val(i) = 2*sum( diag(PRE.dG{i})./diag(PRE.G));
   end

end