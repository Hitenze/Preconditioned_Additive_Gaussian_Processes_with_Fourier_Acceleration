function [val] = afn_trace(PRE)
%% [val] = afn_trace(PRE)
%  author: Tianshi Xu <xuxx1180@umn.edu>
%  date: 09/23/23
%  brief:   Get the trace of M^{-1}dM/dtheta
%
%  input:
%           PRE:     AFN struct returned by afn_setup
%
%   output:
%           val:     the trace value
%
%  example:
%           val = nfftgp.kernels.preconds.afn_trace(afn);
%           Compute the trace of M^{-1}dM/dtheta

   if(isfield(PRE, 'RAN') && ~isempty(PRE.RAN))
      val = nfftgp.kernels.preconds.ran_trace(PRE.RAN);
      return;
   end

   % the diagonal of U is L and inv(G)
   % the diagonal of dU/dtheta is dL'/dtheta and -invG^{-T} dG'/dtheta invG^{-T}
   
   % first get the diagonal of U
   n = length(PRE.perm);
   n1 = size(PRE.L11.L, 1);
   n2 = n - n1;

   diagU = zeros(n, 1);

   num_grads = numel(PRE.K12.dK);

   diagdU = cell(num_grads, 1);
   val = zeros(num_grads, 1);

   diagU(1:n1) = diag(PRE.L11.L);
   diagU(n1+1:end) = 1./diag(PRE.GS.G);

   % then get the diagonal of dU/dtheta
   for i = 1:num_grads
      diagdU{i} = zeros(n, 1);
      diagdU{i}(1:n1) = diag(PRE.L11.dL{i});
      diagdU{i}(n1+1:end) = -1./diag(PRE.GS.G).^2 .* diag(PRE.GS.dG{i});
      val(i) = 2*sum(diagdU{i}./diagU);
   end
end