function [val] = nys_trace(PRE)
%% [val] = nys_trace(PRE)
%  author: Tianshi Xu <xuxx1180@umn.edu>
%  date: 09/23/23
%  brief:   Get the trace of M^{-1}dM/dtheta
%
%  input:
%           PRE:     RAN preconditioner returned by ran_setup
%
%  output:
%           val:     the trace value
%
%  example:
%           val = nfftgp.kernels.preconds.ran_trace(ran);
%           Compute the trace of M^{-1}dM/dtheta

   % first compute the sum of diagonal of dM/dtheta
   % this can be computed using

   if(isfield(PRE, 'trace'))
      val = PRE.trace;
      return;
   end

   n = size(PRE.K1, 2);
   k = size(PRE.K1, 1);

   % get some useful fields
   % first of all we need the inverse cholesky factorization of K11
   % no inverse chol detected, compute it
   L11 = chol(PRE.K11+k*eps(norm(PRE.K11))*eye(k), 'upper');
   % next form the L so that NYS = L*L'
   L = PRE.K1' / L11;
   
   % now we have L, we can compute the trace

   num_grads = numel(PRE.dK1);
   val = zeros(num_grads, 1);

   % the first step is to get the diagonal of the preconditioner

   for i = 1:num_grads-1
      % the gradient with respect to l and f
      dL = PRE.dK1{i}' / L11;
      dKL = L * ( L11' \ ( PRE.dK11{i} / L11 ) );
      val(i) = 2*sum(sum(dL .* L))-sum(sum(dKL .* L));
   end

   % and the last term
   val(num_grads) = n*PRE.f2;

   % now the second term
   LP = L / (PRE.eta*eye(k) + L'*L);
   dLP = cell(num_grads, 1);
   for i = 1:num_grads
      dLP{i} = zeros(n,k);
   end

   for j = 1:k
      % we DO NOT want permutation in this dvp
      dLPj = nfftgp.kernels.preconds.ran_dvp(PRE, L(:,j), 1);
      for i = 1:num_grads
         dLP{i}(:,j) = dLPj{i};
      end
   end
   
   for i = 1:num_grads
      val(i) = (val(i) - sum(sum(dLP{i} .* LP))) / PRE.eta;
   end

   PRE.trace = val;

end