function [y] = afn_dvp(PRE, x)
%% [y] = afn_dvp(PRE, x)
%  author: Tianshi Xu <xuxx1180@umn.edu>
%  date: 09/23/23
%  brief:   Apply the derivative vector multiplication dM/dtheta * x
%
%  input:
%           PRE:     AFN struct returned by afn_setup
%           x:       the vector to be multiplied
%
%   output:
%           y:       the result of the multiplication
%
%  example:
%           y = nfftgp.kernels.preconds.afn_dvp(afn, x);
%           Compute the DVP

   if(isfield(PRE, 'RAN') && ~isempty(PRE.RAN))
      y = nfftgp.kernels.preconds.ran_dvp(PRE.RAN, x);
      return;
   end

   n = length(PRE.perm);
   permed_x = zeros(n,1);

   num_grads = numel(PRE.K12.dK);

   for i = 1:n
      permed_x(i) = x(PRE.perm(i));
   end
   k = size(PRE.L11.L,1);

   if k == n
      permed_y = nfftgp.kernels.preconds.chol_solve(PRE.L11, permed_x);
   else
      xu = permed_x(1:k, :);
      xl = permed_x(k+1:end, :);

      % we need to multiply dM/dtheta
      % which is dU'/dtheta * U + U' * dU/dtheta

      % first U * x
      z1u = PRE.L11.L' * xu + PRE.L11.L \ ( PRE.K12.K * xl);
      z1l = PRE.GS.G' \ xl;

      % then apply first dU'/dtheta * U * x
      y1u = cell(num_grads, 1);
      y1l = cell(num_grads, 1);

      for i = 1:num_grads
         y1u{i} = PRE.L11.dL{i} * z1u;
         y1l_i = PRE.L11.L' \ z1u;
         y1l{i} = (PRE.K12.dK{i}' * y1l_i)+...
                  - PRE.K12.K' * ( PRE.L11.L' \ ( PRE.L11.dL{i}' * y1l_i ) )+...
                  - PRE.GS.G \ (PRE.GS.dG{i} * ( PRE.GS.G \ z1l) );
      end

      % then the second part, apply dU/dtheta first 
      y2u = cell(num_grads, 1);
      y2l = cell(num_grads, 1);
      for i = 1:num_grads
         z2l = - PRE.GS.G' \ ( PRE.GS.dG{i}' * (PRE.GS.G' \ xl) );
         y2u_i = PRE.K12.dK{i} * xl-...
                  PRE.L11.dL{i} * ( PRE.L11.L \ (PRE.K12.K * xl) );
         z2u = (PRE.L11.dL{i}' * xu) + PRE.L11.L \ y2u_i;
         y2u{i} = PRE.L11.L * z2u;
         y2l{i} = PRE.K12.K' * (PRE.L11.L' \ z2u) + PRE.GS.G \ z2l;
      end

      permed_y = cell(num_grads, 1);

      for i = 1:num_grads
         permed_y{i} = [y1u{i}; y1l{i}] + [y2u{i}; y2l{i}];
      end
   end

   y = cell(num_grads, 1);
   for i = 1:num_grads
      y{i} = zeros(n,1);
   end

   for i = 1:n
      for j = 1:num_grads
         y{j}(PRE.perm(i)) = permed_y{j}(i);
      end
   end
end