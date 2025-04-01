function [y] = afn_solve(PRE, x, part)
%% [y] = afn_solve(PRE,x)
%  author: Tianshi Xu <xuxx1180@umn.edu>
%  date: 05/20/23
%  brief:   solve with the AFN preconditioner of A, PRE should be obtained by afn_setup
%
%  input:
%           PRE:     AFN struct returned by afn_setup
%           x:       the right-hand-side
%           part:    (optional) if not empty, only solve the part of the solution
%                    Assume the preconditioner is M = L*R
%                    'L':                 solve the left part L\x
%                    'R':                 solve the right part R\x
%
%  output:
%           y:       the solution of the approximate solve
%
%  example:
%           y = nfftgp.kernels.preconds.afn_solve(afn, x);
%           Solve with the full preconditioner
%
%           y = nfftgp.kernels.preconds.afn_solve(afn, x, 'L');
%           z = nfftgp.kernels.preconds.afn_solve(afn, y, 'R');
%           Solve with the split preconditioner

   if nargin < 3
      part = 'N';
   end

   switch part
      
      case 'N'

         if(isfield(PRE, 'RAN') && ~isempty(PRE.RAN))
            y = nfftgp.kernels.preconds.ran_solve(PRE.RAN, x);
            return;
         end

         n = length(PRE.perm);
         permed_x = zeros(n,1);

         for i = 1:n
            permed_x(i) = x(PRE.perm(i));
         end
         k = size(PRE.L11.L,1);

         if k == n
            permed_y = nfftgp.kernels.preconds.chol_solve(PRE.L11, permed_x);
         else
            xu = permed_x(1:k, :);
            xl = permed_x(k+1:end, :);
            
            % \hat g = g - EB^{-1}f
            zl = xl - PRE.K12.K'*(nfftgp.kernels.preconds.chol_solve(PRE.L11, xu));
            
            % v = S^{-1} \hat g
            yl = nfftgp.kernels.preconds.fsai_solve(PRE.GS, zl);
            
            % u = B^{-1}(f - Fv)
            
            yu = nfftgp.kernels.preconds.chol_solve(PRE.L11, xu - PRE.K12.K*yl);
            permed_y = [yu;yl];
         end

         y = zeros(n,1);

         for i = 1:n
            y(PRE.perm(i)) = permed_y(i);
         end
      case 'L'
         % solve with the left part
   
         if(isfield(PRE, 'RAN') && ~isempty(PRE.RAN))
            error('RAN preconditioner is not supported for left solve');
         end

         n = length(PRE.perm);
         permed_x = zeros(n,1);

         for i = 1:n
            permed_x(i) = x(PRE.perm(i));
         end
         k = size(PRE.L11.L,1);

         if k == n
            permed_y = nfftgp.kernels.preconds.chol_solve(PRE.L11, permed_x, 'L');
         else
            xu = permed_x(1:k, :);
            xl = permed_x(k+1:end, :);
            
            yu = nfftgp.kernels.preconds.chol_solve(PRE.L11, xu, 'L');
            zl = xl - PRE.K12.K'*nfftgp.kernels.preconds.chol_solve(PRE.L11, yu, 'R');
            yl = nfftgp.kernels.preconds.fsai_solve(PRE.GS, zl, 'L');
            
            permed_y = [yu;yl];
         end

         y = zeros(n,1);

         for i = 1:n
            y(PRE.perm(i)) = permed_y(i);
         end

      case 'R'
         % solve with the right part

         if(isfield(PRE, 'RAN') && ~isempty(PRE.RAN))
            error('RAN preconditioner is not supported for right solve');
         end

         n = length(PRE.perm);
         permed_x = zeros(n,1);
         
         for i = 1:n
            permed_x(i) = x(PRE.perm(i));
         end
         k = size(PRE.L11.L,1);

         if k == n
            permed_y = nfftgp.kernels.preconds.chol_solve(PRE.L11, permed_x, 'R');
         else
            xu = permed_x(1:k, :);
            xl = permed_x(k+1:end, :);
            
            yl = nfftgp.kernels.preconds.fsai_solve(PRE.GS, xl, 'R');
            zu = xu - nfftgp.kernels.preconds.chol_solve(PRE.L11, PRE.K12.K*yl, 'L');
            yu = nfftgp.kernels.preconds.chol_solve(PRE.L11, zu, 'R');
            
            permed_y = [yu;yl];
         end

         y = zeros(n,1);

         for i = 1:n
            y(PRE.perm(i)) = permed_y(i);
         end

      otherwise
         error('Unknown part type');
   end
end