function [y] = chol_solve(PRE, x, part)
%% [y] = chol_solve(PRE, x, part)
%  author: Tianshi Xu <xuxx1180@umn.edu>
%  date: 05/20/23
%  brief: solve with the Cholesky factorization of A, PRE should be obtained by chol_setup
%
%  input:
%           PRE:     CHOL preconditioner returned by chol_setup
%           x:       the right-hand-side
%           part:    (optional) if not empty, only solve the part of the solution
%                    Assume the preconditioner in split format is applied as L*A*R*x
%                    For chol this is inv(L)*A*inv(L')*x
%                    'L':                 solve the left part L*x
%                    'R':                 solve the right part R*x
%
%  output:
%           y:       the solution of the approximate solve
%
%  example:
%           y = nfftgp.kernels.preconds.chol_solve(chol, x);
%           Solve with the full preconditioner
%
%           y = nfftgp.kernels.preconds.chol_solve(chol, x, 'L');
%           z = nfftgp.kernels.preconds.chol_solve(chol, y, 'R');
%           Solve with the split preconditioner

   if nargin < 3
      part = 'N';
   end

   switch part
      case 'N'
         % we have K = L*L', so y = K\x = L'\(L\x)
         y = PRE.L'\(PRE.L\x);
      case 'L'
         % the left part of the preconditioner inv(L)
         y = PRE.L \ x;
      case 'R'
         % the right part of the preconditioner inv(L')
         y = PRE.L' \ x;
      otherwise
         error('Unknown part type');
   end
end