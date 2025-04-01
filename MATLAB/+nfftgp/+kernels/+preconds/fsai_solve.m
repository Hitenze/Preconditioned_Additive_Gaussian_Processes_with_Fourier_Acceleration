function [y] = fsai_solve(PRE, x, part)
%% [y] = fsai_solve(PRE, x, part)
%  author: Tianshi Xu <xuxx1180@umn.edu>
%  date: 05/20/23
%  brief: solve with the FSAI preconditioner of A, PRE should be obtained by fsai_setup
%
%  input:
%           PRE:     FSAI struct returned by fsai_setup
%           x:       the right-hand-side
%           part:    (optional) if not empty, only solve the part of the solution
%                    Assume the preconditioner is M = L*R
%                    The preconditioned matrix is L\M/R
%                    'L':                 solve the left part L\x
%                    'R':                 solve the right part R\x
%
%  output:
%           y:       the solution of the approximate solve
%
%  example:
%           y = nfftgp.kernels.preconds.fsai_solve(fsai, x);
%           Solve with the full preconditioner
%
%           y = nfftgp.kernels.preconds.fsai_solve(fsai, x, 'L');
%           z = nfftgp.kernels.preconds.fsai_solve(fsai, y, 'R');
%           Solve with the split preconditioner

   if nargin < 3
      part = 'N';
   end

   switch part
      case 'N'
         % the standard FSAI solve
         % we have inv(K) approx PRE.G'*PRE.G
         % thus, y = K\x approx PRE.G'*PRE.G*x

         y = PRE.G' * (PRE.G * x);
      case 'L'
         % solve the left part
         % we have inv(K) approx PRE.G'*PRE.G
         % thus the left par of the preconditioner
         % is PRE.G

         y = PRE.G * x;
      
      case 'R'
         % solve the right part
         % we have inv(K) approx PRE.G'*PRE.G
         % thus the right par of the preconditioner
         % is PRE.G'

         y = PRE.G' * x;
      
      otherwise
         error('Unknown part type');
   end

end