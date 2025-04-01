function [y] = ran_solve(PRE, x, part)
%% [y] = ran_solve(PRE, x, part)
%  author: Tianshi Xu <xuxx1180@umn.edu>
%  date: 05/20/23
%  brief:   solve with the RAN preconditioner of A, PRE should be obtained by ran_setup
%
%  input:
%           PRE:     RAN preconditioner returned by ran_setup
%           x:       the right-hand-side
%           part:    (optional) if not empty, only solve the part of the solution
%                    TODO: current version does not support part
%
%  output:
%           y:       the solution of the approximate solve
%
%  example:
%           y = nfftgp.kernels.preconds.ran_solve(ran, x);
%           Solve with the full preconditioner

   if nargin < 3
      part = 'N';
   end

   switch part
      case 'N'
         perm = PRE.perm;
         n = length(perm);
         eta = PRE.eta;
         U = PRE.U;
         M = PRE.M;
         
         permed_b = zeros(n,1);
         
         for i = 1:n
            permed_b(i) = x(perm(i));
         end
         
         %permed_y = eta*(U*(M.*(U'*permed_b))) + permed_b - U*(U'*permed_b);
         permed_y = (U*(M.*(U'*permed_b))) + (permed_b - U*(U'*permed_b))/eta;

         y = zeros(n,1);
         
         for i = 1:n
            y(perm(i)) = permed_y(i);
         end
      case 'L'
         error('L solve is not supported yet');
      case 'U'
         error('U solve is not supported yet');
      otherwise
         error('Unknown part type');
   end
   
end
   
   