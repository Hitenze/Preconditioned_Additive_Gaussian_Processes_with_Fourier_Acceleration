function [sol,T,its] = cg(A,n,rhs,sol,maxits)
%% [sol,T,its] = cg(A,n,rhs,sol,maxits)
%  note:    This version is modified from fgmres.m in the
%           MATLAB Suite https://www-users.cse.umn.edu/~saad/software/MATLAB_DEMOS.tar.gz
%  author:  Tianshi Xu <xuxx1180@umn.edu>
%  date:    05/30/23
%  brief:   CG with eigenvalue estimation. We remove the convergence check since 
%           we are going to apply a fixed number of iterations.
%
%  input:
%           A:          Matrix or function handle that could compute matvec.
%           n:          Size of the problem.
%           rhs:        Right-hand-side.
%           sol:        Initial guess.
%           maxits:     Maximum number of iterations.
%
%  output:
%           sol:        Solution.
%           T:          Tridiagonal matrix extracted from the Arnoldi process.
%                       Only meaningful if maxits is symmetric.
%           its:        Number of iterations. Although we do not have convergence check,
%                       the Arnoldi might have to stop earlier.
%
%  example:
%           [sol,T,its] = nfftgp.kernels.krylovs.cg(A,n,rhs,sol,maxits)

   its = 0;

   % create space for Hessenberg matrix and Krylov basis
   maxits = min(maxits,n);
   alpha = zeros(maxits,1);
   beta = zeros(maxits-1,1);

   if isa(A,'function_handle') == 0
      r = rhs - A*sol;
   else
      r = rhs - A(sol);
   end
   %%
   r0 = norm(r,2);
   if ((r0 < eps) || (its >= maxits))
      return
   end
   
   p = r;
   
   %-------------------- inner gmres loop
   while (true)
      its = its + 1;
      %%--------------------       modified GS  ;
      if isa(A,'function_handle') == 0
         Ap = A*p;
      else
         Ap = A(p);
      end
      alpha(its) = r0^2/(p'*Ap);
      sol = sol + alpha(its)*p;
      r = r - alpha(its)*Ap;

      r1 = norm(r,2);
      if (r1 < eps || its >= maxits)
         % terminate
         break;
      end
      
      beta(its) = (r1/r0)^2;
      p = r + beta(its)*p;
      
      r0 = r1;
      
   end  %% end of while its loop
   %
   %       now compute Lanczos estimation  ;
   %
   if nargout > 1
      T = zeros(its,its);
      T(1,1) = 1/alpha(1);
      for i = 2:its
         T(i,i) = 1/alpha(i) + beta(i-1)/alpha(i-1);
         T(i,i-1) = sqrt(beta(i-1))/alpha(i-1);
         T(i-1,i) = T(i,i-1);
      end
   end

end
   