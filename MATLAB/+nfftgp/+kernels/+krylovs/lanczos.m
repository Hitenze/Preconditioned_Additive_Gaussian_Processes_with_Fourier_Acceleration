function [sol,T,its] = lanczos(A,n,rhs,sol,maxits,tol,atol,wsize)
%% [sol,T,its] = lanczos(A,n,rhs,sol,maxits,tol,atol,wsize)
%  note:    This version is modified from fgmres.m in the
%           MATLAB Suite https://www-users.cse.umn.edu/~saad/software/MATLAB_DEMOS.tar.gz
%  author:  Tianshi Xu <xuxx1180@umn.edu>
%  date:    09/11/23
%  brief:   Full Orthogonalization Lanzos Method.
%  update:  1. 11/16/23: add restart for accurate eigenvalues computation for special cases
%
%  input:
%           A:          Matrix or function handle that could compute matvec.
%           n:          Size of the problem.
%           rhs:        Right-hand-side.
%           sol:        Initial guess.
%           maxits:     Maximum number of iterations. Default 200.
%           tol:        Tolerance for convergence check based on (relative) residual norm. Default 0.0;
%           atol:       Do we use absolute residual norm?
%           wsize:      Orth window size
%
%  output:
%           sol:        Solution.
%           T:          Tridiagonal matrix extracted from the Arnoldi process.
%                       Only meaningful if maxits is symmetric.
%           its:        Number of iterations. Although we do not have convergence check,
%                       the Arnoldi might have to stop earlier.
%
%  example:
%           [sol,T,its] = nfftgp.kernels.krylovs.lanczos(A,n,rhs,sol,maxits,tol)
%           Lanczos with full orthgonization
%
%           [sol,T,its] = nfftgp.kernels.krylovs.lanczos(A,n,rhs,sol,maxits,tol,atol,2)
%           Lanczos with only two vector orthgonization, this is same as CG

   if(nargin < 5)
      maxits = 200;
   end

   maxits = min(n, maxits);

   if(nargin < 6)
      tol = 0.0;
   end

   if(nargin < 7)
      atol = 1;
   end

   if(nargin < 8)
      wsize = n;
   end
   if(wsize < 1)
      wsize = 1;
   end
   wsize = wsize - 1;

   i = 0;
   its = 0;
   lock = 1;

   % create space for Hessenberg matrix and Krylov basis
   vv = zeros(n,maxits+1);
   hh = zeros(maxits+1,maxits);

   if isa(A,'function_handle') == 0
      vv(1:n,1) = rhs - A*sol;
   else
      vv(1:n,1) = rhs - A(sol);
   end
   %%
   beta = norm(vv(1:n,1),2);

   if ((beta < eps) || (i >= maxits))
      return
   end

   if atol
      % absolute tol
      tol = tol / beta;
   end

   if(tol ~= 0.0)
      % maintain a Cholesky factorization of H since our matrices are SPD
      Ld = zeros(maxits,1);
      Ll = zeros(maxits,1);
   end

   t = 1.0 / beta;
   vv(1:n,1) = vv(1:n,1) * t;
   
   % actually is not needed, remove later
   i = 0;
   %-------------------- inner gmres loop
   while (i < maxits)
      i=i+1;
      i0 = i - 1;
      i1 = i + 1;
      %%--------------------       modified GS  ;
      if isa(A,'function_handle') == 0
         vv(1:n,i1) = A*vv(1:n,i);
      else
         vv(1:n,i1) = A(vv(1:n,i));
      end
      ro = norm(vv(1:n,i1),2);
      for j=max(1,i-wsize):i
         t = vv(1:n,j)'*vv(1:n,i1);
         hh(j,i) = t;
         vv(1:n,i1) = vv(1:n,i1) - t*vv(1:n,j);
      end
      t = norm(vv(1:n,i1),2);
      while(t >= eps && t < ro/sqrt(2))
         ro = t;
         
         for j=max(1,i-wsize):i
            t = vv(1:n,j)'*vv(1:n,i1);
            hh(j,i) = hh(j,i) + t;
            vv(1:n,i1) = vv(1:n,i1) - t*vv(1:n,j);
         end
         t = norm(vv(1:n,i1),2);
      end
      hh(i1,i) = t;
      
      if (hh(i1,i) < eps)
         % too small, early termination
         break;
      end
      
      t = 1.0 / t;
      vv(1:n,i1) = vv(1:n,i1)*t;

      if(tol ~= 0)
         if(i ~= 1)
            Ll(i0) = hh(i,i0) / Ld(i0);
            Ld(i) = sqrt(hh(i,i) - Ll(i0)^2);
            le = 1.0 / Ld(i);
            ls = - ls * Ll(i0) * le;
            res = abs(le * ls) * hh(i1,i) * beta;
         else
            Ld(i) = sqrt(hh(i,i));
            ls = 1.0 / Ld(i);
            res = hh(i1,i) / hh(i,i) * beta;
         end
         
         if(res < tol)
            % reach tol
            break;
         end
      end

   end  %% end of while i loop
   %
   %       now compute solution. first solve upper triangular system.  ;
   %
   its = i;
   Hm = hh(1:i,1:i);
   if(tol ~= 0.0)
      L = spdiags([Ll,Ld],-1:0,i,i);
      y = L' \ ( L \ (beta*eye(i,1)));
   else
      y = Hm \ (beta*eye(i,1));
   end
   sol = sol + vv(:,1:i)*y;
   
   while(i < maxits)
      % in this case, we need to restart from a random vector to rebuild a new Krylov space
      if (hh(i1,i) < eps)
         % if hh(i1,i) is not small we can proceed from it
         lock = i;
         
         vv(1:n,i1) = rand(n,1);
         for j=max(1,i-wsize):lock
            t = vv(1:n,j)'*vv(1:n,i1);
            vv(1:n,i1) = vv(1:n,i1) - t*vv(1:n,j);
         end
         t = norm(vv(1:n,i1),2);
         while(t >= eps && t < ro/sqrt(2))
            ro = t;
            for j=max(1,i-wsize):lock
               t = vv(1:n,j)'*vv(1:n,i1);
               vv(1:n,i1) = vv(1:n,i1) - t*vv(1:n,j);
            end
            t = norm(vv(1:n,i1),2);
         end
         ro = t;

         if (ro < eps)
            % restart fail, quit
            break
         end
         
         vv(1:n,i1) = vv(1:n,i1) / ro;
      end

      % enter the inner loop
      while (i < maxits)
         i=i+1;
         i1 = i + 1;
         %%--------------------       modified GS  ;
         if isa(A,'function_handle') == 0
            vv(1:n,i1) = A*vv(1:n,i);
         else
            vv(1:n,i1) = A(vv(1:n,i));
         end
         ro = norm(vv(1:n,i1),2);
         for j=max(1,i-wsize):lock
            t = vv(1:n,j)'*vv(1:n,i1);
            vv(1:n,i1) = vv(1:n,i1) - t*vv(1:n,j);
         end
         for j=max(lock+1,i-wsize):i
            t = vv(1:n,j)'*vv(1:n,i1);
            hh(j,i) = t;
            vv(1:n,i1) = vv(1:n,i1) - t*vv(1:n,j);
         end
         t = norm(vv(1:n,i1),2);
         while(t >= eps && t < ro/sqrt(2))
            ro = t;
            for j=max(1,i-wsize):lock-1
               t = vv(1:n,j)'*vv(1:n,i1);
               vv(1:n,i1) = vv(1:n,i1) - t*vv(1:n,j);
            end
            for j=max(lock+1,i-wsize):i
               t = vv(1:n,j)'*vv(1:n,i1);
               hh(j,i) = hh(j,i) + t;
               vv(1:n,i1) = vv(1:n,i1) - t*vv(1:n,j);
            end
            t = norm(vv(1:n,i1),2);
         end
         hh(i1,i) = t;
         
         if (hh(i1,i) < eps)
            % too small, early termination
            break;
         end
         
         t = 1.0 / t;
         vv(1:n,i1) = vv(1:n,i1)*t;

      end  %% end of while i loop

   end

   Hm = hh(1:i,1:i);
   T = tril(Hm,1);
   T = (T+T')/2;

end