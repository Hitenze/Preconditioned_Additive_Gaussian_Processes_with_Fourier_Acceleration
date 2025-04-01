function [sol,T,its] = planczos(A,n,prefunc,rhs,sol,maxits,tol,atol,wsize)
%% [sol,T,its] = planczos(A,n,prefunc,rhs,sol,maxits,tol,atol,wsize)
%  note:    This version is modified from fgmres.m in the
%           MATLAB Suite https://www-users.cse.umn.edu/~saad/software/MATLAB_DEMOS.tar.gz
%  author:  Tianshi Xu <xuxx1180@umn.edu>
%  date:    10/18/23
%  brief:   Preconditioned Full Orthogonalization Lanczos Method.
%  update:  1. 11/16/23: add restart for accurate eigenvalues computation for special cases
%
%  input:
%           A:          Matrix or function handle that could compute matvec.
%           n:          Size of the problem.
%           prefunc:    Preconditioner. If empty, no preconditioning is used.
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
%           [sol,T,its] = nfftgp.kernels.krylovs.planczos(A,n,prefunc,rhs,sol,maxits,tol,atol,wsize)

   if(nargin < 6)
      maxits = 200;
   end

   maxits = min(n, maxits);

   if(nargin < 7)
      tol = 0.0;
   end

   if(nargin < 8)
      atol = 1;
   end

   if(nargin < 9)
      wsize = n;
   end
   if(wsize < 1)
      wsize = 1;
   end
   wsize = wsize - 1;

   if(isempty(prefunc))
      [sol,T,its] = nfftgp.kernels.krylovs.lanczos(A,n,rhs,sol,maxits,tol,atol);
      return;
   end
   
   i = 0;
   its = 0;
   lock = 1;

   % create space for Hessenberg matrix, Krylov basis, and the preconditioned basis
   vv = zeros(n,maxits+1);
   zz = zeros(n,maxits+1);
   hh = zeros(maxits+1,maxits);

   if isa(A,'function_handle') == 0
      zz(1:n,1) = rhs - A*sol;
   else
      zz(1:n,1) = rhs - A(sol);
   end
   
   if isa(prefunc,'function_handle') == 0
      vv(1:n,1) = prefunc*zz(1:n,1);
   else
      vv(1:n,1) = prefunc(zz(1:n,1));
   end

   %% we check convergence using the original residual
   %beta = norm(vv(1:n,1),2);
   beta = sqrt(vv(1:n,1)'*zz(1:n,1));
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
   zz(1:n,1) = zz(1:n,1) * t;

   % actually is not needed, remove later
   i = 0;
   %-------------------- inner gmres loop
   while (i < maxits)
      i=i+1;
      i0 = i - 1;
      i1 = i + 1;
      % A matvec
      if isa(A,'function_handle') == 0
         z = A*vv(1:n,i);
      else
         z = A(vv(1:n,i));
      end

      % orthogonalization in the sense of M-norm
      % note that currently we have
      % vv = M^{-1}A base and zz = A base

      ro = norm(z,2);
      for j=max(1,i-wsize):i
         t = z'*vv(1:n,j); % M inner product of M^{-1}A space
         hh(j,i) = t;
         z = z - t*zz(1:n,j); % update by A space since v is currently in A space
      end
      t = norm(z,2);
      while(t >= eps && t < ro/sqrt(2))
         ro = t;
         for j=max(1,i-wsize):i
            t = z'*vv(1:n,j);
            hh(j,i) = hh(j,i) + t;
            z = z - t*zz(1:n,j);
         end
         t = norm(z,2);
      end
      
      if (t < eps)
         % too small, early termination
         break;
      end

      % next apply preconditioner to the remaining part
      if isa(prefunc,'function_handle') == 0
         v = prefunc*z;
      else
         v = prefunc(z);
      end
      
      hh(i1,i) = sqrt(v'*z);

      if (hh(i1,i) < eps)
         % too small, early termination
         break;
      end

      t = 1.0 / hh(i1,i);
      vv(:,i1) = v * t;
      zz(:,i1) = z * t;

      % we maintain a Cholesky factorization of Hm
      if(tol ~= 0)
         if(i ~= 1)
            Ll(i0) = hh(i,i0) / Ld(i0);
            Ld(i) = sqrt(hh(i,i) - Ll(i0)^2);
            le = 1.0 / Ld(i);
            ls = - ls * Ll(i0) * le;
            res = abs(le * ls) * hh(i1,i) * beta * norm(zz(:,i1));
         else
            Ld(i) = sqrt(hh(i,i));
            ls = 1.0 / Ld(i);
            res = hh(i1,i) / hh(i,i) * beta * norm(zz(:,i1));
         end
         if(res < tol)
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
      if (hh(i1,i) < eps || t < eps)
         % if hh(i1,i) is not small we can proceed from it
         lock = i;
         
         zz(1:n,i1) = rand(n,1);
         for j=max(1,i-wsize):lock
            t = vv(1:n,j)'*zz(1:n,i1);
            zz(1:n,i1) = zz(1:n,i1) - t*zz(1:n,j);
         end
         t = norm(zz(1:n,i1),2);
         while(t >= eps && t < ro/sqrt(2))
            ro = t;
            for j=max(1,i-wsize):lock
               t = vv(1:n,j)'*zz(1:n,i1);
               zz(1:n,i1) = zz(1:n,i1) - t*zz(1:n,j);
            end
            t = norm(zz(1:n,i1),2);
         end
         ro = t;

         if (ro < eps)
            % restart fail, quit
            break
         end

         
         if isa(prefunc,'function_handle') == 0
            vv(1:n,i1) = prefunc*zz(1:n,i1);
         else
            vv(1:n,i1) = prefunc(zz(1:n,i1));
         end

         beta = sqrt(vv(1:n,i1)'*zz(1:n,i1));

         if(beta < eps)
            % restart fail, quit
            break
         end

         t = 1.0 / beta;
         
         zz(1:n,i1) = zz(1:n,i1) * t;
         vv(1:n,i1) = vv(1:n,i1) * t;
      end
         
      while (i < maxits)
         i=i+1;
         i1 = i + 1;
         % A matvec
         if isa(A,'function_handle') == 0
            z = A*vv(1:n,i);
         else
            z = A(vv(1:n,i));
         end

         % orthogonalization in the sense of M-norm
         % note that currently we have
         % vv = M^{-1}A base and zz = A base

         ro = norm(z,2);
         for j=max(1,i-wsize):lock
            t = z'*vv(1:n,j); % M inner product of M^{-1}A space
            z = z - t*zz(1:n,j); % update by A space since v is currently in A space
         end
         for j=max(lock+1,i-wsize):i
            t = z'*vv(1:n,j); % M inner product of M^{-1}A space
            hh(j,i) = t;
            z = z - t*zz(1:n,j); % update by A space since v is currently in A space
         end
         t = norm(z,2);
         while(t >= eps && t < ro/sqrt(2))
            ro = t;
            for j=max(1,i-wsize):lock
               t = z'*vv(1:n,j);
               z = z - t*zz(1:n,j);
            end
            for j=max(lock+1,i-wsize):i
               t = z'*vv(1:n,j);
               hh(j,i) = hh(j,i) + t;
               z = z - t*zz(1:n,j);
            end
            t = norm(z,2);
         end
         
         if (t < eps)
            % too small, early termination
            break;
         end

         % next apply preconditioner to the remaining part
         if isa(prefunc,'function_handle') == 0
            v = prefunc*z;
         else
            v = prefunc(z);
         end
         
         hh(i1,i) = sqrt(v'*z);

         if (hh(i1,i) < eps)
            % too small, early termination
            break;
         end

         t = 1.0 / hh(i1,i);
         vv(:,i1) = v * t;
         zz(:,i1) = z * t;

      end  %% end of while i loop

   end

   Hm = hh(1:i,1:i);
   T = tril(Hm,1);
   T = (T+T')/2;

end
