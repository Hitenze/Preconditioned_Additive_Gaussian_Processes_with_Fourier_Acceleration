function [sol,res,its] = fgmrez(A,n,precfun,rhs,sol,atol,tolIts,maxits,im,outputG)
%% [sol,res,its] = fgmrez(A,n,precfun,rhs,sol,atol,tolIts,maxits,im,outputG)
%  note:    This version is modified from fgmres.m in the
%           MATLAB Suite https://www-users.cse.umn.edu/~saad/software/MATLAB_DEMOS.tar.gz
%  author:  Tianshi Xu <xuxx1180@umn.edu>
%  date:    05/30/23
%  brief:   Restarted Complex flexible GMRES.
%
%  input:
%           A:       Matrix or function handle that could compute matvec.
%           n:       Size of the problem.
%           precfun: Preconditioning function, takes the form y = precfun(x).
%           rhs:     Right-hand-side.
%           sol:     Initial guess.
%           atol:    Use absolute residual?
%           tolIts:  Stopping tolerance.
%           maxits:  Maximum number of iterations.
%           im:      Dimension of Krylov subspace (for restart).
%           outputG: (Optional) Output res at each step? (default: 1)
%
%  output:
%           sol:    Solution.
%           res:    Residual after teach iteration.
%           its:    Number of its.
%
%  example:
%           [sol,res,its] = nfftgp.krylovs.fgmrez(A,n,precfun,rhs,sol,tolIts,maxits,im,outputG)
%           Just normal FGMRES

   if nargin < 10
      outputG = 1;
   end
   its = 0;
   
   % main loop
   while (its < maxits)
      if isa(A,'function_handle') == 0
         vv(1:n,1) = rhs - A*sol;
      else
         vv(1:n,1) = rhs - A(sol);
      end
      %%
      ro = norm(vv(1:n,1),2);
      res(its+1) = ro;
      %W(1) = max(abs(vv(1,1))/norm(res,Inf),1e-10)
      if its == 0 
         if ~atol
            tol1=tolIts*ro;
         else
            tol1=tolIts;
         end
      end
      if ((ro <= tol1) || (its >= maxits))
         return
      end
      t = 1.0/ ro;
      vv(1:n,1) = vv(1:n,1) * t;
      %       initialize 1-st term  of rhs of hessenberg system..  ;
      rs(1) = ro;
      %%-------------------- print its/residual info
      if (outputG)
         fprintf(1,' its %d  res %e \n',its,ro)
      end
      i = 0;
      %-------------------- inner gmres loop
      while ((i < im)  &&  (ro  >  tol1)  &&  (its < maxits))
         i=i+1;
         its = its + 1;
         i1 = i + 1;
         z = feval(precfun,vv(:,i));
         Z(:,i) = z;
         %%--------------------       modified GS  ;
         if isa(A,'function_handle') == 0
            vv(1:n,i1) = A*z;
         else
            vv(1:n,i1) = A(z);
         end
         for j=1:i
            t = vv(1:n,j)'*vv(1:n,i1);
            hh(j,i) = t;
            vv(1:n,i1) = vv(1:n,i1) - t*vv(1:n,j);
         end
         t = norm(vv(1:n,i1),2);
         hh(i1,i) = t;
         if (t  ~= 0.0)
            t = 1.0 / t;
            vv(1:n,i1) = vv(1:n,i1)*t;
         end %% IF
         %%
         if (i ~= 1)
            %
            %--------------------previous rots. on i-th column of h  ;
            %
            for k=2:i
               k1 = k-1;
               t = hh(k1,i);
               hh(k1,i) = conj(c(k1))*t + s(k1)*hh(k,i);
               hh(k,i) = -s(k1)*t + c(k1)*hh(k,i);
            end;  %% FOR
         end  ; %% IF
         %%
         gam = sqrt(abs(hh(i,i))^2 + abs(hh(i1,i))^2);
         if (gam  == 0.0)
            gam = tolmac;
         end
         %
         %       determine plane rotation and update rhs of ls pb   ;
         %
         c(i) = hh(i,i)/gam;
         s(i) = hh(i1,i)/gam;
         rs(i1) = -s(i)*rs(i);
         rs(i) =  conj(c(i))*rs(i);
         %
         %-------------------- test for convergence-  ;
         %
         hh(i,i) = conj(c(i))*hh(i,i) + s(i)*hh(i1,i);
         ro = abs(rs(i1));
         res(its+1) = ro;
         %         w1=ro/norm(res,Inf);
         %         W(its+1) = max(w1,1e-10);
         if (outputG)
            fprintf(1,' its %d  res %e \n',its,ro)
         end
      end  %% end of while (im) loop
      %
      %       now compute solution. first solve upper triangular system.  ;
      %
      rs(i) = rs(i)/hh(i,i);
      for  k=i-1:-1:1
         t=rs(k);
         for j=k+1:i;
            t = t-hh(k,j)*rs(j);
         end
         rs(k) = t/hh(k,k);
      end
      %
      %       done with back substitution..  ;
      %       now form linear combination to get solution  ;
      %         ;
      for j=1:i
         sol = sol +rs(j)* Z(1:n,j);
      end
      if ((ro  <=  tol1) || (its >= maxits))
         %fprintf(1,' total its %d  final res %e \n',its,ro)
         return;
      end
   end  %% end while -- restart

end
