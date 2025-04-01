function [Y2, std] = biased_gp_prediction(f, l, mu, X1, X2, Y1, kernelfun, matfun, solver, maxits, tol, precond_setup, precond_split)
%% [Y2, std] = biased_gp_prediction(f, l, mu, X1, X2, Y1, kernelfun, matfun, solver, maxits, tol, precond_setup, precond_split)
%  author: Tianshi Xu <xuxx1180@umn.edu>
%  date: 10/18/23
%  brief: Prediction based on the current GP values
%
%  input:
%           f:             variance scale AFTER transform
%                          length scale AFTER transform
%                          noise level AFTER transform
%           X1:            training data
%           X2:            prediction data
%           Y1:            training labels
%           kernelfun:     kernel function (optional), default is gaussianKernel
%           matfun:        matrix vector multiplication function (optional), default is gaussianKernelMat
%  output:
%           Y2:            prediction labels
%           std:           standard deviation of the prediction

   % kernel function
   % no derivative needed
   n1 = size(X1,1);
   n2 = size(X2,1);
   n = n1 + n2;

   if(nargin < 9)
      solver = [];
   end
   if(nargin < 10)
      maxits = 200;
   end
   if(nargin < 11)
      tol = 0.0;
   end
   if(nargin < 12)
      precond_setup = [];
   end
   if(nargin < 13)
      precond_split = false;
   end

   if nargout == 2
      kernelfunc = @(f, l, mu)kernelfun([X1;X2], f, l, mu, {}, 0);
      KMat = matfun(kernelfunc(f, l, mu), [], []);

      K11 = KMat.K(1:n1,1:n1);
      K12 = KMat.K(1:n1,n1+1:end);
      K22 = KMat.K(n1+1:end,n1+1:end);

      % TODO: BCG is needed
      if(isempty(solver))
         Y2 = K12'*(K11\Y1);
         Cov = K22 - K12'*(K11\K12);
         std = sqrt(abs(diag(Cov)));
      else
         if(isempty(precond_setup))
            iY1 = solver( K11, n1, Y1, zeros(n1,1), maxits, tol);
            Y2 = K12'*iY1;
            iK12 = zeros(n1,n2);
            for i = 1:n2
               iK12(:,i) = solver( K11, n1, K12(:,i), zeros(n1,1), maxits, tol);
            end
            Cov = K22 - K12'*iK12;
            std = sqrt(abs(diag(Cov)));
         else
            kernelprec = @(f, l, mu)kernelfun(X1, f, l, mu, {}, 0);
            Kernel = kernelprec(f, l, mu);
            PRE = precond_setup(Kernel);
            if(precond_split)
               preK11 = @(x)(precond_solve(PRE, K11*(precond_solve(PRE, x, 'R')), 'L'));
               LY1 = precond_solve(PRE, Y1, 'L');
               RiY1 = solver( preK11, n1, LY1, zeros(n1,1), maxits, tol);
               iY1 = precond_solve(PRE, RiY1, 'R');
               Y2 = K12'*iY1;
               iK12 = zeros(n1,n2);
               for i = 1:n2
                  LK12i = precond_solve(PRE, K12(:,i), 'L');
                  RK12i = solver( preK11, n1, LK12i, zeros(n1,1), maxits, tol);
                  iK12(:,i) = precond_solve(PRE, RK12i, 'R');
               end
               Cov = K22 - K12'*iK12;
               std = sqrt(abs(diag(Cov)));
            else
               prefunc = @(x) PRE.solve_func(PRE,x);
               iY1 = solver( K11, n1, prefunc, Y1, zeros(n1,1), maxits, tol);
               Y2 = K12'*iY1;
               iK12 = zeros(n1,n2);
               for i = 1:n2
                  iK12(:,i) = solver( K11, n1, prefunc, K12(:,i), zeros(n1,1), maxits, tol);
               end
               Cov = K22 - K12'*iK12;
               std = sqrt(abs(diag(Cov)));
            end
         end
      end
   else
      printf("out one");
      kernelfunc = @(f, l, mu)kernelfun([X1;X2], f, l, mu, {}, 0);
      KMat = matfun(kernelfunc(f, l, mu), 1:n1, 1:n);

      K11 = KMat.K(:,1:n1);
      K12 = KMat.K(:,n1+1:end);

      % TODO: BCG is needed
      if(isempty(solver))
         Y2 = K12'*(K11\Y1);
      else
         if(isempty(precond_setup))
            iY1 = solver( K11, n1, Y1, zeros(n1,1), maxits, tol);
            Y2 = K12'*iY1;
         else
            kernelprec = @(f, l, mu)kernelfun(X1, f, l, mu, {}, 0);
            Kernel = kernelprec(f, l, mu);
            PRE = precond_setup(Kernel);
            if(precond_split)
               preK11 = @(x)(precond_solve(PRE, K11*(precond_solve(PRE, x, 'R')), 'L'));
               LY1 = precond_solve(PRE, Y1, 'L');
               RiY1 = solver( preK11, n1, LY1, zeros(n1,1), maxits, tol);
               iY1 = precond_solve(PRE, RiY1, 'R');
               Y2 = K12'*iY1;
            else
               prefunc = @(x) PRE.solve_func(PRE,x);
               iY1 = solver( K11, n1, prefunc, Y1, zeros(n1,1), maxits, tol);
               Y2 = K12'*iY1;
            end
         end
      end
   end
end
