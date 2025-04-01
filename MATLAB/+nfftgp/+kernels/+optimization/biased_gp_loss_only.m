
function [L] = biased_gp_loss_only(x, X, Y, kernelfun, matfun, transforms, masks, solver, maxits, nvecs, precond_setup, precond_split)
%% [L] = biased_gp_loss_only(x, X, Y, kernelfun, matfun, transforms, masks, solver, maxits, nvecs, precond_setup, precond_split)
%  author: Tianshi Xu <xuxx1180@umn.edu>
%  date: 00/23/23
%  brief: computing the without its gradient with respect to the hyperparameters
%         the formula of the loss is given by
%         L = 0.5*( Y'*K*Y + log(det(K)) + n*log(2*pi) )
%         note that we apply an extra transform to the hyperparameters so this has to be appended
%         this implementation does not require writing preconditioner in split form if used
%
%  input:
%           x:             Array of length 3, assume to be column vector
%                          x(1): variance scale before transform (unconstrained), transform(f) is the actual variance scale
%                          x(2): length scale before transform (unconstrained), transform(l) is the actual length scale
%                          x(3): noise level before transform (unconstrained), transform(mu) is the actual noise level
%           X:             training data
%           Y:             training labels
%           kernelfun:     gaussianKernel or maternKernel or whatever you defined
%           matfun:        gaussianKernelMat or maternKernelMat or whatever you defined
%           transforms:    transform function (change the problem to a unconstrained problem) return both value and gradient
%                          single function: all the hyperparameters share the same transform
%                          cell: each hyperparameter has its own transform
%           masks:         Same length as [f,l,mu]. If marker(i) = 0 then [f,l,mu](i) is fixed.
%           solver:        solver for the linear system (optional)
%           maxits:        maximum number of iterations (optional)
%           nvecs:         number of Lanczos vectors (optional)
%           precond_setup: preconditioner setup function, a wraper that only takes kernel as input (optional)
%           precond_split: if the preconditioner support split solve (optional)
%                          if support, we can save memory in the situation where full orthgonalization is used
%
%  output:
%           L:          Loss
%           L_grad:     Gradient of the loss, column vector

   f = x(1);
   l = x(2);
   mu = x(3);

   if(nargin < 4)
      kernelfun = @nfftgp.kernels.kernels.gaussianKernel;
   end

   if(nargin < 5)
      matfun = @nfftgp.kernels.kernels.gaussianKernelMat;
   end

   if(nargin < 6)
      transform1 = @(val)nfftgp.kernels.optimization.transform(val, 'softplus');
      transform2 = @(val)nfftgp.kernels.optimization.transform(val, 'softplus');
      transform3 = @(val)nfftgp.kernels.optimization.transform(val, 'softplus');
      transforms = {transform1; transform2; transform3};
   end

   if(length(transforms) == 1)
      transforms = {transforms; transforms; transforms};
   end

   if(nargin < 7)
      masks = ones(length(x0), 1);
   end
   masks(masks~=0) = 1;

   if(nargin < 8)
      solver = [];
   end

   if(nargin < 9)
      maxits = 5;
   end

   if(nargin < 10)
      nvecs = 5;
   end

   % if no preconditioner is provided we do not use preconditioning
   if(nargin < 11)
      precond_setup = [];
   end

   if(nargin < 12)
      % by default we assume that split solve is not supported
      precond_split = false;
   end
   
   % size of the dataset
   n = size(X,1);
   
   % kernel function
   kernelfunc = @(f, l, mu)kernelfun(X, f, l, mu, {}, 1);
   matfunc = @(kernel)matfun(kernel, [], []);
   
   % extract transformation, we can have different transformation for different parameters
   [f] = transforms{1}(f);
   [l] = transforms{2}(l);
   [mu] = transforms{3}(mu);
   
   % create kernel matrix
   Kernel = kernelfunc(f, l, mu);
   KMat = matfunc(Kernel);
   K = KMat.K;
   if(~isempty(precond_setup))
      % setup preconditioner is provided
      PRE = precond_setup(Kernel);
   end
   
   %% compute the inverse quadaratic term L1
   % this is a non bcg version
   if(isempty(solver))
      % exact solve with "\" operator

      iKY = K\Y;

      L1 = Y'*iKY;
      
   else

      % approximate solve with zero initial guess

      if(isempty(precond_setup))
         
         % no preconditioner
         
         iKY = solver( K, n, Y, zeros(n,1), maxits);

         L1 = Y'*iKY;
         
      else
         
         % use preconditioner
         if(precond_split)
            % split preconditioner, in this case we
            % use the standard CG/Lanczos with the 
            % preconditioned kernel matrix

            % preK(x) is L \ (K * (R \ x))
            % in order to solve this linear system we need
            % to also modify the right-hand side by L
            preK = @(x)(precond_solve(PRE, K*(precond_solve(PRE, x, 'R')), 'L'));
            Y = precond_solve(PRE, Y, 'L');

            % solve the linear system
            iKY = solver( preK, n, Y, zeros(n,1), maxits);

            L1 = Y'*iKY;
         
         else

            % in this case we use the PCG/PLanczos directly

            prefunc = @(x) PRE.solve_func(PRE,x);
            
            iKY = solver( K, n, prefunc, Y, zeros(n,1), maxits);

            L1 = Y'*iKY;
         end
         
      end
   end
   
   %% compute the log determinant term L2
   %  this is again a non bcg version
   if(isempty(solver))
      % this is the exact version
      L2 = sum(log(abs(eig(K))));
   else
      % approximate log determinant
      if(isempty(precond_setup))
         % non preconditioned version
         [val_cell, dval_cell] = nfftgp.krylovs.lanquad({K, dK}, n, maxits, nvecs, solver, 'logdet');
         L2 = val_cell{1};
      else
         % preconditioned version
         [val_cell, dval_cell] = nfftgp.krylovs.lanquad({K, dK}, n, maxits, nvecs, solver, PRE, precond_split, 'logdet');
         
         L2 = val_cell{1};
      end

   end
   
   L = 0.5*(L1 + L2 + n*log(2*pi));
   if ~isreal(L)
      warning("Complex loss")
      disp(L1)
      disp(L2)
      disp(plogdet)
   end
   
   L = L / n;

end