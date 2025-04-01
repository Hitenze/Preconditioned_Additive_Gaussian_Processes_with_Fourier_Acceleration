
function [L, L_grad, cell_L, cell_L_grad] = biased_gp_loss(x, X, Y, kernelfun, matfun, transforms, masks, solver, maxits, nvecs, precond_setup, precond_split)
%% [L, L_grad, cell_L, cell_grad] = biased_gp_loss(x, X, Y, kernelfun, matfun, transforms, masks, solver, maxits, nvecs, precond_setup, precond_split)
%  author: Tianshi Xu <xuxx1180@umn.edu>
%  date: 00/23/23
%  brief: computing the loss and the gradient of the loss
%         and its gradient with respect to the hyperparameters
%         the formula of the loss is given by
%         L = 0.5*( Y'*K*Y + log(det(K)) + n*log(2*pi) )
%         dL = 0.5*( Y'*K^{-1}*dK*K^{-1}*Y + trace(dK*K^{-1}) )
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
%           nvecs:         number of Lanczos vectors or the initial guess matrix (optional)
%                          1. A number, then we use it as the number of Lanczos vectors
%                          2. A struct with filed 'Z', then we use nvecs.Z as the initial guess matrix
%                             nvecs.nvecs is the number of Lanczos vectors (only the first nvecs.nvecs columns of Z are used)
%           precond_setup: preconditioner setup function, a wraper that only takes kernel as input (optional)
%           precond_split: if the preconditioner support split solve (optional)
%                          if support, we can save memory in the situation where full orthgonalization is used
%
%  output:
%           L:             Loss
%           L_grad:        Gradient of the loss, column vector
%           cell_L:        (Optional) Return cell (L1, L2)
%           cell_L_grad:   (Optional) Return grad of cell (L1, L2)

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
   [f, df] = transforms{1}(f);
   [l, dl] = transforms{2}(l);
   [mu, dmu] = transforms{3}(mu);
   
   % create kernel matrix
   Kernel = kernelfunc(f, l, mu);
   KMat = matfunc(Kernel);
   K = KMat.K;
   dK = KMat.dK;
   dKf = dK{1};
   dKl = dK{2};
   dKmu = dK{3};
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
      
      L1_grad = zeros(3,1);
      
      L1_grad(1) = iKY'*dKf*iKY*df;
      L1_grad(2) = iKY'*dKl*iKY*dl;
      L1_grad(3) = iKY'*dKmu*iKY*dmu;
      
   else

      % approximate solve with zero initial guess

      if(isempty(precond_setup))
         
         % no preconditioner
         
         iKY = solver( K, n, Y, zeros(n,1), maxits);

         L1 = Y'*iKY;
         L1_grad = zeros(3,1);
         
         L1_grad(1) = iKY'*dKf*iKY*df;
         L1_grad(2) = iKY'*dKl*iKY*dl;
         L1_grad(3) = iKY'*dKmu*iKY*dmu;
         
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
            
            iKY = precond_solve(PRE, iKY, 'R');

            L1_grad = zeros(3,1);
            
            L1_grad(1) = iKY'*dKf*iKY*df;
            L1_grad(2) = iKY'*dKl*iKY*dl;
            L1_grad(3) = iKY'*dKmu*iKY*dmu;
         
         else

            % in this case we use the PCG/PLanczos directly

            prefunc = @(x) PRE.solve_func(PRE,x);
            
            iKY = solver( K, n, prefunc, Y, zeros(n,1), maxits);

            L1 = Y'*iKY;
            
            L1_grad = zeros(3,1);
            
            L1_grad(1) = iKY'*dKf*iKY*df;
            L1_grad(2) = iKY'*dKl*iKY*dl;
            L1_grad(3) = iKY'*dKmu*iKY*dmu;
         end
         
      end
   end
   
   %% compute the log determinant term L2
   %  this is again a non bcg version
   if(isempty(solver))
      % this is the exact version
      L2 = sum(log(abs(eig(K))));
      L2_grad = zeros(3,1);
   
      L2_grad(1) = trace(K\dKf)*df;
      L2_grad(2) = trace(K\dKl)*dl;
      L2_grad(3) = trace(K\dKmu)*dmu;
      
      if nargout > 2
         L2_hist = [];
         L2_grad_hist = [];
      end

   else
      % approximate log determinant
      if(isempty(precond_setup))
         % non preconditioned version
         [val_cell, dval_cell] = nfftgp.kernels.krylovs.lanquad({K, dK}, n, maxits, nvecs, solver, 'logdet');
         L2 = val_cell{1};
         %L2 = sum(log(abs(eig(K))));
         L2_grad = dval_cell{1};
         
         L2_grad(1) = L2_grad(1)*df;
         L2_grad(2) = L2_grad(2)*dl;
         L2_grad(3) = L2_grad(3)*dmu;
         
         if nargout > 2
            L2_hist = val_cell{2};
            L2_grad_hist = dval_cell{2};
            L2_grad_hist(1,:) = L2_grad_hist(1,:)*df;
            L2_grad_hist(2,:) = L2_grad_hist(2,:)*dl;
            L2_grad_hist(3,:) = L2_grad_hist(3,:)*dmu;
         end

      else
         % preconditioned version
         [val_cell, dval_cell] = nfftgp.kernels.krylovs.lanquad({K, dK}, n, maxits, nvecs, solver, PRE, precond_split, 'logdet');
         
         L2 = val_cell{1};
         %L2 = sum(log(abs(eig(K))));
         L2_grad = dval_cell{1};
         
         L2_grad(1) = L2_grad(1)*df;
         L2_grad(2) = L2_grad(2)*dl;
         L2_grad(3) = L2_grad(3)*dmu;
         
         if nargout > 2
            L2_hist = val_cell{2};
            L2_grad_hist = dval_cell{2};
            L2_grad_hist(1,:) = L2_grad_hist(1,:)*df;
            L2_grad_hist(2,:) = L2_grad_hist(2,:)*dl;
            L2_grad_hist(3,:) = L2_grad_hist(3,:)*dmu;
         end

      end

   end
   
   L = 0.5*(L1 + L2 + n*log(2*pi));
   %L = L2;

   if ~isreal(L)
      warning("Complex loss")
      disp(L1)
      disp(L2)
      disp(plogdet)
   end

   % set unused gradients to zero
   L_grad = 0.5*( - L1_grad + L2_grad);
   %L_grad = L2_grad;
   L_grad = L_grad(:).*masks(:);
   
   L = L / n;
   L_grad = L_grad / n;

   if nargout > 2
      cell_L = {L1 * 0.5 / n, L2 * 0.5 / n, L2_hist * 0.5 / n};
      cell_L_grad = { - L1_grad * 0.5 / n, L2_grad * 0.5 / n, L2_grad_hist * 0.5 / n};
   end
   
end