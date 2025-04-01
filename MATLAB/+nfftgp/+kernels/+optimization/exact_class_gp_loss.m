
function [L_total, L_grad_total] = exact_class_gp_loss(x, mu2, X, Ys, kernelfun, matfun, transforms, masks)
%% [L_total, L_grad_total] = exact_class_gp_loss(x, mu2, X, Ys, kernelfun, matfun, transforms, masks)
%  author: Tianshi Xu <xuxx1180@umn.edu>
%  date: 01/30/24
%  brief: computing the loss and the gradient of the loss
%         and its gradient with respect to the hyperparameters
%         the formula of the loss is given by
%         L = 0.5*( Y'*K*Y + log(det(K)) + n*log(2*pi) )
%         dL = 0.5*( Y'*K^{-1}*dK*K^{-1}*Y + trace(dK*K^{-1}) )
%         note that we apply an extra transform to the hyperparameters so this has to be appended
%         this implementation does not require writing preconditioner in split form if used
%
%  input:
%           x:             Array of length 2*num_classes, assume to be column vector
%                          x(1:num_classes): variance scale before transform (unconstrained), transform(f) is the actual variance scale
%                          x(num_classes+1:end): length scale before transform (unconstrained), transform(l) is the actual length scale
%           mu2:           fixed noise variance matrix
%           X:             training data
%           Ys:            training labels of size n * num_classes
%           kernelfun:     gaussianKernel or maternKernel or whatever you defined
%           matfun:        gaussianKernelMat or maternKernelMat or whatever you defined
%           transforms:    transform function (change the problem to a unconstrained problem) return both value and gradient
%                          single function: all the hyperparameters share the same transform
%                          cell: each hyperparameter has its own transform
%           masks:         Same length as [f,l,mu]. If marker(i) = 0 then [f,l,mu](i) is fixed.
%
%  output:
%           L_total:       Total loss is the sum of each indevidual loss
%           L_grad_total:  Gradient of the total loss with respect to all the parameters of each kernel

   nx = length(x);
   num_classes = nx / 3;

   if(nargin < 5)
      kernelfun = @nfftgp.kernels.kernels.gaussianKernelFixedNoise;
   end

   if(nargin < 6)
      matfun = @nfftgp.kernels.kernels.gaussianKernelFixedNoise;
   end

   if(nargin < 7)
      transform1 = @(val)nfftgp.kernels.optimization.transform(val, 'softplus');
      transform2 = @(val)nfftgp.kernels.optimization.transform(val, 'softplus');
      transform3 = @(val)nfftgp.kernels.optimization.transform(val, 'softplus');
      transforms = {transform1; transform2; transform3};
   end

   if(length(transforms) == 1)
      transforms = {transforms; transforms; transforms};
   end

   if(nargin < 8)
      masks = ones(3, 1);
   end
   masks(masks~=0) = 1;

   % size of the dataset
   n = size(X,1);
   
   % kernel function
   kernelfunc = @(f, l, mu)kernelfun(X, f, l, mu, mu2, 1);
   matfunc = @(kernel)matfun(kernel, [], []);
   
   % extract transformation, we can have different transformation for different parameters
   fs = x(1:num_classes);
   dfs = zeros(num_classes,1);
   ls = x(num_classes+1:2*num_classes);
   dls = zeros(num_classes,1);
   mus = x(2*num_classes+1:end);
   dmus = zeros(num_classes,1);
   for i = 1:num_classes
      [fs(i), dfs(i)] = transforms{1}(fs(i));
      [ls(i), dls(i)] = transforms{2}(ls(i));
      [mus(i), dmus(i)] = transforms{3}(mus(i));
   end
   
   % create kernel matrix
   Kernel = kernelfunc(fs, ls, mus);
   KMat = matfunc(Kernel);
   Ks = KMat.K;
   dKs = KMat.dK;

   % exact solve with "\" operator
   L_total = 0;
   L_grad_total = zeros(nx,1);
   for i = 1:num_classes
      K = Ks{i};
      dK = dKs{i};
      dKf = dK{1};
      dKl = dK{2};
      dKn = dK{3};
      Y = Ys(:,i);

      iKY = K\Y;

      L1 = Y'*iKY;
      
      L1_grad = zeros(3,1);
      
      L1_grad(1) = iKY'*dKf*iKY*dfs(i);
      L1_grad(2) = iKY'*dKl*iKY*dls(i);
      L1_grad(3) = iKY'*dKn*iKY*dmus(i);
      
      L2 = sum(log(abs(eig(K))));
      L2_grad = zeros(3,1);

      L2_grad(1) = trace(K\dKf)*dfs(i);
      L2_grad(2) = trace(K\dKl)*dls(i);
      L2_grad(3) = trace(K\dKn)*dmus(i);

      L = 0.5*(L1 + L2 + n*log(2*pi));
      
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

      L_total = L_total + L;
      L_grad_total(i) = L_grad(1);
      L_grad_total(num_classes+i) = L_grad(2);
      L_grad_total(2*num_classes+i) = L_grad(3);
   end
end