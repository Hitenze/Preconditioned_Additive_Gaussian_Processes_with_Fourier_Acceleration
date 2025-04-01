function [KMat] = gaussianKernelFixedNoiseMat( kernel, permr, permc)
%% [KMat] = gaussianKernelFixedNoiseMat( kernel, permr, permc)
%  author: Tianshi Xu <xuxx1180@umn.edu>
%  date: 05/18/23
%  brief:   compute the gaussian kernel matrix, possible with derivatives
%           w.r.t. length scale and noise level
%           kernel formula: k(x, y) = f^2 * exp(-||x - y||^2 / (2l^2)) + mu1 * I + diag(mu2)
%
%  inputs:
%           kernel:        Gaussian kernel struct
%                          kernel.X:            n * d matrix, input matrix
%                          kernel.f:            variance scale vector
%                          kernel.l:            lengthscale vector
%                          kernel.mu:           noise level vector
%                          kernel.kernelfunc:   kernel function kernel( str, permr, permc, require_grad)
%                          kernel.require_grad: determine whether the gradient is required
%                          kernel.params:       second noise matrix of size n * num_class which is fixed
%                          kernel.K:            (optilnal) kernel matrix
%                          kernel.dK:           (optilnal) cell array of gradient matrices {dKl, dKn}
%           permr:         (optional) row permutation
%           permc:         (optional) column permutation
%                          if permr == [], generate the kernel matrix K(X,X) with noise
%                          if permr ~= [] && permc == [], generate the kernel matrix K(X(permr,:),X(permr,:)) with noise
%                          if permr ~= [] && permc ~= [], generate the kernel matrix K(X(permr,:),X(permc,:)) without noise
%
%  outputs:
%           KMat:          Kernel matrix struct
%                          KMat.K:              kernel matrix
%                          KMat.dK:             (optional) cell array of gradient matrices {dKf, dKl, dKn}
%
%
%  example:
%           kernelg = nfftgp.kernels.kernels.gaussianKernelFixedNoiseMat(X, f, l, mu, {}, 1);
%           KMatg = nfftgp.kernels.kernels.gaussianKernelFixedNoiseMat(kernelg, [], []);
%           Create kernel matrix with derivative

   if(nargin < 2)
      permr = [];
   end

   if(nargin < 3)
      permc = [];
   end

   %%---------------------------------------------------------
   %  Formulas:
   %  k(x,y) = f^2 * (exp(-||x-y||^2 / (2l^2))) + mu1 * I + diag(mu2)
   %  dk(x,y)/df = 2 * f * exp(-||x-y||^2 / (2l^2))
   %  dk(x,y)/dl = f^2 * 1/l^3 * ||x-y||^2 * exp(-||x-y||^2 / (2l^2))
   %  dk(x,y)/dmu = I if x == y, 0 if x ~= y
   %%---------------------------------------------------------

   f = kernel.f;
   l = kernel.l;
   mu = kernel.mu;
   mu2 = kernel.params;
   require_grad = kernel.require_grad;

   num_classes = length(f);

   if isempty(permr)
      % k(X,X), with noise
      X = kernel.X;
      if(~isempty(kernel.K))
         error("does not support in this test version");
      else
         n = size(X,1);
         XXT = X * X';
         XX = sum(X.^2, 2);

         D2 = bsxfun(@plus, bsxfun(@plus, -2 * XXT, XX), XX');

         KMat.K = cell(num_classes, 1);
         if(require_grad)
            KMat.dK = cell(num_classes, 1);
         end
         for i = 1:num_classes
            KMat.K{i} = f(i)^2*exp(-D2 / (2*l(i)^2)) + mu(i) * eye(n) + diag(mu2(:,i));
            
            % Note: here the second noise is fixed, and each element has different noise level
            if(require_grad)
               if(~isempty(kernel.dK))
                  error("does not support in this test version");
               else
                  dKf = KMat.K{i} * 2.0 / f(i);
                  dKl = f(i)^2*D2.*exp(-D2 / (2*l(i)^2))/l(i)^3;
                  dKn = eye(n);
                  KMat.dK{i} = {dKf, dKl, dKn};
               end
            end
         end
      end
   else
      error("does not support in this test version");
   end

end