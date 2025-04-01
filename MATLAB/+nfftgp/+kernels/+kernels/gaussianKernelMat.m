function [KMat] = gaussianKernelMat( kernel, permr, permc)
%% [KMat] = gaussianKernelMat( kernel, permr, permc)
%  author: Tianshi Xu <xuxx1180@umn.edu>
%  date: 05/18/23
%  brief:   compute the gaussian kernel matrix, possible with derivatives
%           w.r.t. length scale and noise level
%           kernel formula: k(x, y) = f^2 * [ exp(-||x - y||^2 / (2l^2)) + mu * I]
%
%  inputs:
%           kernel:        Gaussian kernel struct
%                          kernel.X:            n * d matrix, input matrix
%                          kernel.f:            variance scale
%                          kernel.l:            lengthscale
%                          kernel.mu:           noise level
%                          kernel.kernelfunc:   kernel function kernel( str, permr, permc, require_grad)
%                          kernel.require_grad: determine whether the gradient is required
%                          kernel.params:       not used
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
%                          KMat.dK:             (optional) cell array of gradient matrices {dKl, dKn}
%
%
%  example:
%           kernelg = nfftgp.kernels.kernels.gaussianKernel(X, f, l, mu, {}, 1);
%           KMatg = nfftgp.kernels.kernels.gaussianKernelMat(kernelg, [], []);
%           Create kernel matrix with derivative

   if(nargin < 2)
      permr = [];
   end

   if(nargin < 3)
      permc = [];
   end

   %%---------------------------------------------------------
   %  Formulas:
   %  k(x,y) = f^2 * (exp(-||x-y||^2 / (2l^2)) + mu * I)
   %  dk(x,y)/df = 2 * f * exp(-||x-y||^2 / (2l^2))
   %  dk(x,y)/dl = f^2 * 1/l^3 * ||x-y||^2 * exp(-||x-y||^2 / (2l^2))
   %  dk(x,y)/dmu = f^2 * I if x == y, 0 if x ~= y
   %%---------------------------------------------------------

   f = kernel.f;
   l = kernel.l;
   mu = kernel.mu;
   require_grad = kernel.require_grad;

   if isempty(permr)
      % k(X,X), with noise
      X = kernel.X;
      if(~isempty(kernel.K))
         KMat.K = kernel.K;
         if(kernel.require_grad)
            if(~isempty(kernel.dK))
               KMat.dK = kernel.dK;
            else
               n = size(X,1);
               XXT = X * X';
               XX = sum(X.^2, 2);

               D2 = bsxfun(@plus, bsxfun(@plus, -2 * XXT, XX), XX');
               
               dKf = KMat.K * 2 / f;
               dKl = f^2*D2.*exp(-D2 / (2*l^2))/l^3;
               dKn = f^2*eye(n);
               KMat.dK = {dKf, dKl, dKn};
            end
         end
      else
         n = size(X,1);
         XXT = X * X';
         XX = sum(X.^2, 2);

         D2 = bsxfun(@plus, bsxfun(@plus, -2 * XXT, XX), XX');
         KMat.K = f^2*(exp(-D2 / (2*l^2)) + mu * eye(n));

         if(require_grad)
            if(~isempty(kernel.dK))
               KMat.dK = kernel.dK;
            else
               dKf = KMat.K * 2 / f;
               dKl = f^2*D2.*exp(-D2 / (2*l^2))/l^3;
               dKn = f^2*eye(n);
               KMat.dK = {dKf, dKl, dKn};
            end
         end
      end
   else
      if isempty(permc)
         % k(X,X), with noise
         X = kernel.X(permr, :);
         n = size(X,1);
         XXT = X * X';
         XX = sum(X.^2, 2);
      
         D2 = bsxfun(@plus, bsxfun(@plus, -2 * XXT, XX), XX');
         KMat.K = f^2*(exp(-D2 / (2*l^2)) + mu * eye(n));
      
         if(require_grad)
            dKf = KMat.K * 2 / f;
            dKl = f^2*D2.*exp(-D2 / (2*l^2))/l^3;
            dKn = f^2*eye(n);
            KMat.dK = {dKf, dKl, dKn};
         end
      else
         % k(X,Y), without noise
         X = kernel.X(permr, :);
         Y = kernel.X(permc, :);
         XY = X * Y';
         XX = sum(X.^2, 2);
         YY = sum(Y.^2, 2);

         D2 = bsxfun(@plus, bsxfun(@plus, -2 * XY, XX), YY');
         KMat.K = f^2*exp(-D2 / (2*l^2));

         if(require_grad)
            dKf = KMat.K * 2 / f;
            dKl = f^2*D2.*exp(-D2 / (2*l^2))/l^3;
            dKn = zeros(size(KMat.K));
            KMat.dK = {dKf, dKl, dKn};
         end
      end
   end

end