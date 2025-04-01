function [kernel] = matern12Kernel(X, f, l, mu, params, require_grad, form_matrix)
%% [KMat] = matern12Kernel(X, f, l, mu, params, require_grad, form_matrix)
%  author: Tianshi Xu <xuxx1180@umn.edu>
%  date: 04/20/24
%  brief:   create the gussian kernel struct
%           possibly compute the gaussian kernel matrix
%           possibly with derivatives w.r.t. length scale and noise level
%           kernel formula: k(x, y) = f^2 * exp(-||x-y||/l)
%
%  inputs:
%           X:             n * d matrix, n is the number of points, d is the dimension
%           f:             variance scale
%           l:             lengthscale
%           mu:            noise level
%           params:        (optional) not used
%           require_grad:  (optional) whether to compute the derivatives, default is false
%           form_matrix:   (optional) whether to form the kernel matrix, default is false
%                          if set to true, the kernel matrix and its derivatives will be
%                          stored
%
%  outputs:
%           kernel:        Matern kernel struct
%                          kernel.X:            n * d matrix, input matrix
%                          kernel.f:            variance scale
%                          kernel.l:            lengthscale
%                          kernel.mu:           noise level
%                          kernel.kernelfunc:   kernel function kernel( str, permr, permc)
%                          kernel.require_grad: determine whether the gradient is required
%                          kernel.params:       not used
%                          kernel.K:            (optilnal) kernel matrix
%                          kernel.dK:           (optilnal) cell array of gradient matrices {dKl, dKn}
%
%  example:
%           kernelg = nfftgp.kernels.kernels.matern12Kernel(X, f, l, mu, {}, 1);
%           KMatg = nfftgp.kernels.kernels.matern12KernelMat(kernelg, [], []);
%           Create kernel matrix with derivative

   if(nargin < 4 || nargin > 7)
      warning("Must have 4/5/6/7 input arguments");
      kernel.X = [];
      kernel.f = 1.0;
      kernel.l = 1.0;
      kernel.mu = 0.01;
      kernel.kernelfunc = @nfftgp.kernels.kernels.matern12KernelMat;
      kernel.params = {};
      kernel.require_grad = 0;
      return;
   end

   if(nargin < 5)
      params = {};
   end

   if(nargin < 6)
      require_grad = 0;
   end

   if(nargin < 7)
      form_matrix = 0;
   end

   kernel.X = X;
   kernel.f = f;
   kernel.l = l;
   kernel.mu = mu;
   kernel.kernelfunc = @nfftgp.kernels.kernels.matern12KernelMat;
   kernel.require_grad = require_grad;
   kernel.params = params;

   if(form_matrix)
      KMat = kernel.kernelfunc(X, f, l, mu, require_grad);
      kernel.K = KMat.K;
      kernel.dK = KMat.dK;
   else
      kernel.K = [];
      kernel.dK = {};
   end

end