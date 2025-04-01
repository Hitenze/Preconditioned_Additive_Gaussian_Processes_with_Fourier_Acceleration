function [kernel] = gaussianKernelFixedNoise(X, f, l, mu, params, require_grad)
%% [kernel] = gaussianKernelFixedNoise(X, f, l, mu, params, require_grad)
%  author: Tianshi Xu <xuxx1180@umn.edu>
%  date: 05/18/23
%  brief:   create the gussian kernel struct
%           possibly compute the gaussian kernel matrix
%           possibly with derivatives w.r.t. length scale and noise level
%           kernel formula: k(x, y) = f^2 * [ exp(-||x - y||^2 / (2l^2))] + mu1 * I + diag(mu2)
%
%  inputs:
%           X:             n * d matrix, n is the number of points, d is the dimension
%           f:             variance scale vector of size num_class
%           l:             lengthscale vector of size num_class
%           mu:            noise vector of size num_class
%           params:        second noise matrix of size n * num_class which is fixed
%           require_grad:  (optional) whether to compute the derivatives, default is false
%
%  outputs:
%           kernel:        Gaussian kernel struct
%                          kernel.X:            n * d matrix, input matrix
%                          kernel.f:            variance scale vector
%                          kernel.l:            lengthscale vector
%                          kernel.mu:           noise level vector
%                          kernel.kernelfunc:   kernel function kernel( str, permr, permc)
%                          kernel.require_grad: determine whether the gradient is required
%                          kernel.params:       second noise matrix of size n * num_class which is fixed
%                          kernel.K:            (optilnal) kernel matrix
%                          kernel.dK:           (optilnal) cell array of gradient matrices {dKl, dKn}
%
%  example:
%           kernelg = nfftgp.kernels.kernels.gaussianKernel(X, f, l, mu, {}, 1);
%           KMatg = nfftgp.kernels.kernels.gaussianKernelMat(kernelg, [], []);
%           Create kernel matrix with derivative

   if(nargin < 4 || nargin > 7)
      warning("Must have 4/5/6/7 input arguments");
      kernel.X = [];
      kernel.f = [];
      kernel.l = [];
      kernel.mu = [];
      kernel.kernelfunc = @nfftgp.kernels.kernels.gaussianKernelFixedNoiseMat;
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

   kernel.X = X;
   kernel.f = f;
   kernel.l = l;
   kernel.mu = mu;
   kernel.kernelfunc = @nfftgp.kernels.kernels.gaussianKernelFixedNoiseMat;
   kernel.require_grad = require_grad;
   kernel.params = params;

   kernel.K = [];
   kernel.dK = {};

end