function [kernel] = additiveKernel(X, f, l, mu, params, require_grad, form_matrix)
   %% [kernel] = additiveKernel(X, f, l, mu, params, require_grad, form_matrix)
   %  author: Tianshi Xu <xuxx1180@umn.edu>
   %  date: 05/18/23
   %  brief:   create an additive kernel struct which is a sum of multiple kernels
   %           kernel = (K1 + K2 + ... + Kn)/n where Ki is the i-th kernel
   %
   %  inputs:
   %           X:             n * d matrix, n is the number of points, d is the dimension
   %           f:             variance scale
   %           l:             lengthscale
   %           mu:            noise level
   %           params:        struct of parameters
   %                          params.windows:      cell array of windows, each window is a subset of {1, 2, ..., d}
   %                                               which specifies the data for each matrix
   %                          params.kernelstrfunc:kernel structure function, all kernels must have the same function
   %                          params.kernelfunc:   kernel matrix function, all kernels must have the same function
   %           require_grad:  (optional) whether to compute the derivatives, default is false
   %           form_matrix:   (optional) whether to form the kernel matrix, default is false
   %                          if set to true, the kernel matrix and its derivatives will be
   %                          stored
   %
   %  outputs:
   %           kernel:        Gaussian kernel struct
   %                          kernel.X:            n * d matrix, input matrix
   %                          kernel.f:            variance scale
   %                          kernel.l:            lengthscale
   %                          kernel.mu:           noise level
   %                          kernel.kernels:      cell array of structs of kernels for each window
   %                          kernel.kernelfunc:   kernel function kernel( str, permr, permc)
   %                          kernel.require_grad: determine whether the gradient is required
   %                          kernel.params:       not used
   %                          kernel.K:            (optilnal) kernel matrix
   %                          kernel.dK:           (optilnal) cell array of gradient matrices {dKl, dKn}
   %
   %  example:
   %           kernelg = nfftgp.kernels.kernels.gaussianKernel(X, f, l, mu, {}, 1);
   %           KMatg = nfftgp.kernels.kernels.gaussianKernelMat(kernelg, [], []);
   %           Create kernel matrix with derivative
   
      if(nargin < 5 || nargin > 7)
         warning("Must have 5/6/7 input arguments");
         kernel.X = [];
         kernel.f = 1.0;
         kernel.l = 1.0;
         kernel.mu = 0.01;
         kernel.kernels = {};
         kernel.kernelfunc = @nfftgp.kernels.kernels.additiveKernelMat;
         kernel.params = {};
         kernel.require_grad = 0;
         return;
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
      kernel.kernelfunc = @nfftgp.kernels.kernels.additiveKernelMat;
      kernel.require_grad = require_grad;
      kernel.params = params;
      
      nkernels = length(params.windows);
      kernel.kernels = cell(nkernels, 1);
      for i = 1:nkernels
         window = params.windows{i};
         kernel.kernels{i} = params.kernelstrfunc(X(:, window), f, l, mu, [], require_grad, form_matrix);
      end

      if(form_matrix)
         KMat = kernel.kernelfunc(X, f, l, mu, require_grad);
         kernel.K = KMat.K;
         kernel.dK = KMat.dK;
      else
         kernel.K = [];
         kernel.dK = {};
      end
   
   end