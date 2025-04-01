function [KMat] = additiveKernelMat( kernel, permr, permc)
   %% [KMat] = additiveKernelMat( kernel, permr, permc)
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
   %                          kernel.kernels:      cell array of structs of kernels for each window
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
      %  Directly add the kernel matrices
      %%---------------------------------------------------------
      
      if(isempty(kernel.kernels))
         KMat.K = [];
         KMat.dK = [];
         return;
      else
         KMat = kernel.kernels{1}.kernelfunc( kernel.kernels{1}, permr, permc);
         for i = 2:length(kernel.kernels)
            KMati = kernel.kernels{i}.kernelfunc( kernel.kernels{i}, permr, permc);
            KMat.K = KMat.K + KMati.K;
            if(kernel.require_grad)
               KMat.dK{1} = KMat.dK{1} + KMati.dK{1};
               KMat.dK{2} = KMat.dK{2} + KMati.dK{2};
               KMat.dK{3} = KMat.dK{3} + KMati.dK{3};
            end
         end
         nkernels = length(kernel.kernels);
         KMat.K = KMat.K / nkernels;
         if(kernel.require_grad)
            KMat.dK{1} = KMat.dK{1} / nkernels;
            KMat.dK{2} = KMat.dK{2} / nkernels;
            KMat.dK{3} = KMat.dK{3} / nkernels;
         end
      end
   end