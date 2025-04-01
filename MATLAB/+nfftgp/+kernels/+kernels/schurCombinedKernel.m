function [kernel] = schurCombinedKernel(X, f, l, mu, params, require_grad, form_matrix)
%% [kernel] = maternKernel(X, f, l, mu, params, require_grad, form_matrix)
%  author: Tianshi Xu <xuxx1180@umn.edu>
%  date: 05/18/23
%  brief:   create the Schur complement kernel struct
%           possibly compute the gaussian kernel matrix
%           possibly with derivatives w.r.t. length scale and noise level
%           kernel formula: k(x, y) = C - G12'*G12 where G12'*G12 = E^T*B^{-1}*E
%               | B     E |
%           K = | E^T   C | => S = C - E^T*B^{-1}*E
%
%  inputs:
%           X:             n * d matrix, n is the number of points, d is the dimension
%           f:             variance scale
%           l:             lengthscale
%           mu:            noise level
%           params:        cell of parameters for building the Schur complement
%                          1. kernel function for the kernel of K
%                          2. the permutation
%                          3. n1 the size of the first block
%           require_grad:  (optional) whether to compute the derivatives, default is false
%           form_matrix:   (optional) whether to form the kernel matrix, default is false
%                          if set to true, the kernel matrix and its derivatives will be
%                          stored
%
%  outputs:
%           kernel:        Schur combined kernel struct
%                          kernel.X:            n * d matrix, input matrix
%                          kernel.l:            lengthscale
%                          kernel.mu:           noise level
%                          kernel.kernelfunc:   kernel function kernel( str, permr, permc)
%                          kernel.require_grad: determine whether the gradient is required
%                          kernel.params:       cell of parameters for building the Schur complement
%                                               1. kernel struct for the kernel of K22
%                                               2. the GK12 matrix
%                                               3. (optional) cell of the GdK12 matrices
%                                               4. (optional) cell of the GdK11GK12 matrices
%                                               5. chol struct for K11
%                                               6. kernel matrix struct for K12
%                          kernel.K:            (optilnal) kernel matrix
%                          kernel.dK:           (optilnal) cell array of gradient matrices {dKl, dKn}
%
%  example: I do not recomment to use this function directly
%

   if(nargin < 5 || nargin > 6)
      warning("Must have 5/6 input arguments");
      kernel.X = [];
      kernel.f = 1.0;
      kernel.l = 1.0;
      kernel.mu = 0.01;
      kernel.kernelfunc = @nfftgp.kernels.kernels.schurCombinedKernelMat;
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

   Kkernel = params{1};
   perm = params{2};
   n1 = params{3};

   perm1 = perm(1:n1);
   perm2 = perm(n1+1:end);

   kernel.X = X(perm2,:);
   kernel.l = l;
   kernel.mu = mu;
   kernel.kernelfunc = @nfftgp.kernels.kernels.schurCombinedKernelMat;
   kernel.require_grad = require_grad;
   kernel.perms = perm2;

   % next build the kernel parameters
   if(require_grad && ~Kkernel.require_grad)
      error("Schur complement requires gradient but the K kernel does not");
   end

   % setup the chol of K11
   K11Chol = nfftgp.kernels.preconds.chol_setup(Kkernel, perm1);
   % extract K12 matrix, possibly with gradient
   K12Mat = Kkernel.kernelfunc(Kkernel, perm1, perm2);
   % next, compute G*K12
   GK12 = K11Chol.L \ K12Mat.K;
   if(require_grad)
      % if require gradient, we need to compute 
      % 1. G*dK12
      % 2. G*dK11*G'*G*K12 = G*dK11*G'*GK12
      num_grads = numel(K12Mat.dK);
      GdK12 = cell(num_grads, 1);
      GdK11GK12 = cell(num_grads, 1);
      for i = 1:num_grads
         GdK12{i} = K11Chol.L \ K12Mat.dK{i};
         GdK11GK12{i} = K11Chol.GdKG{i} * GK12;
      end
      % update the dataset to K22
      Kkernel.X = kernel.X;
      kernel.params = {Kkernel, GK12, GdK12, GdK11GK12, K11Chol, K12Mat};
   else
      % update the dataset to K22
      Kkernel.X = kernel.X;
      kernel.params = {Kkernel, GK12, [], [], K11Chol, K12Mat};
   end

   if(form_matrix)
      KMat = nfftgp.kernels.kernels.schurCombinedKernelMat(X, f, l, mu, require_grad);
      kernel.K = KMat.K;
      kernel.dK = KMat.dK;
   else
      kernel.K = [];
      kernel.dK = {};
   end

end
