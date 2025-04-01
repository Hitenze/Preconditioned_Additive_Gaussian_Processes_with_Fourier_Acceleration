function [KMat] = schurCombinedKernelMat( kernel, permr, permc)
%% [KMat] = schurCombinedKernel( kernel, permr, permc)
%  author: Tianshi Xu <xuxx1180@umn.edu>
%  date: 05/20/23
%  brief:   compute the Schur Complement kernel matrix, possible with derivatives
%           w.r.t. length scale and noise level
%           kernel formula: k(x, y) = C - G12'*G12 where G12'*G12 = E^T*B^{-1}*E
%               | B     E |
%           K = | E^T   C | => S = C - E^T*B^{-1}*E
%
%  inputs:
%           X:             n * d matrix, n is the number of points, d is the dimension
%           permr:         X1 is X(permr, :), if set to [], then permr = 1:n
%           permc:         X2 is X(permc, :), if set to [], then permc = permr
%           l_cell:        cell structure as {l, kernel, GK12}
%                          if require_grad, also need L^{-1}K12, L^{-1}dK12, and L^{-1}dK11/dxL^{-T}L^{-1}K12
%                          cell structure as {l, kernel, GK12, {L^{-1}dK12}, {L^{-1}dK11/dxL^{-T}L^{-1}K12}}
%           mu:            noise level
%           require_grad:  (optional) whether to compute the derivatives, default is false
%           form_matrix:   (optional) whether to form the kernel matrix, default is true
%                          if set to false, the matrix won't be formed, only all the other
%                          fields will be filled
%
%  outputs:
%           KMat:          Kernel matrix struct
%                          KMat.K:              kernel matrix
%                          KMat.dK:             (optional) cell array of gradient matrices {dKl, dKn}
%
%  example: I do not recommend to use this function directly

   if(nargin < 2)
      permr = [];
   end

   if(nargin < 3)
      permc = [];
   end

   %l = kernel.l;
   %mu = kernel.mu;
   require_grad = kernel.require_grad;

   %%---------------------------------------------------------
   %  Formula of the gradient is:
   %  dS/dx = dK22 - dK12'*inv(K11)*K12 - K12'*inv(K11)*dK12 + K12'*inv(K11)*dK11/dx*inv(K11)*K12 
   %%---------------------------------------------------------

   if isequal(kernel.params{1}.kernelfunc, @nfftgp.kernels.kernels.additiveKernelMat)
      K22Mat = kernel.params{1}.kernelfunc(kernel.params{1},kernel.perms(permr),kernel.perms(permc));
   else
      K22Mat = kernel.params{1}.kernelfunc(kernel.params{1},permr,permc);
   end
   
   if isempty(permr)
      % k(X,X), with noise
      GK12 = kernel.params{2};
      KMat.K = K22Mat.K - GK12'*GK12;
      
      if(require_grad)
         GdK12 = kernel.params{3};
         GdK11GK12 = kernel.params{4};
         num_grads = numel(K22Mat.dK);
         KMat.dK = cell(num_grads, 1);
         for i = 1:num_grads
            GK12GdK12 = GK12'*GdK12{i};
            KMat.dK{i} = K22Mat.dK{i} - GK12GdK12 - GK12GdK12' + GK12'*GdK11GK12{i};
         end
      end
   else
      if isempty(permc)
         % k(X,Y), without noise
         GK12 = kernel.params{2}(:, permr);
      
         KMat.K = K22Mat.K - GK12'*GK12;
      
         if(require_grad)
            GdK12 = kernel.params{3};
            GdK11GK12 = kernel.params{4};
            num_grads = numel(K22Mat.dK);
            KMat.dK = cell(num_grads, 1);
            for i = 1:num_grads
               GdK12i = GdK12{i}(:, permr);
               GdK11GK12i = GdK11GK12{i}(:, permr);
               GK12GdK12i = GK12'*GdK12i;
               KMat.dK{i} = K22Mat.dK{i} - GK12GdK12i - GK12GdK12i' + GK12'*GdK11GK12i;
            end
         end
      else
         GK12r = kernel.params{2}(:, permr);
         GK12c = kernel.params{2}(:, permc);
      
         KMat.K = K22Mat.K - GK12r'*GK12c;
      
         if(require_grad)
            GdK12 = kernel.params{3};
            GdK11GK12 = kernel.params{4};
            num_grads = numel(K22Mat.dK);
            KMat.dK = cell(num_grads, 1);
            for i = num_grads
               GdK12ri = GdK12{i}(:, permr);
               GdK12ci = GdK12{i}(:, permc);
               GdK11GK12ci = GdK11GK12{i}(:, permc);
               KMat.dK{i} = K22Mat.dK{i} - GK12r'*GdK12ci - GdK12ri'*GK12c + GK12r'*GdK11GK12ci;
            end
         end
      end
   end
end
