function [KMat] = maternKernelMat( kernel, permr, permc)
%% [KMat] = maternKernelMat( kernel, permr, permc)
%  author: Tianshi Xu <xuxx1180@umn.edu>
%  date: 05/18/23
%  brief:   compute the matern32 kernel matrix, possible with derivatives
%           w.r.t. length scale and noise level
%           kernel formula: k(x, y) = f^2*[(1+sqrt(3||x-y||/l) * exp(-sqrt(3)*||x-y||/l) + mu*I]
%
%  inputs:
%           kernel:        Matern kernel struct
%                          kernel.X:            n * d matrix, input matrix
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

   if(nargin < 2)
      permr = [];
   end

   if(nargin < 3)
      permc = [];
   end

   f = kernel.f;
   l = kernel.l;
   mu = kernel.mu;
   require_grad = kernel.require_grad;

   %%---------------------------------------------------------
   %  Formulas are:
   %  k(x,y) = f^2 * [(1 + sqrt(3) / l * ||x-y||) * exp(-sqrt(3)*||x-y|| / l) + mu * I ]
   %  dk(x,y)/df = 2 * f * [(1 + sqrt(3) / l * ||x-y||) * exp(-sqrt(3)*||x-y|| / l) + mu * I ]
   %  dk(x,y)/dl = f^2 * 3 / l^3 * ||x-y||^2 * exp(-sqrt(3)*||x-y|| / l)
   %  dk(x,y)/dmu = f^2*I if x == y, 0 if x ~= y
   %%---------------------------------------------------------

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
               D2(D2<0) = 0;
               D = sqrt(D2);
                  
               dKf = KMat.K * 2 / f;
               dKl = f^2*3*D2.*exp(-1.7320508075688772935 * D / l)/l^3;
               dKn = f^2*eye(n);
               KMat.dK = {dKf, dKl, dKn};
            end
         end
      else
         n = size(X,1);
         XXT = X * X';
         XX = sum(X.^2, 2);

         D2 = bsxfun(@plus, bsxfun(@plus, -2 * XXT, XX), XX');
         D2(D2<0) = 0;
         D = sqrt(D2);
         KMat.K = f^2*(( 1 + 1.7320508075688772935 / l * D )  .* exp( -1.7320508075688772935 * D / l) + mu * eye(n));

         if(require_grad)
            dKf = KMat.K * 2 / f;
            dKl = f^2*3*D2.*exp(-1.7320508075688772935 * D / l)/l^3;
            dKn = f^2*eye(n);
            KMat.dK = {dKf, dKl, dKn};
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
         D2(D2<0) = 0;
         D = sqrt(D2);
         KMat.K = f^2*(( 1 + 1.7320508075688772935 / l * D )  .* exp( -1.7320508075688772935 * D / l) + mu * eye(n));
      
         if(require_grad)
            dKf = KMat.K * 2 / f;
            dKl = f^2*3*D2.*exp(-1.7320508075688772935 * D / l)/l^3;
            dKn = f^2*eye(n);
            KMat.dK = {dKf, dKl, dKn};
         end
      else
         % k(X,Y), without noise
         Y = kernel.X(permc, :);
         X = kernel.X(permr, :);
         XY = X * Y';
         XX = sum(X.^2, 2);
         YY = sum(Y.^2, 2);

         D2 = bsxfun(@plus, bsxfun(@plus, -2 * XY, XX), YY');
         D2(D2<0) = 0;
         D = sqrt(D2);
         KMat.K = f^2*( 1 + 1.7320508075688772935 / l * D ) .* exp( -1.7320508075688772935 * D / l);

         if(require_grad)
            dKf = KMat.K * 2 / f;
            dKl = f^2*3*D2.*exp(-1.7320508075688772935 * D / l)/l^3;
            dKn = zeros(size(KMat.K));
            KMat.dK = {dKf, dKl, dKn};
         end
      end
   end

end