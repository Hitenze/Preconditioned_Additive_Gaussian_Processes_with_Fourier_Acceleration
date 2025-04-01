function [PRE] = chol_setup(str, permr, shift)
%% [PRE] = chol_setup(str, permr, shift)
%  author: Tianshi Xu <xuxx1180@umn.edu>
%  date: 05/18/23
%  brief: setup the Cholesky factorization of A
%
%  input:
%           str:     Kernel matrix or kernel struct
%                    1. kernel struct
%                       struct returned by any kernel functions
%                    2. kernel matrix struct
%                       struct returned by any kernel matrix functions
%           permr:   (optional) row permutation of the kernel matrix, we apply chol on K(permr, permr)
%           shift:   (optional) shift of the kernel matrix, typically used for numerical stability
%                    the default value is eps(norm(KMat.K))
%
%  output:
%           PRE:     Cholesky factorization struct
%                    Struct detail hidden, do not recommend user to access the struct directly
%
%  example:
%           X = nfftgp.kernels.utils.generate_pts(20, 20);
%           kernelg = nfftgp.kernels.kernels.gaussianKernel(X, 1.33, 1.2, 0.01, {}, 1);
%           chol1 = nfftgp.kernels.preconds.chol_setup(kernelg);
%           Set up the Cholesky factorization of with kernel struct
%
%           KMatg = nfftgp.kernels.kernels.gaussianKernelMat(kernelg, [], []);
%           chol2 = nfftgp.kernels.preconds.chol_setup(KMatg);
%           Set up the Cholesky factorization of with kernel matrix struct

   PRE.name = 'CHOL';

   %%---------------------------------------------------------
   %  compute the gradient of L when required
   %  formula of the gradient is given as follows:
   %  dL/dx = L*PHI(L^{-1}*dK/dx*L^{-T})
   %  the function PHI is defined as follows:
   %              { 0, if i < j
   %  PHI(A_ij) = { A_ij/2, if i = j
   %              { A_ij, if i > j
   %%---------------------------------------------------------

   if(nargin < 2)
      permr = [];
   end

   if(isfield(str, 'X'))
      % in this case, str is a kernel struct
      Kmat = str.kernelfunc(str, permr, []);
      n = size(Kmat.K,1);

      if(nargin < 3)
         shift = sqrt(n)*eps(norm(Kmat.K));
      end
      
      PRE.L = chol(Kmat.K+shift*eye(n), 'lower');
      if(str.require_grad)
         PRE.dK = Kmat.dK;
         PRE.dL = cell(size(Kmat.dK));
         PRE.GdKG = cell(size(Kmat.dK));
         for i = 1 : numel(Kmat.dK)
            PRE.GdKG{i} = PRE.L \ (Kmat.dK{i} / PRE.L');
            PRE.dL{i} = PRE.L * PHI(PRE.GdKG{i});
         end
      end
   else
      % in this case, str is a kernel matrix struct
      if(isempty(permr))
         n = size(str.K,1);

         if(nargin < 3)
            shift = sqrt(n)*eps(norm(str.K));
         end

         PRE.L = chol(str.K+shift*eye(n), 'lower');
         if(isfield(str, 'dK') && ~isempty(str.dK))
            PRE.dK = str.dK;
            PRE.dL = cell(size(str.dK));
            PRE.GdKG = cell(size(str.dK));
            for i = 1 : numel(str.dK)
               PRE.GdKG{i} = PRE.L \ (str.dK{i} / PRE.L');
               PRE.dL{i} = PRE.L * PHI(PRE.GdKG{i});
            end
         end      
      else
         n = length(permr);

         Kpp = str.K(permr, permr);

         if(nargin < 3)
            shift = sqrt(n)*eps(norm(Kpp));
         end

         PRE.L = chol(Kpp+shift*eye(n), 'lower');
         if(isfield(KMat, 'dK') && ~isempty(str.dK))
            PRE.dK = str.dK;
            PRE.dL = cell(size(str.dK));
            PRE.GdKG = cell(size(str.dK));
            for i = 1 : numel(str.dK)
               PRE.GdKG{i} = PRE.L \ (str.dK{i}(permr, permr) / PRE.L');
               PRE.dL{i} = PRE.L * PHI(PRE.GdKG{i});
            end
         end
      end
   end
   
   % setup the derivative vector producd, trace, and solve function
   PRE.dvp_func = @nfftgp.kernels.preconds.chol_dvp;
   PRE.trace_func = @nfftgp.kernels.preconds.chol_trace;
   PRE.solve_func = @nfftgp.kernels.preconds.chol_solve;
   PRE.logdet_func = @nfftgp.kernels.preconds.chol_logdet;

end

function PHI_K = PHI(K)
   PHI_K = tril(K,-1) + diag(diag(K)/2);
end