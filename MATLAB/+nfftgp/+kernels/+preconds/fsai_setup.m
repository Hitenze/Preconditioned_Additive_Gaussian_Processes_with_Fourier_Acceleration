function [PRE] = fsai_setup( str, permr, lfil)
%% [PRE] = fsai_setup( str, permr, lfil)
%  author: Tianshi Xu <xuxx1180@umn.edu>
%  date: 05/20/23
%  brief:   setup the FSAI preconditioner of A
%           the pattern is selected by finding lfil neighbors for each column
%           above the diagonal
%           experimental code written in another formula
%
%  input:
%           str:        Kernel struct returned by any kernel functions
%           permr:      row permutation of the kernel matrix, we apply chol on K(permr, permr), can be []
%           lfil:       fill level for the FSAI preconditioner
%
%  output:
%           PRE:     FSAI struct
%                    Struct detail hidden, do not recommend user to access the struct directly
%
%  example:
%           X = nfftgp.kernels.utils.generate_pts(20, 20);
%           kernelg = nfftgp.kernels.kernels.gaussianKernel(X, 1.33, 1.2, 0.01, {}, 1);
%           fsai = nfftgp.kernels.preconds.fsai_setup(kernelg, [], 5);
%           Set up the FSAI of with kernel struct

   PRE.name = 'FSAI';

   %%---------------------------------------------------------
   %  compute the gradient of L when required
   %%---------------------------------------------------------

   if(~isfield(str, 'X'))
      error("Matrix FSAI unimplemented");
   end

   if(isempty(permr))
      KMat = str;
   else
      str.X = str.X(permr,:);
      KMat = str;
   end

   % first build the pattern
   P = knnpattern(KMat.X, lfil);

   %% next compute the FSAI
   if(~KMat.require_grad)
      % no need gradient
      % compute the lower triangular FSAI of A using the upper triangular pattern P

      n = size(KMat.X, 1);
      P = triu(P, 1); % only use the upper triangular P
      maxnnz = nnz(P)+n;

      Gi = zeros(maxnnz,1);
      Gj = zeros(maxnnz,1);
      Ga = zeros(maxnnz,1);
      Gnnz = 0;

      %% main loop
      for i = 1:n
         Pi = [find(P(:,i)); i]; % get the pattern with diagonal
         
         nz = length(Pi);
         AiMat = KMat.kernelfunc(KMat, Pi);

         Ai = AiMat.K; % the submatrix with diagonal
         ei = zeros(nz,1);
         ei(end) = 1.0;
         
         % K^{-1}*e
         iKe = Ai \ ei;
         % sqrt(e'*K^{-1}*e)
         dd = sqrt(ei'*iKe);

         % now start adding gi to G
         ns = Gnnz + 1;
         ne = ns + nz - 1;
         
         % store G
         Gi(ns:ne) = i*ones(nz,1);
         Gj(ns:ne) = Pi;
         Ga(ns:ne) = iKe/dd;
         
         Gnnz = Gnnz + nz;
         
      end

      PRE.G = sparse(Gi(1:Gnnz),Gj(1:Gnnz),Ga(1:Gnnz),n,n);

   else
      % need gradient
      % TODO: need a field to store this value
      num_grads = 3;

      n = size(KMat.X, 1);
      P = triu(P, 1); % only use the upper triangular P
      maxnnz = nnz(P)+n;

      Gi = zeros(maxnnz,1);
      Gj = zeros(maxnnz,1);
      Ga = zeros(maxnnz,1);

      dGa = cell(num_grads,1);
      for i = 1:num_grads
         dGa{i} = zeros(maxnnz,1);
      end

      Gnnz = 0;

      %% main loop
      for i = 1:n
         Pi = [find(P(:,i)); i]; % get the pattern with diagonal
         
         nz = length(Pi);
         AiMat = KMat.kernelfunc(KMat, Pi);

         Ai = AiMat.K; % the submatrix with diagonal
         ei = zeros(nz,1);
         ei(end) = 1.0;
         
         % K^{-1}*e
         iKe = Ai \ ei;
         % sqrt(e'*K^{-1}*e)
         dd = sqrt(ei'*iKe);
         iKe = iKe/dd;

         % now start adding gi to G
         ns = Gnnz + 1;
         ne = ns + nz - 1;
         
         % store G
         Gi(ns:ne) = i*ones(nz,1);
         Gj(ns:ne) = Pi;
         Ga(ns:ne) = iKe;
         
         % next compute the gradient
         for j = 1:num_grads
            dAi = AiMat.dK{j};

            % -K^{-1}*dK*K^{-1}*e
            idKe = - Ai \ (dAi * iKe);

            % -e'*K^{-1}*dK*K^{-1}*e
            ddd = ei'*idKe;

            dGa{j}(ns:ne) = idKe - ddd/2/dd*iKe;
            
         end

         Gnnz = Gnnz + nz;
         
      end
      
      PRE.G = sparse(Gi(1:Gnnz),Gj(1:Gnnz),Ga(1:Gnnz),n,n);
      PRE.dG = cell(num_grads,1);
      for i = 1:num_grads
         PRE.dG{i} = sparse(Gi(1:Gnnz),Gj(1:Gnnz),dGa{i}(1:Gnnz),n,n);
      end
   end

   % setup the derivative vector producd, trace, and solve function
   PRE.dvp_func = @nfftgp.kernels.preconds.fsai_dvp;
   PRE.trace_func = @nfftgp.kernels.preconds.fsai_trace;
   PRE.solve_func = @nfftgp.kernels.preconds.fsai_solve;
   PRE.logdet_func = @nfftgp.kernels.preconds.fsai_logdet;

end

function P = knnpattern(X, lfil)
   % this function build the knn pattern from X following the default order
   [n,~] = size(X);

   if(n <= lfil + 1)
      % in this case we use the full matrix
      P = ones(n,n);
   else
      % in this case we need to find the knn pattern
      P_nnz = n * (lfil + 1) - lfil * ( lfil + 1 ) / 2;
      P_i = zeros(1,P_nnz);
      P_j = zeros(1,P_nnz);
      P_a = ones(1,P_nnz);
      idx = 1;
      for i = 1:lfil+1
         % the first lfils are filled
         P_i(idx:idx+i-1) = 1:i;
         P_j(idx:idx+i-1) = i;
         idx = idx + i;
      end
      for i = lfil+2:n
         % the rest are computed by knn
         P_i(idx:idx+lfil-1) = knnsearch(X(1:i-1,:),X(i,:),'K',lfil);
         P_i(idx+lfil) = i;
         P_j(idx:idx+lfil) = i;
         idx = idx + lfil + 1;
      end
      P = sparse(P_i,P_j,P_a,n,n);
   end
end

   