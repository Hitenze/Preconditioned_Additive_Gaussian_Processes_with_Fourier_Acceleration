function [PRE] = afn_setup(str, permr, maxrank, lfil, m, numests, noran, trueest)
%% [PRE] = afn_setup(str, permr, maxrank, lfil, m, numests, noran, trueest)
%  author: Tianshi Xu <xuxx1180@umn.edu>
%  date: 05/20/23
%  brief:   setup the AFN preconditioner of A
%
%  input:
%           str:        Kernel struct returned by any kernel functions
%           permr:      the permutation of the preconditioner
%           maxrank:    the max rank of the AFN preconditioner
%           lfil:       the level of fill-in for the Schur complement FSAI
%           m:          the number of points used in the rank estimation
%           numests:    the number of times we estimate the rank
%           noran:      if set to 1, we disable the switch to RAN when the rank is too small,
%           trueest:    TODO: do we estimate the true rank? 
%                       Generally if we found that the rank is larger than maxrank, we can skip the remaining rank estimation
%                       However, sometime we want to have a more accurate estimation of the rank. In this case, we can set trueest to 1
%
%  output:
%           PRE:        AFN struct
%                       Struct detail hidden, do not recommend user to access the struct directly
%
%  example:
%           X = nfftgp.kernels.utils.generate_pts(20, 20);
%           kernelg = nfftgp.kernels.kernels.gaussianKernel(X, 1.33, 1.2, 0.01, {}, 1);
%           afn = nfftgp.kernels.preconds.afn_setup(kernelg, [], 5, 5, 5, 5, 1);
%           Setup AFN preconditioner, always use AFN, do not switch to RAN
%

   PRE.name = 'AFN';

   if(nargin < 5)
      m = 500;
   end

   if(nargin < 6)
      numests = 5;
   end

   if(nargin < 7)
      noran = 0;
   end

   if(~isfield(str, 'X'))
      error("Matrix FSAI unimplemented");
   end

   if(isempty(permr))
      KMat = str;
   else
      str.X = str.X(permr,:);
      KMat = str;
   end

   n = size(KMat.X, 1);

   %% rank estimation
   k = rank_estimation(KMat, maxrank, m, numests);

   %fprintf("Estimated rank: %d\n", k);
   k = min(k, maxrank);

   %% FPS
   permk = nfftgp.kernels.utils.fps_par(KMat.X, k);
   permn = nfftgp.utils.expand_perm(permk, n);
   PRE.perm = permn;

   if noran
      % Skip RAN, always use AFN formula
      k = max(k,1); % TODO: check if this is necessary
      params = {KMat, permn, k};
      % TODO: currently do not require grad
      kernels = nfftgp.kernels.kernels.schurCombinedKernel(KMat.X, KMat.f, KMat.l, KMat.mu, params, KMat.require_grad);
      PRE.L11 = kernels.params{5};
      PRE.K12 = kernels.params{6};

      PRE.GS = nfftgp.kernels.preconds.fsai_setup(kernels, [], lfil);

   else
      if k < maxrank
         % in this case we use the RAN preconditioner
         % fprintf("TEST Message: Switch to RAN with rank %d\n", k);
         PRE.RAN = nfftgp.kernels.preconds.ran_setup( KMat, permn, k);
      else
         params = {KMat, permn, k};
         % TODO: currently do not require grad
         kernels = nfftgp.kernels.kernels.schurCombinedKernel(KMat.X, KMat.f, KMat.l, KMat.mu, params, KMat.require_grad);
         PRE.L11 = kernels.params{5};
         PRE.K12 = kernels.params{6};

         PRE.GS = nfftgp.kernels.preconds.fsai_setup(kernels, [], lfil);
         
         if(~isreal(PRE.GS))
            % if FSAI break down
            %fprintf("FSAI break down, switch to RAN\n");
            %fprintf("Kernel parameters: f = %g, l = %g, mu = %g\n", KMat.f, KMat.l, KMat.mu);
            PRE.RAN = nfftgp.kernels.preconds.ran_setup( KMat, permn, maxrank);
         end

      end
   end

   % setup the derivative vector producd, trace, and solve function
   PRE.dvp_func = @nfftgp.kernels.preconds.afn_dvp;
   PRE.trace_func = @nfftgp.kernels.preconds.afn_trace;
   PRE.solve_func = @nfftgp.kernels.preconds.afn_solve;
   PRE.logdet_func = @nfftgp.kernels.preconds.afn_logdet;

end

function [k] = rank_estimation(kernel, max_k, m, numests)
% [k] = rank_estimation(X, kernel, noise_level, rho, nsamples)
% estimate the rank used in the AFN preconditioner
% inputs:
%       kernel:         kernel struct
%       max_k:          maximum rank we accepted
%       m:              how many points we sample
%       numests:        redo the estimation multiple times
% outputs:
%       k:              estimated rank (AFN then use FPS to select k points from X)

if nargin < 4
   numests = 1;
end

if nargin < 3
      m = 500;
end

if nargin < 2
      max_k = 2000;
end

%% first run the scaled rank estimation

   k_scaled = rank_estimation_scaled(kernel, m, numests);

   if k_scaled > max_k
      % rank too large, return immediately
      k = k_scaled;
      return;
   else
      % rank not large enogough, continue with the unscaled version
      % to estimate the numerical rank
      k = rank_estimation_unscaled(kernel, max_k, m, numests);
   end

end

function [err] = NysError(K, k)
   % helper function to compute the absolute Nystrom error
   % building with the first k points
   %
   % K: kernerl matrix (FPS ordered)
   % k: rank
   %
   K11 = K(1:k,1:k);
   K1 = K(:,1:k);
   Knys = K1*(K11\K1');
   err = norm(K-Knys);
end
   
function [k] = rank_estimation_scaled(kernel, m, numests)
% [k] = rank_estimation_scaled(kernel, m, numests)
% estimate the rank used in the AFN preconditioner
% this scaled version sample a subset and scale
% the subsampled dataset is assume to have similar eigenvalue curve
% as the original one
%
% inputs:
%       kernel:         kernel struct
%       m:              how many points we sample
%       numests:        redo the estimation multiple times
% outputs:
%       k:              estimated rank (AFN then use FPS to select k points from X)

   X = kernel.X;
   noise_level = kernel.mu;

   [n, d] = size(X);
   m = min(m, n);

   % global subsampled rank
   r = 0;

   %% main loop
   for i = 1:numests
         
      % sample a subset and scale
      Xm = X(randsample(n, m),:)*(m/n)^(1/d);
      
      % reorder using FPS
      [ permfps, ~] = nfftgp.kernels.utils.fps_par( Xm, m);

      % add shift to make Nystrom stable
      kernel.X = Xm;
      KmMat = kernel.kernelfunc(kernel, permfps);
      Km = (KmMat.K - eye(m)*noise_level) / (kernel.f^2);
      nu = sqrt(m)*eps(norm(Km,2));
      Km = Km + eye(m)*nu;
         
      % next search for the rank reduces relative Nystrom error to below 0.1
      Km_nrm = norm(Km,'fro');
      
      % naive bisection search
      % other search strategies also works
      r_s = 1;
      r_e = m;

      while r_s < r_e
         r_c = floor((r_s+r_e)/2);
         if NysError(Km, r_c)/Km_nrm < 0.1
            r_e = r_c;
         else
            r_s = r_c+1;
         end
      end

      % now we have the first k such that relerr_k < 0.1
      % accumulate it, in this matlab version I use the max
      % among them, also possible to use the average
      r = max(r, r_s);

   end % end of nsamples loop    

   k = ceil(r*n/m);

end

function [k] = rank_estimation_unscaled(kernel, max_k, m, numests)
% [k] = rank_estimation_unscaled(kernel, max_k, m, numests)
% estimate the rank used in the AFN preconditioner
% 
% this function is called when the scaled version indicates that the problem is
% a low-rank problem, i.e., the rank is likely to be smaller than max_k
%
% in this case, we sample few dataset of size max_k and estimate the rank
% using its eigenvalue curve.
%
% note that max_k might again be large, so we scale the dataset again in this estimation
%
% inputs:
%       kernel:         kernel struct
%       max_k:          maximum rank used in the scaled version
%       m:              how many points we sample
%       numests:        redo the estimation multiple times
% outputs:
%       k:              estimated rank (AFN then use FPS to select k points from X)
      
   X = kernel.X;

   % the kernel is f^2 K + mu I
   % we scale it to K + (mu/f^2) I
   noise_level = kernel.mu / (kernel.f^2);

   [n, d] = size(X);
   m = min(m, n);
   max_k = min(max_k, n);
   
   % global subsampled rank
   r = 0;
   
   if(max_k < m)
      max_k = m;
   end

   %% main loop
   for i = 1:numests
      
      % sample a subset of size max_k
      Xmaxk = X(randsample(n, max_k),:);

      % max_k might still be too large, we subsample again from it
      if(max_k > m)
         Xm = Xmaxk(randsample(max_k, m),:)*(m/max_k)^(1/d);
      else
         Xm = Xmaxk;
      end
      
      % eivenvalue curve, eig and svd should be the same
      kernel.X = Xm;
      XmMat = kernel.kernelfunc(kernel); 
      s = svd(XmMat.K);
      
      % compute the numerical rank
      r = max(r, sum(s>1.1*noise_level));

   end % end of nsamples loop    
   
   k = floor(r*max_k/m);
      
end
      
      