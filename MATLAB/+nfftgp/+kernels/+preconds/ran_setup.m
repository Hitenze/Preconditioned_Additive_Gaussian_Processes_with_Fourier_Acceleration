function [PRE] = ran_setup( str, perm, k)
%% [PRE] = ran_setup( str, perm, k)
%  author: Tianshi Xu <xuxx1180@umn.edu>
%  date: 05/20/23
%  brief:   setup the RAN preconditioner of A
%
%  input:
%           str:        Kernel struct returned by any kernel functions
%           perm:       the permutation of the preconditioner
%           k:          the rank of the RAN preconditioner
%
%  output:
%           PRE:        RAN struct
%                       Struct detail hidden, do not recommend user to access the struct directly
%
%  example:
%           X = nfftgp.kernels.utils.generate_pts(20, 20);
%           kernelg = nfftgp.kernels.kernels.gaussianKernel(X, 1.33, 1.2, 0.01, {}, 1);
%           ran = nfftgp.kernels.preconds.ran_setup(kernelg, [], 5);
%           Set up the RAN of with kernel struct

   if(~isfield(str, 'X'))
      error("Matrix RAN unimplemented");
   end

   n = size(str.X,1);
   if(isempty(perm))
      perm = randperm(n);
   end

   noise_level = str.mu;
   str.mu = 0.0;
   KMat = str.kernelfunc(str, perm, []);
   
   k = min(n, k);
   %k=79
   idxu = 1:k;

   I11 = eye(k);
   
   K11 = KMat.K(idxu,idxu);
   K1 = KMat.K(idxu,:);
   if(str.require_grad)
      num_grads = numel(KMat.dK);
      dK11 = cell(num_grads,1);
      dK1 = cell(num_grads,1);
      for i = 1 : numel(KMat.dK)
         dKi = KMat.dK{i};
         dK11{i} = dKi(idxu,idxu);
         dK1{i} = dKi(idxu,:);
      end
      % warning: gradient to mu is 0 since this is not used when forming the preconditioner!
      dK11{3} = zeros(size(dK11{3}));
      dK1{3} = zeros(size(dK1{3}));
   end

   %M * A1
   nu = sqrt(n)*eps(norm(K1,2));
   M = K1'/chol(K11+nu*I11);
   [U,S,~] = svds(M,k);
   
   S = max(diag(S).^2-nu,0);
   S0 = sqrt(S);

   %eta = S(end)+noise_level;
   f2 = str.f*str.f;
   eta = f2*noise_level;

   M = (S+eta).^(-1);

   PRE = struct("K1",K1,"K11",K11,"U",U,"S",S,"M",M,"S0",S0,"perm",perm,"eta",eta,"f2",f2,"name","RAN");
   if(str.require_grad)
      PRE.dK1 = dK1;
      PRE.dK11 = dK11;
   end
   
   % setup the derivative vector producd, trace, and solve function
   PRE.dvp_func = @nfftgp.kernels.preconds.ran_dvp;
   PRE.trace_func = @nfftgp.kernels.preconds.ran_trace;
   PRE.solve_func = @nfftgp.kernels.preconds.ran_solve;
   PRE.logdet_func = @nfftgp.kernels.preconds.ran_logdet;

end