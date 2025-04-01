function [PRE] = nys_setup( str, n, noise, k)
%% [PRE] = nys_setup( str, n, noise, k)
%  author: Tianshi Xu <xuxx1180@umn.edu>
%  date: 12/01/23
%  brief:   setup the Projection based Nystrom preconditioner of K+noise*I (inpute matrix is with noise)!
%
%  input:
%           str:        Kernel struct or matvec handle
%                       1. Kernel struct returned by any kernel functions
%                       2. matvec handle, @(x) A*x
%                          In this case no gradient information is available
%                       3. cell of matvec handles {matvec, dvp}
%           n:          size of the kernel matrix
%           noise:      the noise level of the kernel matrix.
%                       Note that your kernel matvec is K+noise*I
%           k:          the rank of the RAN preconditioner
%
%  output:
%           PRE:        RAN struct
%                       Struct detail hidden, do not recommend user to access the struct directly
%
%  example:
%           X = nfftgp.kernels.utils.generate_pts(20, 20);
%           kernelg = nfftgp.kernels.kernels.gaussianKernel(X, 1.33, 1.2, 0.01, {}, 1);
%           ran = nfftgp.kernels.preconds.nys_setup(kernelg, 5);
%           Set up the NYS of with kernel struct
%
%           KMatg = nfftgp.kernels.kernels.gaussianKernelMat(kernelg, [], []);
%           Kmv = @(x) KMatg.K*x;
%           ran2 = nfftgp.kernels.preconds.nys_setup(KMatg, 5);

   % first check if str is a cell or function handle
   if(iscell(str))
      % in this case, must be a cell of matvec handles
      % must have function handle matvec and dvp
      if(~isfield(str, 'matvec') || ~isfield(str, 'dvp'))
         error('nfftgp:kernels:preconds:nys_setup:input', 'Input cell must have matvec and dvp field');
      end
      % random matrix
      [W,~] = qr(rand(n, k), 0);

      % random projection
      Y = zeros(n, k);
      for i = 1 : k
         Y(:,i) = str.matvec(W(:,i)) - noise*W(:,i);
      end

      nu = eps(norm(Y,'fro'));
      Y = Y + nu*W;
      M = Y / chol(W'*Y);

      [U,S,~] = svds(M,k);

      S = max(diag(S).^2-nu,0);
      S0 = sqrt(S);

      f2 = str.f*str.f;
      eta = f2*noise;
   
      M = (S+eta).^(-1);
   
      PRE = struct("U",U,"S",S,"M",M,"S0",S0,"eta",eta,"f2",f2,"name","NYS");

   elseif(isa(str, 'function_handle'))
      % in this case, str is the matvec handle
      % random matrix
      [W,~] = qr(rand(n, k), 0);

      % random projection
      Y = zeros(n, k);
      for i = 1 : k
         Y(:,i) = str(W(:,i)) - noise*W(:,i);
      end

      nu = eps(norm(Y,'fro'));
      Y = Y + nu*W;
      M = Y / chol(W'*Y);

      [U,S,~] = svds(M,k);

      S = max(diag(S).^2-nu,0);
      S0 = sqrt(S);

      f2 = str.f*str.f;
      eta = f2*noise;

      M = (S+eta).^(-1);

      PRE = struct("U",U,"S",S,"M",M,"S0",S0,"eta",eta,"f2",f2,"name","NYS");
      
   else
      % in this case, str is the kernel struct
      
      str.mu = 0.0;
      KMat = str.kernelfunc(str, [], []);
      str.mu = noise;
      
      % random matrix
      [W,~] = qr(rand(n, k), 0);

      % random projection
      Y = KMat.K * W;

      nu = eps(norm(Y,'fro'));
      Y = Y + nu*W;
      M = Y / chol(W'*Y);

      [U,S,~] = svds(M,k);

      S = max(diag(S).^2-nu,0);
      S0 = sqrt(S);

      f2 = str.f*str.f;
      eta = f2*noise;

      M = (S+eta).^(-1);

      PRE = struct("U",U,"S",S,"M",M,"S0",S0,"eta",eta,"f2",f2,"name","NYS");

   end

   % setup the derivative vector producd, trace, and solve function
   PRE.dvp_func = @nfftgp.kernels.preconds.nys_dvp;
   PRE.trace_func = @nfftgp.kernels.preconds.nys_trace;
   PRE.solve_func = @nfftgp.kernels.preconds.nys_solve;
   PRE.logdet_func = @nfftgp.kernels.preconds.nys_logdet;

end