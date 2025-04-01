function varargout = lanquad(varargin)
%% function varargout = lanquad(varargin)
%  author:  Tianshi Xu <xuxx1180@umn.edu>
%  date:    09/23/23
%  brief:   Stochastic Lanczos Quadrature Approximation. Fast estimation of tr(f(A)) via Stochastic Lanczos Quadrature 
%           S. Ubaru, Jie Chen, and Yousef Saad SIAM Journal on Matrix Analysis and Applications (SIMAX), 38(4), 1075–1099, 2017
%
%  input:
%           A:          matrix or function handle that could compute matvec with A.
%                       Can also be a cell as {A, dA} where A is the same and dA is a cell
%                       of derivatives of A.
%           n:          the size of the problem.
%           m:          number of Lanczos steps for each vector.
%           nvecs:      total vectors
%           args:       possible arguments
%           operation:  operations to be performed
%                       "trace": trace estimation
%                       "logdet": log determinant estimation
%                       "schatten": schatten norm estimation
%
%  outputs:
%           when A is a matrix:
%              val:     the final estimation
%              vals:    the estimations for each vector
%           when A is a cell:
%              val:     cell {val, vals}
%              dval:    cell {dval, dvals}
%
%  example:
%           [val,vals] = nfftgp.kernels.krylovs.lanquad(K, n, maxits, nvecs, 'trace');
%
%           [val,vals] = nfftgp.kernels.krylovs.lanquad(K, n, maxits, nvecs, 'logdet');
%
%           [val_cell, dval_cell] = nfftgp.kernels.krylovs.lanquad({K, dK}, n, maxits, nvecs, solver, PRE, precond_split, 'logdet');
%           K:             matrix or matvec function handle
%           dK:            cell of derivatives of K
%           precond_split: Is L and R solves supported?

   % Extract the inputs
   A = varargin{1}; % matrix
   n = varargin{2}; % size
   m = varargin{3}; % number of Lanczos steps
   nvecs = varargin{4}; % number of vectors

   % if nvecs is a struct, get nvecs.Z
   if(isstruct(nvecs))
      Z = nvecs.Z;
      nvecs = nvecs.nvecs;
   end
   
   vals = zeros(nvecs,1);
   val = 0;

   % we A is a cell we also handle the derivatives
   % in this case the fisrt possible arguments is
   % the solver option, either "pcg" or "planczos"
   if(iscell(A))
      dA = A{2};
      A = A{1};
      solver = varargin{5};

      if(~isempty(dA))
         num_grads = length(dA);
         dvals = zeros( num_grads, nvecs);
         dval = zeros( num_grads, 1);
      else
         num_grads = 0;
      end

      if(length(varargin) == 8)
         % in this case the preconditioner is also provided
         PRE = varargin{6};
         precond_split = varargin{7};
         prefunc = @(x) PRE.solve_func(PRE, x);
         if(precond_split)
            % redefine A if preconditioner is used
            A = @(x)(precond(PRE, A*(precond(PRE, x, 'R')), 'L'));
         end
         %if isa(A,'function_handle') == 0
         %   A = @(x)(precond(PRE, A*(precond(PRE, x, 'R')), 'L'));
         %else
         %   A = @(x)(precond(PRE, A(precond(PRE, x, 'R')), 'L'));
         %end
         logdetP = PRE.logdet_func(PRE); % logdet of the preconditioner
         if(num_grads > 0)
            traceP = PRE.trace_func(PRE); % trace of M^{-1}dM
         else
            traceP = zeros(num_grads,1);
         end
      else
         logdetP = 0;
         traceP = zeros(num_grads,1);
      end

   end

   % main loop
   if(exist('dA','var'))
      % possibly with derivatives
      for i = 1:nvecs
         % if Z is defined, use it
         if(exist('Z','var'))
            z = Z(:,i);
         else
            z = nfftgp.kernels.utils.radamacher(n);
         end
         %z = Z(i,:)';
         if(exist('prefunc','var') && ~precond_split)
            % in this case use precond
            [iAz, T] = solver( A, n, prefunc, z, zeros(n,1), m);
         else
            % standard solve or split preconditioner
            [iAz, T] = solver( A, n, z, zeros(n,1), m);
         end
         
         [V,D] = eig(T);
         [vals(i), dvals(:,i)] = estimation_with_grad( dA, z, iAz, V, D, varargin(6:end));
         vals(i) = vals(i)*n;
         val = val + vals(i);
         dval = dval + dvals(:,i);
      end
      val = val/nvecs + logdetP;
      dval = dval/nvecs + traceP;
      varargout = {{val, vals + logdetP}, {dval, dvals + logdetP}};
   else
      % only the matrix, no preconditioner, no derivative
      for i = 1:nvecs
         % if Z is defined, use it
         if(exist('Z','var'))
            z = Z(:,i);
         else
            z = nfftgp.kernels.utils.radamacher(n);
         end
         [~,T,~] = nfftgp.kernels.krylovs.lanczos(A, n, z, zeros(n,1), m, 0.0, 0);
         [V,D] = eig(T);
         vals(i) = estimation(V, D, varargin(5:end))*n;
         val = val + vals(i);
      end
      val = val/nvecs;
      varargout = {val, vals};
   end
end

function [est] = estimation(V, D, args)
% do the estimation
   operation = args{end};
   switch operation
      case 'trace'
         % trace estimation
         est = traceest(V, D);
      case 'logdet'
         % log determinant estimation
         est = logdetest(V, D);
      case 'schatten'
         % schatten norm estimation
         p = args{1};
         est = schattnormest(V, D, p);
      otherwise
         error('Unsupported operation');
   end

end

function [est,dest] = estimation_with_grad( dA, z, iAz, V, D, args)
% do the estimation
   operation = args{end};

   if(length(args) > 1)
      % in this case the preconditioner is also provided
      PRE = args{1};
      precond_split = args{2};
   else
      PRE = [];
      precond_split = false;
   end

   switch operation
      case 'logdet'
         % log determinant estimation
         if(~isempty(dA))
            dAz = {dA{1}*z, dA{2}*z, dA{3}*z};
         else
            dAz = [];
         end
         [est,dest] = logdetest_with_grad( z, dAz, iAz, V, D, PRE, precond_split);
      otherwise
         error('Unsupported operation');
   end
end

function [est] = traceest(V, D)
% estimate trace
   lam = abs(diag(D));
   tau = V(1,:).^2;
   est = tau*lam;
end

function [est] = logdetest(V, D)
% estimate log determinant
   lam = abs(diag(D));
   tau = V(1,:).^2;
   est = tau*log(lam);
end

function [est,dest] = logdetest_with_grad( z, dAz, iAz, V, D, PRE, precond_split)
% estimate log determinant and its gradient
% this function does not include trace of preconditioners

   % First logdet, same form
   lam = abs(diag(D));
   tau = V(1,:).^2;
   est = tau*log(lam);

   % next trace, sligltly different here
   if(isempty(PRE))
      % no preconditioner is used
      % (K\z)'*(dK*z) / n
      num_grads = length(dAz);
      dest = zeros(num_grads,1);
      for i = 1:num_grads
         dest(i) = iAz'*dAz{i};
      end
   else
      % in this case, preconditioner is used
      % we want to have 
      % trace(M^{-1}dM) + [(K\z)'*(dK*z) - (M\z)'*(dM*z)]/n
      % trace(M^{-1}dM) is not computed in this function

      % next we add the estimation
      num_grads = length(dAz);
      dest = zeros(num_grads,1);
      if(precond_split)
         % when split preconditioner is used
         % TO BE IMPLEMENTED
         % TIANSHI: I think something is not quite right here
         iAz = PRE.solve_func(PRE, iAz, 'R');
         for i = 1:num_grads
            dest(i) = iAz'*dAz{i};
         end
      else
         for i = 1:num_grads
            dest(i) = iAz'*dAz{i};
         end
      end

      % and subtract the estimation for the preconditioner
      % which is z'M^{-1}dM/dt*z
      if(num_grads > 0)
         iMz = PRE.solve_func(PRE, z);
         dMz = PRE.dvp_func(PRE, z);
      end
      for i = 1:num_grads
         dest(i) = dest(i) - iMz'*dMz{i};
      end
   end
end

function [est] = schattnormest(V, D, p)
% estimate schatten norm
   lam = abs(diag(D));
   tau = V(1,:).^2;
   est = (tau.^p)*lam.^(1/2);
end