function [f, l, mu, fval, x, objective_history, parameter_history] = biased_gp(x0, X, Y, kernelfun, matfun, transforms, masks, optimizer, opt_maxits, solver, maxits, nvecs, precond_setup, precond_split)
%% [f, l, mu, fval, x, objective_history, parameter_history] = biased_gp(x0, X, Y, kernelfun, matfun, transforms, masks, optimizer, opt_maxits, solver, maxits, nvecs, precond_setup, precond_split)
%  author: Tianshi Xu <xuxx1180@umn.edu>
%  date: 06/16/23
%  brief: training GP using biased model to learn hyperparameters [f, l, mu]
%         f is the scale, l is the lengthscale, and mu is the noise level
%         note that all those values are subject to transform and the inputs
%         are values BEFORE TRANSFORM, we have transforms{i}(x0(i)) equal to the
%         actual values input into the kernel function
%
%  input:
%           x0:            Initial value before transform, assume to be column vector
%           X:             training data
%           Y:             training labels
%           kernelfun:     kernel function (optional), default is gaussianKernel
%           matfun:        matrix vector multiplication function (optional), default is gaussianKernelMat
%           transforms:    transform functions (change the problem to a unconstrained problem) return both value and gradient (optional)
%                          default is the softplus function
%                          single function: all the hyperparameters share the same transform
%                          cell: each hyperparameter has its own transform
%           masks:         Mask array, same length as x0 (optional) 
%                          If marker(i) = 0 then x(i) is fixed, otherwise x(i) is free.
%                          By default all the hyperparameters are free
%           solver:        solver for the linear system (optional)
%           optimizer:     optimizer (optional)
%                          0. LBFGS (funcmin in matlab)
%                          1. Adam
%           opt_maxits:    maximum number of iterations for the optimizer (optional)
%           solver:        solver for the linear system (optional)
%           maxits:        maximum number of iterations (optional)
%           nvecs:         number of Lanczos vectors (optional)
%           precond_setup: preconditioner setup function, a wraper that only takes kernel as input (optional)
%                          by default no preconditioner is used
%           precond_split: if the preconditioner support split solve (optional)
%                          if support, we can save memory in the situation where full orthgonalization is used
%  output:
%           f:       variance scale
%           l:       Lengthscale
%           mu:      Noise level
%           fval:    Final loss value
%           x:       the hyperparameters before transform

   if(nargin < 4)
      kernelfun = @nfftgp.kernels.kernels.gaussianKernel;
   end

   if(nargin < 5)
      matfun = @nfftgp.kernels.kernels.gaussianKernelMat;
   end

   if(nargin < 6)
      transform1 = @(val)nfftgp.kernels.optimization.transform(val, 'softplus');
      transform2 = @(val)nfftgp.kernels.optimization.transform(val, 'softplus');
      transform3 = @(val)nfftgp.kernels.optimization.transform(val, 'softplus');
      transforms = {transform1; transform2; transform3};
   end

   if(length(transforms) == 1)
      transforms = {transforms; transforms; transforms};
   end
   
   if(nargin < 7)
      masks = ones(length(x0), 1);
   end
   masks(masks~=0) = 1;

   if(nargin < 8)
      optimizer = 0;
   end

   if(nargin < 9)
      opt_maxits = 100;
   end

   if(nargin < 10)
      solver = [];
   end
   
   if(nargin < 11)
      maxits = 5;
   end
   
   if(nargin < 12)
      nvecs = 5;
   end
   
   % if no preconditioner is provided we do not use preconditioning
   if(nargin < 13)
      precond_setup = [];
   end
   
   if(nargin < 14)
      % by default we assume that split solve is not supported
      precond_split = false;
   end
   
   % define the loss function
   Loss = @(hyperparameters) nfftgp.kernels.optimization.biased_gp_loss(hyperparameters, X, Y, kernelfun, matfun, transforms, masks, solver, maxits, nvecs, precond_setup, precond_split);
   %Cost = @(hyperparameters) nfftgp.kernels.optimization.biased_gp_loss_only(hyperparameters, X, Y, kernelfun, matfun, transforms, masks, solver, maxits, nvecs, precond_setup, precond_split);

   Problem.cost = @ploss;
   function f = ploss(x)
      [f,~] = Loss(x);
   end
         
   Problem.grad = @pgrad;
   function g = pgrad(x)
      [~, g] = Loss(x);
   end  

   Problem.full_grad = @pfull_grad;
   function g = pfull_grad(x)
      g = pgrad(x);
   end

   Problem.dim = 3;
   Problem.name = @() 'biasedGP';    
   Problem.samples = @()40; 
   Problem.hessain_w_independent = @() false;
   Problem.type = "afn";
   
   switch(optimizer)
      case 0
         % gradient descent options
         options = optimoptions('fminunc','Algorithm','quasi-newton','SpecifyObjectiveGradient',true,'Display','iter-detailed', 'MaxFunctionEvaluations', opt_maxits);
         
         % Initialize variables to store iteration history
         options.OutputFcn = @outputFunction;
         iteration_history = [];
         objective_history = [];
         parameter_history = [];

         [x, fval] = fminunc(Loss, x0, options);
         %[x, parameter_history] = nfftgp.kernels.optimization.adam(Loss, x0', 0.01, 100, 0.9, 0.999, 1e-08);
         %x = x';
         %fval = Loss(x);
      case 1
         % adam
         maxit = opt_maxits;
         tol = 1e-12;
         x_adam = x0;
         [f_val, grad] = Loss(x0);
         c_adam = f_val;
         m = 0; v = 0;
         beta1 = 0.9; beta2 = 0.999; eps = 1.e-8;
         lr = 0.1;
         for it = 2:maxit+1
            x_old = x_adam(:,it-1);
            g = grad;
            m = beta1*m + (1-beta1)*g;
            denom_m = 1 - beta1^(it-1);
            v = beta2*v + (1-beta2)*g.^2;
            denom_v = 1 - beta2^(it-1);
            x_new = x_old - lr * (m/denom_m) ./ (sqrt(v/denom_v) + eps);
            x_adam(:,it) = x_new;
            [f_val, grad] = Loss(x_new);
            c_adam(it) = f_val;
            rho = norm(grad);
            %lr = exp(-it*0.1) * lr;
            %if(mod(it, 10) == 0)
            %   lr = lr * 0.5;
            %end
            fprintf('it %3d  ff %e  rho %e\n',it-1, c_adam(it),rho);
            if rho < tol; break, end
         end
         parameter_history = x_adam';
         objective_history = c_adam;
         fval = f_val;
         x = x_new;
      case 2
         
         % extract options
         options.step_init = 0.01;
         options.tol = 1.0e-12;
         options.max_iter = opt_maxits;
         options.verbose = false;
         options.store_x = true;
         options.mem_size = 50;

         [x, info] = nfftgp.optimization.lbfgs(Loss, x0, options);
         
         parameter_history = info.x';
         objective_history = info.cost;
         fval = info.cost(end);
      case 3
        
        %% general options for optimization algorithms 
        rng(127);  
        options.w_init = x0;
        % set verbose mode        
        options.verbose = true;  
        % set maximum number of iterations
        options.max_iter = opt_maxits; 
        % set store history of solutions
        options.store_w = true;
     
     
        %% many other optimizers can be found in GDLibrary
        %% perform GD with backtracking line search 
        %options.step_alg = 'backtracking';
        [x, info_list_gd] = gd(Problem, options); 
        %save('info_list_gd.mat','info_list_gd')
        fval = info_list_gd.cost(end);
        
        parameter_history = info_list_gd.w';
        objective_history = info_list_gd.cost;
        %% perform L-BFGS with strong wolfe line search
        %options.step_alg = 'strong_wolfe';                  
        %[w_lbfgs, info_list_lbfgs] = lbfgs(Problem, options);                  
        %save('info_list_lbfgs.mat','info_list_lbfgs')
        %% plot all
        %close all;
        
        % display epoch vs cost/gnorm
        %display_graph('iter','cost', {'GD-BKT', 'LBFGS-WOLFE'}, {w_gd, w_lbfgs}, {info_list_gd, info_list_lbfgs});
        % display optimality gap vs grads
        %display_graph('iter','gnorm', {'GD-BKT', 'LBFGS-WOLFE'}, {w_gd, w_lbfgs}, {info_list_gd, info_list_lbfgs});
      case 4
         %% simple gradient descent
         % adam
         maxit = opt_maxits;
         tol = 1e-12;
         x_gd = x0;
         [f_val, grad] = Loss(x0);
         objective_history = f_val;
         lr = 0.2;
         for it = 2:maxit+1
            x_old = x_gd(:,it-1);
            g = grad;
            x_new = x_old - lr * g;
            x_gd(:,it) = x_new;
            [f_val, grad] = Loss(x_new);
            objective_history(it) = f_val;
            rho = norm(grad);
            %lr = exp(-it*0.1) * lr;
            %if(mod(it, 10) == 0)
            %   lr = lr * 0.5;
            %end
            fprintf('it %3d  ff %e  rho %e\n',it-1, objective_history(it),rho);
            if rho < tol; break, end
         end
         parameter_history = x_gd';
         fval = f_val;
         x = x_new;
   end
      
   f = transforms{1}(x(1));
   
   l = transforms{2}(x(2));
   
   mu = transforms{3}(x(3));
      
   % Define the custom output function
   function stop = outputFunction(x, optimValues, state)
      stop = false;  % Continue optimization
      
      if isequal(state, 'iter')
         % This block will execute at each iteration
         iteration_history = [iteration_history; optimValues.iteration];
         objective_history = [objective_history; optimValues.fval];
         parameter_history = [parameter_history; x'];
      end
   end
   
end