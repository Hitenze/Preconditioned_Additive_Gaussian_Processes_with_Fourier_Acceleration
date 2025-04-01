function [f, l, mu, mu2, Ys, fval, x, objective_history, parameter_history] = exact_class_gp(x0, X, Y, kernelfun, matfun, transforms, masks, optimizer, mits)
%% [f, l, mu, mu2, Ys, fval, x, objective_history, parameter_history] = exact_class_gp(x0, X, Y, kernelfun, matfun, transforms, masks, optimizer, mits)
%  author: Tianshi Xu <xuxx1180@umn.edu>
%  date: 01/30/24
%  brief: training GP classification using biased model to learn hyperparameters [f, l, mu]
%         f is the scale, l is the lengthscale, and mu is the noise level
%         note that all those values are subject to transform and the inputs
%         are values BEFORE TRANSFORM, we have transforms{i}(x0(i)) equal to the
%         actual values input into the kernel function
%
%  input:
%           x0:            Initial value before transform for ALL classes
%           X:             training data
%           Y:             training labels, 
%                          It is recommended that Y is between 1 and num_classes
%                          Ohterwise you need to manually transform the prediction output
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
%           mits:          maximum number of iterations (optional)
%  output:
%           f:       variance scale
%           l:       Lengthscale
%           mu:      Noise level
%           fval:    Final loss value
%           x:       the hyperparameters before transform

   if(nargin < 4)
      kernelfun = @nfftgp.kernels.kernels.gaussianKernelFixedNoise;
   end

   if(nargin < 5)
      matfun = @nfftgp.kernels.kernels.gaussianKernelFixedNoiseMat;
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
      mits = 100;
   end

   n = length(Y);

   % Transform the labels
   minY = min(Y);
   Y = Y - minY + 1;
   num_classes = max(Y);
   
   alpha = 0.01;
   alphas = alpha*ones(n,num_classes);

   fprintf("Num classes detected: %d\n",num_classes);
   
   alphas(sub2ind(size(alphas), (1:n)', Y)) = 1.0 + alpha;
   mu2 = log(1./alphas+1);
   Ys = log(alphas) - 0.5*mu2;

   % define the loss function
   Loss = @(hyperparameters) nfftgp.kernels.optimization.exact_class_gp_loss(hyperparameters, mu2, X, Ys, kernelfun, matfun, transforms, masks);
   
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

   x_init = zeros(3*num_classes, 1);
   if(length(x0) == 3)
      x_init(1:num_classes) = x0(1);
      x_init(num_classes+1:2*num_classes) = x0(2);
      x_init(2*num_classes+1:3*num_classes) = x0(3);
   else
      x_init = x0(1);
   end

   Problem.dim = 3*num_classes;
   Problem.name = @() 'classificationGP';    
   Problem.samples = @()40; 
   Problem.hessain_w_independent = @() false;
   Problem.type = "exact";
   
   switch(optimizer)
      case 0
         % gradient descent options
         options = optimoptions('fminunc','Algorithm','quasi-newton','SpecifyObjectiveGradient',true,'Display','iter-detailed', 'MaxFunctionEvaluations', mits);
         
         % Initialize variables to store iteration history
         options.OutputFcn = @outputFunction;
         iteration_history = [];
         objective_history = [];
         parameter_history = [];

         [x, fval] = fminunc(Loss, x_init, options);
         %[x, parameter_history] = nfftgp.kernels.optimization.adam(Loss, x0', 0.01, 100, 0.9, 0.999, 1e-08);
         %x = x';
         %fval = Loss(x);
      case 1
         % adam
         maxit = mits;
         tol = 1e-12;
         x_adam = x_init;
         [f_val, grad] = Loss(x_init);
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
         options.max_iter = mits;
         options.verbose = false;
         options.store_x = true;
         options.mem_size = 50;

         [x, info] = nfftgp.optimization.lbfgs(Loss, x_init, options);
         
         parameter_history = info.x';
         objective_history = info.cost;
         fval = info.cost(end);
      case 3
        
        %% general options for optimization algorithms 
        rng(127);  
        options.w_init = x_init;
        % set verbose mode        
        options.verbose = true;  
        % set maximum number of iterations
        options.max_iter = mits; 
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
         x = x_init;
         parameter_history = [];
         objective_history = [];
         fval = [];
      end
   
   f = zeros(num_classes, 1);
   l = zeros(num_classes, 1);
   mu = zeros(num_classes, 1);

   for i = 1:num_classes
      f(i) = transforms{1}(x(i));
      l(i) = transforms{2}(x(num_classes+i));
      mu(i) = transforms{3}(x(2*num_classes+i));
   end
      
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