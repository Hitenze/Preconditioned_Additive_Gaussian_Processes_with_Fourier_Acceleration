% preconditioned loss vs non preconditioned loss% In this test, we evaluate the loss estimation with real datasets
% The idea of this test is similar to that of test0_ubkrylovs
clear all;

% dataset:
seed_data = 815;
n = 3000; % test data size
data_func = @(X) sum(sin(2*pi*X).*exp(X) + X.^2, 2) + 0.1 * randn(size(X,1), 1);

% lanquad options
seed = 123;
min_maxits = 1;
max_maxits = 10;

% create fixed Z
rng(321);
nvecs.nvecs = 5;
nvecs.Z = nfftgp.kernels.utils.radamacher(n, nvecs.nvecs);

solve = @nfftgp.kernels.krylovs.lanczos;
solve_p = @nfftgp.kernels.krylovs.planczos;

    
% transform
transform1 = @(val)nfftgp.kernels.optimization.transform(val, 'softplus');
transform2 = @(val)nfftgp.kernels.optimization.transform(val, 'softplus');
transform3 = @(val)nfftgp.kernels.optimization.transform(val, 'softplus');
transforms = {transform1; transform2; transform3};

% preconditioner option
maxrank = 100;
schur_lfil = 100;
nsamples = 50;
nsamples_r = 5;
% simplify the setup to only take kernel as input
precond_setup = @(kernel)nfftgp.kernels.preconds.afn_setup(kernel, [], maxrank, schur_lfil, nsamples, nsamples_r, 1);
precond_solve = @nfftgp.kernels.preconds.afn_solve;

%% create the dataset
%X = linspace(0, 1, n)';

rng(seed_data)

%X1 = nfftgp.kernels.utils.generate_pts(n, 30);
%X2 = nfftgp.kernels.utils.generate_pts(n, 30);
X1 = rand(n, 3);
X2 = rand(n, 3);
windows = {[1,2,3],[4,5,6]};
X = [X1,X2];
Y = data_func(X);

%% kernel function
% since we always require gradient we simplify it
params.windows = windows;
params.kernelstrfunc = @nfftgp.kernels.kernels.gaussianKernel;
params.kernelfunc = @nfftgp.kernels.kernels.gaussianKernelMat;
kernel_func =  @(X, f, l, mu, ~, ~)nfftgp.kernels.kernels.additiveKernel(X, f, l, mu, params, 1);
kernelmat_func = @nfftgp.kernels.kernels.additiveKernelMat;
kernel = @(f, l, mu)nfftgp.kernels.kernels.additiveKernel(X, f, l, mu, params, 1);
KMat = @(kernel)nfftgp.kernels.kernels.additiveKernelMat(kernel, [], []);

%% Problem parameter
x0 = [ nfftgp.kernels.optimization.transform(2.0, 'softplus',1),... 
        nfftgp.kernels.optimization.transform(1.0, 'softplus',1),... 
        nfftgp.kernels.optimization.transform(1.0, 'softplus',1)]';
%x0 = [ 1.0, -1.5, -0.5]';
%x0 = [ 1.0, 17.5, -37.5]';
marker0 = [1,1,1]';

f0 = transforms{1}(x0(1));
l0 = transforms{2}(x0(2));
mu0 = transforms{3}(x0(3));


% exact GP

rng(seed)
[L1, L_grad1, L1_hist, L_grad1_hist] = nfftgp.kernels.optimization.biased_gp_loss(x0, X, Y, kernel_func, kernelmat_func, transforms, marker0);

L1_dl = L_grad1(2);

% biased GP

ntests = max_maxits - min_maxits + 1;
L2s = zeros(ntests,1);
L2s_sem = zeros(ntests,1);
L2s_dl = zeros(ntests,1);
L2s_sem_dl = zeros(ntests,1);

for maxits = min_maxits:max_maxits
   rng(seed)
   [L2, L_grad2, L2_hist, L_grad2_hist] = nfftgp.kernels.optimization.biased_gp_loss(x0, X, Y, kernel_func, kernelmat_func, transforms, marker0, solve, maxits, nvecs);
   L2s(maxits-min_maxits+1) = L2;
   L2s_sem(maxits-min_maxits+1) = std(L2_hist{3})/sqrt(nvecs.nvecs);
   L2s_dl(maxits-min_maxits+1) = L_grad2(2);
   L2s_sem_dl(maxits-min_maxits+1) = std(L_grad2_hist{3}(2,:))/sqrt(nvecs.nvecs);
end

% precond-biased GP
L3s = zeros(ntests,1);
L3s_sem = zeros(ntests,1);
L3s_dl = zeros(ntests,1);
L3s_sem_dl = zeros(ntests,1);

for maxits = min_maxits:max_maxits
   rng(seed)
   [L3, L_grad3, L3_hist, L_grad3_hist] = nfftgp.kernels.optimization.biased_gp_loss(x0, X, Y, kernel_func, kernelmat_func, transforms, marker0, solve_p, maxits, nvecs, precond_setup, false);
   L3s(maxits-min_maxits+1) = L3;
   L3s_sem(maxits-min_maxits+1) = std(L3_hist{3})/sqrt(nvecs.nvecs);
   L3s_dl(maxits-min_maxits+1) = L_grad3(2);
   L3s_sem_dl(maxits-min_maxits+1) = std(L_grad3_hist{3}(2,:))/sqrt(nvecs.nvecs);
end

%%

figure(1);
clf;

plot(min_maxits:max_maxits,zeros(ntests,1),'k-', 'LineWidth', 3);

% Calculate 95% confidence intervals
ts = tinv([0.025  0.975],nvecs.nvecs-1);
L2_err = L2s - L1;
L2s_ci = L2_err' + ts'*L2s_sem';
L2s_u = L2s_ci(2,:);
L2s_l = L2s_ci(1,:);
L3_err = L3s - L1;
L3s_ci = L3_err' + ts'*L3s_sem';
L3s_u = L3s_ci(2,:);
L3s_l = L3s_ci(1,:);

fill([min_maxits:max_maxits, fliplr(min_maxits:max_maxits)], [L2s_u, fliplr(L2s_l)], [0, 0, 1], 'FaceAlpha', 0.3, 'linestyle', 'none');
hold on;
plot(min_maxits:max_maxits, L2_err, 'b-', 'LineWidth', 3); % Plot the main curve on top

fill([min_maxits:max_maxits, fliplr(min_maxits:max_maxits)], [L3s_u, fliplr(L3s_l)], [1, 0, 0], 'FaceAlpha', 0.3, 'linestyle', 'none'); 
plot(min_maxits:max_maxits, L3_err, 'r-', 'LineWidth', 3); % Plot the main curve on top
xlabel("Iteration counts","fontSize",15);
ylabel("Error","fontSize",15);
%external.symlog('y', -2);

legend("Unpreconditioned 95%","Unpreconditioned Mean","Preconditioned 95%","Preconditioned Mean",'location','best',"fontSize",15);
title("$\tilde{Z}(\mathbf{\theta})$","Interpreter","latex","fontSize",20);

figure(2);
clf;

% Calculate 95% confidence intervals
L2_err_dl = L2s_dl - L1_dl;
L2s_ci_dl = L2_err_dl' + ts'*L2s_sem_dl';
L2s_u_dl = L2s_ci_dl(2,:);
L2s_l_dl = L2s_ci_dl(1,:);
L3_err_dl = L3s_dl - L1_dl;
L3s_ci_dl = L3_err_dl' + ts'*L3s_sem_dl';
L3s_u_dl = L3s_ci_dl(2,:);
L3s_l_dl = L3s_ci_dl(1,:);


fill([min_maxits:max_maxits, fliplr(min_maxits:max_maxits)], [L2s_u_dl, fliplr(L2s_l_dl)], [0, 0, 1], 'FaceAlpha', 0.3, 'linestyle', 'none');
hold on;
plot(min_maxits:max_maxits, L2_err_dl, 'b-', 'LineWidth', 3); % Plot the main curve on top

fill([min_maxits:max_maxits, fliplr(min_maxits:max_maxits)], [L3s_u_dl, fliplr(L3s_l_dl)], [1, 0, 0], 'FaceAlpha', 0.3, 'linestyle', 'none'); 
plot(min_maxits:max_maxits, L3_err_dl, 'r-', 'LineWidth', 3); % Plot the main curve on top
xlabel("Iteration counts","fontSize",15);
ylabel("Error","fontSize",15);
%external.symlog('y', -2);

legend("Unpreconditioned 95%","Unpreconditioned Mean","Preconditioned 95%","Preconditioned Mean",'location','best',"fontSize",15);
title("$\frac{\partial \tilde{Z}(\mathbf{\theta})}{\partial l}$","Interpreter","latex","fontSize",20);

