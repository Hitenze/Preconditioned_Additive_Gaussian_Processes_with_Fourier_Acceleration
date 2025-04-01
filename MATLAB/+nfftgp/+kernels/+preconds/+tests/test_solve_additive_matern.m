
rng(815);

n = 3000;
lg = sqrt(100);
ls = -1.5:0.1:3.5;
ls = 10.^ls;
f = 1.0;
mu = 0.01;

X1 = nfftgp.kernels.utils.generate_pts(n, 30);
X2 = nfftgp.kernels.utils.generate_pts(n, 30);

windows = {[1,2,3],[4,5,6]};
X = [X1,X2];

ntests = length(ls);
Ks = cell(ntests,1);
sols_noprecond = cell(ntests,1);
sols_afn = cell(ntests,1);
res_noprecond = cell(ntests,1);
res_afn = cell(ntests,1);
all_its_noprecond = zeros(ntests,1);
all_its_afn = zeros(ntests,1);

%% gaussian
for i = 1:ntests
   tic;
   l = ls(i);
   paramsg.windows = windows;
   paramsg.kernelstrfunc = @nfftgp.kernels.kernels.matern12Kernel;
   paramsg.kernelfunc = @nfftgp.kernels.kernels.matern12KernelMat;
   kernelg = nfftgp.kernels.kernels.additiveKernel(X, f, l, mu, paramsg, 1);
   KMatg = nfftgp.kernels.kernels.additiveKernelMat(kernelg, [], []);
   Ks{i} = KMatg.K;

   rng(127);
   maxrank = 300;
   schur_lfil = 100;
   nsamples = 500;
   nsamples_r = 5;
   AFN_PREg = nfftgp.kernels.preconds.afn_setup( kernelg, [], maxrank, schur_lfil, nsamples, nsamples_r, 1);

   %% GMRES

   rng(906);

   rhs = rand(n,1) - 0.5;
   x0 = zeros(n,1);

   maxits = 200;
   print_level = false;
   atol = false;

   precfun_noprecg = @(x) x;
   [noprecond_gmres_solg, noprecond_gmres_resg, noprecond_gmres_itsg] = nfftgp.krylovs.fgmrez ( KMatg.K, n, precfun_noprecg, rhs, x0, atol, 1e-04, maxits, maxits, print_level);
   %[noprecond_gmres_solg, noprecond_gmres_resg, noprecond_gmres_itsg] = nfftgp.kernels.krylovs.planczos( KMatg.K, n, precfun_noprecg, rhs, x0, maxits, 1e-04, atol, 2);
   sols_noprecond{i} = noprecond_gmres_solg;
   res_noprecond{i} = noprecond_gmres_resg;
   all_its_noprecond(i) = noprecond_gmres_itsg;

   precfun_afng = @(x) nfftgp.kernels.preconds.afn_solve(AFN_PREg, x);
   [afn_gmres_solg, afn_gmres_resg, afn_gmres_itsg] = nfftgp.krylovs.fgmrez ( KMatg.K, n, precfun_afng, rhs, x0, atol, 1e-04, maxits, maxits, print_level);
   %[afn_gmres_solg, afn_gmres_resg, afn_gmres_itsg] = nfftgp.kernels.krylovs.planczos( KMatg.K, n, precfun_afng, rhs, x0, maxits, 1e-04, atol, 2);
   sols_afn{i} = afn_gmres_solg;
   res_afn{i} = afn_gmres_resg;
   all_its_afn(i) = afn_gmres_itsg;

   fprintf("l = %f, its_noprecond = %f, its_afn = %f\n", l, all_its_noprecond(i), all_its_afn(i));
   toc;
end

%% Plotting
fig = figure(1);
clf;
semilogx(ls,all_its_noprecond,'r','LineWidth',2);
hold on;
semilogx(ls,all_its_afn,'b-.','LineWidth',2);
title("Mat\'ern Kernel",'fontsize',25,'interpreter','latex');
legend("NoPrecond","AFN",'location','best','fontsize',20);
xlabel("$l$",'interpreter','latex','fontsize',20);
ylabel("Iterations counts",'fontsize',20);
fig.Position = [100, 100, 800, 600];
