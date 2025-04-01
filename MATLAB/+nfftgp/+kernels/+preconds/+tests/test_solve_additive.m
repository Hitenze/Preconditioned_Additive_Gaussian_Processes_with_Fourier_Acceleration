
rng(815);

n = 3000;
lg = sqrt(100);
lm = 10;
f = 1.2;
mu = 0.01;

X1 = nfftgp.kernels.utils.generate_pts(n, 30);
X2 = nfftgp.kernels.utils.generate_pts(n, 30);

windows = {[1,2,3],[4,5,6]};
X = [X1,X2];
%windows = {[1,2,3]};
%X = X1;


%% gaussian
paramsg.windows = windows;
paramsg.kernelstrfunc = @nfftgp.kernels.kernels.gaussianKernel;
paramsg.kernelfunc = @nfftgp.kernels.kernels.gaussianKernelMat;
kernelg = nfftgp.kernels.kernels.additiveKernel(X, f, lg, mu, paramsg, 1);
KMatg = nfftgp.kernels.kernels.additiveKernelMat(kernelg, [], []);

%% matern
paramsm.windows = windows;
paramsm.kernelstrfunc = @nfftgp.kernels.kernels.maternKernel;
paramsm.kernelfunc = @nfftgp.kernels.kernels.maternKernelMat;
kernelm = nfftgp.kernels.kernels.additiveKernel(X, f, lm, mu, paramsm, 1);
KMatm = nfftgp.kernels.kernels.additiveKernelMat(kernelm, [], []);

%% AFN
rng(127);
maxrank = 100;
schur_lfil = 100;
nsamples = 500;
nsamples_r = 5;
AFN_PREg = nfftgp.kernels.preconds.afn_setup( kernelg, [], maxrank, schur_lfil, nsamples, nsamples_r, 0);
AFN_PREm = nfftgp.kernels.preconds.afn_setup( kernelm, [], maxrank, schur_lfil, nsamples, nsamples_r, 0);

%% GMRES

rng(906);

rhs = rand(n,1) - 0.5;
x0 = zeros(n,1);

maxits = 50;
print_level = false;
atol = false;

precfun_noprecg = @(x) x;
[noprecond_gmres_solg, noprecond_gmres_resg, noprecond_gmres_itsg] = nfftgp.krylovs.fgmrez ( KMatg.K, n, precfun_noprecg, rhs, x0, atol, 1e-04, maxits, maxits, print_level);

precfun_noprecm = @(x) x;
[noprecond_gmres_solm, noprecond_gmres_resm, noprecond_gmres_itsm] = nfftgp.krylovs.fgmrez ( KMatm.K, n, precfun_noprecm, rhs, x0, atol, 1e-04, maxits, maxits, print_level);

precfun_afng = @(x) nfftgp.kernels.preconds.afn_solve(AFN_PREg, x);
[afn_gmres_solg, afn_gmres_resg, afn_gmres_itsg] = nfftgp.krylovs.fgmrez ( KMatg.K, n, precfun_afng, rhs, x0, atol, 1e-04, maxits, maxits, print_level);

precfun_afnm = @(x) nfftgp.kernels.preconds.afn_solve(AFN_PREm, x);
[afn_gmres_solm, afn_gmres_resm, afn_gmres_itsm] = nfftgp.krylovs.fgmrez ( KMatm.K, n, precfun_afnm, rhs, x0, atol, 1e-04, maxits, maxits, print_level);

%% Plotting
figure(1)
clf;
subplot(1,2,1);
semilogy(noprecond_gmres_resg,'r','LineWidth',2);
hold on;
semilogy(afn_gmres_resg,'b-.','LineWidth',2);
title("GMRES-Gaussian");
legend("NoPrecond","AFN",'location','best');

subplot(1,2,2);
semilogy(noprecond_gmres_resm,'r','LineWidth',2);
hold on;
semilogy(afn_gmres_resm,'b-.','LineWidth',2);
title("GMRES-Matern");
legend("NoPrecond","AFN)",'location','best');