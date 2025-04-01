rng(815);
n = 500;
X1 = nfftgp.kernels.utils.generate_pts(n, 21);
X2 = nfftgp.kernels.utils.generate_pts(n, 21);
X3 = nfftgp.kernels.utils.generate_pts(n, 21);

windows = {[1,3],[2,5],[4,6]};
X = [X1(:,1),X2(:,1),X1(:,2),X3(:,1),X2(:,2),X3(:,2)];

f = 0.37;
l = 2.33;
mu = 0.017;
deltaf = f / 10000;
deltal = l / 10000;
deltan = mu / 10000;

%% gaussian
paramsg.windows = windows;
paramsg.kernelstrfunc = @nfftgp.kernels.kernels.gaussianKernel;
paramsg.kernelfunc = @nfftgp.kernels.kernels.gaussianKernelMat;
kernelg = nfftgp.kernels.kernels.additiveKernel(X, f, l, mu, paramsg, 1);
KMatg = nfftgp.kernels.kernels.additiveKernelMat(kernelg, [], []);

kernelgf = nfftgp.kernels.kernels.additiveKernel(X, f+deltaf, l, mu, paramsg, 1);
KMatgf = nfftgp.kernels.kernels.additiveKernelMat(kernelgf, [], []);

kernelgl = nfftgp.kernels.kernels.additiveKernel(X, f, l+deltal, mu, paramsg, 1);
KMatgl = nfftgp.kernels.kernels.additiveKernelMat(kernelgl, [], []);

kernelgn = nfftgp.kernels.kernels.additiveKernel(X, f, l, mu+deltan, paramsg, 1);
KMatgn = nfftgp.kernels.kernels.additiveKernelMat(kernelgn, [], []);

%% matern
paramsm.windows = windows;
paramsm.kernelstrfunc = @nfftgp.kernels.kernels.maternKernel;
paramsm.kernelfunc = @nfftgp.kernels.kernels.maternKernelMat;
kernelm = nfftgp.kernels.kernels.additiveKernel(X, f, l, mu, paramsm, 1);
KMatm = nfftgp.kernels.kernels.additiveKernelMat(kernelm, [], []);

kernelmf = nfftgp.kernels.kernels.additiveKernel(X, f+deltaf, l, mu, paramsm, 1);
KMatmf = nfftgp.kernels.kernels.additiveKernelMat(kernelmf, [], []);

kernelml = nfftgp.kernels.kernels.additiveKernel(X, f, l+deltal, mu, paramsm, 1);
KMatml = nfftgp.kernels.kernels.additiveKernelMat(kernelml, [], []);

kernelmn = nfftgp.kernels.kernels.additiveKernel(X, f, l, mu+deltan, paramsm, 1);
KMatmn = nfftgp.kernels.kernels.additiveKernelMat(kernelmn, [], []);

%% finite difference vs first order gradient
try
   assert(norm((KMatgf.K-KMatg.K)/deltaf-KMatg.dK{1})/norm(KMatg.dK{1}) < 1e-03);
   KMatgdf = sum(sum(KMatg.K.*KMatg.dK{1}))/norm(KMatg.K,'fro');
   assert(abs((norm(KMatgf.K,'fro') - norm(KMatg.K,'fro'))/deltaf-KMatgdf)/KMatgdf < 1e-03);
   assert(norm((KMatgl.K-KMatg.K)/deltal-KMatg.dK{2})/norm(KMatg.dK{2}) < 1e-03);
   KMatgdn = sum(sum(KMatg.K.*KMatg.dK{2}))/norm(KMatg.K,'fro');
   assert(abs((norm(KMatgl.K,'fro') - norm(KMatg.K,'fro'))/deltal-KMatgdn)/KMatgdn < 1e-03);
   assert(norm((KMatgn.K-KMatg.K)/deltan-KMatg.dK{3})/norm(KMatg.dK{3}) < 1e-03);
   KMatgdm = sum(sum(KMatg.K.*KMatg.dK{3}))/norm(KMatg.K,'fro');
   assert(abs((norm(KMatgn.K,'fro') - norm(KMatg.K,'fro'))/deltan-KMatgdm)/KMatgdm < 1e-03);
   assert(norm((KMatmf.K-KMatm.K)/deltaf-KMatm.dK{1})/norm(KMatm.dK{1}) < 1e-03);
   KMatmdf = sum(sum(KMatm.K.*KMatm.dK{1}))/norm(KMatm.K,'fro');
   assert(abs((norm(KMatmf.K,'fro') - norm(KMatm.K,'fro'))/deltaf-KMatmdf)/KMatmdf < 1e-03);
   assert(norm((KMatml.K-KMatm.K)/deltal-KMatm.dK{2})/norm(KMatm.dK{2}) < 1e-03);
   KMatmdn = sum(sum(KMatm.K.*KMatm.dK{2}))/norm(KMatm.K,'fro');
   assert(abs((norm(KMatml.K,'fro') - norm(KMatm.K,'fro'))/deltal-KMatmdn)/KMatmdn < 1e-03);
   assert(norm((KMatmn.K-KMatm.K)/deltan-KMatm.dK{3})/norm(KMatm.dK{3}) < 1e-03);
   KMatmdm = sum(sum(KMatm.K.*KMatm.dK{3}))/norm(KMatm.K,'fro');
   assert(abs((norm(KMatmn.K,'fro') - norm(KMatm.K,'fro'))/deltan-KMatmdm)/KMatmdm < 1e-03);
catch
   error("test_kernel_additive: failed");
end
fprintf("test_kernel_additive: passed\n");