# Preconditioned Additive Gaussian Processes with Fourier Acceleration

## Brief
Scaling Gaussian Processes to high-dimensional, large datasets remains a significant challenge. This research code explores the use of Adaptive Factorized Nystr&#246;m (AFN) Preconditioned Gaussian Processes based on Nonequispaced FFT (NFFT) and Additive Kernels. This package is built on top of the [NFFT package](https://github.com/NFFT/nfft). We study the use of additive kernels, which are well-suited for high-dimensional problems, with NFFT-based matrix-vector multiplications, a key operation in Gaussian process inference and prediction. This code is intended for researchers in the field of Gaussian Processes and kernel methods. The code is not optimized for performance and is only used for research purposes to show the potential for near-linear time complexity and reduced memory footprint of the proposed method.

## Inatall

1. Goto folder src

2. Install fftw at ../fftw/build with --enable-shared --enable-openmp --enable-threads.

3. Install nfft at ../nfft/build with --enable-all --enable-openmp.

4. Install openblas at ../OpenBLAS/build with omp. 

5. Install experiment code with

```bash
./configure --with-openblas
make
```

## Tests

1. Add library to PATH

```bash
. environment.sh
```
or
```bash
source environment.sh
```

2. Install tests by go to corresponding folder and type

```bash
make
```

## Notes

1. This code is provided as-is, without any warranty or guarantee of correctness, reliability, or suitability for any purpose.

2. Due to the experimental nature, the code may contain memory leaks that could impact performance for large datasets or extended use.

3. Documentation may be incomplete or outdated. Please refer to the code comments for details.

## References
