#ifndef NFFT4GP_PROTOS_HEADER_H
#define NFFT4GP_PROTOS_HEADER_H

/**
 * @file _protos.h
 * @brief Wrapper of protos
 */

/* BLAS/LAPACK */

#ifdef NFFT4GP_USING_FLOAT32

#ifdef NFFT4GP_USING_MKL

#include "mkl.h"
#define NFFT4GP_DSYSV       ssysv
#define NFFT4GP_DSYEV       ssyev
#define NFFT4GP_DAXPY       saxpy
#define NFFT4GP_DDOT        sdot
#define NFFT4GP_DGEMV       sgemv
#define NFFT4GP_DSYMV       ssymv
#define NFFT4GP_DPOTRF      spotrf
#define NFFT4GP_TRTRS       strtrs
#define NFFT4GP_DTRTRI      strtri
#define NFFT4GP_DTRMM       strmm
#define NFFT4GP_DTRSM       strsm
#define NFFT4GP_DGESVD      sgesvd
#define NFFT4GP_DGESVDX     sgesvdx
#define NFFT4GP_DGEMM       sgemm
#define NFFT4GP_DGESV       sgesv
#define NFFT4GP_DGETRF      sgetrf
#define NFFT4GP_DGETRI      sgetri
#define NFFT4GP_DLANGE      slange
#define NFFT4GP_DLANTB      slantb
#define NFFT4GP_DLANSY      slansy
#define NFFT4GP_DSTEV       sstev
#else
#define NFFT4GP_DSYSV       ssysv_
#define NFFT4GP_DSYEV       ssyev_
#define NFFT4GP_DAXPY       saxpy_
#define NFFT4GP_DDOT        sdot_
#define NFFT4GP_DGEMV       sgemv_
#define NFFT4GP_DSYMV       ssymv_
#define NFFT4GP_DPOTRF      spotrf_
#define NFFT4GP_TRTRS       strtrs_
#define NFFT4GP_DTRTRI      strtri_
#define NFFT4GP_DTRMM       strmm_
#define NFFT4GP_DTRSM       strsm_
#define NFFT4GP_DGESVD      sgesvd_
#define NFFT4GP_DGESVDX     sgesvdx_
#define NFFT4GP_DGEMM       sgemm_
#define NFFT4GP_DGESV       sgesv_
#define NFFT4GP_DGETRF      sgetrf_
#define NFFT4GP_DGETRI      sgetri_
#define NFFT4GP_DLANGE      slange_
#define NFFT4GP_DLANTB      slantb_
#define NFFT4GP_DLANSY      slansy_
#define NFFT4GP_DSTEV       sstev_
#endif // #ifdef NFFT4GP_USING_MKL

#ifdef NFFT4GP_USING_ARPACK

#define NFFT4GP_DSAUPD ssaupd_
#define NFFT4GP_DSEUPD sseupd_

#endif

#else // #ifdef NFFT4GP_USING_FLOAT32

#ifdef NFFT4GP_USING_MKL

#include "mkl.h"
#define NFFT4GP_DSYSV       dsysv
#define NFFT4GP_DSYEV       dsyev
#define NFFT4GP_DAXPY       daxpy
#define NFFT4GP_DDOT        ddot
#define NFFT4GP_DGEMV       dgemv
#define NFFT4GP_DSYMV       dsymv
#define NFFT4GP_DPOTRF      dpotrf
#define NFFT4GP_TRTRS       dtrtrs
#define NFFT4GP_DTRTRI      dtrtri
#define NFFT4GP_DTRMM       dtrmm
#define NFFT4GP_DTRSM       dtrsm
#define NFFT4GP_DGESVD      dgesvd
#define NFFT4GP_DGESVDX     dgesvdx
#define NFFT4GP_DGEMM       dgemm
#define NFFT4GP_DGESV       dgesv
#define NFFT4GP_DGETRF      dgetrf
#define NFFT4GP_DGETRI      dgetri
#define NFFT4GP_DLANGE      dlange
#define NFFT4GP_DLANTB      dlantb
#define NFFT4GP_DLANSY      dlansy
#define NFFT4GP_DSTEV       dstev
#else
#define NFFT4GP_DSYSV       dsysv_
#define NFFT4GP_DSYEV       dsyev_
#define NFFT4GP_DAXPY       daxpy_
#define NFFT4GP_DDOT        ddot_
#define NFFT4GP_DGEMV       dgemv_
#define NFFT4GP_DSYMV       dsymv_
#define NFFT4GP_DPOTRF      dpotrf_
#define NFFT4GP_TRTRS       dtrtrs_
#define NFFT4GP_DTRTRI      dtrtri_
#define NFFT4GP_DTRMM       dtrmm_
#define NFFT4GP_DTRSM       dtrsm_
#define NFFT4GP_DGESVD      dgesvd_
#define NFFT4GP_DGESVDX     dgesvdx_
#define NFFT4GP_DGEMM       dgemm_
#define NFFT4GP_DGESV       dgesv_
#define NFFT4GP_DGETRF      dgetrf_
#define NFFT4GP_DGETRI      dgetri_
#define NFFT4GP_DLANGE      dlange_
#define NFFT4GP_DLANTB      dlantb_
#define NFFT4GP_DLANSY      dlansy_
#define NFFT4GP_DSTEV       dstev_
#endif // #ifdef NFFT4GP_USING_MKL

#ifdef NFFT4GP_USING_ARPACK

#define NFFT4GP_DSAUPD dsaupd_
#define NFFT4GP_DSEUPD dseupd_

#endif

#endif // #ifdef NFFT4GP_USING_FLOAT32

#ifndef NFFT4GP_USING_MKL

void NFFT4GP_DSYSV(char *uplo, int *n, int *nrhs, NFFT4GP_DOUBLE *a, int *lda, int *ipiv, NFFT4GP_DOUBLE *b, int *ldb, NFFT4GP_DOUBLE *work, int *lwork, int *info);

void NFFT4GP_DSYEV(char *jobz, char *uplo, int *n, NFFT4GP_DOUBLE *a, int *lda, NFFT4GP_DOUBLE *w, NFFT4GP_DOUBLE *work, int *lwork, int *info);

void NFFT4GP_DAXPY(int *n, const NFFT4GP_DOUBLE *alpha, const NFFT4GP_DOUBLE *x, int *incx, NFFT4GP_DOUBLE *y, int *incy);

NFFT4GP_DOUBLE NFFT4GP_DDOT(int *n, NFFT4GP_DOUBLE *x, int *incx, NFFT4GP_DOUBLE *y, int *incy);

void NFFT4GP_DGEMV(char *trans, int *m, int *n, const NFFT4GP_DOUBLE *alpha, const NFFT4GP_DOUBLE *a,
               int *lda, const NFFT4GP_DOUBLE *x, int *incx, const NFFT4GP_DOUBLE *beta, NFFT4GP_DOUBLE *y, int *incy);

void NFFT4GP_DSYMV(char *uplo, int *n, NFFT4GP_DOUBLE *alpha, NFFT4GP_DOUBLE *a, int *lda, NFFT4GP_DOUBLE *x, int *incx, NFFT4GP_DOUBLE *beta, NFFT4GP_DOUBLE *y, int *incy);

void NFFT4GP_DPOTRF(char *uplo, int *n, NFFT4GP_DOUBLE *a, int *lda, int *info);

void NFFT4GP_TRTRS(char *uplo, char *trans, char *diag, int *n, int *nrhs, NFFT4GP_DOUBLE *a, int *lda, NFFT4GP_DOUBLE *b, int *ldb, int *info);

void NFFT4GP_DTRTRI( char *uplo, char *diag, int *n, NFFT4GP_DOUBLE *a, int *lda, int *info);

void NFFT4GP_DTRMM( char *side, char *uplo, char *transa, char *diag, int *m, int *n, NFFT4GP_DOUBLE *alpha, NFFT4GP_DOUBLE *a, int *lda, NFFT4GP_DOUBLE *b, int *ldb);

void NFFT4GP_DTRSM( char *side, char *uplo, char *transa, char *diag, int *m, int *n, NFFT4GP_DOUBLE *alpha, NFFT4GP_DOUBLE *a, int *lda, NFFT4GP_DOUBLE *b, int *ldb);

void NFFT4GP_DGESVD(char *jobu, char *jobvt, int *m, int *n, NFFT4GP_DOUBLE *a, int *lda, NFFT4GP_DOUBLE *s, NFFT4GP_DOUBLE *u, int *ldu,
                int *vt, int *ldvt, NFFT4GP_DOUBLE *work, int *lwork, int *info);

void NFFT4GP_DGESVDX( char *jobu, char *jobvt, char *range, int *m, int *n, NFFT4GP_DOUBLE *a, int *lda, NFFT4GP_DOUBLE *vl, NFFT4GP_DOUBLE *vu,
                  int *il, int *iu, int *ns, NFFT4GP_DOUBLE *s, NFFT4GP_DOUBLE *u, int *ldu, int *vt, int *ldvt, NFFT4GP_DOUBLE *work, int *lwork, int *info);

void NFFT4GP_DGEMM(char *transa, char *transb, int *m, int *n, int *k, const NFFT4GP_DOUBLE *alpha, const NFFT4GP_DOUBLE *a, int *lda,
               const NFFT4GP_DOUBLE *b, int *ldb, const NFFT4GP_DOUBLE *beta, NFFT4GP_DOUBLE *c, int *ldc);

void NFFT4GP_DGESV( int *n, int *nrhs, NFFT4GP_DOUBLE *a, int *lda, int *ipiv, NFFT4GP_DOUBLE *b, int *ldb, int *info);

void NFFT4GP_DGETRF( int *m, int *n, NFFT4GP_DOUBLE *a, int *lda, int *ipiv, int *info);

void NFFT4GP_DGETRI( int *n, NFFT4GP_DOUBLE *a, int *lda, int *ipiv, NFFT4GP_DOUBLE *work, int *lwork, int *info);

NFFT4GP_DOUBLE NFFT4GP_DLANGE( char *norm, int *m, int *n, NFFT4GP_DOUBLE *A, int *lda, NFFT4GP_DOUBLE *work);

NFFT4GP_DOUBLE NFFT4GP_DLANTB( char *norm, char *uplo, char *diag, int *n, int *k, NFFT4GP_DOUBLE *A, int *lda, NFFT4GP_DOUBLE *work);

NFFT4GP_DOUBLE NFFT4GP_DLANSY( char *norm, char *uplo, int *n, NFFT4GP_DOUBLE *A, int *lda, NFFT4GP_DOUBLE *work);

void NFFT4GP_DSTEV( char *jobz, int *n, NFFT4GP_DOUBLE *d, NFFT4GP_DOUBLE *e, NFFT4GP_DOUBLE *z, int *ldz, NFFT4GP_DOUBLE *work, int *info);

#endif // #ifndef NFFT4GP_USING_MKL

#ifdef NFFT4GP_USING_ARPACK

void NFFT4GP_DSAUPD( int* ido, char* bmat, int* n, char* which, int* nev, NFFT4GP_DOUBLE* tol, NFFT4GP_DOUBLE* resid, int* ncv, NFFT4GP_DOUBLE* v, int* ldv, int* iparam, int* ipntr, NFFT4GP_DOUBLE* workd, NFFT4GP_DOUBLE* workl, int* lworkl, int* info);

void NFFT4GP_DSEUPD( int* rvec, char* howmny, int* select, NFFT4GP_DOUBLE* d, NFFT4GP_DOUBLE* z, int* ldz, NFFT4GP_DOUBLE* sigma, char* bmat, int* n, char* which, int* nev, NFFT4GP_DOUBLE* tol, NFFT4GP_DOUBLE* resid, int* ncv, NFFT4GP_DOUBLE* v, int* ldv, int* iparam, int* ipntr, NFFT4GP_DOUBLE* workd, NFFT4GP_DOUBLE* workl, int* lworkl, int* info);

#endif // #ifdef NFFT4GP_USING_ARPACK

#endif