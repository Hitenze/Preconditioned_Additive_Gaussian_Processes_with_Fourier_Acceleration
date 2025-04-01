function [val] = chol_logdet(PRE)
%% [val] = chol_logdet(PRE)
%  author: Tianshi Xu <xuxx1180@umn.edu>
%  date: 09/23/23
%  brief:   Get the logdeterminant of the CHOL preconditioner
%
%  input:
%           PRE:     CHOL preconditioner returned by chol_setup.m
%
%  output:
%           val:     the logdet value
%
%  example:
%           val = nfftgp.kernels.preconds.chol_logdet(chol);
%           Compute the logdet of the CHOL preconditioner

   val = 2*sum(log(diag(PRE.L)));
    
end