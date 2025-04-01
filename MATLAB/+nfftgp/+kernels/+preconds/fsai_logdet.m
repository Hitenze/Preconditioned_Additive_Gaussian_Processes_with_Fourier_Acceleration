function [val] = fsai_logdet(PRE)
%% [val] = fsai_logdet(PRE)
%  author: Tianshi Xu <xuxx1180@umn.edu>
%  date: 09/23/23
%  brief:   Get the logdeterminant of the preconditioner
%
%  input:
%           PRE:     FSAI struct returned by fsai_setup
%
%  output:
%           val:     the logdet value
%
%  example:
%           val = nfftgp.kernels.preconds.fsai_logdet(fsai);
%           Compute the logdet of the FSAI preconditioner

   val = 2*sum(log(1./diag(PRE.G)));

end