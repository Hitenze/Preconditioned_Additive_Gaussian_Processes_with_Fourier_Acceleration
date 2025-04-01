function [val] = nys_logdet(PRE)
%% [val] = nys_logdet(PRE)
%  author: Tianshi Xu <xuxx1180@umn.edu>
%  date: 09/23/23
%  brief:   Get the logdeterminant of the preconditioner
%
%  input:
%           PRE:     RAN preconditioner returned by ran_setup
%
%  output:
%           val:     the logdet value
%
%  example:
%           val = nfftgp.kernels.preconds.ran_trace(ran);
%           Compute the logdet of the RAN preconditioner

   
   % if RAN, the eigenvalues are known
   % Thus, we only need to include the sigma I part
   n = size(PRE.K1, 2);
   k = size(PRE.K1, 1);

   val = sum(log(PRE.S+PRE.eta))+(n-k)*log(PRE.eta);
   
end