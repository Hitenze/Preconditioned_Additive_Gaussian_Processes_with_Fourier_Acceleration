function [val] = afn_logdet(PRE)
%% [val] = afn_logdet(PRE)
%  author: Tianshi Xu <xuxx1180@umn.edu>
%  date: 09/23/23
%  brief:   Get the logdeterminant of the preconditioner
%
%  input:
%           PRE:     AFN struct returned by afn_setup
%
%   output:
%           val:     the logdet value
%
%  example:
%           val = nfftgp.kernels.preconds.afn_logdet(afn);
%           Compute the logdet of the AFN preconditioner

   if(isfield(PRE, 'RAN') && ~isempty(PRE.RAN))
      val = nfftgp.kernels.preconds.ran_logdet(PRE.RAN);
      return;
   end
   
   % if AFN, we simply check the L and G
   % det(AB) = det(A)det(B)
   % thus logdet(LL') = 2*logdet(L)
   val = 2*(sum(log(diag(PRE.L11.L))) + sum(log(1./diag(PRE.GS.G))));
   
end