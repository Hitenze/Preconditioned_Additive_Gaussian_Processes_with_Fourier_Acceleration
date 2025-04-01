function [perm2] = expand_perm(perm,n)
%% [perm2] = expand_perm(perm,n)
%  author: Tianshi Xu <xuxx1180@umn.edu>
%  date: 05/20/23
%  brief:   expand perm \in [1,n] of length k to perm2 of length n
%  
%  inputs:
%           perm:    current permutation array
%           n:       total number of points
%
%  outputs:
%           perm2:   expanded permutation array
%
%  example:
%           [permn] = nfftgp.utils.expand_perm([1,4,2],5);
%           Expand the permutation [1,4,2] to [1,4,2,3,5]
%
%           [permn] = nfftgp.utils.expand_perm([],10);
%           Create array 1:10

   perm = perm(:);
   k = length(perm);
   marker = zeros(n,1);
   
   for i = 1:k
       marker(perm(i)) = 1;
   end
   
   perm2 = zeros(n,1);
   perm2(1:k) = perm;
   
   idx = 1;
   for i = k+1:n
       while(marker(idx) ~= 0)
           idx = idx + 1;
       end
       marker(idx) = 1;
       perm2(i) = idx;
   end
   
end
   
   