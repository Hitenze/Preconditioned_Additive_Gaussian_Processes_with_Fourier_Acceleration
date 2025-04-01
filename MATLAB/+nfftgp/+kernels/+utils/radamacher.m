function [z] = radamacher(n,d)
%% [z] = radamacher(n,d)
%  author: Tianshi Xu <xuxx1180@umn.edu>
%  date: 12/08/23
%  brief:   create a random radamacher vector (n by d)
%
%  inputs:
%           n:  the length of the vector
%           d:  the number of vectors (default 1)
%
%  outputs:
%           z:  the random radamacher vector (n by 1)
%
%  expample:
%           z = nfftgp.kernels.utils.radamacher(20);
%           Create random radamacher vector of length 20

   if nargin < 2
      d = 1;
   end

   z = randi([0,1],n,d);
   z = z*2-1;
            
end
   