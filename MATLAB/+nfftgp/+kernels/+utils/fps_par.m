function [perm, dists] = fps_par(X, k, tol)
%% [perm, dists] = fps_par(X, k, tol)
%  author: Tianshi Xu <xuxx1180@umn.edu>
%  date: 05/20/23
%  brief:   Naive FPS ordering algorithm
%
%  inputs:
%           X:       n * d matrix, n is the number of points, d is the dimension
%           k:       number of points to sample
%           tol:     (optional) tolerance of the fill-distance to stop the algorithm, default 0.0
%
%  outputs:
%           perm:    selected point indices
%           dists:   fill-distance after adding each point
%
%  example:
%           [perm, dists] = nfftgp.kernels.utils.fps_par(X, k, tol);
%           Find the k points using FPS, or stop when the fill-distance is less than tol

   if(nargin < 3)
      tol = 0.0;
   end

   %% the first step is to selected the center point
   [n, ~] = size(X);
   k = max(1, k);
   k = min(k, n);
   perm = zeros(1, k);
   [~,perm(1)] = pdist2(X, mean(X), 'Euclidean', 'Smallest', 1);

   if nargout > 1 || tol > 0
      % create the dists array when needed
      dists = zeros(1, k);
      dists(1) = max(pdist2(X(perm(1), :), X(setdiff(1:n, perm(1)), :), 'Euclidean', 'Smallest', 1));
   end

   if k > 1
      % we have more than one point, construct the distance vector
      max_dists = pdist2(X, X(perm(1), :), 'Euclidean');
      for i = 2:k
         % get the current furthest point
         [~, perm(i)] = max(max_dists);
         % update the distance
         max_dists = min(max_dists, pdist2(X, X(perm(i), :), 'Euclidean'));
         if nargout > 1 || tol > 0
            % compute the current fill distance
            if i < n
               dists(i) = max(pdist2(X(perm(1:i), :), X(setdiff(1:n, perm(1:i)), :), 'Euclidean', 'Smallest', 1));
            else
               dists(i) = 0;
            end
            if(dists(i) < tol)
               k = i;
               ind = ind(1:k);
               if nargout > 1 || tol > 0
                  dists = dists(1:k);
               end
            end
         end
      end
   end
end