function [perm, dists, S] = fps_seq(X,rank,rho,tol,pattern_tol)
%% [perm, dists, S] = fps_seq(X,rank,rho,tol,pattern_tol)
%  author: Tianshi Xu <xuxx1180@umn.edu>
%  date: 05/20/23
%  brief:   Naive FPS ordering algorithm
%
%  inputs:
%           X:             n * d matrix, n is the number of points, d is the dimension
%           rank:          number of points to select
%           rho:           (optional) parameter (>=1), default 2.1
%                          - large: slow, more accurate pattern
%                          - small: fast, inaccurate pattern
%           tol:           (optional) tolerance of the fill-distance to stop the algorithm, default 0.0
%           pattern_tol:   (optional) tolerance of the pattern, default -1.0
%
%   outputs:
%           perm:          selected point indices
%           dists:         fill-distance after adding each point
%           S:             the pattern
%
%  example:
%           [perm, dists] = nfftgp.kernels.utils.fps_seq(X, k, 2.1, 0.0, -1.0);
%           Find the k points using FPS, using radius 2.1, with tolerance 0.0, and pattern tolerance -1.0

   %% Check
   n = size(X,1);
   d = size(X,2);

   if (nargin < 3)
      rho = 2.1;
   end

   if (nargin < 4)
      tol = 0.0;
   end

   if (nargin < 5)
      pattern_tol = -1.0;
   end

   perm = [];
   S = [];

   if (n == 0 || d == 0)
      return
   end

   % visit by column
   XT = X';

   %% Init perm, l, p, and c
   perm = zeros(1,n);

   % lv is the vector of l
   % dists is for reference only
   lv = zeros(n,1);
   dists = zeros(n,1);

   % heap l is defined with 
   %  -ld: distance of heap
   %  -li: index
   %  -lii: reverse index
   ld = zeros(n,1);
   li = zeros(n,1);
   lii = zeros(n,1);

   % parent set p is defined with
   %  -p: the cell for parents
   %  -plen: the length of the cell of each node
   p = cell(n,1);
   plen = zeros(n,1);

   % child array ci and cd
   %  -ci: index
   %  -cd: distance
   %     sort in ascending order
   ci = cell(n,1);
   cd = cell(n,1);
   for i = 1:n
         ci{i} = [];
         cd{i} = [];
   end

   if pattern_tol > 0.0
         % pattern array ci22
         ci2 = cell(n,1);
         for i = 1:n
            ci2{i} = [];
         end
   end

   %% Set the first node to be the center one
   meanX = mean(XT,2);

   % find the one most close to the center
   distance = zeros(n,1);
   for i = 1:n
         distance(i) = norm(XT(:,i)-meanX);
   end
   [~,i1] = min(distance);

   % add to perm
   idx = 1;
   perm(idx) = i1;

   if max(distance) < tol
         perm = i1;
         dists = max(distance);
         S = 1;
         return;
   end

   % let the center node cover all other nodes
   lv(i1) = 2*rho*max(max(distance), pattern_tol);
   dists(idx) = max(distance);

   %% Init heap to be the distance to the center one
   l_heap_len = 0;
   xi1 = XT(:,i1);
   for i = 1:n
         % the i-th point
         % put into the heap
         if(i ~= i1)
            l_heap_len = l_heap_len + 1;
            ld(l_heap_len) = norm(xi1-XT(:,i));
            lv(i) = ld(l_heap_len);
            li(l_heap_len) = i;
            lii(i) = l_heap_len;
            [ld,li,lii] = ilukmaxheapadd(ld,li,lii,l_heap_len);
         end
         
   end

   % update c(hild) and p(parent)
   cd{i1} = zeros(n,1);
   ci{i1} = zeros(n,1);
   for j = 1:n
         % first put into cd and ci without order
         cd{i1}(j) = norm(xi1-XT(:,j));
         ci{i1}(j) = j;
         % put into parten without order
         p{j} = i1;
         plen(j) = 1;
         if cd{i1}(j) < pattern_tol
            ci2{i1} = [ci2{i1},j];
         end
   end

   [~,ord] = sort(cd{i1},'ascend');
   ci{i1} = ci{i1}(ord);
   cd{i1} = cd{i1}(ord);

   %% remaining indices
   while(l_heap_len > 0)
         %Get the root of the heap, remove it, and restore the heap propert
         l = ld(1); % (i,l) = pop(H)
         i = li(1);
         
         [ld,li,lii] = ilukmaxheapremove(ld,li,lii,l_heap_len);
         l_heap_len = l_heap_len - 1;
         
         if l < tol || idx >= rank
            perm = perm(1:idx);
            dists = dists(1:idx);
            break;
         end
         
         lv(i) = max(l, pattern_tol); % rho*l[i] is the search distance
         
         xi = XT(:,i);

         % Select the parent node that has all possible 
         %  children of i amongst its children, and is closest to i
         kk = 0;
         dik = inf;
         for jj = 1:plen(i)
            % search within j \in p(i)
            % such that dist(i,j)+rho*l(i)\leq rho*l(j)
            % find one that min dist(i,j)
            
            j = p{i}(jj);
            
            dij = norm(xi-XT(:,j));
            if dij + rho*lv(i) <= rho*lv(j)
               % candidate
               if dij < dik
                     kk = jj;
                     dik = dij;
               end
            end
         end
         
         % Loop through those children of k that 
         %  are close enough to k to possibly be children of i
         if kk > 0
            k = p{i}(kk);
            
            xk = XT(:,k);
            
            lencik = length(ci{k});
            for jj = 1:lencik
               j = ci{k}(jj);
               djk = norm(xk-XT(:,j));
               if djk > dik + rho*lv(i)
                     break;
               end
               
               % decrease (H,j,dist(i,j))
               dij = norm(xi-XT(:,j));
               [ld,li,lii] = ilukmaxheapdecrease(ld,li,lii,l_heap_len,j,dij);
               
               if dij < rho*lv(i)
                     % push (c[i],j)
                     % push (p[j],i)
                     % this is slow, allocate in advance
                     ci{i} = [ci{i};j];
                     cd{i} = [cd{i};dij];
                     if dij < pattern_tol
                        ci2{i} = [ci2{i},j];
                     end
                     p{j} = [p{j};i];
                     plen(j) = plen(j) + 1;
               end
               
            end
         end

         idx = idx + 1;
         perm(idx) = i;
         dists(idx) = l;
         [~,ord] = sort(cd{i},'ascend');
         ci{i} = ci{i}(ord);
         cd{i} = cd{i}(ord);
   end

   S = eye(idx,idx);

   if pattern_tol  > 0
         rperm = - ones(n,1);
         for i = 1:idx
            rperm(perm(i)) = i; 
         end

         for ii = 1:idx
            i = perm(ii);
            for jj = 1:length(ci2{i})
               j = rperm(ci2{i}(jj));
               if j >= ii
                     S(ii,j) = 1;
                     S(j,j) = 1;
               end
            end
         end
   elseif pattern_tol == 0
         rperm = - ones(n,1);
         for i = 1:idx
            rperm(perm(i)) = i; 
         end

         for ii = 1:idx
            i = perm(ii);
            for jj = 1:length(ci{i})
               j = rperm(ci{i}(jj));
               if j >= ii
                     S(ii,j) = 1;
                     S(j,j) = 1;
               end
            end
         end
   end

end

function [heap,idx,iidx] = ilukmaxheapadd(heap,idx,iidx,len)
   %function [heap,idx,iidx] = ilukmaxheapadd(heap,idx,iidx,len)
   %   Add to heap, the value should be already at the
   %   end of the heap
   
   while (len > 1)
       p = floor(len / 2);
       if(heap(p) < heap(len))
           idx([p,len]) = idx([len,p]);
           iidx([idx(p),idx(len)]) = [p,len];
           heap([p,len]) = heap([len,p]);
           len = p;
       else
           break;
       end
   end
   
end   

function [heap,idx,iidx] = ilukmaxheapdecrease(heap,idx,iidx,len,a,b)
   %[heap,idx,iidx] = ilukmaxheapdecrease(heap,idx,iidx,len,a,b)
   %   decrease the value of the heap with index a to b
   
   p = iidx(a);
   if(p <= 0 || heap(p) <= b)
       % warning("not in heap or heap value increase, function will terminate.");
       % do nothing
       return;
   end
   heap(p) = b;
   l = 2*p;
   
   while (l < len)
       r = 2*p+1;
   
       if(r < len && heap(l) < heap(r))
           l = r;
       end
   
       if(heap(l) > heap(p))
           idx([p,l]) = idx([l,p]);
           iidx([idx(p),idx(l)]) = [p,l];
           heap([p,l]) = heap([l,p]);
           p = l;
           l = 2*p;
       else
           break;
       end
   end
   
end 

function [heap,idx,iidx] = ilukmaxheapremove(heap,idx,iidx,len)
   %[heap,idx,iidx] = ilukmaxheapremove(heap,idx,iidx,len)
   %   Remove from heap
   
   idx([1,len]) = idx([len,1]);
   iidx([idx(1),idx(len)]) = [1,0];
   heap([1,len]) = heap([len,1]);
   
   p = 1;
   l = 2;
   
   while (l < len)
       r = 2*p+1;
   
       if(r < len && heap(l) < heap(r))
           l = r;
       end
   
       if(heap(l) > heap(p))
           idx([p,l]) = idx([l,p]);
           iidx([idx(p),idx(l)]) = [p,l];
           heap([p,l]) = heap([l,p]);
           p = l;
           l = 2*p;
       else
           break;
       end
   end
   
end   
   