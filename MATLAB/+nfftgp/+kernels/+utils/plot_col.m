function plot_col( Ki, X, tol, newfig)
%% plot_col( Ki, X, tol, newfig)
%  author: Tianshi Xu <xuxx1180@umn.edu>
%  date: 06/06/23
%  brief:   plot a column of a matrix with its actually physical location
%
%  inputs:
%           Ki:     the column of a matrix
%           X:      2D or 3D points
%           tol:        Optinal, relative tolerance for plot.
%                       Above this value is red, below is black. 
%                       Default 0.01.
%           newfig:     Optinal, whether to open a new figure.
%                       Default 1.
%
%  example:
%           X = nfftgp.kernels.utils.generate_pts(100, 20);
%           kernelg = nfftgp.kernels.kernels.gaussianKernel(X, 0.5, 1.2, 0.33, {}, 0);
%           KMatg = nfftgp.kernels.kernels.gaussianKernelMat(kernelg, [], []);
%           K = KMatg.K;
%           nfftgp.kernels.utils.plot_col( K(:,10), X)

   [~, d] = size(X);

   if nargin < 3
      tol = 0.01;
   end

   if nargin < 4
      newfig = true;
   end

   if(d == 2)
      plot2d_col( Ki, X, tol,newfig);
   elseif(d == 3)
      plot3d_col( Ki, X, tol,newfig);
   else
      warning("dimension not supported\n");
   end

   end

   function [] = plot2d_col( Ki, X, tol,newfig)

   smallp = 5;
   largep = 20;

   n = length(Ki);

   Ki = abs(Ki);
   mk = max(Ki);

   tol = tol * mk;

   logmk = log(mk);
   logtol = log(tol);
   step = (logmk - logtol) / 6;

   tol1 = exp(logtol + step);
   tol2 = exp(logtol + 2 * step);
   tol3 = exp(logtol + 3 * step);
   tol4 = exp(logtol + 4 * step);
   tol5 = exp(logtol + 5 * step);

   % next we split the region into multiple parts
   if newfig
      figure;
   end

   if(Ki(1) < tol)
       plot(X(1,1),X(1,2), 'k.', 'MarkerSize', smallp);
   elseif(Ki(1)<tol1)
       plot(X(1,1),X(1,2), 'm.', 'MarkerSize', smallp);
   elseif(Ki(1)<tol2)
       plot(X(1,1),X(1,2), 'b.', 'MarkerSize', smallp);
   elseif(Ki(1)<tol3)
       plot(X(1,1),X(1,2), 'c.', 'MarkerSize', smallp);
   elseif(Ki(1)<tol4)
       plot(X(1,1),X(1,2), 'g.', 'MarkerSize', smallp);
   elseif(Ki(1)<tol5)
       plot(X(1,1),X(1,2), 'y.', 'MarkerSize', smallp);
   else
       plot(X(1,1),X(1,2), 'r.', 'MarkerSize', largep);
   end
   hold on;

   for i = 2:n
      
      if(Ki(i) < tol)
         plot(X(i,1),X(i,2), 'k.', 'MarkerSize', smallp);
      elseif(Ki(i)<tol1)
         plot(X(i,1),X(i,2), 'm.', 'MarkerSize', smallp);
      elseif(Ki(i)<tol2)
         plot(X(i,1),X(i,2), 'b.', 'MarkerSize', smallp);
      elseif(Ki(i)<tol3)
         plot(X(i,1),X(i,2), 'c.', 'MarkerSize', smallp);
      elseif(Ki(i)<tol4)
         plot(X(i,1),X(i,2), 'g.', 'MarkerSize', smallp);
      elseif(Ki(i)<tol5)
         plot(X(i,1),X(i,2), 'y.', 'MarkerSize', smallp);
      else
         plot(X(i,1),X(i,2), 'r.', 'MarkerSize', largep);
      end
   end

   patch1 = patch('FaceColor', 'r');
   patch2 = patch('FaceColor', 'y');
   patch3 = patch('FaceColor', 'g');
   patch4 = patch('FaceColor', 'c');
   patch5 = patch('FaceColor', 'b');
   patch6 = patch('FaceColor', 'm');
   patch7 = patch('FaceColor', 'k');

   set(patch1, 'Visible', 'off');
   set(patch2, 'Visible', 'off');
   set(patch3, 'Visible', 'off');
   set(patch4, 'Visible', 'off');
   set(patch5, 'Visible', 'off');
   set(patch6, 'Visible', 'off');
   set(patch7, 'Visible', 'off');

   legend([patch1, patch2, patch3, patch4, patch5, patch6, patch7], ...
      {">"+num2str(tol5), num2str(tol4)+"~"+num2str(tol5), num2str(tol3)+"~"+num2str(tol4), ...
      num2str(tol2)+"~"+num2str(tol3), num2str(tol1)+"~"+num2str(tol2), ...
      num2str(tol)+"~"+num2str(tol1), "<"+num2str(tol)}, 'Location', 'eastoutside');

   pos = get(gcf, 'Position');
   pos(3) = pos(3) + 0.15 * pos(3);
   set(gcf, 'Position', pos);

   end

   function [] = plot3d_col( Ki, X, tol,newfig)

   n = size(X,1);

   if nargin < 3
      newfig = 1;
   end

   smallp = 5;
   largep = 20;

   if newfig
      figure;
   end
   plot3(X(1,1),X(1,2),X(1,3),'b.', 'MarkerSize', smallp);
   hold on;

   for i = 2:n
      plot3(X(i,1),X(i,2),X(i,3),'b.', 'MarkerSize', smallp);
   end

   if ~isempty(Xk)
      k = size(Xk,1);
      for i = 1:k
         plot3(Xk(i,1),Xk(i,2),Xk(i,3),'r.', 'MarkerSize', largep);
      end
   end

end

