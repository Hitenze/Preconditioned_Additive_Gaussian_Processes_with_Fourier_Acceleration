function [Y2, Y2s, std] = exact_class_gp_prediction(f, l, mu, mu2, X1, X2, Y1, kernelfun, matfun)
%% [Y2, Y2s, std] = exact_class_gp_prediction(f, l, mu, mu2, X1, X2, Y1, kernelfun, matfun)
%  author: Tianshi Xu <xuxx1180@umn.edu>
%  date: 01/30/24
%  brief: Prediction based on the current GP values
%
%  input:
%           f:             variance scale vector AFTER transform
%           l:             length scale vector AFTER transform
%           mu:            noise level vector AFTER transform
%           mu2:           fixed noise level matrix
%           X1:            training data
%           X2:            prediction data
%           Y1:            training labels
%           kernelfun:     kernel function (optional), default is gaussianKernel
%           matfun:        matrix vector multiplication function (optional), default is gaussianKernelMat
%  output:
%           Y2:            prediction labels (integer class number)
%           Y2s:           transformed labels (prediction value for each class)
%           std:           standard deviation of the prediction

    % kernel function
    % no derivative needed
    n1 = size(X1,1);
    n2 = size(X2,1);
    n = n1 + n2;

    num_classes = length(f);

    mu2_extend = zeros(n, num_classes);
    mu2_extend(1:n1,:) = mu2;

    if nargout == 3
        kernelfunc = @(f, l, mu)kernelfun([X1;X2], f, l, mu, mu2_extend, 0);
        KMat = matfun(kernelfunc(f, l, mu), [], []);

        Y2s = zeros(n2, num_classes);
        std = zeros(n2, num_classes);

        for i = 1:num_classes
            K = KMat.K{i};
            K11 = K(1:n1,1:n1);
            K12 = K(1:n1,n1+1:end);
            K22 = K(n1+1:end,n1+1:end);

            Y2s(:,i) = K12'*(K11\Y1(:,i));
            Cov = K22 - K12'*(K11\K12);
            
            std(:,i) = sqrt(abs(diag(Cov)));
        end
    else
        kernelfunc = @(f, l, mu)kernelfun([X1;X2], f, l, mu, mu2_extend, 0);
        KMat = matfun(kernelfunc(f, l, mu), [], []);

        Y2s = zeros(n2, num_classes);

        for i = 1:num_classes
            K = KMat.K{i};
            K11 = K(1:n1,1:n1);
            K12 = K(1:n1,n1+1:end);

            Y2s(:,i) = K12'*(K11\Y1(:,i));
        end
    end

    Y2 = zeros(n2, 1);

    % set Y2(i) to be the largest class in Y2s(i,:)
    for i = 1:n2
        [~, Y2(i)] = max(Y2s(i,:));
    end

end
    