function [probability] = exact_class_gp_prob(f, l, mu, mu2, X1, X2, Y2s, kernelfun, matfun, nsamples)
%% [probability] = exact_class_gp_prob(f, l, mu, mu2, X1, X2, Y2s, kernelfun, matfun, nsamples)
%  author: Tianshi Xu <xuxx1180@umn.edu>
%  date: 01/30/24
%  brief: Get the probability of each class for the prediction
%
%  input:
%           f:             variance scale vector AFTER transform
%           l:             length scale vector AFTER transform
%           mu:            noise level vector AFTER transform
%           mu2:           fixed noise level matrix
%           X1:            training data
%           X2:            prediction data
%           Y2s:           prediction values returned by exact_class_gp_prediction
%           kernelfun:     kernel function
%           matfun:        matrix vector multiplication function
%           nsamples:      number of samples
%  output:
%           probability:   the predicted probability of each class

    n1 = size(X1,1);
    n2 = size(X2,1);
    n = n1 + n2;

    num_classes = length(f);

    mu2_extend = zeros(n, num_classes);
    mu2_extend(1:n1,:) = mu2;

    kernelfunc = @(f, l, mu)kernelfun([X1;X2], f, l, mu, mu2_extend, 0);
    KMat = matfun(kernelfunc(f, l, mu), [], []);

    samples = randn(n2, num_classes, nsamples);

    for i = 1:num_classes
        K = KMat.K{i};
        K11 = K(1:n1,1:n1);
        K12 = K(1:n1,n1+1:end);
        K22 = K(n1+1:end,n1+1:end);
        Cov2 = K22 - mu(i)*eye(n2) - K12'*(K11\K12);
        %Cov2 = K22 - K12'*(K11\K12);
        Cov2 = (Cov2 + Cov2')/2;
        
        samples(:,i,:) = mvnrnd(zeros(n2,1), Cov2, nsamples)' + Y2s(:,i);
    end

    samples = exp(samples);

    probability = zeros(n2, num_classes);
    for i = 1:n2
        for j = 1:nsamples
            sumi = sum(samples(i,:,j));
            probability(i,:) = probability(i,:) + samples(i,:,j)/sumi;
        end
        probability(i,:) = probability(i,:)/nsamples;
    end
end
    