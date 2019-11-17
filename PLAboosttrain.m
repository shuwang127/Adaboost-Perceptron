function [ weights, alphas, accplot ] = PLAboosttrain( data, label, ncls, iter )
% Train an adaboost classifier with perceptron.
% Input : data - N * dim
%         label - N * 1
%         ncls - 1 * 1 number of classifiers.
%         iter - 1 * 1 number of iterative for sub-classifier.
% Output: weights - ncls * (dim+1)
%         alphas - ncls * 1
%         accplot - 1 * ncls
% Shu Wang, 2019-11-17.

%% data preparation.
num = size(data, 1);
dim = size(data, 2);
dist = ones(num, 1) / num;

%% training parameters.
weights = [];
alphas = [];
accplot = [];

%% Adaboost
for t = 1 : ncls
    % train a classifier.
    [ w, ~, ~ ] = PLAtrain( data, label, iter, dist );    % train w with dist
    weights(end+1, :) = w;
    % get alpha
    h = PLA( [data, ones(num, 1)], w );             % get predictions
    I = (h ~= label);                               % get misclassification
    e = dist' * I;
    a = 0.5 * log((1 - e) / e);
    alphas(end+1, :) = a;
    % update distribution.
    dist = dist .* exp(- a * label .* h);
    dist = dist / sum(dist);
    % evaluation
    h = PLAboost(data, weights, alphas);
    accplot(end+1) = sum(h == label) / num;
end

%plot(accplot);

end

