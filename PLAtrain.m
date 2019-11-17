function [ wbest, Ein, acc ] = PLAtrain( data, label, iter, dist )
% Train a perceptron classifier (Pocket) with distribution for samples
% Input : data - N * dim
%         label - N * 1
%         iter - 1 * 1
%         dist - N * 1
% Output: wbest - 1 * (dim + 1)
%         Emin - In-sample error.
%         acc - Training accuracy.
% Shu Wang, 2019-11-16.

%% check the aguement.
if nargin <= 3
    dist = ones(size(label)) / length(label);
end

%% data preparation.
num = size(data, 1);
dim = size(data, 2);
data = [ data, ones(num, 1) ];  % extend data with x_0 = 1
w = zeros(1, dim + 1);          % init the weight vector

%% training parameters.
Ein = inf * ones(1, iter);

%% PLA 'Pocket' algorithm.
for cnt = 1 : iter
    % prediction
    h = PLA( data, w );      % get predictions.
    % evaluation
    err = sum(dist' * ((h - label) .^ 2));
    if (err < min(Ein)) || (cnt == 1)
        Ein(cnt) = err;
        wbest = w;
    else
        Ein(cnt) = Ein(cnt-1);
    end
    % update
    index = find(h ~= label);   % get index for prediction ~= label.
    if isempty(index)           % if no sample misclassified
        break;
    end
    idx = index(randperm(numel(index), 1)); % randomly select one sample.
    w = w + label(idx) * data(idx, :); % update the weight.
end

%% visualization.
% plot(Ein);
acc = sum(sign( data * wbest' ) == label) / num;

end
