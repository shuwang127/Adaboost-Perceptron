function [ predicts ] = PLAboost( data, weights, alphas )
% Implementation of the boost function for perceptrons.
% Input : data - N * dim
%         weights - T * (dim+1)
%         alphas - T * 1
% Output: predicts - N * 1

T = length(alphas);
[num, dim] = size(data);

out = 0;
for i = 1 : T
    out = out + alphas(i) * PLA([data, ones(num,1)], weights(i, :));
end

predicts = sign(out);

end

