function [ predicts ] = PLAboost( data, weights, alphas )
%PLABOOST Summary of this function goes here
%   Detailed explanation goes here

T = length(alphas);
[num, dim] = size(data);

out = 0;
for i = 1 : T
    out = out + alphas(i) * PLA([data, ones(num,1)], weights(i, :));
end

predicts = sign(out);

end

