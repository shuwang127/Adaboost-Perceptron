function [ predicts ] = PLA( data, weight )
% Output the results for Perceptron.
% Input : data      - N * (dim + 1) padded with 1.
%         weight    - 1 * (dim + 1)
% Output: predicts  - N * 1  [-1, +1]

predicts = sign( data * weight' );

end

