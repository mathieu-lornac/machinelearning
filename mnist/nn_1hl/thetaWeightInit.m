%% Initialisation of the weights in a random way
%% Weights are uniformly distributed between [-epsilon,epsilon]

function [out]= thetaWeightInit(dim1, dim2, epsilon = 0.12)
  out = rand(dim1, dim2) .* (2 * epsilon) - epsilon;
end
