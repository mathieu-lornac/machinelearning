%% Sigmoid fonction
%% Applies the sigmoid function on every element of the data given
function out = sigmoid(data)
  out = 1.0 ./ (1.0 + exp(-data));
end
