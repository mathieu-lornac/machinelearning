%% Prediction for a dataset
%% Returns the labels associated
function predict = predict()
  predict.predict = @prediction;
  predict.success = @success;
end

function [y, J] = prediction(nn_params, input_layer_size,...
					   hidden_layer_size, output_layer_size,
					   X)
  %% Unrolling of the neural net Thetas
  Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)),
                 hidden_layer_size, (input_layer_size + 1));

  Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end),
                   output_layer_size, (hidden_layer_size + 1));

  %% Feed forwarding on the network to get our predictions
  m = rows(X);
  X = [ones(m, 1) X]; %% Bias input layer
  Z2 = X * Theta1';  %%[6000, 501]
  A2 = sigmoid(Z2);  %%[6000, 501]
  A2 = [ones(m, 1) A2]; %% Bias hidden layer
  Z3 = A2 * Theta2'; %%[6000, 10]
  HX = sigmoid(Z3);  %%[6000, 10]
  [vals, y] = max(HX, [], 2);

  J = 0;
  outputLayerSize = size(HX)(2);
  for i = 1:m
	yPredict = ((1:outputLayerSize) == y(i))';
	hx = HX(i, :);
	s1 = - log(hx) * yPredict;
	s2 = - log(1 - hx) * (1 - yPredict);
	J += sum(s1 + s2);
  end
  J /= m;
end

%% Returns the ratio of correctly predicted data
function e = success(refData, predData)
  e = mean(double(refData == predData));
end
