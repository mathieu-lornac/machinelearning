%% Computation of the cost for a dataset
%% X is the dataset to compute the cost on
%% y is a vector containing the labels
%% lambda is the regularisation parameter
%% Theta1 is the matrix weights between input layer and hidden layer
%% Theta2 is the matric weights between hidden layer and output layer
%%
%% Returns the cost j
%% The gradients of the 2 layers
function [J, ThetaGrad] = costFunction(nn_params, input_layer_size,...
									   hidden_layer_size, output_layer_size,
									   X, y, lambda)
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

  %% Cost Computation. Linear regression formula
  %% - log(hx) if y == 1
  %% - log(1 - hx) if y == 0
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

  %% Cost regularization
  %% Taking individual square values and adding them to the global cost
  %% Exempting bias weights regularisation
  t1 = Theta1(:, 2:size(Theta1, 2));
  t2 = Theta2(:, 2:size(Theta2, 2));
  J += lambda * (sum(sum((t1 .* t1))) +  sum(sum((t2 .* t2)))) / (2 * m);


  %% Back propagation
  Delta2 = zeros(size(Theta2));
  Delta1 = zeros(size(Theta1));

  for i = 1:m
	%% Taking predictions, substraction of the output vector
	delta3 = HX(i, :)' .- ([1:outputLayerSize] == y(i))';
	%% Updating Delta2 matrix
	%% Computing delta on output layer * Activation values of hidden layer
	%% Evaluation of cell responsabilities in the hidden layer
	Delta2 = Delta2 + (delta3 * A2(i, :));

	%% Theta2' * delta3 => incorrect weight shift considering correct activation of cells
	%% mutltiplied by the derivative term of it's value
	delta2 = (Theta2' * delta3) .* sigmoidgradient([1, Z2(i, :)])';
	%% Remove bias related terms
	delta2 = delta2(2:end);
	%% Reporting deltas on layer 1
	Delta1 = Delta1 + delta2 * X(i, :);
  end
  Delta1 = Delta1 ./ m;
  Delta2 = Delta2 ./ m;

  %% Regularization of the weights
  RegTheta2 = Theta2;
  RegTheta2(:, 1) = 0;
  RegTheta1 = Theta1;
  RegTheta1(:, 1) = 0;

  %% Update of Theta values
  Theta2Grad = Delta2 + lambda .* RegTheta2 ./ m;
  Theta1Grad = Delta1 + lambda .* RegTheta1 ./ m;

  % Unroll gradients
  ThetaGrad = [Theta1Grad(:) ; Theta2Grad(:)];
end


function out = sigmoidgradient(z)
  sigs = sigmoid(z);
  out = sigs .* (1 - sigs);
end
