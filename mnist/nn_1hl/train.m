%% Linear regression test

%% Reset the contexts
clear; close all; clc

%% Algorithm global variables
inputLayerSize = 784
hiddenLayerSize = str2num(argv(){1})
outputLayerSize = 10
%% Regularization param
lambdaArg= str2num(argv(){2})

%% Data loading
[Xtrain, ytrain, Xvalid, yvalid, Xtest, ytest] = loadData("../data/train-images-idx3-ubyte",
														  "../data/train-labels-idx1-ubyte",
														  "../data/t10k-images-idx3-ubyte",
														  "../data/t10k-labels-idx1-ubyte");


%% Reducing train data
% Xtrain = Xtrain(9000:10000, :);
% ytrain = ytrain(9000:10000);

## printf("4th element is labelled: %d\n", y(4))
## printf("It's features have the value\n")
## plotNumber(X(:, 4))
## pause


%% Training of the NN with the data
%% TODO does not work with fminunc but should. See why
iterLoops = 100;
options = optimset('GradObj', 'on', 'MaxIter', iterLoops);


lambdas = [lambdaArg]
%% lambdas = [0.001,0.003,0.01,0.03,0.1,0.3,0.6,1]

% Iteration with different lambdas
for lambda = lambdas
  printf("Training with lambda value: %d\n", lambda)
  testCost = 0;
  validCost = 0;

  [NewTheta, predictions, validations, iterations] = model.load(inputLayerSize, hiddenLayerSize, outputLayerSize, lambda);

  %% Training
  easyCostFunction = @(p) costFunction(p, inputLayerSize, hiddenLayerSize,
										 outputLayerSize, Xtrain, ytrain, lambda);
  [NewTheta, iterCost] = fmincg(easyCostFunction, NewTheta, options);

  %% Testing the performance on test set
  [yPred, testCost] = predict.predict(NewTheta, inputLayerSize,
									  hiddenLayerSize, outputLayerSize, Xtest);
  iterSuccess = predict.success(yPred, ytest);

  %% Testing the performance on validation set
  [yPredValid, validCost] = predict.predict(NewTheta, inputLayerSize, hiddenLayerSize, outputLayerSize, Xvalid);
  iterValidSuccess = predict.success(yPredValid, yvalid);

  %% tracking model cost and performance
  predictions = [predictions ; iterSuccess];
  validations = [validations ; iterValidSuccess];
  iterations = [iterations ; iterations(size(iterations))(1) + iterLoops];

  %% Performance display
  printf("NN Performance with lambda %d:\n", lambda)
  printf("\tTraining iterations : %d\n", iterations(size(iterations, 1)))
  printf("\tSuccess on valid set: %d\n", iterValidSuccess * 100.0)
  printf("\tSuccess on test set: %d\n", iterSuccess * 100.0)

  %% Saving new model
  model.save(lambda, hiddenLayerSize, NewTheta, predictions, validations, iterations)
end
