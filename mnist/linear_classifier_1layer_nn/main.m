%% Linear regression test

%% Reset the contexts
clear; close all; clc

%% Algorithm global variables
lambda = 0.15 %% Regularization parameter
inputLayerSize = 784
hiddenLayerSize = 500
outputLayerSize = 10

%% Data loading
[Xtrain, ytrain, Xtest, ytest] = loadData("../data/train-images-idx3-ubyte", "../data/train-labels-idx1-ubyte",
										  "../data/t10k-images-idx3-ubyte", "../data/t10k-labels-idx1-ubyte");
ytest = ytest';

%% Extracting validation data from Train data
validationSetSize = 1000
Xvalid = Xtrain((size(Xtrain, 1) - validationSetSize): size(Xtrain, 1), :);
yvalid = ytrain((size(ytrain, 1) - validationSetSize) : size(ytrain, 1));
Xtrain = Xtrain(1 : size(Xtrain, 1) - validationSetSize, :);
ytrain = ytrain(1 : size(ytrain, 1) - validationSetSize);

%% Reducing train data
Xtrain = Xtrain(1:200, :);
ytrain = ytrain(1:200);

## printf("4th element is labelled: %d\n", y(4))
## printf("It's features have the value\n")
## plotNumber(X(:, 4))
## pause


%% Instanciation of the NN with random weights
Theta1 = thetaWeightInit(hiddenLayerSize, inputLayerSize + 1);
Theta2 = thetaWeightInit(outputLayerSize, hiddenLayerSize + 1);

%% Loading of already precomputed weights
%% TODO

Theta = [Theta1(:); Theta2(:)];

%% Computation of the cost and gradients of the initial state
[J, Grads] = costFunction(Theta, inputLayerSize, hiddenLayerSize,
						  outputLayerSize, Xtrain, ytrain, lambda);


%% Training of the NN with the data
%% TODO does not work with fminunc but should. See why
NewTheta = Theta;
iterLoops = 1;
options = optimset('GradObj', 'on', 'MaxIter', iterLoops);

costs = [J];
testCosts=[];
validCosts=[];
lambdas = []
modelPredTest= [];
modelPredValid=[];

% Iteration with different lambdas
for lambda = [0.001,0.003,0.01]%%,0.03,0.1,0.3,0.6,1]
  printf("Testing with lambda value: %d", lambda)
  testCost = 0;
  validCost = 0;

  %% Loading from the disk this network params
  if exist(sprintf("./params/lambda.%d.mat", lambda), "file")
	load("-binary", sprintf("./params/lambda.%d.mat", lambda))
	sprintf("Weigh file found on disk for lambda %d\n", lambda)
	sprintf("Read val for iterations: ")
	iterations
  else
	sprintf("No file for lambda %d\n", lambda)
	NewTheta = Theta;
	predictions=[0];
	validations=[0];
	iterations=[0];
  end


  easyCostFunction = @(p) costFunction(p, inputLayerSize, hiddenLayerSize,
										 outputLayerSize, Xtrain, ytrain, lambda);

  %% Training
  [NewTheta, iterCost] = fmincg(easyCostFunction, NewTheta, options);

  %% Testing the performance on test set
  [yPred, testCost] = predict.predict(NewTheta, inputLayerSize,
									  hiddenLayerSize, outputLayerSize, Xtest);
  iterSuccess = predict.success(yPred, ytest);

  %% Testing the performance on validation set
  [yPredValid, validCost] = predict.predict(NewTheta, inputLayerSize, hiddenLayerSize, outputLayerSize, Xvalid);
  iterValidSuccess = predict.success(yPredValid, yvalid);

  %% tracking model cost and performance
  costs = [costs; iterCost(size(iterCost))(1)];
  predictions = [predictions ; iterSuccess];
  validations = [validations ; iterValidSuccess];
  iterations = [iterations ; iterations(size(iterations))(1) + iterLoops];

  %% Performance display
  printf("NN Performance with lambda %d:\n", lambda)
  printf("\tTraining iterations : %d\n", iterations(size(iterations, 1)))
  printf("\tSuccess on valid set: %d\n", iterValidSuccess * 100.0)
  printf("\tSuccess on test set: %d\n", iterSuccess * 100.0)

  %% Accumulation of test and validation costs and success
  testCosts = [testCosts; testCost];
  validCosts = [validCosts; validCost];
  modelPredTest = [modelPredTest; iterSuccess];
  modelPredValid = [modelPredValid; iterValidSuccess];
  modelIterations = iterations(size(iterations))(1)
  lambdas = [lambdas; lambda];

  %% Backup to disk of this network params
  save("-binary", sprintf("./params/lambda.%d.mat", lambda), "NewTheta", "iterations",
	   "predictions", "validations")
end
myPlots.cost(lambdas, testCosts,  validCosts, modelIterations, modelPredTest, modelPredValid)
pause
