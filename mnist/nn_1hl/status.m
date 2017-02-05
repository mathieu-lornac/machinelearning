%% Reads the weight files on the disk
%% Displays information on the different nets performances

%% Reset the contexts
clear; close all; clc

%% Algorithm global variables
inputLayerSize = 784
hiddenLayerSize = 500
outputLayerSize = 10

testCosts=[];
validCosts=[];
modelPredTest= [];
modelPredValid=[];
modelIterations=[];

%% Data loading
[Xtrain, ytrain, Xvalid, yvalid, Xtest, ytest] = loadData("../data/train-images-idx3-ubyte",
														  "../data/train-labels-idx1-ubyte",
														  "../data/t10k-images-idx3-ubyte",
														  "../data/t10k-labels-idx1-ubyte");


allIterations=[];
allPredictions=[];
allValidations=[];

trainingDataLen = 100
%% Iteration with different lambdas
lambdas = [0.001,0.003,0.01,0.03,0.1,0.3,0.6,1]
for lambda = lambdas
  testCost = 0;
  validCost = 0;

  [NewTheta, predictions, validations, iterations] = model.load(inputLayerSize, hiddenLayerSize, outputLayerSize, lambda, 1);


  allIterations = [allIterations; resize(iterations, trainingDataLen, 1)'];
  allPredictions = [allPredictions; resize(predictions, trainingDataLen, 1)'];
  allValidations = [allValidations; resize(validations, trainingDataLen, 1)'];

  %% Computing perf and cost on test set
  [yPred, testCost] = predict.predict(NewTheta, inputLayerSize,
									  hiddenLayerSize, outputLayerSize, Xtest);
  iterSuccess = predict.success(yPred, ytest);

  %% Computing perf and cost on validation set
  [yPredValid, validCost] = predict.predict(NewTheta, inputLayerSize,
											hiddenLayerSize, outputLayerSize, Xvalid);
  iterValidSuccess = predict.success(yPredValid, yvalid);

  %% Model performance synthesis display
  printf("##########################\n")
  printf("NN Performance with lambda %d:\n", lambda)
  printf("\tTraining iterations : %d\n", iterations(size(iterations, 1)))
  printf("\tSuccess on validation set: %d\n", validations(size(validations, 1)) * 100.0)
  printf("\tSuccess on test set: %d\n", predictions(size(predictions, 1)) * 100.0)
  printf("\tCost on validation set: %d\n", testCost)
  printf("\tCost on test set: %d\n", validCost)
  printf("\n")


  %% Synthesis over different lambdas
  testCosts = [testCosts; testCost];
  validCosts = [validCosts; validCost];
  modelPredTest = [modelPredTest; iterSuccess];
  modelPredValid = [modelPredValid; iterValidSuccess];
  modelIterations = [modelIterations; iterations(size(iterations))(1)];

end
myPlots.cost(lambdas, testCosts,  validCosts, modelIterations, modelPredTest, modelPredValid)
pause
%% Plotting model info
myPlots.training(lambdas, allIterations, allPredictions, allValidations)
pause
