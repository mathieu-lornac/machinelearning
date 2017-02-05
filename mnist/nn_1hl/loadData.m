%% Linear regression test
%% Data read

% Function parameters: the paths to the files that contain the data
% X returns a matrix with 784 features. Each feature represents a pixel of the original image
%
function [Xtrain, ytrain, Xvalid, yvalid, Xtest, ytest] = loadData(XPath, yPath, XtestPath, ytestPath)
  Xtrain = (loadMNISTImages(XPath))';
  ytrain = loadMNISTLabels(yPath);
  Xtest = (loadMNISTImages(XtestPath))';
  ytest = loadMNISTLabels(ytestPath)';

  %% Extracting validation data from Train data
  validationSetSize = 1000
  Xvalid = Xtrain((size(Xtrain, 1) - validationSetSize): size(Xtrain, 1), :);
  yvalid = ytrain((size(ytrain, 1) - validationSetSize) : size(ytrain, 1));
  Xtrain = Xtrain(1 : size(Xtrain, 1) - validationSetSize, :);
  ytrain = ytrain(1 : size(ytrain, 1) - validationSetSize);
  ytest = ytest';
end
