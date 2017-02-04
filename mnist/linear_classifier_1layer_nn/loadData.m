%% Linear regression test
%% Data read

% Function parameters: the paths to the files that contain the data
% X returns a matrix with 784 features. Each feature represents a pixel of the original image
%
function [X, y, Xtest, ytest] = loadData(XPath, yPath, XtestPath, ytestPath)
  X = (loadMNISTImages(XPath))';
  y = loadMNISTLabels(yPath);
  Xtest = (loadMNISTImages(XtestPath))';
  ytest = loadMNISTLabels(ytestPath)';
end
