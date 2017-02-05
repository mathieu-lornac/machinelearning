%% Loading of a model from the disk

function funcs = model()
  funcs.load = @loader;
  funcs.save = @saver;
end

function [NewTheta, predictions, validations, iterations] = loader(inputLayerSize, hiddenLayerSize,
																   outputLayerSize, lambda,
																   silent = 0)
  if silent == 0
	printf("Loading model: ./params/lambda.%d_hlsize.%d.mat\n", lambda, hiddenLayerSize)
  end
  if exist(sprintf("./params/lambda.%d_hlsize.%d.mat", lambda, hiddenLayerSize), "file")
	if silent == 0
	  printf("Weight file found on disk for lambda %d\n", lambda)
	end
   	load("-binary", sprintf("./params/lambda.%d_hlsize.%d.mat", lambda, hiddenLayerSize));
  else
  	printf("No file for lambda %d\n", lambda)
  	%% Instanciation of the NN with random weights
  	NewTheta = [thetaWeightInit(hiddenLayerSize, inputLayerSize + 1)(:); thetaWeightInit(outputLayerSize, hiddenLayerSize + 1)(:)];
  	predictions=[0];
  	validations=[0];
  	iterations=[0];
  end
  if silent == 0
	printf("Load done\n")
  end
end

%% Saves to disk model data
function saver(lambda, hiddenLayerSize, NewTheta, predictions, validations, iterations)
  printf("Saving model to: ./params/lambda.%d_hlsize.%d.mat...", lambda, hiddenLayerSize)
  save("-binary", sprintf("./params/lambda.%d_hlsize.%d.mat", lambda, hiddenLayerSize),
	   "NewTheta", "iterations", "predictions", "validations");
  printf("  done\n")
end
