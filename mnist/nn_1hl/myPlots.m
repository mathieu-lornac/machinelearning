%% Contains different plots functions usefull to understand what is happening
%% inside the neural net
function funcs = myPlots()
  funcs.cost = @cost;
  funcs.training = @training;
end

function cost(lambda, costValid, costTest, iterations, successTest, successValid)
  subplot(3, 1, 1)
  plot(lambda, costValid, lambda, costTest);
  legend("validation set", "test set")
  title("Cost function on valid and test set")
  subplot(3, 1, 3)
  plot(lambda, iterations)
  legend("Training iterations")
  title("#Iterations for the different models")
  subplot(3, 1, 2)
  plot(lambda, successTest, lambda, successValid);
  legend("test set", "validation set")
  title("Prediction success of the different models")
end

function training(lambdas, allIterations, allPredictions, allValidations)
  subplot(1, 1, 1)
  %%	subplot(size(allIterations, 1), 1, i)
  plot(allIterations', allPredictions');

  ylabel("Prediction success")
  xlabel("iterations")
  legs = [];
  for i=1:size(lambdas, 2)
	legs = [legs; sprintf("%d", lambdas(i))];
  end
  legend(legs);
  title("Prediction success of model with diffent lambda values")
end
