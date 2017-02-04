%% Contains different plots functions usefull to understand what is happening
%% inside the neural net
function funcs = myPlots()
  funcs.cost = @cost;
end

function cost(lambda, costValid, costTest, iterations, successTest, successValid)
  subplot(3, 1, 1)
  plot(lambda, costValid, lambda, costTest);
  legend("Cost on validation set", "Cost on test set")
  title("Model performance with != regularizations")
  subplot(3, 1, 3)
  plot(lambda, iterations)
  legend("Training iterations")
  title("#Iterations for the different models")
  subplot(3, 1, 2)
  plot(lambda, successTest, lambda, successValid);
  legend("On Test set", "On Validation set")
  title("Prediction success of the different models")
end
