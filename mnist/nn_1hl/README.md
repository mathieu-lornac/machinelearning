
# Playing with the mnist data with a one hidden layer neural network

2 main files:
* train.m
* status.m

train.m can be used to train models. It reads from model files located in the params folder.
A model file is named with it's learning parameters (hidden layer size and regularization).

Train a model:
`
octave-cli train.m hiddenLayerSize Lamdba
`


status.m shows important information on the different models.
* Comparison of the performances with different learning rates
* Evolution of the model with it's training

Show stats:
`
octave-cli status.m hiddenLayerSize
`

# Testing with an arbitrary hidden layer size of 500

## Testing with different values of regularization parameters

Best value after 200 iterations is 0.01

Training is computationnaly costly and does not really benefit from multi cores architecture
However, it is possible to launch training with different values of regularisation parameter in //

## Performance results

`
Training iterations : 213
Success on validation set: 88.7113
Success on test set: 87.74
Cost on validation set: 0.665109
Cost on test set: 0.61958
`

# Testing with an arbitrary hidden layer size of 300

## Testing with different values of regularization parameters
Best value after 200 iterations is
