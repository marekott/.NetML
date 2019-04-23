# .NetML
Project of librarie containing machine learning models written in C#.

Ready:
1. Feedforward neural network - model able to learn with backpropagation algorithm. Example:

var neuralNetwork = new FeedForwardNeuralNetwork(numberOfInputs, numberOfOutputs, hiddenLayers);
neuralNetwork.SetWeights(randomNumbers);
neuralNetwork.BackPropagationTrain(traningData, maxEpochs, learningRate);
double accuracyAfterTraning = neuralNetwork.GetAccuracy(testData);
var result = neuralNetwork.ComputeOutput(input);

TODO:
1.1 Random generated initial weights and biases.
1.2 Other activations functions in neurons classes.
1.3 Momentum


## Sources
https://visualstudiomagazine.com/Articles/2015/04/01/Back-Propagation-Using-C.aspx?Page=1
