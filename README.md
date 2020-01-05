# .NetML
Librarie containing machine learning models written in C# (.NET Framework 4.6.1).

# Getting started
1. Feedforward neural network - model able to learn with backpropagation algorithm. Example:
```csharp
var neuralNetwork = new FeedForwardNeuralNetwork(numberOfInputs, numberOfOutputs, hiddenLayers);
neuralNetwork.SetWeights(randomNumbers);
neuralNetwork.BackPropagationTrain(traningData, maxEpochs, learningRate);
double accuracyAfterTraning = neuralNetwork.GetAccuracy(testData);
var result = neuralNetwork.ComputeOutput(input);
```

# Dependencies
```
Install-Package FileDeserializer -Version 1.0.0
```

# TODO
- Random generated initial weights and biases.
- Other activations functions in neurons classes.
- Momentum

## Sources
https://visualstudiomagazine.com/Articles/2015/04/01/Back-Propagation-Using-C.aspx?Page=1
