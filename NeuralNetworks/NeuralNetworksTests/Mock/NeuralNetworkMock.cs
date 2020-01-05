using System;
using NeuralNetworks.Network;

namespace NeuralNetworksTests.Mock
{
	public class NeuralNetworkMock : NeuralNetwork
	{
		public NeuralNetworkMock(int networkInputs, int numberOfOutputs, int[] hiddenLayers = null, int inputNeuronInputs = 1) : base(networkInputs, numberOfOutputs, hiddenLayers, inputNeuronInputs)
		{
		}

		public override void BackPropagationTrain(string traningDataFile, int maxEpochs, double learningRate)
		{
			throw new NotImplementedException();
		}

		public override double[] ComputeOutput(double[] input)
		{
			throw new NotImplementedException();
		}

		public override double GetAccuracy(string fileWithData)
		{
			throw new NotImplementedException();
		}
	}
}
