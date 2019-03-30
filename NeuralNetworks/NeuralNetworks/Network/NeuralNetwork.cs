using System;
using FileDeserializer.CSV;
using NeuralNetworks.Neurons;

namespace NeuralNetworks.Network
{
	public class NeuralNetwork
	{
		private readonly NeuronInputLayer[] _inputLayer;
		private readonly NeuronHiddenLayer[][] _hiddenLayer;
		private readonly NeuronOutputLayer[] _outputLayer;

		public int InputLayerNeuronsNumber => _inputLayer.Length;
		public int HiddenLayersNumber => _hiddenLayer.Length;
		public int OutputLayerNeuronsNumber => _outputLayer.Length;

		public NeuralNetwork(int numberOfInputs, int[] hiddenLayers, int numberOfOutputs, Csv fileWithWeights)
		{
			_inputLayer = new NeuronInputLayer[numberOfInputs];
			for (int i = 0; i < numberOfInputs; i++)
			{
				_inputLayer[i] = new NeuronInputLayer(i,1);
			}

			_hiddenLayer = new NeuronHiddenLayer[hiddenLayers.Length][];
			for (int i = 0; i < _hiddenLayer.Length; i++)
			{
				_hiddenLayer[i] = new NeuronHiddenLayer[hiddenLayers[i]];
				for (int j = 0; j < _hiddenLayer[i].Length; j++)
				{
					_hiddenLayer[i][j] = new NeuronHiddenLayer(j);
				}
			}

			_outputLayer = new NeuronOutputLayer[numberOfOutputs];
			for (int i = 0; i < numberOfOutputs; i++)
			{
				_outputLayer[i] = new NeuronOutputLayer(i);
			}
		}


		public double[] ComputeOutput(double[] input)
		{
			throw new NotImplementedException();
		}
	}
}
