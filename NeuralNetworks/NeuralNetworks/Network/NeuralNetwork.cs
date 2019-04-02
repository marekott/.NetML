using System.Linq;
using FileDeserializer.CSV;
using NeuralNetworks.Neurons;

namespace NeuralNetworks.Network
{
	public class NeuralNetwork
	{
		private readonly NeuronInputLayer[] _inputLayer;
		private readonly NeuronHiddenLayer[][] _hiddenLayer;
		private readonly NeuronOutputLayer[] _outputLayer;
		//TODO komentarz reSharpera
		private double[][][] _connectionsWeights;

		public int InputLayerNeuronsNumber => _inputLayer.Length;
		public int HiddenLayersNumber => _hiddenLayer?.Length ?? 0;
		public int OutputLayerNeuronsNumber => _outputLayer.Length;
		public double[][][] ConnectionsWeights => _connectionsWeights;

		public NeuralNetwork(int numberOfInputs, int numberOfOutputs, int[] hiddenLayers = null)
		{
			_inputLayer = new NeuronInputLayer[numberOfInputs];
			for (int i = 0; i < numberOfInputs; i++)
			{
				_inputLayer[i] = new NeuronInputLayer(i,1);
			}

			if (hiddenLayers != null)
			{
				_hiddenLayer = new NeuronHiddenLayer[hiddenLayers.Length][];
				for (int i = 0; i < _hiddenLayer.Length; i++)
				{
					_hiddenLayer[i] = new NeuronHiddenLayer[hiddenLayers[i]];
					for (int j = 0; j < _hiddenLayer[i].Length; j++)
					{
						_hiddenLayer[i][j] = new NeuronHiddenLayer(j);
					}
				}
			}
			
			_outputLayer = new NeuronOutputLayer[numberOfOutputs];
			for (int i = 0; i < numberOfOutputs; i++)
			{
				_outputLayer[i] = new NeuronOutputLayer(i);
			}
		}
		
		//TODO Refactor + metoda obliczająca potrzebną liczbę wag i porównująca z tą z pliku + bias
		public void SetWeights(Csv fileWithWeights)
		{
			int currentWeightFromFile = 0;
			var weights = fileWithWeights.Deserialize<double>();
			int numberOfConnectionsLayers = (2 + HiddenLayersNumber) - 1;
			int lastLayerIndex = numberOfConnectionsLayers - 1;
			_connectionsWeights = new double[numberOfConnectionsLayers][][];

			for (int currentLayer = 0; currentLayer < numberOfConnectionsLayers; currentLayer++)
			{
				if (currentLayer == 0)
				{
					_connectionsWeights[currentLayer] = new double[InputLayerNeuronsNumber][];
				}
				else if(currentLayer == lastLayerIndex)
				{
					_connectionsWeights[currentLayer] = new double[OutputLayerNeuronsNumber][];
				}
				else
				{
					_connectionsWeights[currentLayer] = new double[_hiddenLayer[currentLayer - 1].Length][];
				}


				for (int currentNeuron = 0; currentNeuron < _connectionsWeights[currentLayer].Length; currentNeuron++)
				{
					if (currentLayer == 0)
					{
						if (lastLayerIndex == 0)
						{
							_connectionsWeights[currentLayer][currentNeuron] = new double[OutputLayerNeuronsNumber];
						}
						else
						{
							_connectionsWeights[currentLayer][currentNeuron] = new double[_hiddenLayer[currentLayer].Length];
						}
					}

					else if (currentLayer == lastLayerIndex)
					{
						_connectionsWeights[currentLayer][currentNeuron] = new double[OutputLayerNeuronsNumber];
					}
					else
					{
						_connectionsWeights[currentLayer][currentNeuron] = new double[_hiddenLayer[currentLayer].Length];
					}

					for (int currentConnection = 0; currentConnection < _connectionsWeights[currentLayer][currentNeuron].Length; currentConnection++)
					{
						_connectionsWeights[currentLayer][currentNeuron][currentConnection] = weights[currentWeightFromFile++];
					}
				}
			}
		}

		//TODO bias + jakiś test czy liczba inputów równa się liczbie neuronów wejściowych
		public double[] ComputeOutput(double[] input)
		{
			var signalsSums = new double[_connectionsWeights.Length][];

			for (int layer = 0; layer < signalsSums.Length; layer++)
			{
				signalsSums = layer == 0 ? ComputeFirstInnerLayerResponse(input) : ComputeInnerLayerResponse(layer, signalsSums);
			}

			var networkResponse = ComputeOutputLayerResponses(signalsSums.Last());

			return networkResponse;
		}

		private int CountNeuronsInNextLayer(int currentLayer)
		{
			int numberOfNeuronsInNextLayer;
			if (_hiddenLayer != null)
			{
				numberOfNeuronsInNextLayer = _hiddenLayer.Length > currentLayer ? _hiddenLayer[currentLayer].Length : OutputLayerNeuronsNumber;
			}
			else
			{
				numberOfNeuronsInNextLayer = OutputLayerNeuronsNumber;
			}

			return numberOfNeuronsInNextLayer;
		}

		private double[][] ComputeFirstInnerLayerResponse(double[] input)
		{
			const int layer = 0;
			var numberOfNeuronsInNextLayer = CountNeuronsInNextLayer(layer);
			var signalsSums = new double[_connectionsWeights.Length][];
			signalsSums[layer] = new double[numberOfNeuronsInNextLayer];

			for (int connection = 0; connection < numberOfNeuronsInNextLayer; connection++)
			{
				for (int neuron = 0; neuron < InputLayerNeuronsNumber; neuron++)
				{
					signalsSums[layer][connection] += _inputLayer[neuron].ComputeOutput(input) *
													  _connectionsWeights[layer][neuron][connection];
				}
			}

			return signalsSums;
		}

		private double[][] ComputeInnerLayerResponse(int layer, double[][] signalsSums)
		{
			var numberOfNeuronsInNextLayer = CountNeuronsInNextLayer(layer);
			signalsSums[layer] = new double[numberOfNeuronsInNextLayer];

			for (int connection = 0; connection < numberOfNeuronsInNextLayer; connection++)
			{
				for (int neuron = 0; neuron < _hiddenLayer[layer - 1].Length; neuron++)
				{
					signalsSums[layer][connection] += _hiddenLayer[layer - 1][neuron].ComputeOutput(signalsSums[layer - 1]) *
					                                  _connectionsWeights[layer][neuron][connection];
				}
			}

			return signalsSums;
		}

		private double[] ComputeOutputLayerResponses(double[] signalsSums)
		{
			var networkResponse = new double[OutputLayerNeuronsNumber];
			for (int outNeuron = 0; outNeuron < OutputLayerNeuronsNumber; outNeuron++)
			{
				networkResponse[outNeuron] = _outputLayer[outNeuron].ComputeOutput(signalsSums);
			}

			return networkResponse;
		}
	}
}
