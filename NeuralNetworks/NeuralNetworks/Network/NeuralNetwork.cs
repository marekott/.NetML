using System;
using FileDeserializer.CSV;
using NeuralNetworks.Neurons;

namespace NeuralNetworks.Network
{
	public abstract class NeuralNetwork
	{
		protected readonly NeuronInputLayer[] InputLayer;
		protected readonly NeuronHiddenLayer[][] HiddenLayer;
		protected readonly NeuronOutputLayer[] OutputLayer;
		public double[][][] ConnectionsWeights { get; protected set; }
		public double[][] NeuronsBiases { get; protected set; }
		protected double[][] NeuronsInputs;
		protected double[][] NeuronsOutputs;
		public int InputLayerNeuronsNumber => InputLayer.Length;
		public int HiddenLayersNumber => HiddenLayer?.Length ?? 0;
		public int OutputLayerNeuronsNumber => OutputLayer.Length;

		protected NeuralNetwork(int networkInputs, int numberOfOutputs, int[] hiddenLayers, int inputNeuronInputs)
		{
			InputLayer = CreateNeuronLayer<NeuronInputLayer>(networkInputs, inputNeuronInputs);

			if (hiddenLayers != null)
			{
				HiddenLayer = CreateHiddenNeuronLayers<NeuronHiddenLayer>(hiddenLayers);
			}

			OutputLayer = CreateNeuronLayer<NeuronOutputLayer>(numberOfOutputs);

			NeuronsInputs = CreateNeuronsArray();
			NeuronsOutputs = CreateNeuronsArray();
		}
		
		private static T[] CreateNeuronLayer<T>(int size, int numberOfInputs = 0) 
		{
			var array  = new T[size];
			for (int positionFromTop = 0; positionFromTop < size; positionFromTop++)
			{
				array[positionFromTop] = numberOfInputs == 0 ? (T) Activator.CreateInstance(typeof(T), positionFromTop) : 
					(T)Activator.CreateInstance(typeof(T), positionFromTop, numberOfInputs);
			}

			return array;
		}

		private static T[][] CreateHiddenNeuronLayers<T>(int[] hiddenLayers)
		{
			var array = new T[hiddenLayers.Length][];
			for (int layer = 0; layer < array.Length; layer++)
			{
				array[layer] = new T[hiddenLayers[layer]];
				for (int positionFromTop = 0; positionFromTop < array[layer].Length; positionFromTop++)
				{
					array[layer][positionFromTop] = (T) Activator.CreateInstance(typeof(T), positionFromTop);
				}
			}

			return array;
		}

		public abstract void Train(Csv traningDataFile, int maxEpochs, double learningRate);
		public abstract double[] ComputeOutput(double[] input);
		public abstract double GetAccuracy(Csv fileWithData);

		/// <summary>
		/// Creates two dimensional array in which first dimension represents layer of network and second neurons in that layer. Input layer is skipped.
		/// </summary>
		/// <returns></returns>
		protected double[][] CreateNeuronsArray()
		{
			var numberOfConnectionsLayers = CountConnectionLayers();
			var lastLayerIndex = numberOfConnectionsLayers - 1;
			var neuronsArray = new double[numberOfConnectionsLayers][];

			for (int currentLayer = 0; currentLayer < numberOfConnectionsLayers; currentLayer++)
			{
				neuronsArray[currentLayer] = InitNeuronsDimension(currentLayer, lastLayerIndex);
			}

			return neuronsArray;
		}

		/// <summary>
		/// Help method for CreateNeuronsArray()
		/// </summary>
		/// <param name="layer"></param>
		/// <param name="lastLayerIndex"></param>
		/// <returns></returns>
		private double[] InitNeuronsDimension(int layer, int lastLayerIndex)
		{
			double[] array;
			if (layer == 0)
			{
				array = InitNeuronsFirstDimension(layer);
			}
			else if (layer == lastLayerIndex)
			{
				array = new double[OutputLayerNeuronsNumber];
			}
			else
			{
				array = new double[HiddenLayer[layer].Length];
			}

			return array;
		}

		/// <summary>
		/// Help method for CreateNeuronsArray()
		/// </summary>
		/// <param name="layer"></param>
		/// <returns></returns>
		private double[] InitNeuronsFirstDimension(int layer)
		{
			var array = HiddenLayer == null ? new double[OutputLayerNeuronsNumber] : new double[HiddenLayer[layer].Length];

			return array;
		}

		/// <summary>
		/// Creates three dimensional array in which first dimension represents layer of network, second neurons in that layer and third connections of each neuron to neurons in next layer. Output layer is skipped.
		/// </summary>
		/// <returns></returns>
		protected double[][][] CreateWeightsArray()
		{
			int numberOfConnectionsLayers = CountConnectionLayers();
			int lastLayerIndex = numberOfConnectionsLayers - 1;
			var weightsArray = new double[numberOfConnectionsLayers][][];

			for (int currentLayer = 0; currentLayer < numberOfConnectionsLayers; currentLayer++)
			{
				weightsArray[currentLayer] = InitNeuronsDimension(currentLayer);

				for (int currentNeuron = 0; currentNeuron < weightsArray[currentLayer].Length; currentNeuron++)
				{
					weightsArray[currentLayer][currentNeuron] =
						InitConnectionsDimension(currentLayer, lastLayerIndex);
				}
			}
			return weightsArray;
		}
		/// <summary>
		/// Help method for CreateWeightsArray()
		/// </summary>
		/// <param name="layer"></param>
		/// <returns></returns>
		private double[][] InitNeuronsDimension(int layer)
		{
			var array = layer == 0 ? new double[InputLayerNeuronsNumber][] : new double[HiddenLayer[layer - 1].Length][];

			return array;
		}

		/// <summary>
		/// Help method for CreateWeightsArray()
		/// </summary>
		/// <returns></returns>
		private double[] InitConnectionsDimension(int layer, int lastLayerIndex)
		{
			double[] array;
			if (layer == 0)
			{
				array = lastLayerIndex == 0 ? new double[OutputLayerNeuronsNumber] : new double[HiddenLayer[layer].Length];
			}
			else if (layer == lastLayerIndex)
			{
				array = new double[OutputLayerNeuronsNumber];
			}
			else
			{
				array = new double[HiddenLayer[layer].Length];
			}

			return array;
		}

		protected int CountConnectionLayers()
		{
			int numberOfConnectionsLayers = (2 + HiddenLayersNumber) - 1;

			return numberOfConnectionsLayers;
		}

		//TODO losowe ustawianie wag
		/// <summary>
		/// Sets initial weights based on provided file. Weights should be sorted by layers and next by neurons starting with the one at the top of the layer.
		/// </summary>
		/// <param name="fileWithWeights">For 2 inputs and 3 outputs first 3 weights will be assign to the first neuron in input layer. Next 3 for second.</param>
		public void SetWeights(Csv fileWithWeights)
		{
			var weights = fileWithWeights.Deserialize<double>();
			CheckFileLength(weights.Length, CountConnections());
			ConnectionsWeights = CreateWeightsArray();
			int currentWeightFromFile = 0;

			foreach (var layer in ConnectionsWeights)
			{
				foreach (var neuron in layer)
				{
					for (int currentConnection = 0; currentConnection < neuron.Length; currentConnection++)
					{
						neuron[currentConnection] = weights[currentWeightFromFile++];
					}
				}
			}
		}

		//TODO losowe ustawianie biasów
		/// <summary>
		/// Sets initial biases based on provided file. Biases should be sorted by layers and next by neurons starting with the one at the top of the layer. Input layer is skipped.
		/// </summary>
		/// <param name="fileWithBiases">For 2 inputs and 3 outputs first bias will be assign to the first neuron in output layer. Next one for second.</param>
		public void SetBiases(Csv fileWithBiases)
		{
			var biases = fileWithBiases.Deserialize<double>();
			CheckFileLength(biases.Length, CountBiases());
			NeuronsBiases = CreateNeuronsArray();
			int currentBiasFromFile = 0;

			foreach (var layer in NeuronsBiases)
			{
				for (int currentNeuron = 0; currentNeuron < layer.Length; currentNeuron++)
				{
					layer[currentNeuron] = biases[currentBiasFromFile++];
				}
			}
		}

		private static void CheckFileLength(int fileLength, int expectedFileLength)
		{
			if (fileLength < expectedFileLength)
			{
				throw new WrongSourceFileLengthException(expectedFileLength, fileLength);
			}
		}

		public int CountConnections()
		{
			var result = CountFirstLayerConnections();
			if (HiddenLayersNumber == 0)
			{
				return result;
			}

			var innerHiddenConnections = CountHiddenConnections();
			var lastLayerConnections = CountLastLayerConnections();

			result += innerHiddenConnections + lastLayerConnections;

			return result;
		}

		private int CountFirstLayerConnections()
		{
			var result = HiddenLayersNumber == 0 ? InputLayerNeuronsNumber * OutputLayerNeuronsNumber : InputLayerNeuronsNumber * HiddenLayer[0].Length;
			return result;
		}

		private int CountHiddenConnections()
		{
			int result = 0;
			for (int layer = 0; layer < HiddenLayersNumber - 1; layer++)
			{
				result += HiddenLayer[layer].Length * HiddenLayer[layer + 1].Length;
			}

			return result;
		}

		private int CountLastLayerConnections()
		{
			int result = HiddenLayer[HiddenLayersNumber - 1].Length * OutputLayerNeuronsNumber;
			return result;
		}

		public int CountBiases()
		{
			int result = CountHiddenNeurons() + OutputLayerNeuronsNumber;

			return result;
		}

		private int CountHiddenNeurons()
		{
			int result = 0;

			for (int layer = 0; layer < HiddenLayersNumber; layer++)
			{
				result += HiddenLayer[layer].Length;
			}

			return result;
		}
	}
}
