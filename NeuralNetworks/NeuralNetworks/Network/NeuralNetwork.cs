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

		//TODO metoda obliczająca potrzebną liczbę wag i porównująca z tą z pliku + opis metody jak w pliku powinny być ułożone wagi
		//TODO losowe ustawianie wag
		public void SetWeights(Csv fileWithWeights)
		{
			int currentWeightFromFile = 0;
			var weights = fileWithWeights.Deserialize<double>();
			ConnectionsWeights = CreateWeightsArray();

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

		//TODO metoda obliczająca potrzebną liczbę progów i porównująca z tą z pliku (chyba że nie każdy neuron musi mieć bias) + opis metody jak w pliku powinny być ułożone biasy
		//TODO losowe ustawianie biasów
		public  void SetBiases(Csv fileWithBiases)
		{
			int currentBiasFromFile = 0;
			var biases = fileWithBiases.Deserialize<double>();
			NeuronsBiases = CreateNeuronsArray();

			foreach (var layer in NeuronsBiases)
			{
				for (int currentNeuron = 0; currentNeuron < layer.Length; currentNeuron++)
				{
					layer[currentNeuron] = biases[currentBiasFromFile++];
				}
			}
		}
	}
}
