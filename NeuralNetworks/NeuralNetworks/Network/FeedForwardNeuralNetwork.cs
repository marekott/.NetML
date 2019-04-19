using System;
using System.Linq;
using FileDeserializer.CSV;

namespace NeuralNetworks.Network
{
	public class FeedForwardNeuralNetwork : NeuralNetwork
	{
		public FeedForwardNeuralNetwork(int networkInputs, int numberOfOutputs, int[] hiddenLayers = null, int inputNeuronInputs = 1) : base(networkInputs, numberOfOutputs, hiddenLayers, inputNeuronInputs)
		{
		}

		/// <summary>
		/// Method is making corrections to weights and biases of connections between neurons using Backpropagation algorithm.
		/// </summary>
		/// <param name="traningDataFile">File will be deserialized to [,] array. Each row have to contain InputLayerNeuronsNumber + OutputLayerNeuronsNumber elements starting with data which will be passed on input of network.</param>
		/// <param name="maxEpochs">Maximum number of traning iterations.</param>
		/// <param name="learningRate"></param>
		public override void BackPropagationTrain(Csv traningDataFile, int maxEpochs, double learningRate)
		{
			var traningData = traningDataFile.DeserializeByRows<double>();
			CheckFileLength(traningData.GetLength(1), InputLayerNeuronsNumber+OutputLayerNeuronsNumber);
			var connectionWeightDelta = CreateWeightsArray();
			var connectionPartialDerivative = CreateWeightsArray();
			var localGradientErrorSignal = CreateNeuronsArray();
			var biasDelta = CreateNeuronsArray();
			var biasPartialDerivative = CreateNeuronsArray();
			int epoch = 0;

			var randomIndexes = GenerateSortedIndexes(traningData.GetLength(0));

			while (epoch < maxEpochs)
			{
				KnuthShuffle(randomIndexes);

				for (int trainingRecord = 0; trainingRecord < traningData.GetLength(0); trainingRecord++)
				{
					var inputs = ExtractInput(traningData, randomIndexes[trainingRecord]);
					var correctResults = ExtractCorrectResponse(traningData, randomIndexes[trainingRecord]);

					var currentOutput = ComputeOutput(inputs);

					//3.4
					ComputeLocalGradientErrorSignal(localGradientErrorSignal, correctResults, currentOutput);
					
					//3.3
					ComputeConnectionPartialDerivative(connectionPartialDerivative, localGradientErrorSignal);
					
					//3.2
					ComputeConnectionWeightDelta(connectionWeightDelta, connectionPartialDerivative, learningRate);

					//3.1 
					UpdateWeights(connectionWeightDelta);

					if (NeuronsBiases != null)
					{
						//3.5
						ComputeBiasPartialDerivative(biasPartialDerivative, localGradientErrorSignal);
						ComputeBiasDelta(biasDelta, biasPartialDerivative, learningRate);
						UpdateBiases(biasDelta);
					}

				}
				epoch++;
			}
		}

		private static void CheckFileLength(int fileLength, int expectedFileLength)
		{
			if (fileLength != expectedFileLength)
			{
				throw new WrongSourceFileLengthException(expectedFileLength, fileLength);
			}
		}

		private static int[] GenerateSortedIndexes(int length)
		{
			int[] indexes = new int[length];
			for (int i = 0; i < indexes.Length; i++)
			{
				indexes[i] = i;
			}

			return indexes;
		}

		private static void KnuthShuffle(int[] sequence)
		{
			Random random = new Random();
			for (int i = 0; i < sequence.Length; i++)
			{
				var randomIndex = random.Next(i, sequence.Length);
				int temp = sequence[randomIndex];
				sequence[randomIndex] = sequence[i];
				sequence[i] = temp;
			}
		}

		public override double[] ComputeOutput(double[] input)
		{
			CheckIfWeightsAreSet();
			CheckFileLength(input.Length, InputLayerNeuronsNumber);
			var signalsSums = new double[ConnectionsWeights.Length][];

			for (int layer = 0; layer < signalsSums.Length; layer++)
			{
				signalsSums = layer == 0 ? ComputeFirstInnerLayerResponse(input) : ComputeInnerLayerResponse(layer, signalsSums);
			}

			var networkResponse = ComputeOutputLayerResponses(signalsSums.Last());

			return networkResponse;
		}

		private void CheckIfWeightsAreSet()
		{
			if (ConnectionsWeights == null)
			{
				throw new WeightsNotInitializedException();
			}
		}

		private double[][] ComputeFirstInnerLayerResponse(double[] input)
		{
			const int layer = 0;
			var numberOfNeuronsInNextLayer = CountNeuronsInNextLayer(layer);
			var signalsSums = new double[ConnectionsWeights.Length][];
			signalsSums[layer] = new double[numberOfNeuronsInNextLayer];

			for (int connection = 0; connection < numberOfNeuronsInNextLayer; connection++)
			{
				for (int neuron = 0; neuron < InputLayerNeuronsNumber; neuron++)
				{
					signalsSums[layer][connection] += InputLayer[neuron].ComputeOutput(input) *
													  ConnectionsWeights[layer][neuron][connection];
				}

				if (NeuronsBiases != null)
				{
					signalsSums[layer][connection] += NeuronsBiases[layer][connection];
				}
			}

			for (int neuron = 0; neuron < NeuronsInputs[layer].Length; neuron++)
			{
				NeuronsInputs[layer][neuron] = signalsSums[layer][neuron];
			}

			return signalsSums;
		}

		private double[][] ComputeInnerLayerResponse(int layer, double[][] signalsSums)
		{
			var numberOfNeuronsInNextLayer = CountNeuronsInNextLayer(layer);
			signalsSums[layer] = new double[numberOfNeuronsInNextLayer];

			for (int connection = 0; connection < numberOfNeuronsInNextLayer; connection++)
			{
				for (int neuron = 0; neuron < HiddenLayer[layer - 1].Length; neuron++)
				{
					signalsSums[layer][connection] += HiddenLayer[layer - 1][neuron].ComputeOutput(signalsSums[layer - 1]) *
													  ConnectionsWeights[layer][neuron][connection];
					NeuronsOutputs[layer - 1][neuron] =
						HiddenLayer[layer - 1][neuron].ComputeOutput(signalsSums[layer - 1]);
				}

				if (NeuronsBiases != null)
				{
					signalsSums[layer][connection] += NeuronsBiases[layer][connection];
				}
			}

			for (int neuron = 0; neuron < NeuronsInputs[layer].Length; neuron++)
			{
				NeuronsInputs[layer][neuron] = signalsSums[layer][neuron];
			}

			return signalsSums;
		}

		private int CountNeuronsInNextLayer(int currentLayer)
		{
			int numberOfNeuronsInNextLayer;
			if (HiddenLayer != null)
			{
				numberOfNeuronsInNextLayer = HiddenLayer.Length > currentLayer ? HiddenLayer[currentLayer].Length : OutputLayerNeuronsNumber;
			}
			else
			{
				numberOfNeuronsInNextLayer = OutputLayerNeuronsNumber;
			}

			return numberOfNeuronsInNextLayer;
		}

		private double[] ComputeOutputLayerResponses(double[] signalsSums)
		{
			var networkResponse = new double[OutputLayerNeuronsNumber];
			for (int outNeuron = 0; outNeuron < OutputLayerNeuronsNumber; outNeuron++)
			{
				networkResponse[outNeuron] = OutputLayer[outNeuron].ComputeOutput(signalsSums);
				NeuronsOutputs[NeuronsOutputs.Length - 1][outNeuron] = networkResponse[outNeuron]; //TODO to pole jest z dupy, zrob to z glowa
			}

			return networkResponse;
		}

		private void ComputeLocalGradientErrorSignal(double[][] localGradientErrorSignal, double[] correctResults, double[] currentOutput)
		{
			for (int layer = localGradientErrorSignal.Length - 1; layer >= 0; layer--)
			{
				for (int neuron = 0; neuron < localGradientErrorSignal[layer].Length; neuron++)
				{
					double derivative;
					if (layer == localGradientErrorSignal.Length - 1)
					{
						var errorSignal = correctResults[neuron] - currentOutput[neuron]; //3.6
						derivative = (1 - currentOutput[neuron]) * currentOutput[neuron]; //3.7
						localGradientErrorSignal[layer][neuron] = errorSignal * derivative; //3.5
					}
					else
					{
						derivative = (1 + NeuronsOutputs[layer][neuron]) *
						             (1 - NeuronsOutputs[layer][neuron]); //3.8
						double sum = 0.0;
						for (int connection = 0; connection < ConnectionsWeights[layer + 1][neuron].Length; connection++)
						{
							sum = localGradientErrorSignal[layer][neuron] * ConnectionsWeights[layer + 1][neuron][connection];
						}
						localGradientErrorSignal[layer][neuron] = sum * derivative; //3.5
					}
				}
			}
		}

		private void ComputeConnectionPartialDerivative(double[][][] connectionPartialDerivative, double[][] localGradientErrorSignal)
		{
			for (int layer = 0; layer < connectionPartialDerivative.Length; layer++)
			{
				for (int neuron = 0; neuron < connectionPartialDerivative[layer].Length; neuron++)
				{
					for (int connection = 0; connection < connectionPartialDerivative[layer][neuron].Length; connection++)
					{
						//TODO tu może być błąd z indeksowaniem
						connectionPartialDerivative[layer][neuron][connection] = NeuronsInputs[layer][connection] * localGradientErrorSignal[layer][connection];
					}
				}
			}
		}

		private void ComputeConnectionWeightDelta(double[][][] connectionWeightDelta, double[][][] connectionPartialDerivative, double learningRate)
		{
			for (int layer = 0; layer < connectionWeightDelta.Length; layer++)
			{
				for (int neuron = 0; neuron < connectionWeightDelta[layer].Length; neuron++)
				{
					for (int connection = 0; connection < connectionWeightDelta[layer][neuron].Length; connection++)
					{
						connectionWeightDelta[layer][neuron][connection] = learningRate * connectionPartialDerivative[layer][neuron][connection];
					}
				}
			}
		}

		private void UpdateWeights(double[][][] connectionWeightDelta)
		{
			for (int layer = 0; layer < ConnectionsWeights.Length; layer++)
			{
				for (int neuron = 0; neuron < ConnectionsWeights[layer].Length; neuron++)
				{
					for (int connection = 0; connection < ConnectionsWeights[layer][neuron].Length; connection++)
					{
						ConnectionsWeights[layer][neuron][connection] += connectionWeightDelta[layer][neuron][connection];
					}
				}
			}
		}

		private void ComputeBiasPartialDerivative(double[][] biasPartialDerivative, double[][] localGradientErrorSignal)
		{
			for (int layer = localGradientErrorSignal.Length - 1; layer >= 0; layer--)
			{
				for (int neuron = 0; neuron < localGradientErrorSignal[layer].Length; neuron++)
				{
					biasPartialDerivative[layer][neuron] = localGradientErrorSignal[layer][neuron] * 1.0; //input for bias is always 1.0
				}
			}
		}

		private void ComputeBiasDelta(double[][] biasDelta, double[][] biasPartialDerivative, double learningRate)
		{
			for (int layer = 0; layer < biasDelta.Length; layer++)
			{
				for (int neuron = 0; neuron < biasDelta[layer].Length; neuron++)
				{
					biasDelta[layer][neuron] = learningRate * biasPartialDerivative[layer][neuron];
				}
			}
		}

		private void UpdateBiases(double[][] biasDelta)
		{
			for (int layer = 0; layer < NeuronsBiases.Length; layer++)
			{
				for (int neuron = 0; neuron < NeuronsBiases[layer].Length; neuron++)
				{
					NeuronsBiases[layer][neuron] += biasDelta[layer][neuron];
				}
			}
		}

		//TODO dokładne porównanie tj. czy każda zaokrąglona wartosc odp jest równa pożądanej odp
		public override double GetAccuracy(Csv fileWithData)
		{
			int numCorrect = 0;
			int numWrong = 0;
			var data = fileWithData.DeserializeByRows<double>();
			CheckFileLength(data.GetLength(1), InputLayerNeuronsNumber + OutputLayerNeuronsNumber);

			for (int i = 0; i < data.GetLength(0); i++)
			{
				var inputs = ExtractInput(data, i);
				var correctResults = ExtractCorrectResponse(data, i);

				var networkResponse = ComputeOutput(inputs).Select(neuronResponse => Math.Round(neuronResponse)).ToList();
				int maxIndex = networkResponse.FindIndex(neuronResponse => (int)neuronResponse == 1);
				int correctMaxIndex = correctResults.ToList().FindIndex(neuronResponse => (int)neuronResponse == 1);

				if (maxIndex == correctMaxIndex)
				{
					numCorrect++;
				}
				else
				{
					numWrong++;
				}
			}

			return (numCorrect * 1.0) / (numCorrect + numWrong);
		}

		private double[] ExtractInput(double[,] data, int rowIndexToExtract)
		{
			double[] inputs = new double[InputLayerNeuronsNumber];
			for (int i = 0; i <= rowIndexToExtract; i++)
			{
				for (int j = 0; j < InputLayerNeuronsNumber; j++)
				{
					inputs[j] = data[i, j];
				}
			}

			return inputs;
		}

		private double[] ExtractCorrectResponse(double[,] data, int rowIndexToExtract)
		{
			double[] correctResults = new double[OutputLayerNeuronsNumber];
			for (int i = 0; i <= rowIndexToExtract; i++)
			{
				int k = 0;

				for (int j = InputLayerNeuronsNumber; j < data.GetLength(1); j++)
				{
					correctResults[k] = data[i, j];
					k++;
				}
			}

			return correctResults;
		}
	}
}
