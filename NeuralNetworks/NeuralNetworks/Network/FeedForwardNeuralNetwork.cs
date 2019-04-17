using System;
using System.Linq;
using FileDeserializer.CSV;

namespace NeuralNetworks.Network
{
	//TODO popatrz gdzie try catch wrzucić
	public class FeedForwardNeuralNetwork : NeuralNetwork
	{
		public FeedForwardNeuralNetwork(int networkInputs, int numberOfOutputs, int[] hiddenLayers = null, int inputNeuronInputs = 1) : base(networkInputs, numberOfOutputs, hiddenLayers, inputNeuronInputs)
		{
		}

		//TODO asynchroniczna?
		//TODO opis (w jakiej formei dane że dwie ostatnie kolumny to poprawne odp (o ile są dwa neurony wyjściwoe))
		//TODO check czy liczba inputów i outputów równa się schematowi sieci
		public override void Train(Csv traningDataFile, int maxEpochs, double learningRate)
		{
			var traningData = traningDataFile.DeserializeByRows<double>();
			var connectionWeightDelta = CreateWeightsArray();
			var connectionPartialDerivative = CreateWeightsArray();
			var localGradientErrorSignal = CreateNeuronsArray();
			var biasDelta = CreateNeuronsArray();
			var biasPartialDerivative = CreateNeuronsArray();
			int epoch = 0;

			int[] randomIndex = new int[traningData.GetLength(0)];
			for (int i = 0; i < randomIndex.Length; i++)
			{
				randomIndex[i] = i;
			}

			while (epoch < maxEpochs)
			{
				KnuthShuffle(randomIndex);

				for (int trainingRecord = 0; trainingRecord < traningData.GetLength(0); trainingRecord++)
				{
					var inputs = ExtractInput(traningData, randomIndex[trainingRecord]);
					var correctResults = ExtractCorrectResponse(traningData, randomIndex[trainingRecord]);

					var currentOutput = ComputeOutput(inputs);


					//3.4 || 3.5
					for (int layer = localGradientErrorSignal.Length-1; layer >= 0; layer--)
					{
						for (int neuron = 0; neuron < localGradientErrorSignal[layer].Length; neuron++)
						{
							double derivative;
							if (layer == localGradientErrorSignal.Length - 1)
							{
								var errorSignal = correctResults[neuron] - currentOutput[neuron]; //3.6
								derivative = (1 - currentOutput[neuron]) * currentOutput[neuron]; //3.7
								localGradientErrorSignal[layer][neuron] = errorSignal * derivative; //3.5
								if (NeuronsBiases != null)
								{
									biasPartialDerivative[layer][neuron] = localGradientErrorSignal[layer][neuron] * 1.0; //input for bias is always 1.0 and bias array is [][] so it can be computed here
								}
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

								if (NeuronsBiases != null)
								{
									biasPartialDerivative[layer][neuron] = localGradientErrorSignal[layer][neuron] * 1.0; //input for bias is always 1.0 and bias array is [][] so it can be computed here
								}
							}
						}
					}

					//3.3
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

					//3.2
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

					if (NeuronsBiases != null)
					{
						for (int layer = 0; layer < biasDelta.Length; layer++)
						{
							for (int neuron = 0; neuron < biasPartialDerivative[layer].Length; neuron++)
							{
								biasDelta[layer][neuron] = learningRate * biasPartialDerivative[layer][neuron];
							}
						}
					}

					//3.1 weightsUpdate
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
					if (NeuronsBiases != null)
					{
						for (int layer = 0; layer < NeuronsBiases.Length; layer++)
						{
							for (int neuron = 0; neuron < NeuronsBiases[layer].Length; neuron++)
							{
								NeuronsBiases[layer][neuron] += biasDelta[layer][neuron];
							}
						}
					}
				}
				epoch++;
			}
		}
		//TODO do osobnej klasy albo coś
		private void KnuthShuffle(int[] sequence)
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

		//TODO jakiś test czy liczba inputów równa się liczbie neuronów wejściowych
		public override double[] ComputeOutput(double[] input)
		{
			CheckIfWeightsAreSet();
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

		//TODO dokładne porównanie tj. czy każda zaokrąglona wartosc odp jest równa pożądanej odp
		public override double GetAccuracy(Csv fileWithData)
		{
			int numCorrect = 0;
			int numWrong = 0;
			var data = fileWithData.DeserializeByRows<double>();			

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
