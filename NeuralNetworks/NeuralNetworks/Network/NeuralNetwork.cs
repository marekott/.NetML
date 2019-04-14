using System;
using System.Linq;
using FileDeserializer.CSV;
using NeuralNetworks.Neurons;

namespace NeuralNetworks.Network
{
	//TODO popatrz gdzie try catch wrzucić
	public class NeuralNetwork
	{
		private readonly NeuronInputLayer[] _inputLayer;
		private readonly NeuronHiddenLayer[][] _hiddenLayer;
		private readonly NeuronOutputLayer[] _outputLayer;
		//TODO komentarz reSharpera
		private double[][][] _connectionsWeights;
		private double[][] _neuronsBiases;
		private double[][] _neuronsInputs; //TODO dodać testy
		private double[][] _neuronsOutputs; //TODO dodać testy

		public int InputLayerNeuronsNumber => _inputLayer.Length;
		public int HiddenLayersNumber => _hiddenLayer?.Length ?? 0;
		public int OutputLayerNeuronsNumber => _outputLayer.Length;
		public double[][][] ConnectionsWeights => _connectionsWeights; //TODO udostępnia do zmiany, wystaw tylko do odczytu na zewnątrz
		public double[][] NeuronsBiases => _neuronsBiases; //TODO udostępnia do zmiany, wystaw tylko do odczytu na zewnątrz

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
			//TODO trzeba zmianić nazwę metody poniżej bo teraz jest wołana w innym kontekście
			_neuronsInputs = CreateBiasesArray();
			_neuronsOutputs = CreateBiasesArray();
		}
		
		//TODO metoda obliczająca potrzebną liczbę wag i porównująca z tą z pliku + opis metody jak w pliku powinny być ułożone wagi
		//TODO losowe ustawianie wag
		public void SetWeights(Csv fileWithWeights)
		{
			int currentWeightFromFile = 0;
			var weights = fileWithWeights.Deserialize<double>();
			_connectionsWeights = CreateWeightsArray();

			foreach (var layer in _connectionsWeights)
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
		public void SetBiases(Csv fileWithBiases)
		{
			int currentBiasFromFile = 0;
			var biases = fileWithBiases.Deserialize<double>();
			_neuronsBiases = CreateBiasesArray();

			foreach (var layer in _neuronsBiases)
			{
				for (int currentNeuron = 0; currentNeuron < layer.Length; currentNeuron++)
				{
					layer[currentNeuron] = biases[currentBiasFromFile++];
				}
			}
		}

		//TODO jakiś test czy liczba inputów równa się liczbie neuronów wejściowych
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

				if (_neuronsBiases != null)
				{
					signalsSums[layer][connection] += _neuronsBiases[layer][connection];
				}
			}

			for (int neuron = 0; neuron < _neuronsInputs[layer].Length; neuron++)
			{
				_neuronsInputs[layer][neuron] = signalsSums[layer][neuron];
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
					_neuronsOutputs[layer - 1][neuron] =
						_hiddenLayer[layer - 1][neuron].ComputeOutput(signalsSums[layer - 1]);
				}

				if (_neuronsBiases != null)
				{
					signalsSums[layer][connection] += _neuronsBiases[layer][connection];
				}
			}

			for (int neuron = 0; neuron < _neuronsInputs[layer].Length; neuron++)
			{
				_neuronsInputs[layer][neuron] = signalsSums[layer][neuron];
			}

			return signalsSums;
		}

		private double[] ComputeOutputLayerResponses(double[] signalsSums)
		{
			var networkResponse = new double[OutputLayerNeuronsNumber];
			for (int outNeuron = 0; outNeuron < OutputLayerNeuronsNumber; outNeuron++)
			{
				networkResponse[outNeuron] = _outputLayer[outNeuron].ComputeOutput(signalsSums);
				_neuronsOutputs[_neuronsOutputs.Length-1][outNeuron] = networkResponse[outNeuron]; //TODO to pole jest z dupy, zrob to z glowa
			}

			return networkResponse;
		}
		
		//TODO asynchroniczna?
		//TODO opis (w jakiej formei dane że dwie ostatnie kolumny to poprawne odp (o ile są dwa neurony wyjściwoe))
		//TODO for na danych wejściowych jako parrarel? -> chyba nie bo tablice z deltami itd są wspólne
		public void Train(Csv traningDataFile, int maxEpochs, double learningRate, double momentum)
		{
			var traningData = traningDataFile.DeserializeByRows<double>();
			var connectionWeightsCorrection = CreateWeightsArray();
			var connectionWeightDelta = CreateWeightsArray();
			var connectionPartialDerivative = CreateWeightsArray();
			var localGradientErrorSignal = CreateBiasesArray();
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
					////TODO dodać obliczenia dla biasów
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
							}
							else
							{
								derivative = (1 + _neuronsOutputs[layer][neuron]) *
								             (1 - _neuronsOutputs[layer][neuron]); //3.8
								double sum = 0.0;
								for (int connection = 0; connection < _connectionsWeights[layer + 1][neuron].Length; connection++)
								{
									sum = localGradientErrorSignal[layer][neuron] * _connectionsWeights[layer + 1][neuron][connection];
								}
								localGradientErrorSignal[layer][neuron] = sum * derivative; //3.5
							}
						}
					}

					//3.3
					////TODO dodać obliczenia dla biasów
					for (int layer = 0; layer < connectionPartialDerivative.Length; layer++)
					{
						for (int neuron = 0; neuron < connectionPartialDerivative[layer].Length; neuron++)
						{
							for (int connection = 0; connection < connectionPartialDerivative[layer][neuron].Length; connection++)
							{
								//TODO tu może być błąd z indeksowaniem
								connectionPartialDerivative[layer][neuron][connection] = _neuronsInputs[layer][connection] * localGradientErrorSignal[layer][connection];
							}
						}
					}

					//3.2
					////TODO dodać obliczenia dla biasów
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

					//3.1
					////TODO dodać obliczenia dla biasów
					for (int layer = 0; layer < connectionWeightsCorrection.Length; layer++)
					{
						for (int neuron = 0; neuron < connectionWeightsCorrection[layer].Length; neuron++)
						{
							for (int connection = 0;
								connection < connectionWeightsCorrection[layer][neuron].Length;
								connection++)
							{
								connectionWeightsCorrection[layer][neuron][connection] = _connectionsWeights[layer][neuron][connection] * connectionWeightDelta[layer][neuron][connection];
							}
						}
					}
					//weightsUpdate
					//TODO dodać obliczenia dla biasów
					for (int layer = 0; layer < connectionWeightsCorrection.Length; layer++)
					{
						for (int neuron = 0; neuron < connectionWeightsCorrection[layer].Length; neuron++)
						{
							for (int connection = 0; connection < connectionWeightsCorrection[layer][neuron].Length; connection++)
							{
								_connectionsWeights[layer][neuron][connection] +=
									connectionWeightsCorrection[layer][neuron][connection];
							}
						}
					}
					if (_neuronsBiases != null)
					{
						throw new NotImplementedException();
					}
				}
				epoch++;
			}
		}
		//TODO Refactor
		private double[][][] CreateWeightsArray()
		{
			int numberOfConnectionsLayers = CountConnectionLayers();
			int lastLayerIndex = numberOfConnectionsLayers - 1;
			var weightsArray = new double[numberOfConnectionsLayers][][];

			for (int currentLayer = 0; currentLayer < numberOfConnectionsLayers; currentLayer++)
			{
				if (currentLayer == 0)
				{
					weightsArray[currentLayer] = new double[InputLayerNeuronsNumber][];
				}
				else
				{
					weightsArray[currentLayer] = new double[_hiddenLayer[currentLayer - 1].Length][];
				}


				for (int currentNeuron = 0; currentNeuron < weightsArray[currentLayer].Length; currentNeuron++)
				{
					if (currentLayer == 0)
					{
						if (lastLayerIndex == 0)
						{
							weightsArray[currentLayer][currentNeuron] = new double[OutputLayerNeuronsNumber];
						}
						else
						{
							weightsArray[currentLayer][currentNeuron] = new double[_hiddenLayer[currentLayer].Length];
						}
					}

					else if (currentLayer == lastLayerIndex)
					{
						weightsArray[currentLayer][currentNeuron] = new double[OutputLayerNeuronsNumber];
					}
					else
					{
						weightsArray[currentLayer][currentNeuron] = new double[_hiddenLayer[currentLayer].Length];
					}
				}
			}
			return weightsArray;
		}
		//TODO Refactor
		private double[][] CreateBiasesArray()
		{
			int numberOfConnectionsLayers = CountConnectionLayers();
			int lastLayerIndex = numberOfConnectionsLayers - 1;
			var biasesArray = new double[numberOfConnectionsLayers][];

			for (int currentLayer = 0; currentLayer < numberOfConnectionsLayers; currentLayer++)
			{
				if (currentLayer == 0)
				{
					if (_hiddenLayer != null)
					{
						biasesArray[currentLayer] = new double[_hiddenLayer[currentLayer].Length];
					}
					else
					{
						biasesArray[currentLayer] = new double[OutputLayerNeuronsNumber];
					}
				}
				else if (currentLayer == lastLayerIndex)
				{
					biasesArray[currentLayer] = new double[OutputLayerNeuronsNumber];
				}
				else if (_hiddenLayer != null)
				{
					biasesArray[currentLayer] = new double[_hiddenLayer[currentLayer].Length];
				}
			}

			return biasesArray;
		}

		private int CountConnectionLayers()
		{
			int numberOfConnectionsLayers = (2 + HiddenLayersNumber) - 1;

			return numberOfConnectionsLayers;
		}

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

		//TODO troche na pałe bo czysto teoretycznie przy 4 neuronach każdy może zwrócić 0.25 i żaden nie będzie zaokrąglony do 1
		public double GetAccuracy(Csv fileWithData)
		{
			int numCorrect = 0;
			int numWrong = 0;
			var data = fileWithData.DeserializeByRows<double>();			

			for (int i = 0; i < data.GetLength(0); i++)
			{
				//TODO do przemyślenia czy nie przenieść deklaracji poza pętlę
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
