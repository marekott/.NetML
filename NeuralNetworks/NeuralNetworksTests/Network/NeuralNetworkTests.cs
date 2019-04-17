using FileDeserializer.CSV;
using NeuralNetworks.Network;
using NeuralNetworksTests.Mock;
using Xunit;

namespace NeuralNetworksTests.Network
{
	public class NeuralNetworkTests
	{
		private readonly Csv _fileWithWeights2In2Out;
		private readonly Csv _fileWithWeights2In2Hidden2Out;
		private readonly Csv _fileWithWeights2In3Hidden2Hidden2Out;
		private readonly Csv _fileWithRandomWeights;
		private readonly Csv _fileWithBiases2Out;
		private readonly Csv _fileWithBiases2Hidden2Out;
		private readonly Csv _fileWithBiases3Hidden2Hidden2Out;
		private readonly Csv _fileWith3Numbers;

		public NeuralNetworkTests()
		{
			_fileWithWeights2In2Out = new Csv(new MockFileLocator(@"Mock\2_2_Weights.csv"), ';');
			_fileWithWeights2In2Hidden2Out = new Csv(new MockFileLocator(@"Mock\2_2_2_Weights.csv"), ';');
			_fileWithWeights2In3Hidden2Hidden2Out = new Csv(new MockFileLocator(@"Mock\2_3_2_2_Weights.csv"), ';');
			_fileWithRandomWeights = new Csv(new MockFileLocator(@"Mock\Any_Weights.csv"), ';');
			_fileWithBiases2Out = new Csv(new MockFileLocator(@"Mock\2_2_Biases.csv"), ';');
			_fileWithBiases2Hidden2Out = new Csv(new MockFileLocator(@"Mock\2_2_2_Biases.csv"), ';');
			_fileWithBiases3Hidden2Hidden2Out = new Csv(new MockFileLocator(@"Mock\2_3_2_2_Biases.csv"), ';');
			_fileWith3Numbers = new Csv(new MockFileLocator(@"Mock\ThreeNumbers.csv"), ';');
		}

		[Fact]
		public void NeuralNetworkConstructorTestWithSchema_2_2_2()
		{
			int numberOfInputs = 2;
			int[] hiddenLayers = { 2 };
			int numberOfOutputs = 2;

			var neuralNetwork = new NeuralNetworkMock(numberOfInputs, numberOfOutputs, hiddenLayers);

			Assert.Equal(numberOfInputs, neuralNetwork.InputLayerNeuronsNumber);
			Assert.Equal(hiddenLayers.Length, neuralNetwork.HiddenLayersNumber);
			Assert.Equal(numberOfOutputs, neuralNetwork.OutputLayerNeuronsNumber);
		}

		[Fact]
		public void NeuralNetworkConstructorTestWithSchema_2_2()
		{
			int numberOfInputs = 2;
			int numberOfOutputs = 2;

			var neuralNetwork = new NeuralNetworkMock(numberOfInputs, numberOfOutputs);

			Assert.Equal(numberOfInputs, neuralNetwork.InputLayerNeuronsNumber);
			Assert.Equal(0, neuralNetwork.HiddenLayersNumber);
			Assert.Equal(numberOfOutputs, neuralNetwork.OutputLayerNeuronsNumber);
		}

		[Fact]
		public void NeuralNetworkConstructorTestWithSchema_2_3_2_2()
		{
			int numberOfInputs = 2;
			int[] hiddenLayers = { 3, 2 };
			int numberOfOutputs = 2;

			var neuralNetwork = new NeuralNetworkMock(numberOfInputs, numberOfOutputs, hiddenLayers);

			Assert.Equal(numberOfInputs, neuralNetwork.InputLayerNeuronsNumber);
			Assert.Equal(hiddenLayers.Length, neuralNetwork.HiddenLayersNumber);
			Assert.Equal(numberOfOutputs, neuralNetwork.OutputLayerNeuronsNumber);
		}

		[Fact]
		public void NeuralNetworkProperWeightsSetTestWithSchema2_2()
		{
			int numberOfInputs = 2;
			int numberOfOutputs = 2;

			var neuralNetwork = new NeuralNetworkMock(numberOfInputs, numberOfOutputs);
			neuralNetwork.SetWeights(_fileWithWeights2In2Out);


			double[][][] expectedWeights = new double[1][][];
			expectedWeights[0] = new double[2][];
			for (int i = 0; i < expectedWeights[0].Length; i++)
			{
				expectedWeights[0][i] = new double[2];
			}

			expectedWeights[0][0][0] = 0.5;
			expectedWeights[0][0][1] = 0.2;
			expectedWeights[0][1][0] = 0.3;
			expectedWeights[0][1][1] = 1.4;

			Assert.Equal(expectedWeights, neuralNetwork.ConnectionsWeights);
		}

		[Fact]
		public void NeuralNetworkProperWeightsSetTestWithSchema2_2_2()
		{
			int numberOfInputs = 2;
			int[] hiddenLayers = { 2 };
			int numberOfOutputs = 2;

			var neuralNetwork = new NeuralNetworkMock(numberOfInputs, numberOfOutputs, hiddenLayers);
			neuralNetwork.SetWeights(_fileWithWeights2In2Hidden2Out);

			double[][][] expectedWeights = new double[2][][];
			expectedWeights[0] = new double[2][];
			expectedWeights[1] = new double[2][];
			foreach (var layer in expectedWeights)
			{
				for (int j = 0; j < layer.Length; j++)
				{
					layer[j] = new double[2];
				}
			}

			expectedWeights[0][0][0] = 0.5;
			expectedWeights[0][0][1] = 0.2;
			expectedWeights[0][1][0] = 0.3;
			expectedWeights[0][1][1] = 1.4;
			expectedWeights[1][0][0] = 0.6;
			expectedWeights[1][0][1] = 0.3;
			expectedWeights[1][1][0] = 0.5;
			expectedWeights[1][1][1] = 0.7;

			Assert.Equal(expectedWeights, neuralNetwork.ConnectionsWeights);
		}

		[Fact]
		public void NeuralNetworkProperWeightsSetTestWithSchema2_3_2_2()
		{
			int numberOfInputs = 2;
			int[] hiddenLayers = { 3, 2 };
			int numberOfOutputs = 2;
			

			var neuralNetwork = new NeuralNetworkMock(numberOfInputs, numberOfOutputs, hiddenLayers);
			neuralNetwork.SetWeights(_fileWithWeights2In3Hidden2Hidden2Out);

			double[][][] expectedWeights = new double[3][][];
			expectedWeights[0] = new double[2][];
			expectedWeights[1] = new double[3][];
			expectedWeights[2] = new double[2][];

			for (int i = 0; i < expectedWeights[0].Length; i++)
			{
				expectedWeights[0][i] = new double[3];
				expectedWeights[2][i] = new double[2];
			}

			for (int i = 0; i < expectedWeights[1].Length; i++)
			{
				expectedWeights[1][i] = new double[2];
			}

			expectedWeights[0][0][0] = 0.5;
			expectedWeights[0][0][1] = 0.2;
			expectedWeights[0][0][2] = 0.4;
			expectedWeights[0][1][0] = 0.3;
			expectedWeights[0][1][1] = 1.4;
			expectedWeights[0][1][2] = 0.53;

			expectedWeights[1][0][0] = 0.6;
			expectedWeights[1][0][1] = 0.3;
			expectedWeights[1][1][0] = 0.5;
			expectedWeights[1][1][1] = 0.7;
			expectedWeights[1][2][0] = 0.6;
			expectedWeights[1][2][1] = 0.7;

			expectedWeights[2][0][0] = 0.63;
			expectedWeights[2][0][1] = 0.34;
			expectedWeights[2][1][0] = 0.52;
			expectedWeights[2][1][1] = 0.71;

			Assert.Equal(expectedWeights, neuralNetwork.ConnectionsWeights);
		}

		[Fact]
		public void NeuralNetworkWith5HiddenLayersProperDimensionsNumberOfWeightsArrayTest()
		{
			int numberOfInputs = 3;
			int[] hiddenLayers = { 5, 4, 3, 3, 3 };
			int numberOfOutputs = 3;

			var neuralNetwork = new NeuralNetworkMock(numberOfInputs, numberOfOutputs, hiddenLayers);
			neuralNetwork.SetWeights(_fileWithRandomWeights);

			int expectedNumberOfWeightsLayers = 6; //7 neuron layers so there will be 6 layers of connections. Always one less
			int expectedNumberOfNeuronsInFirstLayer = 3;
			int expectedNumberOfNeuronsInSecondLayer = 5;
			int expectedNumberOfNeuronsInThirdLayer = 4;
			int expectedNumberOfNeuronsInFourthLayer = 3;
			int expectedNumberOfNeuronsInFifthLayer = 3;
			int expectedNumberOfNeuronsInSixthLayer = 3;
			int expectedNumberOfConnectionsInFirstLayer = 5; //number of connections is based of number of neurons in next layer
			int expectedNumberOfConnectionsInSecondLayer = 4;
			int expectedNumberOfConnectionsInThirdLayer = 3;
			int expectedNumberOfConnectionsInFourthLayer = 3;
			int expectedNumberOfConnectionsInFifthLayer = 3;
			int expectedNumberOfConnectionsInSixthLayer = 3;

			Assert.Equal(expectedNumberOfWeightsLayers, neuralNetwork.ConnectionsWeights.Length);
			Assert.Equal(expectedNumberOfNeuronsInFirstLayer, neuralNetwork.ConnectionsWeights[0].Length);
			Assert.Equal(expectedNumberOfNeuronsInSecondLayer, neuralNetwork.ConnectionsWeights[1].Length);
			Assert.Equal(expectedNumberOfNeuronsInThirdLayer, neuralNetwork.ConnectionsWeights[2].Length);
			Assert.Equal(expectedNumberOfNeuronsInFourthLayer, neuralNetwork.ConnectionsWeights[3].Length);
			Assert.Equal(expectedNumberOfNeuronsInFifthLayer, neuralNetwork.ConnectionsWeights[4].Length);
			Assert.Equal(expectedNumberOfNeuronsInSixthLayer, neuralNetwork.ConnectionsWeights[5].Length);

			for (int layer = 0; layer < expectedNumberOfNeuronsInFirstLayer; layer++)
			{
				Assert.Equal(expectedNumberOfConnectionsInFirstLayer,
					neuralNetwork.ConnectionsWeights[0][layer].Length);
			}

			for (int layer = 0; layer < expectedNumberOfNeuronsInFirstLayer; layer++)
			{
				Assert.Equal(expectedNumberOfConnectionsInSecondLayer,
					neuralNetwork.ConnectionsWeights[1][layer].Length);
			}

			for (int layer = 0; layer < expectedNumberOfNeuronsInFirstLayer; layer++)
			{
				Assert.Equal(expectedNumberOfConnectionsInThirdLayer,
					neuralNetwork.ConnectionsWeights[2][layer].Length);
			}

			for (int layer = 0; layer < expectedNumberOfNeuronsInFirstLayer; layer++)
			{
				Assert.Equal(expectedNumberOfConnectionsInFourthLayer,
					neuralNetwork.ConnectionsWeights[3][layer].Length);
			}

			for (int layer = 0; layer < expectedNumberOfNeuronsInFirstLayer; layer++)
			{
				Assert.Equal(expectedNumberOfConnectionsInFifthLayer,
					neuralNetwork.ConnectionsWeights[4][layer].Length);
			}

			for (int layer = 0; layer < expectedNumberOfNeuronsInFirstLayer; layer++)
			{
				Assert.Equal(expectedNumberOfConnectionsInSixthLayer,
					neuralNetwork.ConnectionsWeights[5][layer].Length);
			}
		}

		[Fact]
		public void NeuralNetworkProperBiasSetTestWithSchema2_2()
		{
			int numberOfInputs = 2;
			int numberOfOutputs = 2;
			

			var neuralNetwork = new NeuralNetworkMock(numberOfInputs, numberOfOutputs);

			neuralNetwork.SetBiases(_fileWithBiases2Out);

			double[][] expectedBiases = new double[1][];
			expectedBiases[0] = new double[2];
			expectedBiases[0][0] = 0.9;
			expectedBiases[0][1] = 0.1;

			Assert.Equal(expectedBiases, neuralNetwork.NeuronsBiases);
		}

		[Fact]
		public void NeuralNetworkProperBiasSetTestWithSchema2_2_2()
		{
			int numberOfInputs = 2;
			int[] hiddenLayers = { 2 };
			int numberOfOutputs = 2;

			var neuralNetwork = new NeuralNetworkMock(numberOfInputs, numberOfOutputs, hiddenLayers);

			neuralNetwork.SetBiases(_fileWithBiases2Hidden2Out);

			double[][] expectedBiases = new double[2][];
			for (int i = 0; i < expectedBiases.Length; i++)
			{
				expectedBiases[i] = new double[2];
			}

			expectedBiases[0][0] = 0.9;
			expectedBiases[0][1] = 0.1;
			expectedBiases[1][0] = 0.8;
			expectedBiases[1][1] = 0.2;

			Assert.Equal(expectedBiases, neuralNetwork.NeuronsBiases);
		}

		[Fact]
		public void NeuralNetworkProperBiasSetTestWithSchema2_3_2_2()
		{
			int numberOfInputs = 2;
			int[] hiddenLayers = { 3, 2 };
			int numberOfOutputs = 2;

			var neuralNetwork = new NeuralNetworkMock(numberOfInputs, numberOfOutputs, hiddenLayers);

			neuralNetwork.SetBiases(_fileWithBiases3Hidden2Hidden2Out);

			double[][] expectedBiases = new double[3][];
			expectedBiases[0] = new double[3];
			expectedBiases[1] = new double[2];
			expectedBiases[2] = new double[2];

			expectedBiases[0][0] = 0.9;
			expectedBiases[0][1] = 0.1;
			expectedBiases[0][2] = 0.8;
			expectedBiases[1][0] = 0.2;
			expectedBiases[1][1] = 0.5;
			expectedBiases[2][0] = 0.3;
			expectedBiases[2][1] = 0.3;

			Assert.Equal(expectedBiases, neuralNetwork.NeuronsBiases);
		}

		[Fact]
		public void NeuralNetworkProperWeightsArrayDimensionsTestWithSchema2_3_2()
		{
			int numberOfInputs = 2;
			int[] hiddenLayers = { 3 };
			int numberOfOutputs = 2;

			var neuralNetwork = new NeuralNetworkMock(numberOfInputs, numberOfOutputs, hiddenLayers);
			neuralNetwork.SetWeights(_fileWithRandomWeights);

			Assert.Equal(hiddenLayers[0], neuralNetwork.ConnectionsWeights[1].Length);
		}

		[Fact]
		public void NeuralNetworkProperWeightsArrayDimensionsTestWithSchema2_3_4_5()
		{
			int numberOfInputs = 2;
			int[] hiddenLayers = { 3, 4 };
			int numberOfOutputs = 5;

			var neuralNetwork = new NeuralNetworkMock(numberOfInputs, numberOfOutputs, hiddenLayers);
			neuralNetwork.SetWeights(_fileWithRandomWeights);

			Assert.Equal(hiddenLayers[0], neuralNetwork.ConnectionsWeights[1].Length);
			Assert.Equal(hiddenLayers[1], neuralNetwork.ConnectionsWeights[2].Length);
		}

		[Fact]
		public void NeuralNetworkProperBiasArrayDimensionsTestWithSchema2_3_2()
		{
			int numberOfInputs = 2;
			int[] hiddenLayers = { 3 };
			int numberOfOutputs = 2;

			var neuralNetwork = new NeuralNetworkMock(numberOfInputs, numberOfOutputs, hiddenLayers);
			neuralNetwork.SetBiases(_fileWithRandomWeights);

			Assert.Equal(hiddenLayers[0], neuralNetwork.NeuronsBiases[0].Length);
			Assert.Equal(numberOfOutputs, neuralNetwork.NeuronsBiases[1].Length);
		}

		[Fact]
		public void NeuralNetworkProperBiasArrayDimensionsTestWithSchema2_3_4_5()
		{
			int numberOfInputs = 2;
			int[] hiddenLayers = { 3, 4 };
			int numberOfOutputs = 5;

			var neuralNetwork = new NeuralNetworkMock(numberOfInputs, numberOfOutputs, hiddenLayers);
			neuralNetwork.SetBiases(_fileWithRandomWeights);

			Assert.Equal(hiddenLayers[0], neuralNetwork.NeuronsBiases[0].Length);
			Assert.Equal(hiddenLayers[1], neuralNetwork.NeuronsBiases[1].Length);
			Assert.Equal(numberOfOutputs, neuralNetwork.NeuronsBiases[2].Length);
		}

		[Fact]
		public void NeuralNetworkCountConnections3_4Test()
		{
			int numberOfInputs = 3;
			int numberOfOutputs = 4;
			int expectedNumberOfConnections = 12;

			var neuralNetwork = new NeuralNetworkMock(numberOfInputs, numberOfOutputs);

			Assert.Equal(expectedNumberOfConnections, neuralNetwork.CountConnections());
		}

		[Fact]
		public void NeuralNetworkCountConnections3_4_2Test()
		{
			int numberOfInputs = 3;
			int[] hiddenLayers = { 4 };
			int numberOfOutputs = 2;
			int expectedNumberOfConnections = 20;

			var neuralNetwork = new NeuralNetworkMock(numberOfInputs, numberOfOutputs, hiddenLayers);
			Assert.Equal(expectedNumberOfConnections, neuralNetwork.CountConnections());
		}

		[Fact]
		public void NeuralNetworkCountBiases3_4Test()
		{
			int numberOfInputs = 3;
			int numberOfOutputs = 4;
			int expectedNumberOfBiases = 4;

			var neuralNetwork = new NeuralNetworkMock(numberOfInputs, numberOfOutputs);

			Assert.Equal(expectedNumberOfBiases, neuralNetwork.CountBiases());
		}

		[Fact]
		public void NeuralNetworkCountBiases3_4_2Test()
		{
			int numberOfInputs = 3;
			int[] hiddenLayers = { 4 };
			int numberOfOutputs = 2;
			int expectedNumberOfBiases = 6;

			var neuralNetwork = new NeuralNetworkMock(numberOfInputs, numberOfOutputs, hiddenLayers);

			Assert.Equal(expectedNumberOfBiases, neuralNetwork.CountBiases());
		}

		[Fact]
		public void NeuralNetworkWeightsSetWasWrongSourceFileLengthExceptionThrown()
		{
			int numberOfInputs = 2;
			int numberOfOutputs = 2;

			var neuralNetwork = new NeuralNetworkMock(numberOfInputs, numberOfOutputs);

			Assert.Throws<WrongSourceFileLengthException>(() => neuralNetwork.SetWeights(_fileWith3Numbers));
		}

		[Fact]
		public void NeuralNetworkBiasSetWasWrongSourceFileLengthExceptionThrown()
		{
			int numberOfInputs = 2;
			int numberOfOutputs = 4;

			var neuralNetwork = new NeuralNetworkMock(numberOfInputs, numberOfOutputs);

			Assert.Throws<WrongSourceFileLengthException>(() => neuralNetwork.SetBiases(_fileWith3Numbers));
		}
	}
}
