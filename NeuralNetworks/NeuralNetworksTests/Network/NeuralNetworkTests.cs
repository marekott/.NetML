using System.Linq;
using FileDeserializer.CSV;
using NeuralNetworks.Network;
using NeuralNetworksTests.Mock;
using Xunit;

namespace NeuralNetworksTests.Network
{
	public class NeuralNetworkTests
	{
		[Fact]
		public void NeuralNetworkConstructorTestWithSchema_2_2_2()
		{
			int numberOfInputs = 2;
			int[] hiddenLayers = { 2 };
			int numberOfOutputs = 2;

			var neuralNetwork = new NeuralNetwork(numberOfInputs, numberOfOutputs, hiddenLayers);

			Assert.Equal(numberOfInputs, neuralNetwork.InputLayerNeuronsNumber);
			Assert.Equal(hiddenLayers.Length, neuralNetwork.HiddenLayersNumber);
			Assert.Equal(numberOfOutputs, neuralNetwork.OutputLayerNeuronsNumber);
		}

		[Fact]
		public void NeuralNetworkConstructorTestWithSchema_2_2()
		{
			int numberOfInputs = 2;
			int numberOfOutputs = 2;

			var neuralNetwork = new NeuralNetwork(numberOfInputs, numberOfOutputs);

			Assert.Equal(numberOfInputs, neuralNetwork.InputLayerNeuronsNumber);
			Assert.Equal(0,neuralNetwork.HiddenLayersNumber);
			Assert.Equal(numberOfOutputs, neuralNetwork.OutputLayerNeuronsNumber);
		}

		[Fact]
		public void NeuralNetworkConstructorTestWithSchema_2_3_2_2()
		{
			int numberOfInputs = 2;
			int[] hiddenLayers = { 3, 2 };
			int numberOfOutputs = 2;

			var neuralNetwork = new NeuralNetwork(numberOfInputs, numberOfOutputs, hiddenLayers);

			Assert.Equal(numberOfInputs, neuralNetwork.InputLayerNeuronsNumber);
			Assert.Equal(hiddenLayers.Length, neuralNetwork.HiddenLayersNumber);
			Assert.Equal(numberOfOutputs, neuralNetwork.OutputLayerNeuronsNumber);
		}

		[Fact]
		public void NetworkOutputTestWithSchema_2_2()
		{
			int numberOfInputs = 2;
			int numberOfOutputs = 2;
			var fileWithWeights = new Csv(new MockFileLocator(@"Mock\2_2_Weights.csv"), ';');
			double[] input = { 5.0, 3.0 };
			double[] expected = { 0.141851064900488, 0.858148935099512 };

			var neuralNetwork = new NeuralNetwork(numberOfInputs, numberOfOutputs);
			neuralNetwork.SetWeights(fileWithWeights);

			var actual = neuralNetwork.ComputeOutput(input);

			Assert.Equal(expected[0], actual[0], 9);
			Assert.Equal(expected[1], actual[1], 9);
		}

		[Fact]
		public void NetworkOutputTestWithSchema_2_2_2()
		{
			int numberOfInputs = 2;
			int[] hiddenLayers = {2};
			int numberOfOutputs = 2;
			var fileWithWeights = new Csv(new MockFileLocator(@"Mock\2_2_2_Weights.csv"), ';');
			double[] input = {5.0, 3.0 };
			double[] expected = {0.524815756471604, 0.475184243528396};

			var neuralNetwork = new NeuralNetwork(numberOfInputs, numberOfOutputs, hiddenLayers);
			neuralNetwork.SetWeights(fileWithWeights);

			var actual = neuralNetwork.ComputeOutput(input);

			Assert.Equal(expected[0],actual[0],9);
			Assert.Equal(expected[1], actual[1], 9);
		}

		[Fact]
		public void NetworkOutputTestWithSchema_2_3_2_2()
		{
			int numberOfInputs = 2;
			int[] hiddenLayers = { 3,2 };
			int numberOfOutputs = 2;
			var fileWithWeights = new Csv(new MockFileLocator(@"Mock\2_3_2_2_Weights.csv"), ';');
			double[] input = { 5.0, 3.0 };
			double[] expected = {0.523358076385092, 0.476641923614908};

			var neuralNetwork = new NeuralNetwork(numberOfInputs, numberOfOutputs, hiddenLayers);
			neuralNetwork.SetWeights(fileWithWeights);

			var actual = neuralNetwork.ComputeOutput(input);

			Assert.Equal(expected[0], actual[0], 9);
			Assert.Equal(expected[1], actual[1], 9);
		}

		[Fact]
		public void NetworkOutputsSumEqualOneTest()
		{
			int numberOfInputs = 2;
			int numberOfOutputs = 2;
			var fileWithWeights = new Csv(new MockFileLocator(@"Mock\2_2_Weights.csv"), ';');
			double[] input = { 5.0, 3.0 };

			var neuralNetwork = new NeuralNetwork(numberOfInputs, numberOfOutputs);
			neuralNetwork.SetWeights(fileWithWeights);

			var actual = neuralNetwork.ComputeOutput(input);

			Assert.Equal(1.0,actual.Select(p => p).Sum(),2);
		}

		[Fact]
		public void NeuralNetworkProperWeightsSetTestWithSchema2_2()
		{
			int numberOfInputs = 2;
			int numberOfOutputs = 2;
			var fileWithWeights = new Csv(new MockFileLocator(@"Mock\2_2_Weights.csv"), ';');

			var neuralNetwork = new NeuralNetwork(numberOfInputs, numberOfOutputs);
			neuralNetwork.SetWeights(fileWithWeights);


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
			var fileWithWeights = new Csv(new MockFileLocator(@"Mock\2_2_2_Weights.csv"), ';');

			var neuralNetwork = new NeuralNetwork(numberOfInputs, numberOfOutputs, hiddenLayers);
			neuralNetwork.SetWeights(fileWithWeights);

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
			var fileWithWeights = new Csv(new MockFileLocator(@"Mock\2_3_2_2_Weights.csv"), ';');

			var neuralNetwork = new NeuralNetwork(numberOfInputs, numberOfOutputs, hiddenLayers);
			neuralNetwork.SetWeights(fileWithWeights);

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
			int[] hiddenLayers = { 5, 4, 3, 3, 3};
			int numberOfOutputs = 3;
			var fileWithWeights = new Csv(new MockFileLocator(@"Mock\Any_Weights.csv"), ';');

			var neuralNetwork = new NeuralNetwork(numberOfInputs, numberOfOutputs, hiddenLayers);
			neuralNetwork.SetWeights(fileWithWeights);

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
				Assert.Equal(expectedNumberOfConnectionsInFirstLayer, neuralNetwork.ConnectionsWeights[0][layer].Length);
			}
			for (int layer = 0; layer < expectedNumberOfNeuronsInFirstLayer; layer++)
			{
				Assert.Equal(expectedNumberOfConnectionsInSecondLayer, neuralNetwork.ConnectionsWeights[1][layer].Length);
			}
			for (int layer = 0; layer < expectedNumberOfNeuronsInFirstLayer; layer++)
			{
				Assert.Equal(expectedNumberOfConnectionsInThirdLayer, neuralNetwork.ConnectionsWeights[2][layer].Length);
			}
			for (int layer = 0; layer < expectedNumberOfNeuronsInFirstLayer; layer++)
			{
				Assert.Equal(expectedNumberOfConnectionsInFourthLayer, neuralNetwork.ConnectionsWeights[3][layer].Length);
			}
			for (int layer = 0; layer < expectedNumberOfNeuronsInFirstLayer; layer++)
			{
				Assert.Equal(expectedNumberOfConnectionsInFifthLayer, neuralNetwork.ConnectionsWeights[4][layer].Length);
			}
			for (int layer = 0; layer < expectedNumberOfNeuronsInFirstLayer; layer++)
			{
				Assert.Equal(expectedNumberOfConnectionsInSixthLayer, neuralNetwork.ConnectionsWeights[5][layer].Length);
			}
		}

		[Fact]
		public void NeuralNetworkWith5HiddenLayersOutputTest()
		{
			int numberOfInputs = 3;
			int[] hiddenLayers = { 5, 4, 3, 3, 3 };
			int numberOfOutputs = 3;
			var fileWithWeights = new Csv(new MockFileLocator(@"Mock\Any_Weights.csv"), ';');

			double[] input = { 1.0, 1.0, 1.0 };

			var neuralNetwork = new NeuralNetwork(numberOfInputs, numberOfOutputs, hiddenLayers);
			neuralNetwork.SetWeights(fileWithWeights);

			neuralNetwork.ComputeOutput(input);
		}

		[Fact]
		public void NeuralNetworkProperWeightsSetTestWithSchema2_2WithBias()
		{
			int numberOfInputs = 2;
			int numberOfOutputs = 2;
			var fileWithWeights = new Csv(new MockFileLocator(@"Mock\2_2_Weights.csv"), ';');
			var fileWithBiases = new Csv(new MockFileLocator(@"Mock\2_2_Biases.csv"), ';');

			var neuralNetwork = new NeuralNetwork(numberOfInputs, numberOfOutputs);
			neuralNetwork.SetWeights(fileWithWeights);


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

			neuralNetwork.SetBiases(fileWithBiases);

			double[][] expectedBiases = new double[1][];
			expectedBiases[0] = new double[2];
			expectedBiases[0][0] = 0.9;
			expectedBiases[0][1] = 0.1;

			Assert.Equal(expectedBiases, neuralNetwork.NeuronsBiases);
			Assert.Equal(expectedWeights, neuralNetwork.ConnectionsWeights);
		}

		[Fact]
		public void NeuralNetworkProperWeightsSetTestWithSchema2_2_2WithBias()
		{
			int numberOfInputs = 2;
			int[] hiddenLayers = { 2 };
			int numberOfOutputs = 2;
			var fileWithWeights = new Csv(new MockFileLocator(@"Mock\2_2_2_Weights.csv"), ';');
			var fileWithBiases = new Csv(new MockFileLocator(@"Mock\2_2_2_Biases.csv"), ';');

			var neuralNetwork = new NeuralNetwork(numberOfInputs, numberOfOutputs, hiddenLayers);
			neuralNetwork.SetWeights(fileWithWeights);

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

			neuralNetwork.SetBiases(fileWithBiases);

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
			Assert.Equal(expectedWeights, neuralNetwork.ConnectionsWeights);
		}

		[Fact]
		public void NeuralNetworkProperWeightsSetTestWithSchema2_3_2_2WithBias()
		{
			int numberOfInputs = 2;
			int[] hiddenLayers = { 3, 2 };
			int numberOfOutputs = 2;
			var fileWithWeights = new Csv(new MockFileLocator(@"Mock\2_3_2_2_Weights.csv"), ';');
			var fileWithBiases = new Csv(new MockFileLocator(@"Mock\2_3_2_2_Biases.csv"), ';');

			var neuralNetwork = new NeuralNetwork(numberOfInputs, numberOfOutputs, hiddenLayers);
			neuralNetwork.SetWeights(fileWithWeights);

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

			neuralNetwork.SetBiases(fileWithBiases);

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
			Assert.Equal(expectedWeights, neuralNetwork.ConnectionsWeights);
		}

		[Fact]
		public void NetworkOutputTestWithSchema_2_2AndBiases()
		{
			int numberOfInputs = 2;
			int numberOfOutputs = 2;
			var fileWithWeights = new Csv(new MockFileLocator(@"Mock\2_2_Weights.csv"), ';');
			var fileWithBiases = new Csv(new MockFileLocator(@"Mock\2_2_Biases.csv"), ';');
			double[] input = { 5.0, 3.0 };
			double[] expected = { 0.268941421369995, 0.731058578630005 };

			var neuralNetwork = new NeuralNetwork(numberOfInputs, numberOfOutputs);
			neuralNetwork.SetWeights(fileWithWeights);
			neuralNetwork.SetBiases(fileWithBiases);

			var actual = neuralNetwork.ComputeOutput(input);

			Assert.Equal(expected[0], actual[0], 9);
			Assert.Equal(expected[1], actual[1], 9);
		}

		[Fact]
		public void NetworkOutputTestWithSchema_2_2_2AndBiases()
		{
			int numberOfInputs = 2;
			int[] hiddenLayers = { 2 };
			int numberOfOutputs = 2;
			var fileWithWeights = new Csv(new MockFileLocator(@"Mock\2_2_2_Weights.csv"), ';');
			var fileWithBiases = new Csv(new MockFileLocator(@"Mock\2_2_2_Biases.csv"), ';');
			double[] input = { 5.0, 3.0 };
			double[] expected = { 0.668165494750142, 0.331834505249858 };

			var neuralNetwork = new NeuralNetwork(numberOfInputs, numberOfOutputs, hiddenLayers);
			neuralNetwork.SetWeights(fileWithWeights);
			neuralNetwork.SetBiases(fileWithBiases);

			var actual = neuralNetwork.ComputeOutput(input);

			Assert.Equal(expected[0], actual[0], 9);
			Assert.Equal(expected[1], actual[1], 9);
		}

		[Fact]
		public void NetworkOutputTestWithSchema_2_3_2_2AndBiases()
		{
			int numberOfInputs = 2;
			int[] hiddenLayers = { 3, 2 };
			int numberOfOutputs = 2;
			var fileWithWeights = new Csv(new MockFileLocator(@"Mock\2_3_2_2_Weights.csv"), ';');
			var fileWithBiases = new Csv(new MockFileLocator(@"Mock\2_3_2_2_Biases.csv"), ';');
			double[] input = { 5.0, 3.0 };
			double[] expected = { 0.522961404687474, 0.477038595312526 };

			var neuralNetwork = new NeuralNetwork(numberOfInputs, numberOfOutputs, hiddenLayers);
			neuralNetwork.SetWeights(fileWithWeights);
			neuralNetwork.SetBiases(fileWithBiases);

			var actual = neuralNetwork.ComputeOutput(input);

			Assert.Equal(expected[0], actual[0], 9);
			Assert.Equal(expected[1], actual[1], 9);
		}
	}
}
