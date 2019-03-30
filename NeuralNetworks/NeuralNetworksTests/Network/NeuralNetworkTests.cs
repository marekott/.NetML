using FileDeserializer.CSV;
using NeuralNetworks.Network;
using NeuralNetworksTests.Mock;
using Xunit;

namespace NeuralNetworksTests.Network
{
	public class NeuralNetworkTests
	{
		[Fact]
		public void NeuralNetworkConstructorTest()
		{
			int numberOfInputs = 2;
			int[] hiddenLayers = { 2 };
			int numberOfOutputs = 2;
			var fileWithWeights = new Csv(new MockFileLocator("2_2_2_Weights.csv"), ';');


			var neuralNetwork = new NeuralNetwork(numberOfInputs, hiddenLayers, numberOfOutputs, fileWithWeights);

			Assert.Equal(numberOfInputs, neuralNetwork.InputLayerNeuronsNumber);
			Assert.Equal(hiddenLayers.Length, neuralNetwork.HiddenLayersNumber);
			Assert.Equal(numberOfOutputs, neuralNetwork.OutputLayerNeuronsNumber);
		}

		[Fact]
		public void NetworkOutputTestWithSchema_2_2_2()
		{
			int numberOfInputs = 2;
			int[] hiddenLayers = {2};
			int numberOfOutputs = 2;
			var fileWithWeights = new Csv(new MockFileLocator("2_2_2_Weights.csv"), ';');
			double[] input = {5.0, 3.0 };
			double[] expected = {0.368140775, 0.333325921};

			var neuralNetwork = new NeuralNetwork(numberOfInputs, hiddenLayers, numberOfOutputs, fileWithWeights);

			var actual = neuralNetwork.ComputeOutput(input);

			Assert.Equal(expected[0],actual[0],9);
			Assert.Equal(expected[1], actual[1], 9);
		}
	}
}
