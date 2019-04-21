using System.Linq;
using FileDeserializer.CSV;
using NeuralNetworks.Network;
using NeuralNetworksTests.Mock;
using Xunit;

namespace NeuralNetworksTests.Network
{
	public class FeedForwardNeuralNetworkTests
	{
		[Fact]
		public void NetworkOutputTestWithSchema_2_2()
		{
			int numberOfInputs = 2;
			int numberOfOutputs = 2;
			var fileWithWeights = new Csv(new MockFileLocator(@"Mock\2_2_Weights.csv"), ';');
			double[] input = {5.0, 3.0};
			double[] expected = {0.141851064900488, 0.858148935099512};

			var neuralNetwork = new FeedForwardNeuralNetwork(numberOfInputs, numberOfOutputs);
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
			double[] input = {5.0, 3.0};
			double[] expected = {0.524815756471604, 0.475184243528396};

			var neuralNetwork = new FeedForwardNeuralNetwork(numberOfInputs, numberOfOutputs, hiddenLayers);
			neuralNetwork.SetWeights(fileWithWeights);

			var actual = neuralNetwork.ComputeOutput(input);

			Assert.Equal(expected[0], actual[0], 9);
			Assert.Equal(expected[1], actual[1], 9);
		}

		[Fact]
		public void NetworkOutputTestWithSchema_2_3_2_2()
		{
			int numberOfInputs = 2;
			int[] hiddenLayers = {3, 2};
			int numberOfOutputs = 2;
			var fileWithWeights = new Csv(new MockFileLocator(@"Mock\2_3_2_2_Weights.csv"), ';');
			double[] input = {5.0, 3.0};
			double[] expected = {0.523358076385092, 0.476641923614908};

			var neuralNetwork = new FeedForwardNeuralNetwork(numberOfInputs, numberOfOutputs, hiddenLayers);
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
			double[] input = {5.0, 3.0};

			var neuralNetwork = new FeedForwardNeuralNetwork(numberOfInputs, numberOfOutputs);
			neuralNetwork.SetWeights(fileWithWeights);

			var actual = neuralNetwork.ComputeOutput(input);

			Assert.Equal(1.0, actual.Select(p => p).Sum(), 2);
		}		

		[Fact]
		public void NeuralNetworkWith5HiddenLayersOutputTest()
		{
			int numberOfInputs = 3;
			int[] hiddenLayers = {5, 4, 3, 3, 3};
			int numberOfOutputs = 3;
			var randomNumbers = new Csv(new MockFileLocator(@"Mock\Any_Weights.csv"), ';');

			double[] input = {1.0, 1.0, 1.0};

			var neuralNetwork = new FeedForwardNeuralNetwork(numberOfInputs, numberOfOutputs, hiddenLayers);
			neuralNetwork.SetWeights(randomNumbers);

			neuralNetwork.ComputeOutput(input);
		}

		[Fact]
		public void NetworkOutputTestWithSchema_2_2AndBiases()
		{
			int numberOfInputs = 2;
			int numberOfOutputs = 2;
			var fileWithWeights = new Csv(new MockFileLocator(@"Mock\2_2_Weights.csv"), ';');
			var fileWithBiases = new Csv(new MockFileLocator(@"Mock\2_2_Biases.csv"), ';');
			double[] input = {5.0, 3.0};
			double[] expected = {0.268941421369995, 0.731058578630005};

			var neuralNetwork = new FeedForwardNeuralNetwork(numberOfInputs, numberOfOutputs);
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
			int[] hiddenLayers = {2};
			int numberOfOutputs = 2;
			var fileWithWeights = new Csv(new MockFileLocator(@"Mock\2_2_2_Weights.csv"), ';');
			var fileWithBiases = new Csv(new MockFileLocator(@"Mock\2_2_2_Biases.csv"), ';');
			double[] input = {5.0, 3.0};
			double[] expected = {0.668165494750142, 0.331834505249858};

			var neuralNetwork = new FeedForwardNeuralNetwork(numberOfInputs, numberOfOutputs, hiddenLayers);
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
			int[] hiddenLayers = {3, 2};
			int numberOfOutputs = 2;
			var fileWithWeights = new Csv(new MockFileLocator(@"Mock\2_3_2_2_Weights.csv"), ';');
			var fileWithBiases = new Csv(new MockFileLocator(@"Mock\2_3_2_2_Biases.csv"), ';');
			double[] input = {5.0, 3.0};
			double[] expected = {0.522961404687474, 0.477038595312526};

			var neuralNetwork = new FeedForwardNeuralNetwork(numberOfInputs, numberOfOutputs, hiddenLayers);
			neuralNetwork.SetWeights(fileWithWeights);
			neuralNetwork.SetBiases(fileWithBiases);

			var actual = neuralNetwork.ComputeOutput(input);

			Assert.Equal(expected[0], actual[0], 9);
			Assert.Equal(expected[1], actual[1], 9);
		}

		[Fact]
		public void NetworkAccuracyMethodTest()
		{
			int numberOfInputs = 2;
			int numberOfOutputs = 2;
			double expectedAccuracy = 0.333;

			var startingWeights = new Csv(new MockFileLocator(@"Mock\2_2_Weights.csv"), ';');
			var startingBiases = new Csv(new MockFileLocator(@"Mock\2_2_Biases.csv"), ';');
			var data = new Csv(new MockFileLocator(@"Mock\AccuracyTestData.csv"), ';');

			var neuralNetwork = new FeedForwardNeuralNetwork(numberOfInputs, numberOfOutputs);
			neuralNetwork.SetWeights(startingWeights);
			neuralNetwork.SetBiases(startingBiases);

			Assert.Equal(expectedAccuracy, neuralNetwork.GetAccuracy(data),3);
		}

		[Fact]
		public void NetworkTraningProcessAccuracyTest2_2WithBias()
		{
			int numberOfInputs = 2;
			int numberOfOutputs = 2;
			int maxEpochs = 1000;
			double learningRate = 0.10;
			var traningData2Inputs2Outputs = new Csv(new MockFileLocator(@"Mock\traning\data\Traning2Inputs2Outputs.csv"), ';');
			var testData2Inputs2Outputs = new Csv(new MockFileLocator(@"Mock\traning\data\Test2Inputs2Outputs.csv"), ';');
			var randomNumbers = new Csv(new MockFileLocator(@"Mock\Any_Weights.csv"), ';');

			var neuralNetwork = new FeedForwardNeuralNetwork(numberOfInputs, numberOfOutputs);
			neuralNetwork.SetWeights(randomNumbers);
			neuralNetwork.SetBiases(randomNumbers);

			double accuracyBeforeTraning = neuralNetwork.GetAccuracy(testData2Inputs2Outputs);

			neuralNetwork.BackPropagationTrain(traningData2Inputs2Outputs, maxEpochs, learningRate);

			double accuracyAfterTraning = neuralNetwork.GetAccuracy(testData2Inputs2Outputs);

			Assert.True(accuracyAfterTraning > accuracyBeforeTraning, $"accuracyAfterTraning: {accuracyAfterTraning}, accuracyBeforeTraning: {accuracyBeforeTraning}");
		}

		[Fact]
		public void NetworkTraningProcessAccuracyTest2_2WithoutBias()
		{
			int numberOfInputs = 2;
			int numberOfOutputs = 2;
			int maxEpochs = 1000;
			double learningRate = 0.10;
			var traningData2Inputs2Outputs = new Csv(new MockFileLocator(@"Mock\traning\data\Traning2Inputs2Outputs.csv"), ';');
			var testData2Inputs2Outputs = new Csv(new MockFileLocator(@"Mock\traning\data\Test2Inputs2Outputs.csv"), ';');
			var randomNumbers = new Csv(new MockFileLocator(@"Mock\Any_Weights.csv"), ';'); //TODO do wywalenia wszędzie ten plik jak bd losowe wagi generowane

			var neuralNetwork = new FeedForwardNeuralNetwork(numberOfInputs, numberOfOutputs);
			neuralNetwork.SetWeights(randomNumbers);

			double accuracyBeforeTraning = neuralNetwork.GetAccuracy(testData2Inputs2Outputs);

			neuralNetwork.BackPropagationTrain(traningData2Inputs2Outputs, maxEpochs, learningRate);

			double accuracyAfterTraning = neuralNetwork.GetAccuracy(testData2Inputs2Outputs);

			Assert.True(accuracyAfterTraning > accuracyBeforeTraning, $"accuracyAfterTraning: {accuracyAfterTraning}, accuracyBeforeTraning: {accuracyBeforeTraning}");
		}

		[Fact]
		public void NetworkTraningProcessAccuracyTest2_3_2WithBias()
		{
			int numberOfInputs = 2;
			int[] hiddenLayers = { 3 };
			int numberOfOutputs = 2;
			var traningData = new Csv(new MockFileLocator(@"Mock\traning\data\Traning2Inputs2Outputs.csv"), ';');
			var testData = new Csv(new MockFileLocator(@"Mock\traning\data\Test2Inputs2Outputs.csv"), ';');
			var randomNumbers = new Csv(new MockFileLocator(@"Mock\Any_Weights.csv"), ';');
			int maxEpochs = 1000;
			double learningRate = 0.10;

			var neuralNetwork = new FeedForwardNeuralNetwork(numberOfInputs, numberOfOutputs, hiddenLayers);
			neuralNetwork.SetWeights(randomNumbers);
			neuralNetwork.SetBiases(randomNumbers);

			double accuracyBeforeTraning = neuralNetwork.GetAccuracy(testData);

			neuralNetwork.BackPropagationTrain(traningData, maxEpochs, learningRate);

			double accuracyAfterTraning = neuralNetwork.GetAccuracy(testData);

			Assert.True(accuracyAfterTraning > accuracyBeforeTraning, $"accuracyAfterTraning: {accuracyAfterTraning}, accuracyBeforeTraning: {accuracyBeforeTraning}");
		}

		[Fact]
		public void NetworkTraningProcessAccuracyTest2_3_2WithoutBias()
		{
			int numberOfInputs = 2;
			int[] hiddenLayers = { 3 };
			int numberOfOutputs = 2;
			var traningData = new Csv(new MockFileLocator(@"Mock\traning\data\Traning2Inputs2Outputs.csv"), ';');
			var testData = new Csv(new MockFileLocator(@"Mock\traning\data\Test2Inputs2Outputs.csv"), ';');
			var randomNumbers = new Csv(new MockFileLocator(@"Mock\Any_Weights.csv"), ';');
			int maxEpochs = 1000;
			double learningRate = 0.10;

			var neuralNetwork = new FeedForwardNeuralNetwork(numberOfInputs, numberOfOutputs, hiddenLayers);
			neuralNetwork.SetWeights(randomNumbers);

			double accuracyBeforeTraning = neuralNetwork.GetAccuracy(testData);

			neuralNetwork.BackPropagationTrain(traningData, maxEpochs, learningRate);

			double accuracyAfterTraning = neuralNetwork.GetAccuracy(testData);

			Assert.True(accuracyAfterTraning > accuracyBeforeTraning, $"accuracyAfterTraning: {accuracyAfterTraning}, accuracyBeforeTraning: {accuracyBeforeTraning}");
		}

		[Fact]
		public void NetworkTraningProcessAccuracyTest3_3_2WithBias()
		{
			int numberOfInputs = 3;
			int[] hiddenLayers = { 3 };
			int numberOfOutputs = 2;
			var traningData = new Csv(new MockFileLocator(@"Mock\traning\data\Traning3Inputs2Outputs.csv"), ';');
			var testData = new Csv(new MockFileLocator(@"Mock\traning\data\Test3Inputs2Outputs.csv"), ';');
			var randomNumbers = new Csv(new MockFileLocator(@"Mock\Any_Weights.csv"), ';');
			int maxEpochs = 1000;
			double learningRate = 0.10;

			var neuralNetwork = new FeedForwardNeuralNetwork(numberOfInputs, numberOfOutputs, hiddenLayers);
			neuralNetwork.SetWeights(randomNumbers);
			neuralNetwork.SetBiases(randomNumbers);

			double accuracyBeforeTraning = neuralNetwork.GetAccuracy(testData);

			neuralNetwork.BackPropagationTrain(traningData, maxEpochs, learningRate);

			double accuracyAfterTraning = neuralNetwork.GetAccuracy(testData);

			Assert.True(accuracyAfterTraning > accuracyBeforeTraning, $"accuracyAfterTraning: {accuracyAfterTraning}, accuracyBeforeTraning: {accuracyBeforeTraning}");
		}

		[Fact]
		public void NetworkTraningProcessAccuracyTest3_3_2WithoutBias()
		{
			int numberOfInputs = 3;
			int[] hiddenLayers = { 3 };
			int numberOfOutputs = 2;
			var traningData = new Csv(new MockFileLocator(@"Mock\traning\data\Traning3Inputs2Outputs.csv"), ';');
			var testData = new Csv(new MockFileLocator(@"Mock\traning\data\Test3Inputs2Outputs.csv"), ';');
			var randomNumbers = new Csv(new MockFileLocator(@"Mock\Any_Weights.csv"), ';');
			int maxEpochs = 1000;
			double learningRate = 0.10;

			var neuralNetwork = new FeedForwardNeuralNetwork(numberOfInputs, numberOfOutputs, hiddenLayers);
			neuralNetwork.SetWeights(randomNumbers);

			double accuracyBeforeTraning = neuralNetwork.GetAccuracy(testData);

			neuralNetwork.BackPropagationTrain(traningData, maxEpochs, learningRate);

			double accuracyAfterTraning = neuralNetwork.GetAccuracy(testData);

			Assert.True(accuracyAfterTraning > accuracyBeforeTraning, $"accuracyAfterTraning: {accuracyAfterTraning}, accuracyBeforeTraning: {accuracyBeforeTraning}");
		}

		[Fact]
		public void NetworkTraningProcessAccuracyTest3_3_2_4WithBias()
		{
			int numberOfInputs = 3;
			int[] hiddenLayers = { 3, 2 };
			int numberOfOutputs = 4;
			var traningData = new Csv(new MockFileLocator(@"Mock\traning\data\Traning3Inputs4Outputs.csv"), ';');
			var testData = new Csv(new MockFileLocator(@"Mock\traning\data\Test3Inputs4Outputs.csv"), ';');
			var randomNumbers = new Csv(new MockFileLocator(@"Mock\Any_Weights.csv"), ';');
			int maxEpochs = 1000;
			double learningRate = 0.10;

			var neuralNetwork = new FeedForwardNeuralNetwork(numberOfInputs, numberOfOutputs, hiddenLayers);
			neuralNetwork.SetWeights(randomNumbers);
			neuralNetwork.SetBiases(randomNumbers);

			double accuracyBeforeTraning = neuralNetwork.GetAccuracy(testData);

			neuralNetwork.BackPropagationTrain(traningData, maxEpochs, learningRate);

			double accuracyAfterTraning = neuralNetwork.GetAccuracy(testData);

			Assert.True(accuracyAfterTraning > accuracyBeforeTraning,$"accuracyAfterTraning: {accuracyAfterTraning}, accuracyBeforeTraning: {accuracyBeforeTraning}");
		}

		[Fact]
		public void NetworkTraningProcessAccuracyTest3_3_2_4WithoutBias()
		{
			int numberOfInputs = 3;
			int[] hiddenLayers = { 3, 2 };
			int numberOfOutputs = 4;
			var traningData = new Csv(new MockFileLocator(@"Mock\traning\data\Traning3Inputs4Outputs.csv"), ';');
			var testData = new Csv(new MockFileLocator(@"Mock\traning\data\Test3Inputs4Outputs.csv"), ';');
			var randomNumbers = new Csv(new MockFileLocator(@"Mock\Any_Weights.csv"), ';');
			int maxEpochs = 7000;
			double learningRate = 0.90;

			var neuralNetwork = new FeedForwardNeuralNetwork(numberOfInputs, numberOfOutputs, hiddenLayers);
			neuralNetwork.SetWeights(randomNumbers);

			double accuracyBeforeTraning = neuralNetwork.GetAccuracy(testData);

			neuralNetwork.BackPropagationTrain(traningData, maxEpochs, learningRate);

			double accuracyAfterTraning = neuralNetwork.GetAccuracy(testData);

			Assert.True(accuracyAfterTraning > accuracyBeforeTraning, $"accuracyAfterTraning: {accuracyAfterTraning}, accuracyBeforeTraning: {accuracyBeforeTraning}");
		}

		[Fact]
		public void NetworkTraningProcessOnTitanicDataWithoutBias()
		{
			int numberOfInputs = 5;
			int[] hiddenLayers = { 3, 2 };
			int numberOfOutputs = 2;

			var randomNumbers = new Csv(new MockFileLocator(@"Mock\Any_Weights.csv"), ';');
			var traningData = new Csv(new MockFileLocator(@"Mock\traning\data\Titanic_surviving_traning_data.csv"), ';');
			var testData = new Csv(new MockFileLocator(@"Mock\traning\data\Titanic_surviving_test_data.csv"), ';');
			int maxEpochs = 7000;
			double learningRate = 0.10;

			var neuralNetwork = new FeedForwardNeuralNetwork(numberOfInputs, numberOfOutputs, hiddenLayers);
			neuralNetwork.SetWeights(randomNumbers);

			double accuracyBeforeTraning = neuralNetwork.GetAccuracy(testData);

			neuralNetwork.BackPropagationTrain(traningData, maxEpochs, learningRate);

			double accuracyAfterTraning = neuralNetwork.GetAccuracy(testData);

			Assert.True(accuracyAfterTraning > accuracyBeforeTraning, $"accuracyAfterTraning: {accuracyAfterTraning}, accuracyBeforeTraning: {accuracyBeforeTraning}");
			Assert.True(accuracyAfterTraning > 0.5, $"accuracyAfterTraning: {accuracyAfterTraning}, expected more than 0,5.");
		}


		[Fact]
		public void NetworkTraningProcessOnTitanicDataWithBias()
		{
			int numberOfInputs = 5;
			int[] hiddenLayers = { 3, 2 };
			int numberOfOutputs = 2;

			var traningData = new Csv(new MockFileLocator(@"Mock\traning\data\Titanic_surviving_traning_data.csv"), ';');
			var testData = new Csv(new MockFileLocator(@"Mock\traning\data\Titanic_surviving_test_data.csv"), ';');
			var randomNumbers = new Csv(new MockFileLocator(@"Mock\Any_Weights.csv"), ';');
			int maxEpochs = 5000;
			double learningRate = 0.50;

			var neuralNetwork = new FeedForwardNeuralNetwork(numberOfInputs, numberOfOutputs, hiddenLayers);
			neuralNetwork.SetWeights(randomNumbers);
			neuralNetwork.SetBiases(randomNumbers);

			double accuracyBeforeTraning = neuralNetwork.GetAccuracy(testData);

			neuralNetwork.BackPropagationTrain(traningData, maxEpochs, learningRate);

			double accuracyAfterTraning = neuralNetwork.GetAccuracy(testData);

			Assert.True(accuracyAfterTraning > accuracyBeforeTraning, $"accuracyAfterTraning: {accuracyAfterTraning}, accuracyBeforeTraning: {accuracyBeforeTraning}");
			Assert.True(accuracyAfterTraning > 0.5, $"accuracyAfterTraning: {accuracyAfterTraning}, expected more than 0,5.");
		}

		[Fact]
		public void WasWeightsNotInitializedExceptionThrownOnComputeOutputMethodTest()
		{
			int numberOfInputs = 1;
			int numberOfOutputs = 1;
			double[] input = {1.0};

			var neuralNetwork = new FeedForwardNeuralNetwork(numberOfInputs, numberOfOutputs);

			Assert.Throws<WeightsNotInitializedException>(() => neuralNetwork.ComputeOutput(input));
		}

		[Fact]
		public void WasWeightsNotInitializedExceptionThrownOnBackPropagationTrainMethodTest()
		{
			int numberOfInputs = 2;
			int numberOfOutputs = 2;
			int maxEpochs = 5000;
			double learningRate = 0.10;
			var traningData2Inputs2Outputs = new Csv(new MockFileLocator(@"Mock\traning\data\Traning2Inputs2Outputs.csv"), ';');

			var neuralNetwork = new FeedForwardNeuralNetwork(numberOfInputs, numberOfOutputs);

			Assert.Throws<WeightsNotInitializedException>(() => neuralNetwork.BackPropagationTrain(traningData2Inputs2Outputs, maxEpochs, learningRate));
		}


		[Fact]
		public void WasWrongSourceFileLengthExceptionThrownOnBackPropagationTrainMethodToManyTraningDataTest()
		{
			int numberOfInputs = 4;
			int numberOfOutputs = 2;
			int maxEpochs = 5000;
			double learningRate = 0.10;
			var randomNumbers = new Csv(new MockFileLocator(@"Mock\Any_Weights.csv"), ';');
			var traningData = new Csv(new MockFileLocator(@"Mock\traning\data\Titanic_surviving_traning_data.csv"), ';');

			var neuralNetwork = new FeedForwardNeuralNetwork(numberOfInputs, numberOfOutputs);
			neuralNetwork.SetWeights(randomNumbers);

			Assert.Throws<WrongSourceFileLengthException>(() => neuralNetwork.BackPropagationTrain(traningData, maxEpochs, learningRate));
		}

		[Fact]
		public void WasWrongSourceFileLengthExceptionThrownOnBackPropagationTrainMethodNotEnoughTraningDataTest()
		{
			int numberOfInputs = 5;
			int numberOfOutputs = 5;
			int maxEpochs = 5000;
			double learningRate = 0.10;
			var randomNumbers = new Csv(new MockFileLocator(@"Mock\Any_Weights.csv"), ';');
			var traningData = new Csv(new MockFileLocator(@"Mock\traning\data\Titanic_surviving_traning_data.csv"), ';');

			var neuralNetwork = new FeedForwardNeuralNetwork(numberOfInputs, numberOfOutputs);
			neuralNetwork.SetWeights(randomNumbers);

			Assert.Throws<WrongSourceFileLengthException>(() => neuralNetwork.BackPropagationTrain(traningData, maxEpochs, learningRate));
		}

		[Fact]
		public void WasWrongSourceFileLengthExceptionThrownOnComputeOutputMethodToManyInputsTest()
		{
			int numberOfInputs = 4;
			int numberOfOutputs = 2;
			var randomNumbers = new Csv(new MockFileLocator(@"Mock\Any_Weights.csv"), ';');
			double[] input = {3.0, 4.0, 5.0, 3.0, 6.0};


			var neuralNetwork = new FeedForwardNeuralNetwork(numberOfInputs, numberOfOutputs);
			neuralNetwork.SetWeights(randomNumbers);

			Assert.Throws<WrongSourceFileLengthException>(() => neuralNetwork.ComputeOutput(input));
		}

		[Fact]
		public void WasWrongSourceFileLengthExceptionThrownOnComputeOutputMethodNotEnoughInputsTest()
		{
			int numberOfInputs = 4;
			int numberOfOutputs = 2;
			var randomNumbers = new Csv(new MockFileLocator(@"Mock\Any_Weights.csv"), ';');
			double[] input = { 3.0, 4.0, 5.0 };


			var neuralNetwork = new FeedForwardNeuralNetwork(numberOfInputs, numberOfOutputs);
			neuralNetwork.SetWeights(randomNumbers);

			Assert.Throws<WrongSourceFileLengthException>(() => neuralNetwork.ComputeOutput(input));
		}

		[Fact]
		public void WasWrongSourceFileLengthExceptionThrownOnGetAccuracyMethodToManyInputsTest()
		{
			int numberOfInputs = 4;
			int numberOfOutputs = 2;
			var randomNumbers = new Csv(new MockFileLocator(@"Mock\Any_Weights.csv"), ';');
			var testData = new Csv(new MockFileLocator(@"Mock\traning\data\Titanic_surviving_test_data.csv"), ';');

			var neuralNetwork = new FeedForwardNeuralNetwork(numberOfInputs, numberOfOutputs);
			neuralNetwork.SetWeights(randomNumbers);

			Assert.Throws<WrongSourceFileLengthException>(() => neuralNetwork.GetAccuracy(testData));
		}

		[Fact]
		public void WasWrongSourceFileLengthExceptionThrownOnGetAccuracyMethodNotEnoughInputsTest()
		{
			int numberOfInputs = 6;
			int numberOfOutputs = 2;
			var randomNumbers = new Csv(new MockFileLocator(@"Mock\Any_Weights.csv"), ';');
			var testData = new Csv(new MockFileLocator(@"Mock\traning\data\Titanic_surviving_test_data.csv"), ';');


			var neuralNetwork = new FeedForwardNeuralNetwork(numberOfInputs, numberOfOutputs);
			neuralNetwork.SetWeights(randomNumbers);

			Assert.Throws<WrongSourceFileLengthException>(() => neuralNetwork.GetAccuracy(testData));
		}
	}
}
