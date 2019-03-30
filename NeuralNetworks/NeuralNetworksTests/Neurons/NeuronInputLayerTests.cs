using System;
using NeuralNetworks.Neurons;
using Xunit;

namespace NeuralNetworksTests.Neurons
{
	public class NeuronInputLayerTests
	{
		private readonly double[] _inputsWith3Numbers;
		private readonly double[] _inputsWith10Numbers;
		private readonly double[] _inputsWith3Biases;
		private readonly double[] _inputsWith10Biases;

		public NeuronInputLayerTests()
		{
			_inputsWith3Numbers = new[] { 1.0, 2.0, 3.0 };
			_inputsWith10Numbers = new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 };
			_inputsWith3Biases = new[] { 5.0, 10.0, 20.0 };
			_inputsWith10Biases = new[] { 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0 };
		}

		[Fact]
		public void ComputeOutputFirstNeuronInLayerOneInputTest()
		{
			var numberOfInputs = 1;
			var neuronPositionFromTop = 0;

			var neuron = new NeuronInputLayer(neuronPositionFromTop, numberOfInputs);

			var neuronOutput = neuron.ComputeOutput(_inputsWith3Numbers);

			Assert.Equal(_inputsWith3Numbers[neuronPositionFromTop], neuronOutput, 5);
		}

		[Fact]
		public void ComputeOutputSecondNeuronInLayerOneInputTest()
		{
			var numberOfInputs = 1;
			var neuronPositionFromTop = 1;

			var neuron = new NeuronInputLayer(neuronPositionFromTop, numberOfInputs);

			var neuronOutput = neuron.ComputeOutput(_inputsWith3Numbers);

			Assert.Equal(_inputsWith3Numbers[neuronPositionFromTop], neuronOutput, 5);
		}

		[Fact]
		public void ComputeOutputThirdNeuronInLayerOneInputTest()
		{
			var numberOfInputs = 1;
			var neuronPositionFromTop = 2;

			var neuron = new NeuronInputLayer(neuronPositionFromTop, numberOfInputs);

			var neuronOutput = neuron.ComputeOutput(_inputsWith3Numbers);

			Assert.Equal(_inputsWith3Numbers[neuronPositionFromTop], neuronOutput, 5);
		}

		[Fact]
		public void ComputeOutputFirstNeuronInLayerFiveInputsTest()
		{
			var numberOfInputs = 5;
			var neuronPositionFromTop = 0;

			var neuron = new NeuronInputLayer(neuronPositionFromTop, numberOfInputs);

			var neuronOutput = neuron.ComputeOutput(_inputsWith10Numbers);

			Assert.Equal(15.0, neuronOutput, 5);
		}

		[Fact]
		public void ComputeOutputSecondNeuronInLayerFiveInputsTest()
		{
			var numberOfInputs = 5;
			var neuronPositionFromTop = 1;

			var neuron = new NeuronInputLayer(neuronPositionFromTop, numberOfInputs);

			var neuronOutput = neuron.ComputeOutput(_inputsWith10Numbers);

			Assert.Equal(40.0, neuronOutput, 5);
		}

		[Fact]
		public void ComputeOutputFirstNeuronInLayerOneInputAndBiasesTest()
		{
			var numberOfInputs = 1;
			var neuronPositionFromTop = 0;

			var neuron = new NeuronInputLayer(neuronPositionFromTop, numberOfInputs);

			var neuronOutput = neuron.ComputeOutput(_inputsWith3Numbers, _inputsWith3Biases);

			Assert.Equal(6.0, neuronOutput, 5);
		}

		[Fact]
		public void ComputeOutputSecondNeuronInLayerOneInputAndBiasesTest()
		{
			var numberOfInputs = 1;
			var neuronPositionFromTop = 1;

			var neuron = new NeuronInputLayer(neuronPositionFromTop, numberOfInputs);

			var neuronOutput = neuron.ComputeOutput(_inputsWith3Numbers, _inputsWith3Biases);

			Assert.Equal(12.0, neuronOutput, 5);
		}

		[Fact]
		public void ComputeOutputThirdNeuronInLayerOneInputAndBiasesTest()
		{
			var numberOfInputs = 1;
			var neuronPositionFromTop = 2;

			var neuron = new NeuronInputLayer(neuronPositionFromTop, numberOfInputs);

			var neuronOutput = neuron.ComputeOutput(_inputsWith3Numbers, _inputsWith3Biases);

			Assert.Equal(23.0, neuronOutput, 5);
		}

		[Fact]
		public void ComputeOutputFirstNeuronInLayerFiveInputsAndBiasesTest()
		{
			var numberOfInputs = 5;
			var neuronPositionFromTop = 0;

			var neuron = new NeuronInputLayer(neuronPositionFromTop, numberOfInputs);

			var neuronOutput = neuron.ComputeOutput(_inputsWith10Numbers, _inputsWith10Biases);

			Assert.Equal(25.0, neuronOutput, 5);
		}

		[Fact]
		public void ComputeOutputSecondNeuronInLayerFiveInputsAndBiasesTest()
		{
			var numberOfInputs = 5;
			var neuronPositionFromTop = 1;

			var neuron = new NeuronInputLayer(neuronPositionFromTop, numberOfInputs);

			var neuronOutput = neuron.ComputeOutput(_inputsWith10Numbers, _inputsWith10Biases);

			Assert.Equal(49.0, neuronOutput, 5);
		}

		[Fact]
		public void ThrowIndexWasOutOfRangeException()
		{
			var numberOfInputs = 5;
			var neuronPositionFromTop = 1;

			var neuron = new NeuronInputLayer(neuronPositionFromTop, numberOfInputs);

			Assert.Throws<IndexOutOfRangeException>(() => neuron.ComputeOutput(_inputsWith3Numbers, _inputsWith3Biases));
		}
	}
}