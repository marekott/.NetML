using System;
using NeuralNetworks.Neurons;
using Xunit;

namespace NeuralNetworksTests.Neurons
{
	public class NeuronHiddenLayerTests
	{
		private readonly double[] _inputs;
		private readonly double[] _biases;

		public NeuronHiddenLayerTests()
		{
			_inputs = new[] { -30.0, 0.0, 50.0, 60.0, -70.0, 80.0 };
			_biases = new[] { 60.0, -5.0, -50.0 };
		}

		[Fact]
		public void ComputeOutputForFirstNeuronInLayerForInputLessThan0()
		{
			var neuronHiddenLayer = new NeuronHiddenLayer(0);

			double expected = -1.0;
			var result = neuronHiddenLayer.ComputeOutput(_inputs);

			Assert.Equal(expected, result, 2);
		}

		[Fact]
		public void ComputeOutputForSecondNeuronInLayerForInputEqual0()
		{
			var neuronHiddenLayer = new NeuronHiddenLayer(1);

			double expected = 0.0;
			var result = neuronHiddenLayer.ComputeOutput(_inputs);

			Assert.Equal(expected, result, 2);
		}

		[Fact]
		public void ComputeOutputForThirdNeuronInLayerForInputGraterThan0()
		{
			var neuronHiddenLayer = new NeuronHiddenLayer(2);

			double expected = 1.0;
			var result = neuronHiddenLayer.ComputeOutput(_inputs);

			Assert.Equal(expected, result, 2);
		}

		[Fact]
		public void ComputeOutputForFirstNeuronInLayerForInputLessThan0AndBiases()
		{
			var neuronHiddenLayer = new NeuronHiddenLayer(0);

			double expected = 1.0;
			var result = neuronHiddenLayer.ComputeOutput(_inputs, _biases);

			Assert.Equal(expected, result, 2);
		}

		[Fact]
		public void ComputeOutputForSecondNeuronInLayerForInputEqual0AndBiases()
		{
			var neuronHiddenLayer = new NeuronHiddenLayer(1);

			double expected = -1.0;
			var result = neuronHiddenLayer.ComputeOutput(_inputs, _biases);

			Assert.Equal(expected, result, 2);
		}

		[Fact]
		public void ComputeOutputForThirdNeuronInLayerForInputGraterThan0AndBiases()
		{
			var neuronHiddenLayer = new NeuronHiddenLayer(2);

			double expected = 0.0;
			var result = neuronHiddenLayer.ComputeOutput(_inputs, _biases);

			Assert.Equal(expected, result, 2);
		}

		[Fact]
		public void ThrowIndexWasOutOfRangeException()
		{
			var neuronHiddenLayer = new NeuronHiddenLayer(4);

			Assert.Throws<IndexOutOfRangeException>(() => neuronHiddenLayer.ComputeOutput(_inputs, _biases));
		}
	}
}