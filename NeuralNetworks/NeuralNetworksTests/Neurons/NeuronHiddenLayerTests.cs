using System;
using NeuralNetworks.Neurons;
using Xunit;

namespace NeuralNetworksTests.Neurons
{
	public class NeuronHiddenLayerTests
	{
		private readonly double[] _inputs;

		public NeuronHiddenLayerTests()
		{
			_inputs = new[] { -30.0, 0.0, 50.0, 60.0, -70.0, 80.0 };
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
	}
}