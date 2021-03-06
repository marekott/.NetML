﻿using System;
using NeuralNetworks.Neurons;
using Xunit;

namespace NeuralNetworksTests.Neurons
{
	public class NeuronOutputLayerTests
	{
		private readonly double[] _biases;

		public NeuronOutputLayerTests()
		{
			_biases = new[] { 1.0, 1.0 };
		}

		[Fact]
		public void NnOutputFromTwoOutputsFirstNeuronTestForMinus3()
		{
			double[] input = { -3.0, 3.0 };

			double excepted = 0.00247262315663477;
			var actual = new NeuronOutputLayer(0).ComputeOutput(input);

			Assert.Equal(excepted, actual, 15);
		}

		[Fact]
		public void NnOutputFromTwoOutputsSecondNeuronTestFor3()
		{
			double[] input = { -3.0, 3.0 };

			double excepted = 0.997527376843365;
			var actual = new NeuronOutputLayer(1).ComputeOutput(input);

			Assert.Equal(excepted, actual, 15);
		}

		[Fact]
		public void NnOutputFromTwoOutputsFirstNeuronTestForTwice0()
		{
			double[] input = { 0.0, 0.0 };

			double excepted = 0.5;
			var actual = new NeuronOutputLayer(0).ComputeOutput(input);

			Assert.Equal(excepted, actual, 15);
		}

		[Fact]
		public void NnOutputFromTwoOutputsSecondNeuronTestForTwice0()
		{
			double[] input = { 0.0, 0.0 };

			double excepted = 0.5;
			var actual = new NeuronOutputLayer(1).ComputeOutput(input);

			Assert.Equal(excepted, actual, 15);
		}

		[Fact]
		public void NnOutputFromTwoOutputsFirstNeuronTestForDifferentSignsInInput()
		{
			double[] input = { 3.7, -5.4 };

			double excepted = 0.99988834665937;
			var actual = new NeuronOutputLayer(0).ComputeOutput(input);

			Assert.Equal(excepted, actual, 15);
		}

		[Fact]
		public void NnOutputFromTwoOutputsSecondNeuronTestForDifferentSignsInInput()
		{
			double[] input = { 3.7, -5.4 };

			double excepted = 0.000111653340629563;
			var actual = new NeuronOutputLayer(1).ComputeOutput(input);

			Assert.Equal(excepted, actual, 15);
		}

		[Fact]
		public void NnOutputFromThreeOutputs()
		{
			double[] input = { 1.0, 2.0, 3.0 };

			double exceptedFirstNeuron = 0.0900305731703805;
			double exceptedSecondNeuron = 0.244728471054798;
			double exceptedThirdNeuron = 0.665240955774822;

			var actualFirstNeuron = new NeuronOutputLayer(0).ComputeOutput(input);
			var actualSecondNeuron = new NeuronOutputLayer(1).ComputeOutput(input);
			var actualThirdNeuron = new NeuronOutputLayer(2).ComputeOutput(input);

			Assert.Equal(exceptedFirstNeuron, actualFirstNeuron, 15);
			Assert.Equal(exceptedSecondNeuron, actualSecondNeuron, 15);
			Assert.Equal(exceptedThirdNeuron, actualThirdNeuron, 15);
		}

		[Fact]
		public void NnOutputFromFiveOutputs()
		{
			double[] input = { 1.0, 2.0, 3.0, 4.0, 5.0 };

			double exceptedFirstNeuron = 0.0116562309560396;
			double exceptedSecondNeuron = 0.0316849207961243;
			double exceptedThirdNeuron = 0.0861285444362687;
			double exceptedFourthNeuron = 0.234121657252737;
			double exceptedFifthNeuron = 0.636408646558831;

			var actualFirstNeuron = new NeuronOutputLayer(0).ComputeOutput(input);
			var actualSecondNeuron = new NeuronOutputLayer(1).ComputeOutput(input);
			var actualThirdNeuron = new NeuronOutputLayer(2).ComputeOutput(input);
			var actualFourthNeuron = new NeuronOutputLayer(3).ComputeOutput(input);
			var actualFifthNeuron = new NeuronOutputLayer(4).ComputeOutput(input);

			Assert.Equal(exceptedFirstNeuron, actualFirstNeuron, 15);
			Assert.Equal(exceptedSecondNeuron, actualSecondNeuron, 15);
			Assert.Equal(exceptedThirdNeuron, actualThirdNeuron, 15);
			Assert.Equal(exceptedFourthNeuron, actualFourthNeuron, 15);
			Assert.Equal(exceptedFifthNeuron, actualFifthNeuron, 15);
		}
	}
}