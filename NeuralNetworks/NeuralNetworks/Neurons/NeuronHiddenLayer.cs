﻿using static System.Math;

namespace NeuralNetworks.Neurons
{
	public class NeuronHiddenLayer : Neuron
	{
		/// <summary>
		/// </summary>
		/// <param name="neuronPositionFromTop">counting from 0</param>
		public NeuronHiddenLayer(int neuronPositionFromTop) : base(neuronPositionFromTop)
		{
		}

		public override double ComputeOutput(double[] inputs)
		{
			return Tanh(inputs[NeuronPositionFromTop]);
		}
	}
}
