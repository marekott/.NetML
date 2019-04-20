using System;
using System.Linq;

namespace NeuralNetworks.Neurons
{
	public class NeuronOutputLayer : Neuron //TODO druga funkcja aktywacji do wyboru
	{
		/// <summary>
		/// </summary>
		/// <param name="neuronPositionFromTop">counting from 0</param>
		public NeuronOutputLayer(int neuronPositionFromTop) : base(neuronPositionFromTop)
		{
		}

		public override double ComputeOutput(double[] inputs)
		{
			var result = Softmax(inputs);

			return result;
		}

		private double Softmax(double[] inputs)
		{
			var counter = inputs.Select(Math.Exp).Sum();
			var result = Math.Exp(inputs[NeuronPositionFromTop]) / counter;

			return result;
		}
	}
}
