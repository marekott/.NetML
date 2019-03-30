using System;
using System.Linq;

namespace NeuralNetworks.Neurons
{
	public class NeuronOutputLayer : Neuron
	{
		/// <summary>
		/// </summary>
		/// <param name="neuronPositionFromTop">counting from 0</param>
		public NeuronOutputLayer(int neuronPositionFromTop) : base(neuronPositionFromTop)
		{
		}

		public override double ComputeOutput(double[] inputs, double[] biases = null)
		{
			if (biases != null)
			{
				inputs.Add(biases);
			}

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

	public static class ArrayExtensions
	{
		public static void Add(this double[] array, double[] arrayToAdd)
		{
			if (array.IsArrayEqual(arrayToAdd))
			{
				for (int i = 0; i < array.Length; i++)
				{
					array[i] += arrayToAdd[i];
				}
			}

			else
			{
				throw new IndexOutOfRangeException("Arrays haven't the same size.");
			}
		}

		private static bool IsArrayEqual(this double[] array, double[] arrayToCompare)
		{
			return array.GetLength(0) == arrayToCompare.GetLength(0);
		}
	}
}
