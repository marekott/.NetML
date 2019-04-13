using System;
using System.Linq;

namespace NeuralNetworks.Neurons
{
	public class NeuronOutputLayer : Neuron //TODO potrzeba innej funkcji aktywacji bo ta dla większej liczby wyjść niż 4 żadko zwróci coś większego niż 0,5 więc zaokrąglenie uwali do 0, zmiana bd wtedy w konstruktrze sieci żeby szło wybrac funkcję aktywacji
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
