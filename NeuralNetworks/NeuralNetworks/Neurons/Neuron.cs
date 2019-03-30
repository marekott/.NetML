namespace NeuralNetworks.Neurons
{
	public abstract class Neuron
	{
		protected readonly int NeuronPositionFromTop;

		protected Neuron(int neuronPositionFromTop)
		{
			NeuronPositionFromTop = neuronPositionFromTop;
		}
		//TODO Prawdopodobnie bias do wywalenia, bo będzie obliczany w klasie sieci, zweryfikuj
		public abstract double ComputeOutput(double[] inputs, double[] biases = null);
	}
}
