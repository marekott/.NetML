namespace NeuralNetworks.Neurons
{
	public abstract class Neuron
	{
		protected readonly int NeuronPositionFromTop;

		protected Neuron(int neuronPositionFromTop)
		{
			NeuronPositionFromTop = neuronPositionFromTop;
		}
		
		public abstract double ComputeOutput(double[] inputs);
	}
}
