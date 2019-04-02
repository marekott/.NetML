namespace NeuralNetworks.Neurons
{
	public class NeuronInputLayer : Neuron
	{
		private readonly int _numberOfInputs;

		/// <summary>
		/// </summary>
		/// <param name="numberOfInputs">counting from 1</param>
		/// <param name="neuronPositionFromTop">counting from 0</param>
		public NeuronInputLayer(int neuronPositionFromTop, int numberOfInputs) : base(neuronPositionFromTop)
		{
			_numberOfInputs = numberOfInputs;
		}

		public override double ComputeOutput(double[] inputs)
		{
			double inputSum = 0;

			for (int i = _numberOfInputs * NeuronPositionFromTop; i <= _numberOfInputs * (NeuronPositionFromTop + 1) - 1; i++) //for numberOfInputs=5 and neuronPositionFromTop=2 takes arguments from inputs(with Length=10) with indexes from <5:9>)
			{
				inputSum += inputs[i];
			}

			return inputSum;
		}
	}
}
