using System;

namespace NeuralNetworks.Network
{	
	public class WrongSourceFileLengthException : Exception
	{
		public WrongSourceFileLengthException()
		{
		}

		public WrongSourceFileLengthException(int expectedLength, int fileLength) : base(
			$"Wrong file length. Expected length {expectedLength}, actual: {fileLength}")
		{
		}

		public WrongSourceFileLengthException(string message, Exception inner) : base(message, inner)
		{
		}
	}

	public class WeightsNotInitializedException : Exception
	{
		public WeightsNotInitializedException() : base("Weights have to be initialized before using ComputeOutput or BackPropagationTrain methods.")
		{
		}

		public WeightsNotInitializedException(string message) : base(message)
		{
		}

		public WeightsNotInitializedException(string message, Exception inner) : base(message, inner)
		{
		}
	}
}
