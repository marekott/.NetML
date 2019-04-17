using System;

namespace NeuralNetworks.Network
{
	public class WrongSourceFileLengthException : Exception
	{
		public WrongSourceFileLengthException(int expectedLength, int fileLength) : base(
			$"File is to short. Expected length {expectedLength}, actual: {fileLength}")
		{

		}
	}

	public class WeightsNotInitializedException : Exception
	{
		public WeightsNotInitializedException() : base("Weights have to be initialized before using ComputeOutput or Train methods.")
		{

		}
	}
}
