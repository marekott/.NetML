using System;

namespace NeuralNetworks.Network
{	//TODO Refactor wyjatkow
	public class WrongSourceFileLengthException : Exception
	{
		public WrongSourceFileLengthException(int expectedLength, int fileLength) : base(
			$"Wrong file length. Expected length {expectedLength}, actual: {fileLength}")
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
