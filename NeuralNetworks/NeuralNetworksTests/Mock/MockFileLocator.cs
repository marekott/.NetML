using System.IO;
using FileDeserializer.CSV;

namespace NeuralNetworksTests.Mock
{
	internal class MockFileLocator : IFileLocator
	{
		private static string _filePath;

		public MockFileLocator(string fileName)
		{
			var currentDirectory = Directory.GetCurrentDirectory();
			var directory = new DirectoryInfo(currentDirectory);
			_filePath = Path.Combine(directory.FullName, fileName);
		}

		public string GetFileLocation()
		{
			return _filePath;
		}
	}
}
