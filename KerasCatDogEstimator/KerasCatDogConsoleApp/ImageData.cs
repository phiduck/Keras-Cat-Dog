using Microsoft.ML.Data;

namespace KerasCatDogConsoleApp
{
    public class ImageData
    {
        [LoadColumn(0)]
        public string ImagePath;
    }
}
