using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace KerasCatDogConsoleApp
{
    public class Program
    {
        static void Main()
        {
            var settings = new ModelRunSettings()
            {
                ModelLocation = Path.GetFullPath("../../../KerasModel"),
                DataLocation = Path.GetFullPath("../../../TestData"),
                ImageHeight = 180,
                ImageWidth = 180,
                KerasModelInputColumnName = "serving_default_input_1",
                KerasModelOutputColumnName = "StatefulPartitionedCall",
            };

            MLContext context = new MLContext();
            EstimatorChain<TensorFlowTransformer> pipeline = ModelBuilder.CreatePipeline(context, settings);

            // We pass an empty data view into the fit method since we are not fitting anything in our pipeline.
            ITransformer transformer = pipeline.Fit(context.Data.LoadFromEnumerable(new List<ImageData>()));

            var testImages = new ImageData[] {
                new ImageData() { ImagePath = Path.Combine(settings.DataLocation, "cat.jpeg") },
                new ImageData() { ImagePath = Path.Combine(settings.DataLocation, "dog.jpg") }
            };

            IDataView testData = context.Data.LoadFromEnumerable<ImageData>(testImages);
            IDataView predictions = transformer.Transform(testData);
            float[][] scores = predictions.GetColumn<float[]>(settings.KerasModelOutputColumnName).ToArray();

            for (int i = 0; i < testImages.Length; i++)
            {
                float score = scores[i][0];
                Console.WriteLine($"Image {testImages[i].ImagePath } was {100 * (1 - score) : 0.##}% cat and { 100 * score :0.##}% dog");
            }
        }
    }
}
