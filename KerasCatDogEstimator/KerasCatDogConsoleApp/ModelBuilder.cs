using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Image;

namespace KerasCatDogConsoleApp
{
    public static class ModelBuilder
    {
        /// <summary>
        /// Creates a pipeline that loads in an image of any size and resizes it before passing it into the model saved from Keras
        /// </summary>
        /// <param name="context">Machine learning context</param>
        /// <param name="settings">Specification detailing image size and for running model</param>
        /// <returns></returns>
        public static EstimatorChain<TensorFlowTransformer> CreatePipeline(MLContext context, ModelRunSettings settings)
        {
            ImageLoadingEstimator imageLoader = context.Transforms.LoadImages(
                outputColumnName: "originalImage",
                imageFolder: settings.DataLocation,
                inputColumnName: nameof(ImageData.ImagePath));

            ImageResizingEstimator imageResizer = context.Transforms.ResizeImages(
                outputColumnName: "resizedImage",
                imageWidth: settings.ImageWidth,
                imageHeight: settings.ImageHeight,
                inputColumnName: "originalImage");

            ImagePixelExtractingEstimator pixelExtractor = context.Transforms.ExtractPixels(
                outputColumnName: settings.KerasModelInputColumnName,
                inputColumnName: "resizedImage");

            TensorFlowModel model = context.Model.LoadTensorFlowModel(settings.ModelLocation);

            TensorFlowEstimator estimator = model.ScoreTensorFlowModel(settings.KerasModelOutputColumnName, settings.KerasModelInputColumnName);

            return imageLoader.Append(imageResizer).Append(pixelExtractor).Append(estimator);
        }

    }
}
