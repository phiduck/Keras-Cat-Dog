# Image Classification - Working with a model trained using Keras

| ML.NET version | API type          | Status                        | App Type    | Data type | Scenario            | ML Task                   | Algorithms                  |
|----------------|-------------------|-------------------------------|-------------|-----------|---------------------|---------------------------|-----------------------------|
| v1.5.4           | Dynamic API | up-to-date | Console app | Images | Images classification | Keras cat/dog example classifier  | DeepLearning model |


## Creating the Keras Model
The Keras model was created by following the steps in this [keras example](https://keras.io/examples/vision/image_classification_from_scratch/) and saving the model to disk using the following command

```Python
  tf.keras.models.save_model(model, "KerasModel", save_format='tf', include_optimizer=False)
```

The outputted model is stored in KerasModel in the root directory of the project.

## Test Data
We have downloaded an image of a cat and a dog from wikipedia in order to test our Keras model and they are stored in TestData in the root directory of the project.

## Inspecting the model

In order to figure out the input and outputs to our model we use

```
  saved_model_cli show --dir assets --all
```
which should produce an output similar to 

```
signature_def['serving_default']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['input_1'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 180, 180, 3)
        name: serving_default_input_1:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['dense'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 1)
        name: StatefulPartitionedCall:0
  Method name is: tensorflow/serving/predict
```

From the output we can infer that the shape of the input is (-1, 180, 180, 3). The first number is the batch size, in our case -1 to indicate that
None was specified. The remaining figures are 180, 180, 3 indicating that we are expecting an image that has size 180x180 pixels with 3 RGB values.

The input name "serving_default_input_1" and output name "StatefulPartitionedCall" will be used later in order to put the model into our pipeline

## Loading the model into ML.NET

### Creating data objects

We first create an object for storing the image files 

```csharp
  public class ImageData
  {
      [LoadColumn(0)]
      public string ImagePath;
  }
```

Next, we create an object for storing the settings 
```csharp
using System;
using System.Collections.Generic;
using System.Text;

namespace KerasCatDogConsoleApp
{
  public class ModelRunSettings
  {
      public string ModelLocation { get; set; }
      public string DataLocation { get; set; }
      public int ImageHeight { get; set; }
      public int ImageWidth { get; set; }
      public string KerasModelInputColumnName { get; set; }
      public string KerasModelOutputColumnName { get; set; }
  }
}
```

ModelLocation is the directory we created when we saved our Keras model. For convenience we copy this into our project root folder.

DataLocation is the directory containing new images to be classified.

ImageHeight, ImageWidth, KerasModelInputColumnName and KerasModelOutputColumnName store the values we found when we ran `saved_model_cli` on our saved model.

### Creating a pipeline
Out pipeline will use several estimators for loading, resizing and extracting pixels as well as a TensorFlowEstimator to load in the Keras
model. These estimators are put into a method `CreatePipeline` in a new class `ModelBuilder`.

```csharp
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
```

### Using the model to make a prediction
First we begin by setting up our model settings

```csharp
  var settings = new ModelRunSettings()
  {
      ModelLocation = Path.GetFullPath("../../../KerasModel"),
      DataLocation = Path.GetFullPath("../../../TestData"),
      ImageHeight = 180,
      ImageWidth = 180,
      KerasModelInputColumnName = "serving_default_input_1",
      KerasModelOutputColumnName = "StatefulPartitionedCall",
  };
```

Next, we create a new context our pipeline and a new transformer to be used on our test data. We pass an empty data view into the fit method since we are not fitting anything in our pipeline.

```csharp
  MLContext context = new MLContext();
  EstimatorChain<TensorFlowTransformer> pipeline = ModelBuilder.CreatePipeline(context, settings);
  ITransformer transformer = pipeline.Fit(context.Data.LoadFromEnumerable(new List<ImageData>()));
```

To make a prediction with the transformer, we load in some test data, call transform and extract the scores for the test data

```csharp
  var testImages = new ImageData[] {
      new ImageData() { ImagePath = Path.Cosmbine(settings.DataLocation, "cat.jpeg") },
      new ImageData() { ImagePath = Path.Combine(settings.DataLocation, "dog.jpg") }
  };

  IDataView testData = context.Data.LoadFromEnumerable<ImageData>(testImages);
  IDataView predictions = transformer.Transform(testData);
  float[][] scores = predictions.GetColumn<float[]>(settings.KerasModelOutputColumnName).ToArray();
```

Finally we output the scores

```csharp
  for (int i = 0; i < testImages.Length; i++)
  {
      float score = scores[i][0];
      Console.WriteLine($"Image {testImages[i].ImagePath } was {100 * (1 - score) : 0.##}% cat and { 100 * score :0.##}% dog");
  }
```

### Citation
Prediction images are taken from 
> *Wikimedia Commons, the free media repository.* Retrieved Decemeber 20, 2020 from https://commons.wikimedia.org/w/index.php?title=Main_Page&oldid=313158208.

Further information can be found in Wikipedia.md in the TestData directory.
