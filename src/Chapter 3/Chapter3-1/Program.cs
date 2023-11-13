using Chapter3_1;
using Microsoft.ML;
using Microsoft.ML.Vision;
using static Microsoft.ML.DataOperationsCatalog;

var experimentDirectory = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../"));
var workspaceRelativePath = Path.Combine(experimentDirectory, "workspace");
var assetsRelativePath = Path.Combine(experimentDirectory, "assets");

MLContext mlContext = new MLContext();

IEnumerable<Image> images = ImageLoader.LoadImagesFromAllDirectory(folder: assetsRelativePath);

IDataView imageData = mlContext.Data.LoadFromEnumerable(images);

IDataView randomData = mlContext.Data.ShuffleRows(imageData);

var preprocessingPipeline = mlContext.Transforms.Conversion.MapValueToKey(
        inputColumnName: "Label",
        outputColumnName: "LabelAsKey")
    .Append(mlContext.Transforms.LoadRawImageBytes(
        outputColumnName: "Image",
        imageFolder: assetsRelativePath,
        inputColumnName: "ImagePath"));

IDataView preProcessedData = preprocessingPipeline
                    .Fit(randomData)
                    .Transform(randomData);

TrainTestData trainingDataSplitted = mlContext.Data.TrainTestSplit(data: preProcessedData, testFraction: 0.2);
TrainTestData validationTestSplit = mlContext.Data.TrainTestSplit(trainingDataSplitted.TestSet);

IDataView trainSet = trainingDataSplitted.TrainSet;
IDataView validationSet = validationTestSplit.TrainSet;
IDataView testSet = validationTestSplit.TestSet;

var classifierOptions = new ImageClassificationTrainer.Options()
{
    FeatureColumnName = "Image",
    LabelColumnName = "LabelAsKey",
    ValidationSet = validationSet,
    Arch = ImageClassificationTrainer.Architecture.ResnetV2101,
    MetricsCallback = (metrics) => Console.WriteLine(metrics),
    TestOnTrainSet = false,
    ReuseTrainSetBottleneckCachedValues = true,
    ReuseValidationSetBottleneckCachedValues = true
};

var trainingPipeline = mlContext.MulticlassClassification.Trainers.ImageClassification(classifierOptions)
    .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

ITransformer fitTrainedModel = trainingPipeline.Fit(trainSet);

ImageClassificator.ClassifySingleImage(mlContext, testSet, fitTrainedModel);

ImageClassificator.ClassifyListOfImages(mlContext, testSet, fitTrainedModel);

Console.ReadKey();

