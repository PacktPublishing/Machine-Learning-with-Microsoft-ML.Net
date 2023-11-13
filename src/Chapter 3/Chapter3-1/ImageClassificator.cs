using Chapter3_1;
using Microsoft.ML;

public class ImageClassificator
{
    public static void ClassifySingleImage(MLContext mlContext, IDataView data, ITransformer trainedModel)
    {
        PredictionEngine<ModelInput, ModelOutput> predictionEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(trainedModel);

        ModelInput image = mlContext.Data.CreateEnumerable<ModelInput>(data, reuseRowObject: true).First();

        ModelOutput prediction = predictionEngine.Predict(image);

        Console.WriteLine("Hi, I'm classifying a single image");
        OutputPredictionTemplate(prediction);
    }

    public static void ClassifyListOfImages(MLContext mlContext, IDataView data, ITransformer trainedModel)
    {
        IDataView predictionData = trainedModel.Transform(data);

        IEnumerable<ModelOutput> predictions = mlContext.Data.CreateEnumerable<ModelOutput>(predictionData, reuseRowObject: true).Take(10);

        Console.WriteLine("Hi, I'm classifying multiple images");
        foreach (var prediction in predictions)
        {
            OutputPredictionTemplate(prediction);
        }
    }

    private static void OutputPredictionTemplate(ModelOutput prediction)
    {
        string imageName = Path.GetFileName(prediction.ImagePath);
        Console.WriteLine($"Image name: {imageName} | Actual Value: {prediction.Label} | Predicted Value: {prediction.PredictedLabel}");
    }
}