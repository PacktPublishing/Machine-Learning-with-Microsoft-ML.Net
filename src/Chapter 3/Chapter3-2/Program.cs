using Chapter3_2;
using Chapter3_2.Models;
using Microsoft.ML;
using System.Drawing;
using System.Drawing.Drawing2D;
using ObjectDetection.YoloParser;

var assetsRelativePath = @"../../../Assets";
string assetsPath = Extension.GetAbsolutePath(assetsRelativePath);
var modelFilePath = Path.Combine(assetsPath, "Model", "TinyYolo2_model.onnx");
var imagesFolder = Path.Combine(assetsPath, "images");
var outputFolder = Path.Combine(assetsPath, "images", "output");

// Initialize MLContext
MLContext mlContext = new MLContext();

try
{
    // Load Data                    
    IEnumerable<ImageNetData> images = ImageLoader.ReadFromFile(imagesFolder);
    IDataView imageDataView = mlContext.Data.LoadFromEnumerable(images);

    // Create instance of model scorer
    var modelScorer = new OnnxScorer(imagesFolder, modelFilePath, mlContext);

    // Use model to score data
    IEnumerable<float[]> probabilities = modelScorer.Score(imageDataView);

    // Post-process model output
    YoloOutputParser parser = new YoloOutputParser();

    var boundingBoxes =
        probabilities
        .Select(probability => parser.ParseOutputs(probability))
        .Select(boxes => parser.FilterBoundingBoxes(boxes, 5, .5F));

    // Draw bounding boxes for detected objects in each of the images
    for (var i = 0; i < images.Count(); i++)
    {
        string imageFileName = images.ElementAt(i).Label;
        IList<YoloBoundingBox> detectedObjects = boundingBoxes.ElementAt(i);

        Extension.DrawBoundingBox(imagesFolder, outputFolder, imageFileName, detectedObjects);

        Extension.LogDetectedObjects(imageFileName, detectedObjects);
    }
}
catch (Exception ex)
{
    Console.WriteLine(ex.ToString());
}

Console.WriteLine("========= End of Process..Hit any Key ========");



