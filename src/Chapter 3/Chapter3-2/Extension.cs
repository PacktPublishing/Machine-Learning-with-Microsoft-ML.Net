using System.Drawing;
using System.Drawing.Drawing2D;
using ObjectDetection.YoloParser;

namespace Chapter3_2;

public static class Extension
{
    /// <summary>
    /// Draw bounding box on image
    /// </summary>
    /// <param name="inputImageLocation"></param>
    /// <param name="outputImageLocation"></param>
    /// <param name="imageName"></param>
    /// <param name="filteredBoundingBoxes"></param>
    public static void DrawBoundingBox(string inputImageLocation, string outputImageLocation, string imageName, IList<YoloBoundingBox> filteredBoundingBoxes)
    {
        Image image = Image.FromFile(Path.Combine(inputImageLocation, imageName));

        var originalImageHeight = image.Height;
        var originalImageWidth = image.Width;

        foreach (var box in filteredBoundingBoxes)
        {
            // Get Bounding Box Dimensions
            var x = (uint)Math.Max(box.Dimensions.X, 0);
            var y = (uint)Math.Max(box.Dimensions.Y, 0);
            var width = (uint)Math.Min(originalImageWidth - x, box.Dimensions.Width);
            var height = (uint)Math.Min(originalImageHeight - y, box.Dimensions.Height);

            // Resize To Image
            x = (uint)originalImageWidth * x / OnnxScorer.ImageNetSettings.imageWidth;
            y = (uint)originalImageHeight * y / OnnxScorer.ImageNetSettings.imageHeight;
            width = (uint)originalImageWidth * width / OnnxScorer.ImageNetSettings.imageWidth;
            height = (uint)originalImageHeight * height / OnnxScorer.ImageNetSettings.imageHeight;

            // Bounding Box Text
            string text = $"{box.Label} ({(box.Confidence * 100).ToString("0")}%)";

            using (Graphics thumbnailGraphic = Graphics.FromImage(image))
            {
                thumbnailGraphic.CompositingQuality = CompositingQuality.HighQuality;
                thumbnailGraphic.SmoothingMode = SmoothingMode.HighQuality;
                thumbnailGraphic.InterpolationMode = InterpolationMode.HighQualityBicubic;

                // Define Text Options
                Font drawFont = new Font("Arial", 12, FontStyle.Bold);
                SizeF size = thumbnailGraphic.MeasureString(text, drawFont);
                SolidBrush fontBrush = new SolidBrush(Color.Black);
                Point atPoint = new Point((int)x, (int)y - (int)size.Height - 1);

                // Define BoundingBox options
                Pen pen = new Pen(box.BoxColor, 3.2f);
                SolidBrush colorBrush = new SolidBrush(box.BoxColor);

                // Draw text on image 
                thumbnailGraphic.FillRectangle(colorBrush, (int)x, (int)(y - size.Height - 1), (int)size.Width, (int)size.Height);
                thumbnailGraphic.DrawString(text, drawFont, fontBrush, atPoint);

                // Draw bounding box on image
                thumbnailGraphic.DrawRectangle(pen, x, y, width, height);
            }
        }

        if (!Directory.Exists(outputImageLocation))
        {
            Directory.CreateDirectory(outputImageLocation);
        }

        image.Save(Path.Combine(outputImageLocation, imageName));
    }

    /// <summary>
    /// Log detected object
    /// </summary>
    /// <param name="imageName"></param>
    /// <param name="boundingBoxes"></param>
    public static void LogDetectedObjects(string imageName, IList<YoloBoundingBox> boundingBoxes)
    {
        Console.WriteLine($".....The objects in the image {imageName} are detected as below....");

        foreach (var box in boundingBoxes)
        {
            Console.WriteLine($"{box.Label} and its Confidence score: {box.Confidence}");
        }

        Console.WriteLine("");
    }

    /// <summary>
    /// Get full path to the model file
    /// </summary>
    /// <param name="relativePath"></param>
    /// <returns></returns>
    public static string GetAbsolutePath(string relativePath)
    {
        FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
        string assemblyFolderPath = _dataRoot.Directory.FullName;

        string fullPath = Path.Combine(assemblyFolderPath, relativePath);

        return fullPath;
    }
}
