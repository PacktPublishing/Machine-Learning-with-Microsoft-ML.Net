using Chapter3_2.Models;

namespace Chapter3_2;

public static class ImageLoader
{
    public static IEnumerable<ImageNetData> ReadFromFile(string imageFolder)
        {
            return Directory
                .GetFiles(imageFolder)
                .Where(filePath => Path.GetExtension(filePath) != ".md")
                .Select(filePath => new ImageNetData { ImagePath = filePath, Label = Path.GetFileName(filePath) });
        }
}
