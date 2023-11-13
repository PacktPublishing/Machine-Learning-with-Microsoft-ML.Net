using Chapter3_1;
using Microsoft.ML;
using Microsoft.ML.Data;

public static class ImageLoader
{
    /// <summary>
    /// Load images from directory.
    /// </summary>
    /// <param name="folder">Folder name</param>
    /// <param name="useFolderNameAsLabel">Label</param>
    /// <returns></returns>
    public static IEnumerable<Image> LoadImagesFromAllDirectory(string folder)
    {
        var files = Directory.GetFiles(folder, "*",
            searchOption: SearchOption.AllDirectories);

        return GetImagesInformationFromFilesName(files, true);
       
    }

    private static List<Image> GetImagesInformationFromFilesName(string[] files, bool useFolderNameAsLabel = true)
    {
        var images = new List<Image>();

        foreach (var file in files)
        {
            if ((Path.GetExtension(file) != ".jpg") && (Path.GetExtension(file) != ".png"))
                continue;

            var filename = Path.GetFileName(file);

            if (useFolderNameAsLabel)
                filename = Directory.GetParent(file).Name ?? string.Empty;
            else
            {
                for (int index = 0; index < filename.Length; index++)
                {
                    if (!char.IsLetter(filename[index]))
                    {
                        filename = filename.Substring(0, index);
                        break;
                    }
                }
            }

            images.Add(new Image()
            {
                ImagePath = file,
                Label = filename
            });
        }
        return images;
        
    }
}