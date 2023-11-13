using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.Data;

namespace Chapter3_2.Models
{
    public class ImageNetData
    {
        [LoadColumn(0)]
        public string ImagePath;

        [LoadColumn(1)]
        public string Label;

        
    }
}