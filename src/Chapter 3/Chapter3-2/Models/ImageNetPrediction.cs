using Microsoft.ML.Data;

namespace Chapter3_2.Models
{
    public class ImageNetPrediction
    {
        [ColumnName("grid")]
        public float[] PredictedLabels;
    }
}