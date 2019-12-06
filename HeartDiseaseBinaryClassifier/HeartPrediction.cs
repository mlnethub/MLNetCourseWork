using Microsoft.ML.Data;

namespace HeartDiseaseBinaryClassifier
{
    /// <summary>
    /// The HeartPrediction class contains a single heart data prediction.
    /// </summary>
    public class HeartPrediction
    {
        [ColumnName("PredictedLabel")] public bool Prediction;
        public float Probability;
        public float Score;
    }
}