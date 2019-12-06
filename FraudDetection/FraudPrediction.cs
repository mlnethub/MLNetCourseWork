using System;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace FraudDetection
{
    public class FraudPrediction
    {
        [ColumnName("PredictedLabel")] public bool Prediction;
        public float Probability;
        public float Score;
    }
}